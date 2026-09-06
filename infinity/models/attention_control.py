import abc
from typing import List, Dict, Tuple

import torch


class AttentionControlBase(abc.ABC):
    """
    Base class for attention control in InfinityStar.

    Operates on the visual->text sub-matrix of the full attention map
    (the semantic equivalent of cross-attention in joint self-attention).

    Two-phase usage:
      1. Inversion: run with source prompt, controller caches attention maps.
      2. Editing:   call start_edit(), run with edit prompt, controller
                    replaces cached attention for shared tokens.
    """

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, text_len: int, text_offset: int):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, text_len: int = 0, text_offset: int = 0):
        if text_len <= 0:
            return attn
        attn = self.forward(attn, text_len, text_offset)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def start_edit(self):
        pass

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControlBase):
    def forward(self, attn, text_len: int, text_offset: int):
        return attn


class AttentionStore(AttentionControlBase):
    """Caches the visual->text attention sub-matrix at each step/layer."""

    def forward(self, attn, text_len: int, text_offset: int):
        L_q = attn.shape[-2]
        if self.cache_attn and L_q <= self.max_seq_len:
            vt_attn = attn[:, :, :, text_offset : text_offset + text_len]
            # Use scale_ind (set by basic.py) instead of cur_step for keying
            scale_ind = getattr(self, '_current_scale_ind', self.cur_step)
            key = (scale_ind, self.cur_att_layer)
            if key in self.step_store:
                # Accumulate across repetitions: cat along L_q dim
                self.step_store[key] = torch.cat([self.step_store[key], vt_attn.detach()], dim=2)
            else:
                self.step_store[key] = vt_attn.detach()
        return attn

    def start_edit(self):
        self.cache_attn = False
        self.cur_step = 0
        self.cur_att_layer = 0

    def reset(self):
        super().reset()
        self.step_store = {}
        self.cache_attn = True

    def __init__(self, max_seq_len: int = 512):
        super().__init__()
        self.step_store: Dict[Tuple[int, int], torch.Tensor] = {}
        self.cache_attn = True
        self.max_seq_len = max_seq_len


class AttentionRefine(AttentionStore):
    """
    During editing, replaces the visual->text attention for shared tokens
    with cached values from the inversion phase.

    mapper[i] = j means edit token i maps to source token j.  -1 = no mapping.
    alphas[i] = blend weight: 1.0 = fully cached, 0.0 = fully new.
    """

    def forward(self, attn, text_len: int, text_offset: int):
        if self.cache_attn:
            return super().forward(attn, text_len, text_offset)

        key = (self.cur_step, self.cur_att_layer)
        if key not in self.step_store:
            return attn

        attn_base_vt = self.step_store[key]
        L_q = attn.shape[-2]
        L_q_base = attn_base_vt.shape[-2]
        num_pix = min(L_q, L_q_base)

        if num_pix <= 0:
            return attn

        base_text_len = attn_base_vt.shape[-1]
        new_attn = attn.clone()

        for edit_idx, (base_idx, alpha) in enumerate(zip(self.mapper, self.alphas)):
            if base_idx < 0 or edit_idx >= text_len or base_idx >= base_text_len:
                continue
            if alpha <= 0:
                continue
            base_col = attn_base_vt[:, :, :num_pix, base_idx]
            edit_col_pos = text_offset + edit_idx
            if edit_col_pos >= attn.shape[-1]:
                continue
            edit_col = attn[:, :, :num_pix, edit_col_pos]
            new_attn[:, :, :num_pix, edit_col_pos] = alpha * base_col + (1 - alpha) * edit_col

        return new_attn

    def __init__(self, mapper: List[int], alphas: List[float], max_seq_len: int = 512):
        super().__init__(max_seq_len=max_seq_len)
        self.mapper = mapper
        self.alphas = alphas


def build_token_mapping(
    tokenizer,
    src_prompt: str,
    edit_prompt: str,
    word_mapping: Dict[str, str],
) -> Tuple[List[int], List[float]]:
    """
    Build per-token mapper and alpha arrays for AttentionRefine.

    Returns:
        mapper: mapper[i] = j means edit token i maps to source token j; -1 = new token.
        alphas: 1.0 for shared tokens, 0.0 for changed tokens.
    """

    def tokenize(prompt):
        tokens = tokenizer(
            text=[prompt], max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids = tokens["input_ids"][0].tolist()
        mask = tokens["attention_mask"][0]
        actual_len = int(mask.sum().item())
        decoded = [tokenizer.decode([tid]).lower().strip() for tid in ids[:actual_len]]
        return ids[:actual_len], decoded, actual_len

    def find_token_indices(word, token_ids, decoded):
        word_tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")
        word_ids = word_tokens["input_ids"][0].tolist()
        if word_ids:
            for i in range(len(token_ids) - len(word_ids) + 1):
                if token_ids[i : i + len(word_ids)] == word_ids:
                    return list(range(i, i + len(word_ids)))
        w = word.lower().strip()
        for i, tok in enumerate(decoded):
            if w == tok.strip():
                return [i]
        return []

    src_ids, src_decoded, src_len = tokenize(src_prompt)
    edit_ids, edit_decoded, edit_len = tokenize(edit_prompt)

    changed_src_indices = set()
    changed_edit_indices = set()
    edit_to_src_map = {}

    for src_word, edit_word in word_mapping.items():
        si = find_token_indices(src_word, src_ids, src_decoded)
        ei = find_token_indices(edit_word, edit_ids, edit_decoded)
        if si and ei:
            for idx in si:
                changed_src_indices.add(idx)
            for idx in ei:
                changed_edit_indices.add(idx)
            for j in range(min(len(si), len(ei))):
                edit_to_src_map[ei[j]] = si[j]
            print(f"[AttnCtrl] Token mapping (P2P replace): '{src_word}' indices {si} -> '{edit_word}' indices {ei}")
        else:
            print(f"[AttnCtrl] WARNING: could not find mapping for '{src_word}' -> '{edit_word}'")

    mapper = []
    alphas = []
    src_ptr = 0
    edit_ptr = 0
    while edit_ptr < edit_len:
        if edit_ptr in changed_edit_indices:
            if edit_ptr in edit_to_src_map:
                mapper.append(edit_to_src_map[edit_ptr])
                alphas.append(1.0)
            else:
                mapper.append(-1)
                alphas.append(0.0)
            edit_ptr += 1
            continue
        if src_ptr < src_len and src_ptr in changed_src_indices:
            src_ptr += 1
            continue
        if src_ptr < src_len:
            mapper.append(src_ptr)
            alphas.append(1.0)
            src_ptr += 1
        else:
            mapper.append(-1)
            alphas.append(0.0)
        edit_ptr += 1

    return mapper, alphas


def make_controller(
    tokenizer,
    prompts: List[str],
    word_mapping: Dict[str, str],
    max_seq_len: int = 512,
) -> AttentionRefine:
    src_prompt, edit_prompt = prompts[0], prompts[1]
    mapper, alphas = build_token_mapping(tokenizer, src_prompt, edit_prompt, word_mapping)
    return AttentionRefine(mapper=mapper, alphas=alphas, max_seq_len=max_seq_len)


def register_attention_control(model, controller, max_seq_len: int = 512, min_seq_len: int = 0, max_layers: int = None, scale_whitelist: set = None):
    """
    Register the controller on SelfAttention modules in the model.
    max_layers: only enable on the first N modules (to avoid OOM on later layers with large sequences).
    """
    count = 0
    enabled = 0
    for m in model.modules():
        if hasattr(m, "use_ptp_debug"):
            count += 1
            if max_layers is not None and enabled >= max_layers:
                continue
            m.use_ptp_debug = True
            m._max_debug_seq_len = max_seq_len
            m._min_debug_seq_len = min_seq_len
            m._attn_scale_whitelist = scale_whitelist
            m.ptp_controller = controller
            enabled += 1
    controller.num_att_layers = enabled
    print(f"[AttnCtrl] Registered controller on {enabled}/{count} SelfAttention modules (min_seq_len={min_seq_len}, max_seq_len={max_seq_len}, max_layers={max_layers}, scale_whitelist={scale_whitelist})")


def reset_attention_control(model):
    """Remove controller and disable debug path on all modules."""
    for m in model.modules():
        if hasattr(m, "use_ptp_debug"):
            m.use_ptp_debug = False
        if hasattr(m, "ptp_controller"):
            delattr(m, "ptp_controller")
