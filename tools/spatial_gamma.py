"""
Attention-Guided Spatial Gamma Map (v5).

Phase 1: FirstRepAttentionStore captures rep 0, all enabled layers (default 5).
Phase 2: Per-scale gamma maps from multi-layer averaged attention + sigmoid schedule.
Large scales beyond max_seq_len are upsampled from the highest captured resolution.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from infinity.models.attention_control import AttentionStore


class FirstRepAttentionStore(AttentionStore):
    """Stores attention from the first repetition only, all enabled layers.
    When cfg>1, batch dim is [cond, uncond] — only store cond ([:1])."""

    def forward(self, attn, text_len: int, text_offset: int):
        L_q = attn.shape[-2]
        if not self.cache_attn or L_q > self.max_seq_len:
            return attn

        scale_ind = getattr(self, '_current_scale_ind', self.cur_step)
        key = (scale_ind, self.cur_att_layer)

        if key not in self.step_store:
            vt_attn = attn[:1, :, :, text_offset: text_offset + text_len]
            self.step_store[key] = vt_attn.detach()

        return attn


def find_token_indices(tokenizer, prompt, word):
    tokens = tokenizer(text=[prompt], max_length=512, padding="max_length",
                       truncation=True, return_tensors="pt")
    ids = tokens["input_ids"][0].tolist()
    mask = tokens["attention_mask"][0]
    actual_len = int(mask.sum().item())
    ids = ids[:actual_len]

    word_tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")
    word_ids = word_tokens["input_ids"][0].tolist()
    if word_ids:
        for i in range(len(ids) - len(word_ids) + 1):
            if ids[i:i + len(word_ids)] == word_ids:
                return list(range(i, i + len(word_ids)))
    decoded = [tokenizer.decode([tid]).lower().strip() for tid in ids]
    w = word.lower().strip()
    for i, tok in enumerate(decoded):
        if w == tok.strip():
            return [i]
    return []


def _extract_attn_1d(step_store, si, token_indices, num_tokens):
    """Extract 1D attention for target token(s) at scale si, averaged across all stored layers."""
    layer_maps = []
    for layer_id in range(100):
        key = (si, layer_id)
        if key not in step_store:
            continue
        vt = step_store[key]  # [B, H, L_q, text_len]
        L_q = vt.shape[2]
        if L_q < num_tokens:
            continue
        vt_use = vt[:, :, :num_tokens, :]
        avg = vt_use.float().mean(dim=(0, 1))  # [num_tokens, text_len]
        word_attn = avg[:, token_indices].sum(dim=-1)  # [num_tokens]
        layer_maps.append(word_attn.cpu())

    if not layer_maps:
        return None
    return torch.stack(layer_maps).mean(dim=0)


def _normalize(t):
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-8:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo)


def _interpolate_2d(src_map, src_h, src_w, tgt_h, tgt_w):
    inp = src_map.view(1, 1, src_h, src_w).float()
    out = F.interpolate(inp, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
    return out.view(-1)


def _interpolate_3d(src_map, src_pn, src_h, src_w, tgt_pn, tgt_h, tgt_w):
    inp = src_map.view(1, 1, src_pn, src_h, src_w).float()
    out = F.interpolate(inp, size=(tgt_pn, tgt_h, tgt_w), mode='trilinear', align_corners=False)
    return out.view(-1)


def _upsample_attn(src_attn, src_shape, tgt_shape):
    src_pn, src_h, src_w = src_shape
    tgt_pn, tgt_h, tgt_w = tgt_shape
    if src_pn == 1 and tgt_pn == 1:
        return _interpolate_2d(src_attn, src_h, src_w, tgt_h, tgt_w)
    if src_pn == tgt_pn:
        frames = src_attn.view(src_pn, src_h, src_w)
        out = []
        for f in range(src_pn):
            out.append(_interpolate_2d(frames[f], src_h, src_w, tgt_h, tgt_w).view(tgt_h, tgt_w))
        return torch.stack(out).view(-1)
    return _interpolate_3d(src_attn, src_pn, src_h, src_w, tgt_pn, tgt_h, tgt_w)


def _attn_sigmoid(attn_normed, center=0.5, width=0.1):
    """Map [0,1] attention through a sigmoid for sharper fg/bg separation."""
    x = (attn_normed - center) / max(width, 1e-6)
    return 1.0 / (1.0 + torch.exp(-x))


def _attn_to_gamma(attn_normed, gamma_high, gamma_low, attn_center=0.5, attn_width=0.1):
    """attn=0 (background) -> gamma_high, attn=1 (foreground) -> gamma_low."""
    s = _attn_sigmoid(attn_normed, attn_center, attn_width)
    return gamma_high + (gamma_low - gamma_high) * s


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _scale_gamma_range(tower_index, num_tower_scales,
                       gamma_start, gamma_end_fg, gamma_end_bg,
                       transition_scale, transition_width):
    """Per-scale gamma envelope via sigmoid schedule."""
    t = tower_index / max(num_tower_scales - 1, 1)
    t_center = transition_scale / max(num_tower_scales - 1, 1)
    s = _sigmoid((t - t_center) / max(transition_width, 1e-6))

    gamma_high = gamma_start + (gamma_end_bg - gamma_start) * s
    gamma_low = gamma_start + (gamma_end_fg - gamma_start) * s

    return gamma_low, gamma_high


def build_spatial_gamma_maps(
    attn_store,
    tokenizer,
    prompt,
    target_word,
    scale_schedule,
    image_scale_repetition,
    video_scale_repetition,
    gamma_start=2.0,
    gamma_end_fg=1.60,
    gamma_end_bg=1.78,
    image_transition=5,
    video_transition=4,
    transition_width=0.06,
    attn_center=0.5,
    attn_width=0.1,
    clip2_mode=False,
):
    """
    Build per-step_ptr gamma maps from rep-0 multi-layer attention + sigmoid schedule.

    Returns: List[Tensor] aligned with idx_Blv_cache_list (one entry per
    (scale, repetition) step).  Each tensor: [num_spatial_tokens], float32, cpu.
    """
    token_indices = find_token_indices(tokenizer, prompt, target_word)
    if not token_indices:
        print(f"[SpatialGamma] WARNING: '{target_word}' not found in prompt")
        token_indices = None

    scales_in_one_clip = len(image_scale_repetition)

    # --- Extract attention per scale (rep 0, multi-layer average) ---
    scale_attn = {}
    for si, (pn_t, h, w) in enumerate(scale_schedule):
        if token_indices is None:
            continue
        num_tokens = pn_t * h * w
        attn_1d = _extract_attn_1d(attn_store.step_store, si, token_indices, num_tokens)
        if attn_1d is not None:
            scale_attn[si] = _normalize(attn_1d)
            n_layers = sum(1 for lid in range(100) if (si, lid) in attn_store.step_store)
            print(f"[SpatialGamma] scale {si:2d} ({pn_t}x{h}x{w}={num_tokens}t): "
                  f"extracted {n_layers}L, range=[{attn_1d.min():.4f}, {attn_1d.max():.4f}]")

    best_image_si = max((si for si in scale_attn if scale_schedule[si][0] == 1), default=None)
    best_video_si = max((si for si in scale_attn if scale_schedule[si][0] > 1), default=None)
    print(f"[SpatialGamma] Extracted {len(scale_attn)} scales. "
          f"best_image={best_image_si}, best_video={best_video_si}")

    # --- Upsample for scales without attention ---
    for si, (pn_t, h, w) in enumerate(scale_schedule):
        if si in scale_attn:
            continue
        is_video = pn_t > 1
        num_tokens = pn_t * h * w
        ref_si = best_video_si if is_video else best_image_si
        if ref_si is None:
            ref_si = best_image_si if is_video else best_video_si
        if ref_si is not None:
            upsampled = _upsample_attn(scale_attn[ref_si],
                                       scale_schedule[ref_si], (pn_t, h, w))
            scale_attn[si] = _normalize(upsampled)
            print(f"[SpatialGamma] scale {si:2d} ({pn_t}x{h}x{w}={num_tokens}t): "
                  f"upsampled from scale {ref_si}")
        else:
            scale_attn[si] = torch.zeros(num_tokens)

    # --- Print sigmoid schedule ---
    print(f"[SpatialGamma] Sigmoid schedule: gamma_start={gamma_start:.2f}, "
          f"gamma_end_fg={gamma_end_fg:.2f}, gamma_end_bg={gamma_end_bg:.2f}, "
          f"image_transition={image_transition}, video_transition={video_transition}, "
          f"width={transition_width:.3f}, attn_center={attn_center:.2f}, attn_width={attn_width:.2f}")

    # --- Build per-step_ptr gamma maps ---
    gamma_map_list = []

    for si, (pn_t, h, w) in enumerate(scale_schedule):
        if clip2_mode:
            is_video = True
            tower_index = si
            t_scale = video_transition
            reps = int(video_scale_repetition[si % len(video_scale_repetition)])
        else:
            is_video = si >= scales_in_one_clip
            tower_index = si - scales_in_one_clip if is_video else si
            t_scale = video_transition if is_video else image_transition
            rel_si = si % scales_in_one_clip
            if si < scales_in_one_clip:
                reps = int(image_scale_repetition[rel_si])
            else:
                reps = int(video_scale_repetition[rel_si])

        num_tower_scales = len(video_scale_repetition) if clip2_mode else scales_in_one_clip
        g_low, g_high = _scale_gamma_range(
            tower_index, num_tower_scales,
            gamma_start, gamma_end_fg, gamma_end_bg,
            t_scale, transition_width,
        )

        attn_for_scale = scale_attn.get(si, torch.zeros(pn_t * h * w))
        gamma_map = _attn_to_gamma(attn_for_scale, g_high, g_low, attn_center, attn_width)

        for rep_idx in range(reps):
            gamma_map_list.append(gamma_map)

        tower_name = "video" if is_video else "image"
        print(f"[SpatialGamma] scale {si:2d} ({tower_name} idx={tower_index:2d}) "
              f"{reps} reps, gamma=[{g_low:.3f}, {g_high:.3f}], "
              f"mean≈{(g_low+g_high)/2:.3f}")

    return gamma_map_list
