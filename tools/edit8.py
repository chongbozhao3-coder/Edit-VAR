# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import sys
import time
import numpy as np
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args
sys.path.append(osp.dirname(osp.dirname(__file__)))
from infinity.utils.video_decoder import EncodedVideoDecord

from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import video_vae_model
from tools.run_infinity import read_video_as_frames
from infinity.models.videovae.modules.quantizer.multiscale_bsq_tp_absorb_patchify import get_latent2scale_schedule


from einops import rearrange


class DummyController:
    """
    占位用的注意力控制器，目前只是原样返回 attn。
    后续可以在此基础上实现 PTP-style 的注意力编辑逻辑。
    """
    def reset(self):
        pass

    def start_edit(self):
        pass

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class SimpleTextAttentionController:
    """
    文本注意力控制器（Joint Attention PTP 版，显式词映射）：
    针对 InfinityStar 的联合自注意力机制设计。
    - 结构控制：在大矩阵的左上角（像素-像素），前 self_replace_layers 层做硬替换，保结构。
    - 语义控制：在大矩阵的右上角（像素-词），对 word_mapping 指定的词做 Cross-Attention 替换。
    
    参数：
        word_mapping: dict，显式指定词映射，如 {"cat": "dog"} 表示把 base_prompt 里的 "cat" 映射到 edit_prompt 里的 "dog"
    """
    def __init__(self, tokenizer, prompts, word_mapping: dict = None, alpha: float = 0.3, max_layers: int = 0, self_replace_layers: int = 4):
        self.tokenizer = tokenizer
        self.prompts = prompts  # [base_prompt, edit_prompt]
        self.word_mapping = word_mapping or {}  # {"cat": "dog", "chair": "chair"}
        self.alpha = float(alpha)
        self.max_layers = int(max_layers)
        self.self_replace_layers = int(self_replace_layers)
        
        self.in_edit_phase = False
        self.attn_base = []
        self._edit_ptr = 0
        self._call_idx = 0
        self._record_idx = 0  # 第一阶段已缓存的层数（用于限制缓存前几层）

        # 获取两个 prompt 的 token 信息（实际长度，去掉 padding）
        self.base_info = self._get_prompt_info(prompts[0])
        self.edit_info = self._get_prompt_info(prompts[1])
        
        # 建立显式词映射：(base_token_indices, edit_token_indices) - 支持多对多
        self.mapping = self._build_explicit_mapping()
        if self.mapping:
            # 打印映射信息（支持多对多）
            mapping_strs = []
            for base_indices, edit_indices in self.mapping:
                base_tokens = [self.base_info['decoded'][i] for i in base_indices]
                edit_tokens = [self.edit_info['decoded'][i] for i in edit_indices]
                mapping_strs.append(f"{base_tokens} -> {edit_tokens}")
            print(f"[PTP] 显式词映射: {mapping_strs}")

    def _get_prompt_info(self, prompt):
        # 用和 encode_prompt 完全一样的方式编码，确保长度一致
        # 参考 run_infinity.py 的 encode_prompt 逻辑
        tokens = self.tokenizer(text=[prompt], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        ids = tokens["input_ids"][0].tolist()
        mask = tokens["attention_mask"][0]
        # 关键：用 mask.sum() 计算实际长度（去掉 padding），和推理时一致
        actual_len = int(mask.sum().item())
        decoded = [self.tokenizer.decode([tid]).lower().strip() for tid in ids[:actual_len]]  # 只解码实际部分
        # 调试：打印前几个 token
        print(f"[PTP_DEBUG] Prompt: {prompt[:50]}...")
        print(f"[PTP_DEBUG] Token count (with padding): {len(ids)}, Actual len (mask.sum()): {actual_len}")
        print(f"[PTP_DEBUG] First 10 actual tokens: {decoded[:10]}")
        return {"ids": ids, "decoded": decoded, "len": actual_len}  # 返回实际长度，不是 padding 后的长度

    def _find_word_token_indices(self, word, decoded_tokens, token_ids=None):
        """
        找到一个词对应的所有连续 token 索引。
        例如 "cat" 可能被分词成 ["cat"] 或 ["ca", "t"]
        返回连续 token 的索引列表。
        
        改进：通过直接 tokenize 词来精确匹配，而不是字符串拼接。
        """
        word_lower = word.lower().strip()
        indices = []
        
        # 策略0（最精确）：直接对词进行 tokenize，然后在 prompt 的 token IDs 中精确匹配
        if token_ids is not None:
            # 对单个词/短语进行 tokenize（不添加特殊 token，不 padding）
            # 注意：对于短语如 "orange tiger"，tokenizer 会将其 tokenize 为多个 token
            word_tokens = self.tokenizer(word, add_special_tokens=False, return_tensors='pt')
            word_token_ids = word_tokens["input_ids"][0].tolist()
            
            if word_token_ids:
                # 在 prompt 的 token IDs 序列中搜索这个 token ID 序列（精确匹配）
                # 只返回第一个完全匹配的位置，确保精确性
                for i in range(len(token_ids) - len(word_token_ids) + 1):
                    if token_ids[i:i+len(word_token_ids)] == word_token_ids:
                        indices = list(range(i, i+len(word_token_ids)))
                        # 验证：检查匹配的 token 解码后是否与目标词一致
                        matched_decoded = [decoded_tokens[j] for j in indices]
                        matched_text = ''.join(matched_decoded).lower().strip()
                        word_clean = word_lower.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        matched_clean = matched_text.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        
                        # 如果匹配的文本和目标词一致（去掉标点后），认为是精确匹配
                        if word_clean == matched_clean or (len(word_clean) > 3 and word_clean in matched_clean):
                            print(f"[PTP_DEBUG] 精确匹配: '{word}' -> token indices {indices} (tokens: {matched_decoded}, token IDs: {word_token_ids})")
                            return indices
                        # 如果 token IDs 完全匹配但解码文本不一致，可能是 tokenizer 的差异，仍然返回
                        # 但打印警告
                        else:
                            print(f"[PTP_DEBUG] Token IDs 匹配但文本不一致: '{word}' -> {matched_decoded} (期望: {word_lower})")
                            # 仍然返回，因为 token IDs 匹配是最可靠的
                            return indices
        
        # 策略1: 尝试找到完整的词（可能跨多个 token）
        # 先尝试找到包含该词的 token 序列（作为策略0的备选）
        best_match = None
        best_match_score = 0
        
        for i in range(len(decoded_tokens)):
            # 检查从位置 i 开始的连续 token 是否能组成完整的词
            combined = ""
            candidate_indices = []
            for j in range(i, min(i + 10, len(decoded_tokens))):  # 最多检查10个连续token（支持更长的短语）
                tok = decoded_tokens[j].strip()
                combined += tok
                candidate_indices.append(j)
                
                # 清理字符串进行比较
                combined_clean = combined.lower().replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('-', '')
                word_clean = word_lower.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('-', '')
                
                # 精确匹配：完全相等
                if combined_clean == word_clean:
                    print(f"[PTP_DEBUG] 字符串精确匹配: '{word}' -> token indices {candidate_indices} (tokens: {[decoded_tokens[k] for k in candidate_indices]})")
                    return candidate_indices
                
                # 部分匹配：计算匹配度
                if word_clean in combined_clean:
                    match_score = len(word_clean) / max(len(combined_clean), 1)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = candidate_indices.copy()
        
        # 如果找到了较好的匹配（匹配度 > 0.8），返回它
        if best_match and best_match_score > 0.8:
            matched_tokens = [decoded_tokens[k] for k in best_match]
            print(f"[PTP_DEBUG] 字符串部分匹配 (score={best_match_score:.2f}): '{word}' -> token indices {best_match} (tokens: {matched_tokens})")
            return best_match
        
        # 策略2: 如果策略1没找到，尝试找单个 token 包含该词
        if not indices:
            for i, tok in enumerate(decoded_tokens):
                tok_clean = tok.lower().strip().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                word_clean = word_lower.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                if word_clean in tok_clean or tok_clean in word_clean:
                    indices = [i]
                    break
        
        return indices
    
    def _build_explicit_mapping(self):
        """
        根据显式的 word_mapping 建立 token 索引映射（支持多对多）。
        word_mapping: {"base_word": "edit_word"}，例如 {"cat": "dog"}
        现在支持：如果 base_word 被分词成 2 个 token，edit_word 被分词成 3 个 token，
        则建立 2->3 的映射关系。
        """
        mapping = []
        if not self.word_mapping:
            return mapping
        
        for base_word, edit_word in self.word_mapping.items():
            base_word_lower = base_word.lower().strip()
            edit_word_lower = edit_word.lower().strip()
            
            # 找到 base_word 对应的所有连续 token 索引（传入 token_ids 以支持精确匹配）
            base_indices = self._find_word_token_indices(
                base_word, 
                self.base_info["decoded"],
                token_ids=self.base_info["ids"][:self.base_info["len"]]  # 只传入实际长度的 token IDs
            )
            
            # 找到 edit_word 对应的所有连续 token 索引（传入 token_ids 以支持精确匹配）
            edit_indices = self._find_word_token_indices(
                edit_word, 
                self.edit_info["decoded"],
                token_ids=self.edit_info["ids"][:self.edit_info["len"]]  # 只传入实际长度的 token IDs
            )
            
            if base_indices and edit_indices:
                mapping.append((base_indices, edit_indices))
                print(f"[PTP] 映射词: '{base_word}' (base_indices={base_indices}, {len(base_indices)} tokens) -> "
                      f"'{edit_word}' (edit_indices={edit_indices}, {len(edit_indices)} tokens)")
            else:
                print(f"[PTP] 警告: 无法找到映射 '{base_word}' -> '{edit_word}' 的 token")
                if not base_indices:
                    print(f"  - 在 base_prompt 中未找到 '{base_word}'")
                if not edit_indices:
                    print(f"  - 在 edit_prompt 中未找到 '{edit_word}'")
        
        return mapping

    def reset(self):
        self.in_edit_phase = False
        self.attn_base = []
        self._edit_ptr = 0
        self._call_idx = 0
        self._record_idx = 0

    def start_edit(self):
        self.in_edit_phase = True
        self._edit_ptr = 0
        self._call_idx = 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """
        attn: [B, H, L, L]，其中：
            - B=2 (CFG 时，前半是条件生成，后半是无条件)
            - H=heads
            - L = N_pix + L_text（视觉 token 数 + 文本 token 数）
        
        在 InfinityStar 中，序列拼接顺序是：[视觉token, 文本token]
        所以 Cross-Attention 区域是：attn[0:N_pix, N_pix:N_pix+L_text]
        """
        # 添加调试：确认 __call__ 被调用
        if len(self.attn_base) == 0 and not self.in_edit_phase:
            print(f"[PTP_DEBUG] ✅ Controller.__call__ 被调用（第一阶段，记录模式）")
        if self.in_edit_phase and self._call_idx == 0:
            print(f"[PTP_DEBUG] ✅ Controller.__call__ 被调用（第二阶段，编辑模式）")
            print(f"[PTP_DEBUG] attn.shape={attn.shape}, in_edit_phase={self.in_edit_phase}, attn_base_len={len(self.attn_base)}")
        
        if not self.in_edit_phase:
            # 第一阶段：只记录，不修改
            # 这里只按序记录所有会被 PTP 控制的层的第一次 attention，用于后续的编辑阶段复用；
            # 不再仅限制前 max_layers 个调用，避免后面那些也做 attention control 的层没有对应的 base cache。
            # 仍保留长度过滤，避免单个超大矩阵导致 OOM。
            L = attn.shape[-1]
            if L <= 1024:
                self.attn_base.append(attn.detach().clone())
            else:
                # 对于大序列，存一个 None 占位，编辑阶段会跳过
                self.attn_base.append(None)

            self._record_idx += 1
            return attn

        if (not self.attn_base) or (self._edit_ptr >= len(self.attn_base)) or (self._call_idx >= self.max_layers):
            self._call_idx += 1
            return attn

        base = self.attn_base[self._edit_ptr]
        
        # 如果 base 是 None（大序列未收集），跳过
        if base is None:
            self._edit_ptr += 1
            self._call_idx += 1
            return attn
        self._edit_ptr += 1
        self._call_idx += 1

        # ========== 核心逻辑：以 base 的 attention map 为基础，把 cat 列替换成 edit 轮里 dog 列 ==========
        L_edit = attn.shape[-1]
        L_base = base.shape[-1]

        # ---- 调试：检查第二轮的长度与防 OOM 的 L 阈值（1024）以及第一轮缓存情况的关系 ----
        if self._call_idx == 1:
            oom_L = 1024
            print(f"[PTP_DEBUG] Edit phase call_idx={self._call_idx}: L_edit={L_edit}, L_base={L_base}, OOM_L={oom_L}")
            if L_edit > oom_L:
                print(f"[PTP_DEBUG][WARN] 当前第二轮 L_edit({L_edit}) > OOM_L({oom_L})，"
                      f"如果这是第一轮，将会被防 OOM 过滤，不缓存 base attn")
            else:
                print(f"[PTP_DEBUG] 当前第二轮 L_edit({L_edit}) <= OOM_L({oom_L})，长度本身不会触发防 OOM 过滤")
            if L_edit != L_base:
                print(f"[PTP_DEBUG][WARN] L_edit({L_edit}) != L_base({L_base})，后续计算将使用两者中的较小值进行裁剪对齐")
        
        # 获取文本的实际长度（去掉 padding，和 encode_prompt 的 lens[0] 一致）
        L_text_edit_actual = self.edit_info["len"]
        L_text_base_actual = self.base_info["len"]
        
        # 推断视觉 token 数量
        N_pix_edit = max(0, L_edit - L_text_edit_actual)
        N_pix_base = max(0, L_base - L_text_base_actual)
        num_pix = min(N_pix_edit, N_pix_base)

        # 额外调试：确认推断出的视觉 token 数是否合理，并结合 OOM 阈值观察是否可能被拦截
        if self._call_idx == 1:
            print(f"[PTP_DEBUG] N_pix_edit={N_pix_edit}, N_pix_base={N_pix_base}, num_pix(clip)={num_pix}")
        
        # 如果推断出的 num_pix 为 0 或异常小，跳过控制
        if num_pix <= 0 or num_pix > L_edit * 0.9:
            if self._call_idx == 1:
                print(f"[PTP_DEBUG] 警告: num_pix={num_pix} 异常，跳过控制 (L_edit={L_edit}, L_text_edit_actual={L_text_edit_actual})")
            return attn

        # 1. 以 edit 的 attention map 为基础（因为文本部分必须对应 edit prompt 的 token）
        new_attn = attn.clone()
        
        # 2. 结构控制：前 self_replace_layers 层，把像素-像素块替换成 base 的
        if self._call_idx <= self.self_replace_layers:
            new_attn[0, :, :num_pix, :num_pix] = base[0, :, :num_pix, :num_pix]

        # 2. 语义控制：把 base attention map 中 cat 对应的列，替换成 edit 轮里 dog 的列
        if self.mapping:
            # 只在第一次调用时打印调试信息
            if self._call_idx == 1:
                print(f"\n[PTP_DEBUG] ========== Layer {self._call_idx} (Edit Phase) ==========")
                print(f"[PTP_DEBUG] L_edit={L_edit}, L_base={L_base}")
                print(f"[PTP_DEBUG] L_text_edit_actual={L_text_edit_actual}, L_text_base_actual={L_text_base_actual}")
                print(f"[PTP_DEBUG] N_pix_edit={N_pix_edit}, N_pix_base={N_pix_base}, num_pix={num_pix}")
                print(f"[PTP_DEBUG] 验证: N_pix_edit + L_text_edit_actual = {N_pix_edit + L_text_edit_actual} (应该 <= L_edit={L_edit})")
                print(f"[PTP_DEBUG] 验证: N_pix_base + L_text_base_actual = {N_pix_base + L_text_base_actual} (应该 <= L_base={L_base})")
            
            for base_indices, edit_indices in self.mapping:
                # 支持多对多映射：base_indices 和 edit_indices 都是列表
                # 例如：base_indices=[5, 6], edit_indices=[7, 8, 9] 表示 base 的 token 5,6 映射到 edit 的 token 7,8,9
                
                if self._call_idx == 1:
                    print(f"\n[PTP_DEBUG] 处理映射: base_indices={base_indices} ({len(base_indices)} tokens) -> "
                          f"edit_indices={edit_indices} ({len(edit_indices)} tokens)")
                
                # 边界检查：检查所有索引是否在有效范围内
                if any(idx >= L_text_base_actual for idx in base_indices):
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ 跳过映射: base_indices 中有索引越界 (>= {L_text_base_actual})")
                    continue
                if any(idx >= L_text_edit_actual for idx in edit_indices):
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ 跳过映射: edit_indices 中有索引越界 (>= {L_text_edit_actual})")
                    continue
                
                # 计算绝对列位置
                base_cols = [N_pix_base + idx for idx in base_indices]
                edit_cols = [N_pix_edit + idx for idx in edit_indices]
                
                # 检查绝对列位置是否越界
                if any(col >= L_base for col in base_cols):
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ 跳过映射: base_cols 中有列越界 (>= {L_base})")
                    continue
                if any(col >= L_edit for col in edit_cols):
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ 跳过映射: edit_cols 中有列越界 (>= {L_edit})")
                    continue
                
                # 多对多映射策略：
                # 1. 从 base 的多个列中提取 attention 值，取平均（或加权平均）
                # 2. 将结果替换到 edit 的多个列中
                try:
                    # 收集 base 的所有相关列
                    base_cols_tensor = []
                    for col_base in base_cols:
                        src_col = base[0, :, :num_pix, col_base]  # [H, num_pix]
                        base_cols_tensor.append(src_col)
                    
                    # 对 base 的多个列取平均，得到融合后的 attention 模式
                    if len(base_cols_tensor) > 1:
                        fused_base_col = torch.stack(base_cols_tensor, dim=0).mean(dim=0)  # [H, num_pix]
                    else:
                        fused_base_col = base_cols_tensor[0]  # [H, num_pix]
                    
                    # 检查数值有效性
                    if torch.isnan(fused_base_col).any():
                        if self._call_idx == 1:
                            print(f"[PTP_DEBUG] ❌ 警告: 检测到 NaN，跳过映射")
                        continue
                    
                    # 将融合后的 attention 应用到 edit 的每个对应列
                    for col_edit in edit_cols:
                        if col_edit >= L_edit:
                            continue
                        # 使用插值：alpha 控制融合强度
                        target_col = attn[0, :, :num_pix, col_edit]  # [H, num_pix]
                        new_col = (1 - self.alpha) * target_col + self.alpha * fused_base_col
                        new_attn[0, :, :num_pix, col_edit] = new_col
                    
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ✅ 成功应用多对多映射: {len(base_indices)} base tokens -> {len(edit_indices)} edit tokens")
                        
                except IndexError as e:
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ IndexError: {e}")
                        print(f"[PTP_DEBUG] base.shape={base.shape}, attn.shape={attn.shape}")
                        print(f"[PTP_DEBUG] base_cols={base_cols}, edit_cols={edit_cols}")
                    continue
                except Exception as e:
                    import traceback
                    if self._call_idx == 1:
                        print(f"[PTP_DEBUG] ❌ 错误: {e}")
                        print(f"[PTP_DEBUG] 错误类型: {type(e).__name__}")
                        print(f"[PTP_DEBUG] 错误堆栈:\n{traceback.format_exc()}")
                    continue
                
                # （已用多对多替换逻辑完成替换）——删除旧的一对一残留逻辑以避免使用未定义的变量

        return new_attn


def register_attn_shape_hooks(infinity_model, max_print: int = 20):
    """
    在 Infinity 模型的注意力层上注册前向 hook，用于调试查看注意力张量形状。
    只打印前 max_print 次调用，避免输出过多。
    """
    hooks = []
    counter = {"n": 0}

    def make_hook(name):
        def _hook(module, inputs, output):
            if counter["n"] >= max_print:
                return
            counter["n"] += 1
            in_shapes = []
            for x in inputs:
                try:
                    in_shapes.append(tuple(x.shape))
                except AttributeError:
                    pass
            out_shapes = []
            if isinstance(output, tuple):
                for x in output:
                    try:
                        out_shapes.append(tuple(x.shape))
                    except AttributeError:
                        pass
            else:
                try:
                    out_shapes.append(tuple(output.shape))
                except AttributeError:
                    pass
            print(f"[ATTN_SHAPE] {name}: in={in_shapes}, out={out_shapes}")
        return _hook

    for n, m in infinity_model.named_modules():
        cls = m.__class__.__name__.lower()
        if "attn" in cls or "attention" in cls:
            hooks.append(m.register_forward_hook(make_hook(n)))
    print(f"[ATTN_SHAPE] registered {len(hooks)} attention hooks, will print up to {max_print} calls")
    return hooks


def enable_ptp_debug_for_self_attention(infinity_model, controller=None, max_layers=None):
    """
    遍历 Infinity 模型，将所有带有 use_ptp_debug 属性的 SelfAttention 模块
    切换到显式 PyTorch 注意力路径，并挂上 ptp_controller（若提供）。

    注意：这会关闭 flash/flex attention，推理速度会变慢，仅用于调试/实验。
    
    Args:
        max_layers: 最多启用前 N 层（用于避免 OOM，前几层序列长度通常较小）
    """
    num_modules = 0
    num_with_controller = 0
    num_enabled = 0
    for m in infinity_model.modules():
        if hasattr(m, "use_ptp_debug"):
            num_modules += 1
            # 如果设置了 max_layers，只启用前 N 层
            if max_layers is not None and num_enabled >= max_layers:
                continue
            m.use_ptp_debug = True
            if controller is not None:
                setattr(m, "ptp_controller", controller)
                num_with_controller += 1
            num_enabled += 1
    print(f"[PTP_DEBUG] enabled use_ptp_debug on {num_enabled}/{num_modules} SelfAttention-like modules" + 
          (f" (限制前 {max_layers} 层)" if max_layers else ""))
    if controller is not None:
        print(f"[PTP_DEBUG] attached controller to {num_with_controller} modules")
    else:
        print(f"[PTP_DEBUG] ⚠️ controller is None, modules will not be controlled")


def enable_ptp_debug_for_self_attention_range(infinity_model, controller=None, start_layer=0, end_layer=None):
    """
    在指定层范围内启用 PTP attention control（逻辑与 enable_ptp_debug_for_self_attention 基本一致，
    只是限定层编号区间），用于比如在第 45 层到某一层也应用同样的 attention control。
    
    Args:
        start_layer: 起始层索引（包含），按具有 use_ptp_debug 属性的模块顺序计数。
        end_layer: 结束层索引（包含）。若为 None，则从 start_layer 一直到最后一层。
    """
    num_modules = 0          # 模型中带 use_ptp_debug 的模块总数
    num_with_controller = 0  # 实际挂上 controller 的模块数
    num_enabled = 0          # 实际启用 use_ptp_debug 的模块数
    cur_idx = 0              # 当前遍历到的“层编号”（只对带 use_ptp_debug 的模块计数）

    for m in infinity_model.modules():
        if hasattr(m, "use_ptp_debug"):
            num_modules += 1

            # 还没到起始层，跳过但计数
            if cur_idx < start_layer:
                cur_idx += 1
                continue

            # 已超过结束层，停止
            if end_layer is not None and cur_idx > end_layer:
                break

            # 在指定范围内：启用显式注意力路径，并挂上 controller
            m.use_ptp_debug = True
            if controller is not None:
                setattr(m, "ptp_controller", controller)
                num_with_controller += 1
            num_enabled += 1
            cur_idx += 1

    print(
        f"[PTP_DEBUG_RANGE] enabled use_ptp_debug on {num_enabled}/{num_modules} "
        f"SelfAttention-like modules in range [{start_layer}, {end_layer}]")
    if controller is not None:
        print(f"[PTP_DEBUG_RANGE] attached controller to {num_with_controller} modules in that range")
    else:
        print(f"[PTP_DEBUG_RANGE] ⚠️ controller is None, modules in that range will not be controlled")


def to_tensor_video(frames_bgr: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    frames_bgr: [T, H, W, 3]，0~255，BGR
    返回: [1, 3, T, H, W]，[-1, 1]，float32
    """
    # BGR -> RGB
    frames_rgb = frames_bgr[:, :, :, ::-1].copy()  # [T, H, W, 3]
    frames = torch.from_numpy(frames_rgb).float() / 255.0  # [0,1]

    # [T, H, W, 3] -> [1, 3, T, H, W]
    frames = frames.permute(0, 3, 1, 2)             # [T,3,H,W]
    frames = frames.unsqueeze(0)                    # [1,T,3,H,W]
    frames = frames.permute(0, 2, 1, 3, 4)          # [1,3,T,H,W]

    # 映射到 [-1,1]
    frames = frames * 2.0 - 1.0
    return frames.to(device)


def build_vae(vae_ckpt: str, device: str = "cuda") -> torch.nn.Module:
    """
    用作者提供的 video_vae_model 构建 VAE（AutoencoderKLCogVideoX），
    schedule_mode / global_args 选一个与原工程一致的典型配置。
    """
      # 这里只给出一份通用的 global_args，你可以根据自己训练配置微调
    class GlobalArgs:
        # 这些值与 Infinity 配置基本一致（semantic 16 维，detail 64 维）
        semantic_scale_dim = 16
        detail_scale_dim = 64
        use_learnable_dim_proj = 1
        detail_scale_min_tokens = 750        # 和项目里 args.detail_scale_min_tokens 一致
        use_feat_proj = 2                    # 训练里常用 2（在 z 上做 proj_down）
        semantic_scales = "[12]"             # 这里只是占位，具体值对纯重建影响不大

    global_args = GlobalArgs()


    schedule_mode = "dynamic"   # 或者 "dense_f8"，二选一先试一个

    # codebook_dim 一般是 64，对应 latent 通道，这里也按项目默认值
    codebook_dim = 64

    vae = video_vae_model(
        vqgan_ckpt=vae_ckpt,
        schedule_mode=schedule_mode,
        codebook_dim=codebook_dim,
        global_args=global_args,
        test_mode=True,   # eval 模式，禁用梯度
    )
    
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def vae_bsq2all_bit_indices(pipe, data, args) -> torch.Tensor:
    video_path = data.get("video_path", None)
    # 计算 num_frames：使用固定的 81 帧（与 infer_video_480p-test1.py 一致）
    num_frames = 81
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    # 优先使用外部覆盖（若提供 h_div_w_template_override），否则按下列优先级计算 video_hw_ratio：
    # 1) args.video_hw_ratio（如果用户显式传入）；
    # 2) 如果 data 中有 video_path，则从视频第一帧读取 H/W 并计算；
    # 3) 回退到兼容常量（1/0.571）。
    if hasattr(args, "h_div_w_template_override") and getattr(args, "h_div_w_template_override") is not None:
        h_div_w_template_ = args.h_div_w_template_override
    else:
        video_hw_ratio = getattr(args, "video_hw_ratio", None)
        if video_hw_ratio is None and video_path is not None:
            try:
                video = EncodedVideoDecord(video_path, os.path.basename(video_path), num_threads=0)
                raw_video, _ = video.get_clip(0, 5, min(81, getattr(video, "_duration", 81)))
                if raw_video is not None and len(raw_video) > 0:
                    first_frame = raw_video[0]
                    # raw_video frame shape could be [H, W, C]
                    Hf, Wf = first_frame.shape[0], first_frame.shape[1]
                    video_hw_ratio = float(Hf) / float(Wf) if Wf != 0 else None
            except Exception:
                video_hw_ratio = None
        if video_hw_ratio is None:
            # 保留向后兼容的选择基准（之前代码在不同地方使用 0.571 或 1/0.571）
            video_hw_ratio = 1.0 / 0.571
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(video_hw_ratio)))]
    
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)    
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None
    if video_path is not None:
        # Video continuation
        video = EncodedVideoDecord(video_path, os.path.basename(video_path), num_threads=0)
        duration = video._duration
        # if mapped_duration != 10:
        #     raise ValueError('Video continuation only support 10 seconds generation.')
        if duration < 5:
            raise ValueError('Input video duration must be longer than 5 seconds.')
        condition_scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(81-1)//4+1]
        cond_tgt_h, cond_tgt_w = condition_scale_schedule[-1][1] * 16, condition_scale_schedule[-1][2] * 16
        raw_video, _ = video.get_clip(0, 5, 81)
        video_T3HW = [transform(Image.fromarray(frame).convert("RGB"), cond_tgt_h, cond_tgt_w) for frame in raw_video]
        video_T3HW = torch.stack(video_T3HW, 0) # [t,3,h,w]
        video_bcthw = video_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(pipe.vae, video_bcthw.cuda(), vae_features=None, self_correction=pipe.self_correction, args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w)
    
    return gt_ls_Bl




def sampling_func_wrapper(idx_Blv_cache, prob_cache, gamma=1.75, scale_schedule=None,
                          front_attn_max_layers=None, attn_control_start=None, attn_control_end=None):
    
    def sampling_func(logits_BlV, si, rng=None):
        if idx_Blv_cache is None:
            return None, None
        
        # 缓存是按“每个 repeat step” append 的，这里用顺序指针逐步消费，避免 repeat_times>1 时错位
        if not hasattr(sampling_func, "_step_ptr"):
            sampling_func._step_ptr = 0
        step_ptr = sampling_func._step_ptr
        if step_ptr >= len(idx_Blv_cache):
            print(f"[WARN] cache exhausted: step_ptr={step_ptr}, cache_len={len(idx_Blv_cache)}")
            return None, None
        i_cache = idx_Blv_cache[step_ptr] if idx_Blv_cache else None
        p_cache = prob_cache[step_ptr] if prob_cache else None
        sampling_func._step_ptr += 1
        
        B, l, V = logits_BlV.shape
        probs = logits_BlV.softmax(dim=-1).view(-1, V)
        
        num_samples = 1

        # 编辑判定规则：
        # - 从 0 到 edit_threshold_1 之间的尺度不算 edit
        # - 从 45 到 edit_threshold_2 之间的尺度不算 edit
        # - 其他尺度都算 edit
        edit_threshold_1 = 5   # 可根据需要修改
        edit_threshold_2 = 55  # 可根据需要修改
        non_edit_flag = ((0 <= si <= edit_threshold_1) or (45 <= si <= edit_threshold_2))
        edit_flag = not non_edit_flag
        
        if edit_flag:
            # 适配视频缓存：flatten 到与当前 logits 的 l 对齐
            # logits_BlV.shape = [B, l, V]，其中 l 已经等于 t*h*w*d_bit（按推理时的 token 序列长度）
            # 缓存 i_cache / p_cache 可能是 [B, t, h, w, d_bit] / [B, t, h, w, d_bit*2] 或 [B, l, d_bit] / [B, l, 2]
            i_cache_flat = i_cache.reshape(B, -1)  # [B, l]
            # p_cache 最后一维是 2（bit 概率），其他维展平到 l
            p_cache_flat = p_cache.reshape(B, -1, 2)  # [B, l, 2]

            if i_cache_flat.shape[1] != l:
                print(f"[WARN] cache length {i_cache_flat.shape[1]} != logits l {l} at scale {si}")
            # 展平到 (B*l, 1) / (B*l, 2) 便于 scatter/gather
            idx_Bl_cache = rearrange(i_cache_flat, "B l->(B l) 1")
            probs_cache = rearrange(p_cache_flat, "B l d->(B l) d")
            
            # modify
            bias = (gamma - torch.gather(probs_cache, 1, idx_Bl_cache)).clamp(min=0)
            
            # visualize_bias(bias, h=int((bias.shape[0]/2)**0.5), w=int((bias.shape[0]/2)**0.5), save_path=f'bias_image_{si}.jpg')
            
            probs = torch.scatter(probs, 1, idx_Bl_cache, torch.gather(probs, 1, idx_Bl_cache)+bias)
            
            probs, idx = torch.topk(probs, k=num_samples, dim=-1)

            # --- 仅在指定尺度统计 bit 翻转并可视化 ---
            # 用户给出的 scale_schedule 明确：第 14 个尺度 (0-based) 为 (1,45,80)，直接锁定
            target_scale_for_stats = 14
            target_t, target_h, target_w = 1, 45, 80
            if scale_schedule is not None and len(scale_schedule) > target_scale_for_stats:
                sc = scale_schedule[target_scale_for_stats]
                if isinstance(sc, (list, tuple)) and len(sc) >= 3:
                    target_t, target_h, target_w = sc[0], sc[1], sc[2]
            # 避免重复保存
            if si == target_scale_for_stats and not hasattr(sampling_func, "_flip_saved"):
                try:
                    # 将原始/新选择的 bit 还原为空间网格
                    # 依据锁定的 t,h,w 直接还原
                    def infer_thwd():
                        return target_t, target_h, target_w, None

                    t_dim, h_dim, w_dim, d_dim = infer_thwd()
                    tokens_spatial = t_dim * h_dim * w_dim
                    if tokens_spatial <= 0 or l % tokens_spatial != 0:
                        print(f"[FlipStats][WARN] 无法可靠还原空间维: l={l}, T={t_dim}, H={h_dim}, W={w_dim}")
                        sampling_func._flip_saved = True
                        return idx[:, :, None], probs
                    # 若 d_dim 未定或不匹配，按 l // tokens_spatial 计算
                    d_dim = d_dim if d_dim is not None else (l // tokens_spatial)
                    orig_bits_grid = i_cache_flat.view(B, t_dim, h_dim, w_dim, d_dim)
                    new_bits_grid = idx.view(B, -1).reshape(B, t_dim, h_dim, w_dim, d_dim)

                    # 统计 bit 翻转（包括 0->1 和 1->0）
                    flip_mask = (orig_bits_grid != new_bits_grid)
                    # 按 bit 维度求和，得到每个潜空间点被翻转的 bit 数
                    flip_map = flip_mask.sum(dim=-1).sum(dim=1)  # [B, h, w]，T=1 时直接 squeeze

                    # 只处理 batch 的第一个样本
                    heatmap = flip_map[0].float().cpu().numpy()
                    heatmap_max = heatmap.max()
                    if heatmap_max > 0:
                        heatmap = (heatmap / heatmap_max * 255).astype(np.uint8)
                    else:
                        heatmap = np.zeros_like(heatmap, dtype=np.uint8)

                    # 伪彩色可视化并保存
                    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    os.makedirs("output/heatmaps", exist_ok=True)
                    save_path = os.path.join("output/heatmaps", f"flip_heatmap_scale{target_scale_for_stats}v1.png")
                    cv2.imwrite(save_path, color_map)
                    print(f"[FlipStats] Saved heatmap to {os.path.abspath(save_path)}")
                except Exception as e:
                    print(f"[FlipStats][WARN] Failed to save heatmap at scale {target_scale_for_stats}: {e}")
                finally:
                    sampling_func._flip_saved = True
            # idx = torch.multinomial(
            #     probs,
            #     num_samples=num_samples,
            #     replacement=True,
            #     generator=rng,
            # )

        else:
            # 非编辑 scale：始终硬替换为 Phase 1 缓存的 token
            # attention control 的效果仅通过 KV cache 间接传导到后续 scale
            i_cache_flat = i_cache.reshape(B, -1)  # [B, l]
            if i_cache_flat.shape[1] != l:
                print(f"[WARN] cache length {i_cache_flat.shape[1]} != logits l {l} at scale {si}")
            idx = i_cache_flat  # [B, l]
            probs = logits_BlV.softmax(dim=-1).view(-1, V)  # [(B*l), V]
        return idx[:,:,None], probs
    
    return sampling_func

def src_img_sampling_func_wrapper(all_bit_indices, scale_schedule=None, args=None, spatial_patchify=False, save_used_indices=None):
    """
    使用 all_bit_indices (gt_ls_Bl) 直接作为采样结果
    all_bit_indices 包含所有 (scale, repeat) 组合的结果，需要使用 cum_scales+repeat_idx 来索引
    save_used_indices: 如果提供，会保存实际使用的 idx（用于调试对比）
    """
    if save_used_indices is None:
        save_used_indices = []
    
    # 计算 repeat_times（与 ar_infer_infinity_elegant 中的逻辑一致）
    if scale_schedule is not None and args is not None:
        from infinity.schedules.dynamic_resolution import get_first_full_spatial_size_scale_index
        import json
        import numpy as np
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
    else:
        # 如果没有提供 scale_schedule 和 args，假设每个 scale 只重复一次
        scales_in_one_clip = len(scale_schedule) if scale_schedule else len(all_bit_indices)
        image_scale_repetition = np.ones(scales_in_one_clip, dtype=int)
        video_scale_repetition = np.ones(scales_in_one_clip, dtype=int)
    
    # 维护计数器：跟踪当前的 si、repeat_idx 和 cum_scales
    current_si = -1
    current_repeat_idx = 0
    cum_scales = 0
    
    def sampling_func(logits_BlV, si, rng=None):
        nonlocal current_si, current_repeat_idx, cum_scales
        
        # logits_BlV: [B, l, V]，其中 l = t*h*w*d 展平后的长度，V=2（bit）
        B, l, V = logits_BlV.shape

        # 如果 si 变化了，更新 cum_scales 并重置 repeat_idx
        if si != current_si:
            # 如果之前有 scale（不是第一次调用），需要更新 cum_scales（加上前一个 scale 的 repeat_times）
            if current_si >= 0 and scale_schedule is not None:
                if current_si < scales_in_one_clip:
                    prev_repeat_times = image_scale_repetition[current_si % scales_in_one_clip]
                else:
                    prev_repeat_times = video_scale_repetition[current_si % scales_in_one_clip]
                # 使用 min 来限制 repeat_times（与 ar_infer_infinity_elegant 中的 infer_repeat_times 一致）
                prev_repeat_times = min(prev_repeat_times, args.max_repeat_times if args else prev_repeat_times)
                cum_scales += prev_repeat_times
            current_si = si
            current_repeat_idx = 0
        
        # 计算当前 scale 的 repeat_times（与 ar_infer_infinity_elegant 中的逻辑一致）
        if scale_schedule is not None:
            if si < scales_in_one_clip:
                repeat_times = image_scale_repetition[si % scales_in_one_clip]
            else:
                repeat_times = video_scale_repetition[si % scales_in_one_clip]
            # 使用 min 来限制 repeat_times（与 ar_infer_infinity_elegant 中的 infer_repeat_times 一致）
            infer_repeat_times = min(repeat_times, args.max_repeat_times if args else repeat_times)
        else:
            infer_repeat_times = 1
        
        # 检查 repeat_idx 是否超出范围
        if current_repeat_idx >= infer_repeat_times:
            raise RuntimeError(f"scale {si}: current_repeat_idx={current_repeat_idx} >= infer_repeat_times={infer_repeat_times}")
        
        # 使用 cum_scales + repeat_idx 来索引 gt_ls_Bl（与 ar_infer_infinity_elegant 第 106 行一致）
        gt_idx = cum_scales + current_repeat_idx
        
        # 检查索引是否在范围内
        if gt_idx >= len(all_bit_indices):
            raise RuntimeError(f"scale {si}, repeat {current_repeat_idx}: gt_idx={gt_idx} >= len(all_bit_indices)={len(all_bit_indices)}")
        
        idx = all_bit_indices[gt_idx]
        
        # 递增 repeat_idx（下次调用时使用）
        current_repeat_idx += 1
        
        # 检查 idx 的 batch 维度是否匹配
        if idx.shape[0] != B:
            raise RuntimeError(f"scale {si}: all_bit_indices[{si}].shape[0]={idx.shape[0]} != B={B}")

        # idx 可能是 [B, t, h, w, d]（视频）或 [B, h, w, d]（图像）
        # 如果开启了 spatial patchify，只处理空间维度的 unpatchify
        if spatial_patchify and idx.ndim == 5:
            idx = rearrange(idx, "B 1 h w d -> B d h w")
            idx = torch.nn.functional.pixel_unshuffle(idx, 2)
            idx = rearrange(idx, "B d h w -> B (h w d)")
        elif idx.ndim >= 3:
            # 通用展平（保持 batch 维，其他维度全部展平）
            # 使用 reshape 而不是 view，更安全
            idx_flat = idx.reshape(B, -1)
            
            # 验证长度是否匹配
            if idx_flat.shape[1] != l:
                print(f"[ERROR] scale {si}: all_bit_indices[{si}].shape={tuple(idx.shape)}")
                print(f"  reshape(B={B}, -1) -> shape={tuple(idx_flat.shape)}, expected l={l}")
                print(f"  idx.numel()={idx.numel()}, idx.numel()//B={idx.numel()//B}, l={l}")
                raise RuntimeError(f"scale {si}: all_bit_indices[{si}].reshape(B, -1).shape[1]={idx_flat.shape[1]} != logits_l={l}")
            
            idx = idx_flat
        else:
            raise RuntimeError(f"scale {si}: all_bit_indices[{si}].ndim={idx.ndim} < 3, unexpected shape")

        # 保存实际使用的 idx（用于调试对比）
        if si < len(save_used_indices):
            save_used_indices[si] = idx.clone().cpu()
        else:
            while len(save_used_indices) <= si:
                save_used_indices.append(None)
            save_used_indices[si] = idx.clone().cpu()

        # 概率按当前 logits 计算，用于缓存
        probs = logits_BlV.softmax(dim=-1).view(-1, V)          # [(B*l), 2]
        probs = rearrange(probs, "(b l s) d -> b l s d", b=B, l=l, s=1)  # [B, l, 1, 2]

        # 返回强制的 idx 和对应概率（与 teacher_sampling_func_wrapper 完全一致）
        return idx[:, :, None].long(), probs

    return sampling_func 


def _init_prompt_rewriter():
    from tools.prompt_rewriter import OpenAIGPTModel
    """Initialize the OpenAI GPT model."""
    # Initialize the OpenAI GPT model
    model_name = 'gpt-4o-2024-08-06'
    ak = os.environ.get("OPEN_API_KEY", "")
    if len(ak) == 0:
        raise ValueError("Please provide your OpenAI API key in the OPEN_API_KEY environment variable.")
    model = OpenAIGPTModel(model_name, ak, if_global=True)
    system_prompt = (
        "You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description to make the video more realistic and beautiful. 0. Preserve ALL information, including style words and technical terms. 1. If the subject is related to person, you need to provide a detailed description focusing on basic visual characteristics of the person, such as appearance, clothing, expression, posture, etc. You need to make the person as beautiful and handsome as possible. When the subject is only one person or object, do not use they to describe him/her/it to avoid confusion with multiple subjects. 2. If the input does not include style, lighting, atmosphere, you can make reasonable associations. 3. We only generate a four-second video based on your descriptions. So do not generate descriptions that are too long, too complex or contain too many activities. 4. You can add some descriptions of camera movements with regards to the scenes and allow the scenes to have very natural and coherent movements. 6. If the input is in Chinese, translate the entire description to English. 7. Output ALL must be in English. 8. Here are some expanded descriptions that can serve as examples: 1. The video begins with a distant aerial view of a winding river cutting through a rocky landscape, with the sun casting a soft glow over the scene. As the camera moves closer, the river's flow becomes more visible, and the surrounding terrain appears more defined. The camera continues to approach, revealing a steep cliff with a person sitting on its edge. The person is positioned near the top of the cliff, overlooking the river below. The camera finally reaches a close-up view, showing the person sitting calmly on the cliff, with the river and landscape fully visible in the background. 2. In a laboratory setting, a machine with a metallic structure and a green platform is seen. A small, clear plastic bottle is positioned on the green platform. The machine has a control panel with red and green lights on the right side. A nozzle is positioned above the bottle, and it begins to dispense liquid into the bottle. The liquid is dispensed in small droplets, and the nozzle moves slightly between each droplet. The background includes other laboratory equipment and a mesh-like structure. 3. The video shows a panoramic view of a cityscape with a prominent building featuring a green dome and ornate architecture in the center. Surrounding the main building are several other structures, including a white building with balconies on the left and a taller building with multiple windows on the right. In the background, there are hills with scattered buildings and greenery. The camera remains stationary, capturing the scene from a fixed position, with no noticeable changes in the environment or the buildings throughout the frames. 4. In a dimly lit room with red and blue lighting, a person holds up a smartphone to record a video of a band performing. The band members are seated, with one holding a guitar and another playing a double bass. The smartphone screen shows the band members being recorded, with the camera capturing their movements and expressions. The background includes a lamp and some furniture, adding to the cozy atmosphere of the scene. 5. In a grassy area with scattered trees, a large tree stands prominently in the center. A lion is perched on a thick branch of this tree, looking out into the distance. The sky is overcast, adding a somber tone to the scene. 6. A man in a green sweater holding a paper turns around and speaks to a group of people seated in a theater. He then points at a man in a yellow sweater sitting in the front row. The man in the yellow sweater looks at the paper in his hand and begins to speak. The man in the green sweater lowers his head and then looks up at the man in the yellow sweater again. 7. An elderly man, wearing a beige sweater over a yellow shirt, is sitting in front of a laptop. He holds a pair of glasses in his right hand and appears to be deep in thought, resting his head on his hand. He then raises the glasses and rubs his eyes with his fingers, showing signs of fatigue. After rubbing his eyes, he places the glasses on his sweater and looks down at the laptop screen. 8. A woman and a child are sitting at a table, each holding a pencil and coloring on a piece of paper. The woman is coloring a green leafy plant, while the child is coloring a red and blue object. The table has several colored pencils, a container filled with more pencils, and a few small colorful blocks. The woman is wearing a striped shirt, and the child is focused on their drawing. 9. A person wearing teal running shoes and colorful socks is running on a wet, sandy surface. The camera captures the movement of their legs and feet as they lift off the ground and land back, creating a clear shadow on the wet sand. The shadow elongates and shifts with each step, indicating the person's motion. The background remains consistent with the wet, textured sand, and the focus is solely on the runner's feet and their shadow. 10. A man is running along the shoreline of a beach, with the ocean waves gently crashing onto the shore. The sun is setting in the background, casting a warm glow over the scene. The man is wearing a light-colored jacket and shorts, and his hair is blowing in the wind as he runs. The water splashes around his legs as he moves forward, and his reflection is visible on the wet sand. The waves create a dynamic and lively atmosphere as they roll in and out."
    )
    gpt_model = OpenAIGPTModel(model_name, ak, if_global=True)
    return gpt_model, system_prompt

class InferencePipe:
    def __init__(self, args):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to('cuda')
        # load infinity
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

        if args.enable_rewriter:
            self.gpt_model, self.system_prompt = _init_prompt_rewriter()   


def perform_inference(pipe, data, args, scale_schedule_override=None, h_div_w_template_override=None, controller=None, phase1_early_stop_si=None, early_stop_bit_indices=None, early_stop_default_prob=0.95, token_merge_from_si=None, token_merge_factor_w=2, token_merge_weight_maps=None, token_merge_to_si=None, token_merge_mode='2to1', layer_skip_from_si=None, layer_skip_chunks=None,
    layer_skip_indices=None, token_prune_from_si=None, token_prune_gamma_maps=None, token_prune_threshold=1.70, token_prune_mode="gamma", token_prune_keep_ratio=0.5):
    
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=5
    num_frames=81

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    
    # 如果提供了 scale_schedule_override，直接使用；否则使用动态计算
    if scale_schedule_override is not None:
        scale_schedule = scale_schedule_override
        print(f"[PERFORM_INF] 使用传入的 scale_schedule，共 {len(scale_schedule)} 个尺度")
    else:
        # 如果提供了 h_div_w_template_override，使用它；否则优先根据输入视频的 H/W 选择模板
        if h_div_w_template_override is not None:
            h_div_w_template_ = h_div_w_template_override
        else:
            video_hw_ratio = getattr(args, "video_hw_ratio", None)
            # 如果没有显式提供 ratio，则尝试从 data['video_path'] 读取第一帧计算 ratio
            if video_hw_ratio is None and data.get("video_path", None) is not None:
                try:
                    vpath = data.get("video_path")
                    video = EncodedVideoDecord(vpath, os.path.basename(vpath), num_threads=0)
                    raw_video, _ = video.get_clip(0, 5, 1)
                    if raw_video is not None and len(raw_video) > 0:
                        f0 = raw_video[0]
                        video_hw_ratio = float(f0.shape[0]) / float(f0.shape[1]) if f0.shape[1] != 0 else None
                except Exception:
                    video_hw_ratio = None
            if video_hw_ratio is not None:
                h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(video_hw_ratio)))]
            else:
                # 向后兼容：回退到旧常量
                h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-0.571))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
        print(f"[PERFORM_INF] 使用动态计算的 scale_schedule (h_div_w_template_={h_div_w_template_:.4f})，共 {len(scale_schedule)} 个尺度")
    
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    generated_image_list = []
    negative_prompt=''
    prompt = f'{prompt}, Close-up on big objects, emphasize scale and detail'
    negative_prompt = ""
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt
    
    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, idx_Blv_cache_list, prob_cache_list= gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt=negative_prompt,
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg, 
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
            sampling_func=args.sampling_func,
            phase1_early_stop_si=phase1_early_stop_si,
            early_stop_bit_indices=early_stop_bit_indices,
            early_stop_default_prob=early_stop_default_prob,
            token_merge_from_si=token_merge_from_si,
            token_merge_factor_w=token_merge_factor_w,
            token_merge_weight_maps=token_merge_weight_maps,
            token_merge_to_si=token_merge_to_si,
            token_merge_mode=token_merge_mode,
            layer_skip_from_si=layer_skip_from_si,
            layer_skip_chunks=layer_skip_chunks,
            layer_skip_indices=layer_skip_indices,
            token_prune_from_si=token_prune_from_si,
            token_prune_gamma_maps=token_prune_gamma_maps,
            token_prune_threshold=token_prune_threshold,
            token_prune_mode=token_prune_mode,
            token_prune_keep_ratio=token_prune_keep_ratio,
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        print(generated_image.shape)
        generated_image_list.append(generated_image)
            
    generated_image = torch.cat(generated_image_list, 2)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    
    return {
            "output": generated_image.cpu().numpy(),
            "elapsed_time": elapsed_time,
            "idx_Blv_cache_list": idx_Blv_cache_list,
            "prob_cache_list": prob_cache_list,
        }


def perform_inference_phase1_fast(pipe, data, args, scale_schedule, h_div_w_template,
                                  early_stop_si=25, default_prob=0.95):
    encode_data = {'seed': data['seed'], 'prompt': data['prompt'], 'video_path': data['video_path']}
    all_bit_indices = vae_bsq2all_bit_indices(pipe, encode_data, args)
    args.sampling_func = src_img_sampling_func_wrapper(
        all_bit_indices, scale_schedule=scale_schedule, args=args, spatial_patchify=False)
    out = perform_inference(pipe, data, args,
        scale_schedule_override=scale_schedule,
        h_div_w_template_override=h_div_w_template,
        phase1_early_stop_si=early_stop_si,
        early_stop_bit_indices=all_bit_indices,
        early_stop_default_prob=default_prob)
    return out, all_bit_indices


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--gamma", type=float, default=1.7)
    cli_parser.add_argument("--gamma_grid", type=str, default="", help="Comma-separated gamma values, e.g. 1.5,1.6,1.7,1.8")
    cli_args, _ = cli_parser.parse_known_args()

    # For optimal performance, enabling the prompt rewriter is recommended.
    # To utilize the GPT model, ensure the following environment variables are set:
    # export OPEN_API_KEY="YOUR_API_KEY"
    # export GLOBAL_AZURE_ENDPOINT="YOUR_ENDPOINT"
    enable_rewriter=0
    checkpoints_dir = './'
    

    # infer args
    args = Args()
    args.pn='0.90M'
    args.fps=16
    args.video_frames=81
    args.model_path=os.path.join(checkpoints_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type='torch_shard' # omnistore
    args.vae_path=os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt=os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type='infinity_qwen8b'
    args.text_channels=2048
    args.dynamic_scale_schedule='infinity_elegant_clip20frames_v2'
    args.bf16=1
    args.use_apg=1
    args.use_cfg=0
    args.cfg=5
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold=0.05
    args.image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption=1
    args.use_two_stage_lfq=1
    args.detail_scale_min_tokens=750
    args.semantic_scales=12
    args.max_repeat_times=10000
    args.enable_rewriter=enable_rewriter
    # ============ 注意力控制相关接口 ============
    args.attn_control = 1         # 是否开启 attention control（当前已启用，此参数暂未使用）
    # 显式词映射，格式："base_word:edit_word"，多个用逗号分隔，如 "cat:dog,chair:chair"
    # 之前这里默认用了长短语 "orange-and-white cat:..."，会导致按 token 粒度找不到完整匹配。
    # 现在改成默认不传，由下面的 fallback 用简单词级映射来兜底。
    args.word_mapping = ""
    args.max_attn_res = 16         # 控制器使用的最大注意力分辨率（暂未使用）
    args.gamma = cli_args.gamma       # 用于编辑阶段 sampling 的 gamma（bit 级别控制）
    # ======================================================================
    # 额外接口：是否在更深的层（例如 45-某一层）启用同样的 attention control
    # 若需要，在此将 enable_late_attn 设为 1，并根据需要修改 late_attn_start / late_attn_end。
    args.enable_late_attn = 1
    args.late_attn_start = 45
    args.late_attn_end = 55
    args.input_video = 'assets/input_videot.mp4'
    # 调试选项：是否打印注意力层的张量形状（仅前若干次调用）
    args.debug_attn_shapes = False  # native generated-video edit test: keep logs compact
    device = 'cuda'

    # load models
    pipe = InferencePipe(args)

    # 启用显式 PyTorch 注意力路径，并挂上增强版的 SimpleTextAttentionController
    base_prompt = "A handsome smiling man gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic. Fixed camera"
    edit_prompt = "Spider-Man inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic. Fixed camera"
    
    # 显式词映射：{"base_word": "edit_word"}
    # 例如 {"cat": "dog"} 表示把 base_prompt 里的 "cat" 映射到 edit_prompt 里的 "dog"
    # 可以通过命令行参数传入，格式：--word_mapping "cat:dog,chair:chair"
    # 这里不再使用长短语默认值，避免 tokenizer 粒度不一致导致“无法找到映射”的警告。
    word_mapping_str = getattr(args, "word_mapping", "")
    word_mapping = {"man": "Spider-Man", "gardener": "Spider-Man"}
    if word_mapping_str:
        for pair in word_mapping_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                base_w, edit_w = pair.split(":", 1)
                word_mapping[base_w.strip()] = edit_w.strip()
    
    # 如果没有从参数获取到，使用默认映射
    # 注意：分词是按子词来的，像 "orange-and-white cat" 会被拆成多个 token：
    # ['orange', '-', 'and', '-', 'white', 'cat']，找不到一个 token 包含整句。
    # 所以这里默认改成用单词级别的映射，保证一定能命中。
    if not word_mapping:
        # 最简单、最稳妥的默认：直接把 "cat" 映射到 "dog"
        word_mapping = {"cat": "dog", "orange":"silver" }
    
    controller = SimpleTextAttentionController(
        pipe.text_tokenizer,
        [base_prompt, edit_prompt],
        word_mapping=word_mapping,  # 显式映射
        alpha=1,               # 语义拉回强度（Cross-Attention 插值系数）
        max_layers=8,            # 控制的总深度（先控制前 8 层，更稳妥，后续可再调高）
        self_replace_layers=0,   # 结构锁定的深度（前 N 层硬替换像素-像素块）
    )
    # 只有当 max_layers > 0 时才启用 ptp_debug，否则保持原始 flash_attn 路径
    if controller.max_layers > 0:
        # 原始逻辑：在前 max_layers 个注意力层上启用 PTP 控制
        enable_ptp_debug_for_self_attention(pipe.infinity, controller=controller, max_layers=controller.max_layers)

        # 额外接口（可选）：在更深的层范围（例如 45-某一层）启用同样的 attention control
        # 默认关闭；若想在 45-args.late_attn_end 层也做相同的控制，将 args.enable_late_attn 设为 1 即可。
        if getattr(args, "enable_late_attn", 0):
            enable_ptp_debug_for_self_attention_range(
                pipe.infinity,
                controller=controller,
                start_layer=getattr(args, "late_attn_start", 45),
                end_layer=getattr(args, "late_attn_end", 60),
            )
    controller.reset()

    # 可选：在 Infinity 模型上注册注意力形状调试 hook（仅打印 shape，不修改数值）
    if getattr(args, "debug_attn_shapes", False):
        register_attn_shape_hooks(pipe.infinity, max_print=10)

    
    prompt = base_prompt
    data = {
        'seed': 41,
        'prompt': prompt,
        'video_path': args.input_video,
    }
    if args.enable_rewriter:
        # Step 1: Rewrite the prompt using GPT
        # rewritten_prompt = prompt
        rewritten_prompt = pipe.gpt_model(
            prompt = ("Rewrite the following video descriptions, add more details of the subject and the camera movement to enhance the quality of the video. Do not use the word 'they' to refer to a single person or object. Concatenate all sentences together, not present them in paragraphs. Please rewrite with concise and clear language: "
                    + prompt),
            system_prompt=pipe.system_prompt,
        )
        print(f"Rewritten prompt: {rewritten_prompt}")
        prompt = rewritten_prompt
    data['prompt'] = prompt

    # 1. 读视频 -> [T,H,W,3] BGR
    frames_bgr = read_video_as_frames(args.input_video)
    if frames_bgr is None:
        raise RuntimeError(f"无法读取视频文件: {args.input_video}")
    print(f"[VAE-BSQ] 读取视频 {args.input_video}, frames={frames_bgr.shape[0]}")

    # 2. 转成 [1,3,T,H,W]，[-1,1]
    video_tensor = to_tensor_video(frames_bgr, device=device)
    print(f"[VAE-BSQ] video_tensor.shape={tuple(video_tensor.shape)}")
    
    # 3. 直接用 VAE 对输入视频做一次重建并保存，便于对照
    vae = build_vae(args.vae_path, device=device)
    with torch.no_grad():
        z_in, _, _ = vae.encode_for_raw_features(
            video_tensor,
            scale_schedule=None,
            return_residual_norm_per_scale=False,
            slice=None,
        )
        x_recon = vae.decode(z_in, slice=True)  # [B,3,T,H,W], 范围约 [-1,1]
        x_recon = (x_recon + 1) / 2
        x_recon = torch.clamp(x_recon, 0, 1)
        x_recon = x_recon.permute(0, 2, 3, 4, 1)  # [B,T,H,W,3]
        x_recon = (x_recon * 255).to(torch.uint8).cpu().numpy()  # RGB
        recon_save_path = osp.join('output', 'input_recon.mp4')
        os.makedirs(osp.dirname(recon_save_path), exist_ok=True)
        save_video(x_recon[0], fps=args.fps, save_filepath=recon_save_path)
        print(f"[VAE-BSQ] 直接重建并保存输入视频: {osp.abspath(recon_save_path)}")
    
    # 4. 横屏实验固定使用 720x1280 对应的模板。不要根据输入/latent 自适应选择，
    # 避免 VAE/decord/latent 轴顺序把 schedule 带偏。
    B, C, T, H, W = video_tensor.shape
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    args.h_div_w_template_override = h_div_w_template_
    print(f"[VAE-BSQ] 强制横屏 h_div_w_template_=0.571，匹配模板: {h_div_w_template_:.4f}")

    num_frames = T
    pt = (num_frames - 1) // 4 + 1
    schedule_dict = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"]
    schedule_pt = pt if pt in schedule_dict else min(schedule_dict.keys(), key=lambda x: abs(x - pt))
    scale_schedule = schedule_dict[schedule_pt]
    print(f"[VAE-BSQ] 强制横屏 scale_schedule: pt={pt} -> {schedule_pt}, 共 {len(scale_schedule)} 个尺度, last={scale_schedule[-1]}")

    all_bit_indices = vae_bsq2all_bit_indices(pipe, data, args)

    _, final_t, final_h, final_w, _ = all_bit_indices[-1].shape
    sched_t, sched_h, sched_w = scale_schedule[-1]
    if (final_h, final_w) != (sched_h, sched_w):
        raise RuntimeError(
            f"[VAE-BSQ] token H/W=({final_h},{final_w}) 与强制横屏 schedule H/W=({sched_h},{sched_w}) 不匹配"
        )
    print(f"[VAE-BSQ] token H/W matches forced horizontal schedule: ({final_h},{final_w})")
    
    # 检查 all_bit_indices 的内容
    print(f"[CHECK] all_bit_indices 检查:")
    print(f"  长度: {len(all_bit_indices)}")
    for si in [0, len(all_bit_indices)//2, len(all_bit_indices)-1]:
        if si < len(all_bit_indices):
            bit_idx = all_bit_indices[si]
            print(f"  scale {si}: shape={tuple(bit_idx.shape)}, dtype={bit_idx.dtype}")
            print(f"    min={bit_idx.min().item()}, max={bit_idx.max().item()}, mean={bit_idx.float().mean().item():.4f}")
            # 检查是否全为0或全为1
            if torch.allclose(bit_idx, torch.zeros_like(bit_idx)):
                print(f"    ⚠️ 警告: scale {si} 全为0！")
            elif torch.allclose(bit_idx, torch.ones_like(bit_idx)):
                print(f"    ⚠️ 警告: scale {si} 全为1！")
            # 检查数值范围（应该是0或1）
            unique_vals = torch.unique(bit_idx)
            print(f"    唯一值: {unique_vals.tolist()}")
    
    # 确保 perform_inference 使用相同的 scale_schedule
    print(f"[CHECK] 生成 all_bit_indices 时使用的 scale_schedule: {scale_schedule[:3] if scale_schedule else None}...")
    print(f"[CHECK] 生成 all_bit_indices 时使用的 h_div_w_template_: {h_div_w_template_:.4f}")
    
    # 保存第一次生成时实际使用的 idx（用于对比）
    first_gen_used_indices = []
    args.sampling_func = src_img_sampling_func_wrapper(all_bit_indices, scale_schedule=scale_schedule, args=args, spatial_patchify=False, save_used_indices=first_gen_used_indices)
    # 传入相同的 scale_schedule 和 h_div_w_template_，确保一致性
    output_dict = perform_inference(pipe, data, args, scale_schedule_override=scale_schedule, h_div_w_template_override=h_div_w_template_)
    save_dir = 'output'
    idx_Blv_cache_list = output_dict['idx_Blv_cache_list']
    prob_cache_list = output_dict['prob_cache_list']
    first_video = output_dict['output']

    save_video(first_video, fps=args.fps, save_filepath=osp.join(save_dir, 'first_video.mp4'))
    
    # 对比原始的 all_bit_indices 和第二次生成时模型自己生成的 bit_indices（从 cache 中获取）
    print(f"[COMPARE] 对比原始的 all_bit_indices 和第二次生成的 bit_indices (从 cache):")
    for si in [0, len(all_bit_indices)//2, len(all_bit_indices)-1]:
        if si < len(all_bit_indices):
            # 原始的 all_bit_indices
            original_idx_raw = all_bit_indices[si]  # [B, t, h, w, d] 或 [B, h, w, d]
            original_idx_flat = original_idx_raw.reshape(1, -1).cpu()
            
            # 从第二次生成的 cache 中获取对应的 idx
            if si < len(idx_Blv_cache_list):
                cache_idx_raw = idx_Blv_cache_list[si]  # [B, t, h, w, d] 或 [B, h, w, d]
                cache_idx_flat = cache_idx_raw.reshape(1, -1).cpu()
                
                # 对比
                if original_idx_flat.shape == cache_idx_flat.shape:
                    diff = (original_idx_flat != cache_idx_flat).sum().item()
                    total = original_idx_flat.numel()
                    match_rate = 1.0 - diff / total
                    print(f"  scale {si}: 匹配率={match_rate*100:.2f}% ({diff}/{total} 不同)")
                    print(f"    original shape={tuple(original_idx_raw.shape)}, cache shape={tuple(cache_idx_raw.shape)}")
                    if diff > 0:
                        # 显示前几个不同的位置
                        diff_mask = (original_idx_flat != cache_idx_flat)
                        diff_count = diff_mask.sum().item()
                        print(f"    不同位置数量: {diff_count}")
                        # 检查是否是顺序问题：如果重新排序后能匹配，说明是顺序问题
                        original_sorted = torch.sort(original_idx_flat.flatten())[0]
                        cache_sorted = torch.sort(cache_idx_flat.flatten())[0]
                        if torch.equal(original_sorted, cache_sorted):
                            print(f"    ⚠️ 数值相同但顺序不同！可能是 flatten 顺序问题")
                        else:
                            print(f"    ⚠️ 数值本身不同")
                            # 显示前几个不同的值
                            diff_indices = torch.nonzero(diff_mask, as_tuple=False)[:5]
                            for pos in diff_indices:
                                print(f"      位置 {pos.tolist()}: original={original_idx_flat[tuple(pos)].item()}, cache={cache_idx_flat[tuple(pos)].item()}")
                else:
                    print(f"  scale {si}: 形状不匹配 original={original_idx_flat.shape}, cache={cache_idx_flat.shape}")

    # ============= 第二阶段：使用按词注意力插值 + gamma 进行编辑生成 =============
    if cli_args.gamma_grid.strip():
        gamma_values = [float(x.strip()) for x in cli_args.gamma_grid.split(',') if x.strip()]
    else:
        gamma_values = [float(getattr(args, "gamma", 1.75))]
    print(f"[GRID] gamma_values={gamma_values}")

    for gamma_value in gamma_values:
        args.gamma = gamma_value
        print(f"[GRID] ===== Start edit gamma={gamma_value:.2f} =====")

        # 切换 controller 到编辑阶段：第二轮开始对 blended_words 对应 token 的 attention 做温和插值
        controller.start_edit()

        # 设置编辑 prompt，其 token 将用于 controller 内按词对齐
        data['prompt'] = edit_prompt

        # 使用第一轮的 idx/prob cache 构造编辑阶段的 sampling_func
        args.sampling_func = sampling_func_wrapper(
            idx_Blv_cache_list,
            prob_cache_list,
            gamma=gamma_value,
            scale_schedule=scale_schedule,
            front_attn_max_layers=getattr(controller, "max_layers", None),
            attn_control_start=getattr(args, "late_attn_start", None),
            attn_control_end=getattr(args, "late_attn_end", None),
        )

        # 第二次生成也使用相同的 scale_schedule，确保一致性
        output_dict_edit = perform_inference(
            pipe,
            data,
            args,
            scale_schedule_override=scale_schedule,
            h_div_w_template_override=h_div_w_template_,
        )

        edited_video = output_dict_edit['output']

        gamma_tag = f"{gamma_value:.1f}".replace('.', 'p')
        gen_video_path = osp.join(os.path.join(save_dir, 'gen_videos'), f'mask113edit7trys_gamma{gamma_tag}.mp4')
        os.makedirs(osp.dirname(gen_video_path), exist_ok=True)
        save_video(edited_video, fps=args.fps, save_filepath=gen_video_path)
        print(f"[GRID] Video generation done: gamma={gamma_value:.2f}, gen_video_path={gen_video_path}")
