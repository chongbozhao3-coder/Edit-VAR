"""Utilities used by the public video-editing entry point."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from einops import rearrange


def reset_attention_debug(model):
    for module in model.modules():
        if hasattr(module, "use_ptp_debug"):
            module.use_ptp_debug = False
        if hasattr(module, "ptp_controller"):
            delattr(module, "ptp_controller")


def spatial_gamma_sampling(idx_cache, prob_cache, gamma_maps, default_gamma=2.0):
    """Build the Phase 2 sampler that biases decisions toward source bits."""

    def sampling_func(logits, scale_index, rng=None):
        del rng
        if idx_cache is None:
            return None, None
        if not hasattr(sampling_func, "step"):
            sampling_func.step = 0
        step = sampling_func.step
        if step >= len(idx_cache):
            return None, None
        source_indices = idx_cache[step]
        source_probs = prob_cache[step]
        sampling_func.step += 1

        batch, length, vocabulary = logits.shape
        edit_probs = logits.softmax(dim=-1).view(-1, vocabulary)
        source_indices = rearrange(
            source_indices.reshape(batch, -1), "B l -> (B l) 1"
        )
        source_probs = rearrange(
            source_probs.reshape(batch, -1, 2), "B l d -> (B l) d"
        )

        if step < len(gamma_maps):
            spatial_gamma = gamma_maps[step].to(edit_probs.device)
            spatial_tokens = spatial_gamma.shape[0]
            bits_per_token = length // spatial_tokens if spatial_tokens else 1
            gamma = (
                spatial_gamma.repeat_interleave(bits_per_token)
                if bits_per_token > 1
                else spatial_gamma
            )
            gamma = gamma.repeat(batch).unsqueeze(1)
        else:
            gamma = torch.full(
                (batch * length, 1), default_gamma, device=edit_probs.device
            )

        source_bit_probs = torch.gather(source_probs, 1, source_indices)
        bias = (gamma - source_bit_probs).clamp(min=0)
        edit_probs = torch.scatter(
            edit_probs,
            1,
            source_indices,
            torch.gather(edit_probs, 1, source_indices) + bias,
        )
        selected_probs, selected_indices = torch.topk(edit_probs, k=1, dim=-1)

        if not hasattr(sampling_func, "printed_scales"):
            sampling_func.printed_scales = set()
        if scale_index not in sampling_func.printed_scales:
            sampling_func.printed_scales.add(scale_index)
            print(
                f"[SPATIAL GAMMA] scale={scale_index}, step={step}, "
                f"range=[{gamma.min().item():.2f}, {gamma.max().item():.2f}], "
                f"mean={gamma.mean().item():.3f}, bits={length}"
            )
        return selected_indices.view(batch, length, 1), selected_probs

    return sampling_func


def save_attention_visualization(
    attention_store,
    tokenizer,
    prompt,
    target_word,
    scale_schedule,
    output_dir,
):
    """Save image- and video-tower attention maps for one source anchor."""
    from tools.spatial_gamma import (
        _extract_attn_1d,
        _normalize,
        _upsample_attn,
        find_token_indices,
    )

    os.makedirs(output_dir, exist_ok=True)
    token_indices = find_token_indices(tokenizer, prompt, target_word)
    if not token_indices:
        raise ValueError(f"Attention anchor {target_word!r} was not found")

    raw_attention = {}
    for scale_index, (frames, height, width) in enumerate(scale_schedule):
        attention = _extract_attn_1d(
            attention_store.step_store,
            scale_index,
            token_indices,
            frames * height * width,
        )
        if attention is not None:
            raw_attention[scale_index] = _normalize(attention)

    best_image_scale = max(
        (i for i in raw_attention if scale_schedule[i][0] == 1), default=None
    )
    best_video_scale = max(
        (i for i in raw_attention if scale_schedule[i][0] > 1), default=None
    )
    attention_by_scale = {}
    for scale_index, (frames, height, width) in enumerate(scale_schedule):
        if scale_index in raw_attention:
            attention_by_scale[scale_index] = raw_attention[scale_index]
            continue
        reference = (
            best_video_scale if frames > 1 else best_image_scale
        )
        if reference is None:
            reference = best_image_scale if best_image_scale is not None else best_video_scale
        if reference is None:
            attention_by_scale[scale_index] = torch.zeros(frames * height * width)
        else:
            attention_by_scale[scale_index] = _normalize(
                _upsample_attn(
                    raw_attention[reference],
                    scale_schedule[reference],
                    (frames, height, width),
                )
            )

    figure, axes = plt.subplots(3, 5, figsize=(22, 13))
    for scale_index in range(15):
        axis = axes[scale_index // 5][scale_index % 5]
        if scale_index >= len(scale_schedule):
            axis.axis("off")
            continue
        frames, height, width = scale_schedule[scale_index]
        values = attention_by_scale[scale_index].numpy().reshape(frames, height, width)
        image = axis.imshow(values[0], cmap="hot", vmin=0, vmax=1, aspect="auto")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        source = "direct" if scale_index in raw_attention else "upsampled"
        axis.set_title(f"scale {scale_index} ({height}x{width}) [{source}]")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f'Image tower attention: "{target_word}"')
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "attention_image.png"), dpi=150)
    plt.close(figure)

    video_count = min(15, len(scale_schedule) - 15)
    if video_count <= 0:
        return
    rows = (video_count + 4) // 5
    figure, axes = plt.subplots(rows, 5, figsize=(22, rows * 4.3), squeeze=False)
    for offset in range(rows * 5):
        axis = axes[offset // 5][offset % 5]
        if offset >= video_count:
            axis.axis("off")
            continue
        scale_index = offset + 15
        frames, height, width = scale_schedule[scale_index]
        values = attention_by_scale[scale_index].numpy().reshape(frames, height, width)
        image = axis.imshow(
            values[frames // 2], cmap="hot", vmin=0, vmax=1, aspect="auto"
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        source = "direct" if scale_index in raw_attention else "upsampled"
        axis.set_title(
            f"scale {scale_index} ({frames}x{height}x{width}) [{source}]"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f'Video tower attention: "{target_word}" (middle frame)')
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "attention_video.png"), dpi=150)
    plt.close(figure)
