#!/usr/bin/env python3
"""Anonymous single-video editing entry point for InfinityStar."""

import argparse
import importlib.util
import json
import os
import os.path as osp
import subprocess
import sys
import time
import warnings


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

# InfinityStar does not use audio. Some environments contain a torchaudio build
# that is incompatible with the installed PyTorch; keep it out of Transformers'
# optional dependency discovery for this process.
_original_find_spec = importlib.util.find_spec


def _find_spec_without_torchaudio(name, *args, **kwargs):
    if name == "torchaudio" or name.startswith("torchaudio."):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_torchaudio

import numpy as np
import torch

# Transformers 5 moved this helper. Retain compatibility with both layouts.
import transformers.modeling_utils as _transformers_modeling_utils

if not hasattr(_transformers_modeling_utils, "load_sharded_checkpoint"):
    from transformers.trainer_utils import load_sharded_checkpoint

    _transformers_modeling_utils.load_sharded_checkpoint = load_sharded_checkpoint

from infinity.models.attention_control import register_attention_control
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta
from tools.edit8 import Args, InferencePipe, perform_inference, perform_inference_phase1_fast
from tools.run_infinity import save_video
from tools.editing_utils import (
    reset_attention_debug,
    save_attention_visualization,
    spatial_gamma_sampling,
)
from tools.spatial_gamma import (
    FirstRepAttentionStore,
    build_spatial_gamma_maps,
    find_token_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training-free, attention-guided editing of one InfinityStar video"
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--target-prompt", required=True)
    parser.add_argument(
        "--target-word",
        required=True,
        help="Source-prompt word or phrase used as the attention anchor",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--name", default="edited")

    parser.add_argument("--model-path", default="infinitystar_8b_720p_weights")
    parser.add_argument("--vae-path", default="infinitystar_videovae.pth")
    parser.add_argument("--text-encoder-path", default="text_encoder/flan-t5-xl-official")

    parser.add_argument("--gamma-start", type=float, default=2.0)
    parser.add_argument("--gamma-fg", type=float, default=1.60)
    parser.add_argument("--gamma-bg", type=float, default=1.78)
    parser.add_argument("--image-transition", type=int, default=5)
    parser.add_argument("--video-transition", type=int, default=4)
    parser.add_argument("--transition-width", type=float, default=0.06)
    parser.add_argument("--attn-center", type=float, default=0.5)
    parser.add_argument("--attn-width", type=float, default=0.1)
    parser.add_argument("--attn-layers", type=int, default=5)
    parser.add_argument("--attn-max-seq-len", type=int, default=1200)
    parser.add_argument("--early-stop-si", type=int, default=25)
    parser.add_argument("--early-stop-default-prob", type=float, default=0.95)
    parser.add_argument(
        "--prune-from-si",
        type=int,
        default=None,
        help="Enable residual-guided token pruning from this scale index (disabled by default)",
    )
    parser.add_argument(
        "--prune-keep-ratio",
        type=float,
        default=0.50,
        help="Fraction of current-scale tokens retained when pruning is enabled",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the resolved configuration without loading models",
    )
    return parser.parse_args()


def validate(cli):
    if not osp.isfile(cli.input_video):
        raise FileNotFoundError(f"Input video not found: {cli.input_video}")
    for name in ("gamma_start", "gamma_fg", "gamma_bg"):
        value = getattr(cli, name)
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 2], got {value}")
    if cli.gamma_fg > cli.gamma_bg:
        warnings.warn("gamma-fg is greater than gamma-bg; the edit region is more constrained")
    if cli.attn_width <= 0 or cli.transition_width <= 0:
        raise ValueError("Attention and transition widths must be positive")
    if cli.early_stop_si < 0:
        raise ValueError("--early-stop-si must be non-negative")
    if cli.prune_from_si is not None and cli.prune_from_si < 0:
        raise ValueError("--prune-from-si must be non-negative")
    if not 0 < cli.prune_keep_ratio <= 1:
        raise ValueError("--prune-keep-ratio must be in (0, 1]")


def build_model_args(cli):
    args = Args()
    args.pn = "0.90M"
    args.fps = 16
    args.video_frames = 81
    args.model_path = cli.model_path
    args.checkpoint_type = "torch_shard"
    args.vae_path = cli.vae_path
    args.text_encoder_ckpt = cli.text_encoder_path
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0
    args.sampling_func = None
    args.input_video = cli.input_video
    return args


def resolved_config(cli):
    return {key: value for key, value in vars(cli).items()}


def main():
    cli = parse_args()
    validate(cli)
    os.makedirs(cli.output_dir, exist_ok=True)
    config_path = osp.join(cli.output_dir, f"{cli.name}_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(resolved_config(cli), handle, indent=2, ensure_ascii=False)
    print(json.dumps(resolved_config(cli), indent=2, ensure_ascii=False))
    if cli.dry_run:
        print(f"[DRY RUN] Configuration saved to {config_path}")
        return

    for path, label in (
        (cli.model_path, "Transformer checkpoint"),
        (cli.vae_path, "video VAE checkpoint"),
        (cli.text_encoder_path, "text encoder"),
    ):
        if not osp.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    args = build_model_args(cli)
    print("[MODEL] Loading checkpoints...")
    pipe = InferencePipe(args)
    print("[MODEL] Ready")

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames
    )
    video_hw_ratio = 720.0 / 1280.0
    h_template = h_div_w_templates[
        np.argmin(np.abs(h_div_w_templates - video_hw_ratio))
    ]
    pt = (args.video_frames - 1) // 4 + 1
    scale_schedule = dynamic_resolution_h_w[h_template][args.pn]["pt2scale_schedule"][pt]
    args.h_div_w_template_override = h_template
    args.video_hw_ratio = video_hw_ratio

    actual_source_prompt = (
        f"<<<t=5s>>>{cli.source_prompt}, "
        "Close-up on big objects, emphasize scale and detail"
    )
    token_indices = find_token_indices(
        pipe.text_tokenizer, actual_source_prompt, cli.target_word
    )
    if not token_indices:
        raise ValueError(
            f"Attention anchor {cli.target_word!r} was not found in the source prompt"
        )

    attn_store = FirstRepAttentionStore(max_seq_len=cli.attn_max_seq_len)
    register_attention_control(
        pipe.infinity,
        attn_store,
        max_seq_len=cli.attn_max_seq_len,
        min_seq_len=0,
        max_layers=cli.attn_layers,
        scale_whitelist=None,
    )
    attn_store.reset()

    source_data = {
        "seed": cli.seed,
        "prompt": cli.source_prompt,
        "video_path": cli.input_video,
    }
    phase1_start = time.time()
    source_output, _ = perform_inference_phase1_fast(
        pipe,
        source_data,
        args,
        scale_schedule=scale_schedule,
        h_div_w_template=h_template,
        early_stop_si=cli.early_stop_si,
        default_prob=cli.early_stop_default_prob,
    )
    phase1_seconds = time.time() - phase1_start
    idx_cache = source_output["idx_Blv_cache_list"]
    prob_cache = source_output["prob_cache_list"]

    attention_dir = osp.join(cli.output_dir, f"{cli.name}_attention")
    save_attention_visualization(
        attention_store=attn_store,
        tokenizer=pipe.text_tokenizer,
        prompt=actual_source_prompt,
        target_word=cli.target_word,
        scale_schedule=scale_schedule,
        output_dir=attention_dir,
    )

    image_repetitions = np.array(json.loads(args.image_scale_repetition))
    video_repetitions = np.array(json.loads(args.video_scale_repetition))
    gamma_maps = build_spatial_gamma_maps(
        attn_store=attn_store,
        tokenizer=pipe.text_tokenizer,
        prompt=actual_source_prompt,
        target_word=cli.target_word,
        scale_schedule=scale_schedule,
        image_scale_repetition=image_repetitions,
        video_scale_repetition=video_repetitions,
        gamma_start=cli.gamma_start,
        gamma_end_fg=cli.gamma_fg,
        gamma_end_bg=cli.gamma_bg,
        image_transition=cli.image_transition,
        video_transition=cli.video_transition,
        transition_width=cli.transition_width,
        attn_center=cli.attn_center,
        attn_width=cli.attn_width,
    )

    reset_attention_debug(pipe.infinity)
    args.sampling_func = spatial_gamma_sampling(
        idx_cache,
        prob_cache,
        gamma_maps=gamma_maps,
        default_gamma=cli.gamma_start,
    )
    target_data = {
        "seed": cli.seed,
        "prompt": cli.target_prompt,
        "video_path": cli.input_video,
    }
    phase2_start = time.time()
    target_output = perform_inference(
        pipe,
        target_data,
        args,
        scale_schedule_override=scale_schedule,
        h_div_w_template_override=h_template,
        token_prune_from_si=cli.prune_from_si,
        token_prune_mode="residual",
        token_prune_keep_ratio=cli.prune_keep_ratio,
    )
    phase2_seconds = time.time() - phase2_start

    output_path = osp.join(cli.output_dir, f"{cli.name}.mp4")
    save_video(target_output["output"][0], fps=args.fps, save_filepath=output_path)

    if not cli.skip_compare:
        compare_path = osp.join(cli.output_dir, f"{cli.name}_compare.mp4")
        command = [
            "ffmpeg", "-y", "-i", cli.input_video, "-i", output_path,
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-an", compare_path,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            warnings.warn("ffmpeg comparison video failed; the edited video is valid")

    print(f"[DONE] Phase 1: {phase1_seconds:.1f}s")
    print(f"[DONE] Phase 2: {phase2_seconds:.1f}s")
    print(f"[DONE] Result: {output_path}")


if __name__ == "__main__":
    main()
