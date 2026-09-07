# 🎬 Edit-VAR: Taming Visual Autoregressive Model for Precise Video Editing

Official implementation of **Edit-VAR**, a training-free and inversion-free video editing framework operating in the discrete token space of a visual autoregressive model.

🔗 [GitHub Repository](https://github.com/chongbozhao3-coder/Edit-VAR)

## 🔍 Method Overview

Edit-VAR performs editing through a two-phase coarse-to-fine generation process. In Phase 1, the source video is encoded into discrete token indices, while source-conditioned token probabilities and cross-attention signals are cached. In Phase 2, the source and target distributions are compared to determine whether each token should be preserved or regenerated. The attention signal produces token-wise preservation strengths, and the scale schedule adjusts these strengths over the autoregressive hierarchy; at late scales, dynamic source likelihoods are replaced by a fixed preservation bias, relaxing source constraints so that the target-conditioned model can regenerate fine details more freely. Residual-guided token pruning is provided as an optional inference acceleration mechanism.

## 🗂️ Repository Structure

```text
.
├── launch.sh                 # Unified launch interface
├── edit_video.py             # Main two-phase editing pipeline
├── tools/
│   ├── spatial_gamma.py      # Attention-guided gamma scheduling
│   ├── editing_utils.py      # Sampling and visualization utilities
│   ├── edit8.py              # Video encoding and inference pipeline
│   └── run_infinity.py       # Model construction and generation utilities
└── infinity/                 # Visual autoregressive backbone
```

## 🤗 Model Checkpoints

Download the InfinityStar checkpoints separately and arrange them as follows:

```text
.
├── infinitystar_8b_720p_weights/
├── infinitystar_videovae.pth
└── text_encoder/
    └── flan-t5-xl-official/
```

## 🛠️ Installation

Install PyTorch first with a CUDA build that supports the target GPU, then install the remaining dependencies. The configuration below is the environment used for the end-to-end release test (Python 3.11, PyTorch 2.7.1, CUDA 12.8, and FlashAttention 2.8.3):

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Do not blindly downgrade an existing environment to an older PyTorch wheel: recent GPU architectures require a wheel compiled for their CUDA compute capability.

The released 8B, 720p, 81-frame configuration is intended for a high-memory CUDA GPU. `ffmpeg` is optional and is used only to create the side-by-side comparison video.

## 🚀 Launch

All editing runs use the unified `launch.sh` entry point:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> bash launch.sh \
  --input-video <INPUT_VIDEO> \
  --source-prompt "<SOURCE_PROMPT>" \
  --target-prompt "<TARGET_PROMPT>" \
  --target-word "<SOURCE_ATTENTION_ANCHOR>" \
  --output-dir <OUTPUT_DIRECTORY> \
  --name <OUTPUT_NAME>
```

`--target-word` specifies the source-side word or phrase used to extract the attention signal and must occur in `--source-prompt`. For object addition, use the source-side word describing the intended insertion location.

The command writes the edited video, attention visualizations, a side-by-side comparison, and the resolved configuration. Set `PYTHON_BIN` when a non-default Python executable is required. Run `bash launch.sh --help` for the complete interface.

## 📥 Inputs and Outputs

The unified interface requires a source video, its source prompt, the target editing prompt, and a source-side attention anchor. The input video must be readable by Decord. The source and target prompts should describe the same scene except for the intended edit.

Each run creates the following files:

```text
<OUTPUT_DIRECTORY>/
├── <OUTPUT_NAME>.mp4
├── <OUTPUT_NAME>_compare.mp4
├── <OUTPUT_NAME>_config.json
└── <OUTPUT_NAME>_attention/
    ├── attention_image.png
    └── attention_video.png
```

The comparison video is omitted when `--skip-compare` is specified or when `ffmpeg` is unavailable. The configuration file records all resolved command-line arguments for reproducibility.

## 🎛️ Main Hyperparameters

The default command reproduces the current paper configuration:

| Option | Default | Effect |
|---|---:|---|
| `--gamma-fg` | 1.60 | Lower values strengthen the edit; higher values preserve more source content. |
| `--gamma-bg` | 1.78 | Higher values more strongly preserve regions outside the edit. |
| `--attn-center` | 0.50 | Lower values expand the edit region; higher values shrink it. |
| `--attn-width` | 0.10 | Controls how sharply attention is mapped to foreground/background strength. |
| `--image-transition` | 5 | Scale-schedule transition center in the image tower. |
| `--video-transition` | 4 | Scale-schedule transition center in the video tower. |
| `--early-stop-si` | 25 | Skips the Phase 1 Transformer at scales `si >= 25`. |
| `--prune-from-si` | disabled | Optionally enables residual-guided token pruning from the selected scale. |
| `--prune-keep-ratio` | 0.50 | Fraction of current-scale tokens retained when pruning is enabled. |

Tune `--gamma-fg` first and verify the attention visualization before interpreting a parameter sweep. Token pruning remains opt-in and can be enabled with `--prune-from-si <SCALE_INDEX> --prune-keep-ratio <RATIO>`.

## ✅ Dry-run Validation

Append `--dry-run` to the unified launch command to validate paths and arguments without loading checkpoints.
