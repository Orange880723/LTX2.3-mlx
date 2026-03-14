# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

Install from source:

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

Supported models:

### LTX-2
[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a 19B parameter video generation model from Lightricks.

## Features

- Text-to-video (T2V) and Image-to-video (I2V) generation
- Three pipeline modes: Distilled, Dev, and Dev Two-Stage
- Synchronized audio-video generation (experimental)
- LoRA support (including HuggingFace repos)
- Prompt enhancement via Gemma
- 2x spatial upscaling for images and videos
- Optimized for Apple Silicon using MLX

## Usage

### Pipelines

mlx-video supports three pipeline types via the `--pipeline` flag:

| Pipeline | Description | CFG | Stages | Speed |
|----------|-------------|-----|--------|-------|
| `distilled` (default) | Fixed sigma schedule, no CFG | No | 2 (8+3 steps) | Fastest |
| `dev` | Dynamic sigmas, constant CFG | Yes | 1 (30 steps) | Medium |
| `dev-two-stage` | Dev + LoRA refinement | Yes (stage 1) | 2 (30+3 steps) | Slowest, highest quality |

### Text-to-Video

```bash
# Distilled (default) - fast, two-stage
uv run mlx_video.generate --prompt "Two dogs wearing sunglasses, cinematic, sunset" -n 97 --width 768

# Dev - single-stage with CFG
uv run mlx_video.generate --pipeline dev --prompt "A cinematic scene" --cfg-scale 3.0

# Dev two-stage - dev + LoRA refinement (highest quality)
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" \
    -n 145 --width 1024 --height 768 \
    --model-repo prince-canuma/LTX-2-dev \
    --cfg-scale 3.0 --lora-strength 0.8 \
    --enhance-prompt
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

### Image-to-Video

```bash
# Distilled I2V
uv run mlx_video.generate --prompt "A person dancing" --image photo.jpg

# Dev I2V
uv run mlx_video.generate --pipeline dev --prompt "Waves crashing" --image beach.png --cfg-scale 3.5
```

### Audio-Video (experimental)

```bash
uv run mlx_video.generate --prompt "Ocean waves crashing" --audio
uv run mlx_video.generate --pipeline dev --prompt "A jazz band playing" --audio --enhance-prompt

# With full guidance (STG + modality_scale, matches PyTorch defaults)
uv run mlx_video.generate --pipeline dev --prompt "Ocean waves crashing" --audio \
    --stg-scale 1.0 --stg-blocks 29 --modality-scale 3.0
```

### LoRA

LoRA weights can be loaded from a file, directory, or HuggingFace repo:

```bash
# From HuggingFace repo
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "Camera dolly out of a forest" \
    --lora-path Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out \
    --lora-strength 1.0

# From local file
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "A scene" \
    --lora-path ./my-lora/weights.safetensors

# From local directory (auto-detects .safetensors file)
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "A scene" \
    --lora-path ./LTX-2-distilled/lora
```

### Upscaling

```bash
# Upscale an image 2x
uv run mlx_video.upscale --input photo.png --output upscaled.png

# Upscale a video 2x
uv run mlx_video.upscale --input video.mp4 --output upscaled.mp4

# Upscale with refinement (higher quality, requires text prompt)
uv run mlx_video.upscale --input video.mp4 --output upscaled.mp4 --refine --prompt "A cinematic scene"
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--pipeline` | `distilled` | Pipeline type: `distilled`, `dev`, or `dev-two-stage` |
| `--height`, `-H` | 512 | Output height (divisible by 64 for two-stage, 32 for dev) |
| `--width`, `-W` | 512 | Output width (divisible by 64 for two-stage, 32 for dev) |
| `--num-frames`, `-n` | 33 | Number of frames (must be 1 + 8*k) |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output-path`, `-o` | output.mp4 | Output video path |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |
| `--text-encoder-repo` | None | Separate text encoder repo (if not in model repo) |
| `--save-frames` | false | Save individual frames as images |
| `--enhance-prompt` | false | Enhance prompt using Gemma |
| `--image`, `-i` | None | Conditioning image for I2V |
| `--image-strength` | 1.0 | Conditioning strength for I2V |
| `--audio`, `-a` | false | Enable synchronized audio generation |
| `--tiling` | `auto` | VAE tiling mode: `auto`, `none`, `aggressive`, `conservative` |
| `--stream` | false | Stream frames as they decode |

**Dev/Dev-Two-Stage options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 30 | Number of denoising steps |
| `--cfg-scale` | 3.0 | CFG guidance scale |
| `--cfg-rescale` | 0.7 | CFG rescale factor (reduces over-saturation) |
| `--negative-prompt` | (default) | Negative prompt for CFG |
| `--apg` | false | Use Adaptive Projected Guidance (more stable for I2V) |
| `--stg-scale` | 0.0 | STG scale (PyTorch default: 1.0, requires `--audio`) |
| `--stg-blocks` | None | Transformer blocks for STG ([29] for LTX-2, [28] for LTX-2.3) |
| `--modality-scale` | 1.0 | Cross-modal guidance scale (PyTorch default: 3.0, requires `--audio`) |

**Dev-Two-Stage LoRA options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-path` | auto-detect | Path to LoRA file, directory, or HuggingFace repo |
| `--lora-strength` | 1.0 | LoRA merge strength |

## How It Works

### Distilled Pipeline (default)
1. **Stage 1**: Generate at half resolution with 8 denoising steps (fixed sigmas)
2. **Upsample**: 2x spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

### Dev Pipeline
1. **Generate**: Full resolution with configurable steps and constant CFG
2. **Decode**: VAE decoder converts latents to RGB video

### Dev Two-Stage Pipeline
1. **Stage 1**: Dev denoising at half resolution with CFG
2. **Upsample**: 2x spatial upsampling via LatentUpsampler
3. **Stage 2**: Distilled refinement at full resolution with LoRA weights (3 steps, no CFG)
4. **Decode**: VAE decoder converts latents to RGB video

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0

## Model Specifications

- **Transformer**: 48 layers, 32 attention heads, 128 dim per head (19B parameters)
- **Latent channels**: 128
- **Text encoder**: Gemma 3 with 3840-dim output
- **Audio**: Synchronized audio-video with separate audio VAE and vocoder

## License

MIT
