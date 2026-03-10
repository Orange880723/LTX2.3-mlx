"""Unified video and audio-video generation pipeline for LTX-2.

Supports both distilled (two-stage with upsampling) and dev (single-stage with CFG) pipelines.
"""

import argparse
import math
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel

# Rich console for styled output
console = Console()


from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality

from mlx_video.utils import to_denoised, load_image, prepare_image_for_encoding, get_model_path
from mlx_video.models.ltx.video_vae.decoder import VideoDecoder
from mlx_video.models.ltx.video_vae import VideoEncoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents
from mlx_video.conditioning import VideoConditionByLatentIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, apply_denoise_mask


class PipelineType(Enum):
    """Pipeline type selector."""
    DISTILLED = "distilled"  # Two-stage with upsampling, fixed sigmas, no CFG
    DEV = "dev"              # Single-stage, dynamic sigmas, CFG


# Distilled model sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Dev model scheduling constants
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR  # 25

# Default negative prompt for CFG (dev pipeline)
# Matches PyTorch LTX-2 reference InferenceConfig default
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    """Compute CFG delta for classifier-free guidance.

    Args:
        cond: Conditional prediction
        uncond: Unconditional prediction
        scale: CFG guidance scale

    Returns:
        Delta to add to unconditional for CFG: (scale - 1) * (cond - uncond)
    """
    return (scale - 1.0) * (cond - uncond)


def apg_delta(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> mx.array:
    """Compute APG (Adaptive Projected Guidance) delta.

    Decomposes guidance into parallel and orthogonal components relative to
    the conditional prediction, providing more stable guidance for I2V.

    Based on: https://arxiv.org/abs/2407.12173

    Args:
        cond: Conditional prediction (x0_pos)
        uncond: Unconditional prediction (x0_neg)
        scale: Guidance strength (same as CFG scale)
        eta: Weight for parallel component (1.0 = keep full parallel)
        norm_threshold: Clamp guidance norm to this value (0 = no clamping)

    Returns:
        Delta to add to unconditional for APG guidance
    """
    guidance = cond - uncond

    # Optionally clamp guidance norm for stability
    if norm_threshold > 0:
        guidance_norm = mx.sqrt(mx.sum(guidance ** 2, axis=(-1, -2, -3), keepdims=True) + 1e-8)
        scale_factor = mx.minimum(mx.ones_like(guidance_norm), norm_threshold / guidance_norm)
        guidance = guidance * scale_factor

    # Project guidance onto cond direction
    batch_size = cond.shape[0]
    cond_flat = mx.reshape(cond, (batch_size, -1))
    guidance_flat = mx.reshape(guidance, (batch_size, -1))

    # Projection coefficient: (guidance · cond) / (cond · cond)
    dot_product = mx.sum(guidance_flat * cond_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(cond_flat ** 2, axis=1, keepdims=True) + 1e-8
    proj_coeff = dot_product / squared_norm

    # Reshape back and compute parallel/orthogonal components
    proj_coeff = mx.reshape(proj_coeff, (batch_size,) + (1,) * (cond.ndim - 1))
    g_parallel = proj_coeff * cond
    g_orth = guidance - g_parallel

    # Combine with eta weighting parallel component
    g_apg = g_parallel * eta + g_orth

    return g_apg * (scale - 1.0)


def ltx2_scheduler(
    steps: int,
    num_tokens: Optional[int] = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    """LTX-2 scheduler for sigma generation (dev model).

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.

    Args:
        steps: Number of inference steps
        num_tokens: Number of latent tokens (F*H*W). If None, uses MAX_SHIFT_ANCHOR
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch sigmas to terminal value
        terminal: Terminal sigma value for stretching

    Returns:
        Array of sigma values of shape (steps + 1,)
    """
    tokens = num_tokens if num_tokens is not None else MAX_SHIFT_ANCHOR
    sigmas = np.linspace(1.0, 0.0, steps + 1)

    # Compute shift based on token count
    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    # Apply shift transformation
    power = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        sigmas = np.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

    # Stretch sigmas to terminal value
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return mx.array(sigmas, dtype=mx.float32)


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for RoPE in pixel space.

    Args:
        batch_size: Batch size
        num_frames: Number of frames (latent)
        height: Height (latent)
        width: Width (latent)
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)
        fps: Frames per second (default 24.0)
        causal_fix: Apply causal fix for first frame (default True)

    Returns:
        Position grid of shape (B, 3, num_patches, 2) in pixel space
        where dim 2 is [start, end) bounds for each patch
    """
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    # Compute temporal division in bfloat16 to match PyTorch's precision behavior
    # This ensures RoPE frequencies are computed identically to the reference implementation
    temporal_coords = mx.array(pixel_coords[:, 0, :, :], dtype=mx.bfloat16)
    fps_bf16 = mx.array(fps, dtype=mx.bfloat16)
    temporal_coords = temporal_coords / fps_bf16
    mx.eval(temporal_coords)
    pixel_coords[:, 0, :, :] = np.array(temporal_coords.astype(mx.float32))

    return mx.array(pixel_coords, dtype=mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE."""
    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


# =============================================================================
# Distilled Pipeline Denoising (no CFG, fixed sigmas)
# =============================================================================

def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    audio_latents: Optional[mx.array] = None,
    audio_positions: Optional[mx.array] = None,
    audio_embeddings: Optional[mx.array] = None,
) -> tuple[mx.array, Optional[mx.array]]:
    """Run denoising loop for distilled pipeline (no CFG)."""
    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    # Keep latents in float32 throughout to avoid quantization noise accumulation.
    latents = latents.astype(mx.float32)
    if enable_audio:
        audio_latents = audio_latents.astype(mx.float32)

    desc = "[cyan]Denoising A/V[/]" if enable_audio else "[cyan]Denoising[/]"
    num_steps = len(sigmas) - 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(desc, total=num_steps)

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast to model dtype for transformer input
            latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            video_modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                sigma=mx.full((b,), sigma, dtype=dtype),
            )

            audio_modality = None
            if enable_audio:
                ab, ac, at, af = audio_latents.shape
                audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

                audio_modality = Modality(
                    latent=audio_flat,
                    timesteps=mx.full((ab, at), sigma, dtype=dtype),
                    positions=audio_positions,
                    context=audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    sigma=mx.full((ab,), sigma, dtype=dtype),
                )

            velocity, audio_velocity = transformer(video=video_modality, audio=audio_modality)
            mx.eval(velocity)
            if audio_velocity is not None:
                mx.eval(audio_velocity)

            # Compute denoised (x0) using per-token timesteps in float32
            # x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

            audio_denoised = None
            if enable_audio and audio_velocity is not None:
                ab, ac, at, af = audio_latents.shape
                audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
                audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
                audio_denoised = audio_latents - sigma_f32 * audio_velocity.astype(mx.float32)

            if state is not None:
                denoised = apply_denoise_mask(denoised, state.clean_latent.astype(mx.float32), state.denoise_mask)

            mx.eval(denoised)
            if audio_denoised is not None:
                mx.eval(audio_denoised)

            # Euler step in float32 (latents stay in float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
                if enable_audio and audio_denoised is not None:
                    audio_latents = audio_denoised + sigma_next_f32 * (audio_latents - audio_denoised) / sigma_f32
            else:
                latents = denoised
                if enable_audio and audio_denoised is not None:
                    audio_latents = audio_denoised

            mx.eval(latents)
            if enable_audio:
                mx.eval(audio_latents)

            progress.advance(task)

    return latents.astype(dtype), audio_latents.astype(dtype) if enable_audio else None


# =============================================================================
# Dev Pipeline Denoising (with CFG, dynamic sigmas)
# =============================================================================

def denoise_dev(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
) -> mx.array:
    """Run denoising loop for dev pipeline with CFG or APG guidance.

    Args:
        use_apg: Use Adaptive Projected Guidance instead of standard CFG.
                 APG decomposes guidance into parallel/orthogonal components
                 for more stable I2V generation.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
    """
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    # Keep latents in float32 throughout the denoising loop to avoid
    # quantization noise accumulation over many steps.
    # Model input is cast to model dtype; all denoising math stays in float32.
    latents = latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    num_steps = len(sigmas_list) - 1

    # Precompute RoPE once
    precomputed_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_rope)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Denoising (CFG)[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast to model dtype for transformer input
            latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1)).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            sigma_array = mx.full((b,), sigma, dtype=dtype)

            # Positive conditioning pass
            video_modality_pos = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
                sigma=sigma_array,
            )
            velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

            # Convert velocity to x0 (denoised) using per-token timesteps
            # Matches PyTorch's X0Model: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (correct regardless of velocity)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_pos_f32 = latents_flat_f32 - timesteps_f32 * velocity_pos.astype(mx.float32)

            if use_cfg:
                # Negative conditioning pass
                video_modality_neg = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=text_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                    sigma=sigma_array,
                )
                velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

                # Convert negative velocity to x0 using per-token timesteps
                x0_neg_f32 = latents_flat_f32 - timesteps_f32 * velocity_neg.astype(mx.float32)

                # Apply guidance to x0 predictions
                # For conditioned tokens: x0_pos = x0_neg = latent, so delta = 0
                if use_apg:
                    # APG: decompose into parallel/orthogonal components for stability
                    x0_guided_f32 = x0_pos_f32 + apg_delta(
                        x0_pos_f32, x0_neg_f32, cfg_scale,
                        eta=apg_eta, norm_threshold=apg_norm_threshold
                    )
                else:
                    # Standard CFG
                    x0_guided_f32 = x0_pos_f32 + (cfg_scale - 1.0) * (x0_pos_f32 - x0_neg_f32)
            else:
                x0_guided_f32 = x0_pos_f32

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            denoised = mx.reshape(mx.transpose(x0_guided_f32, (0, 2, 1)), (b, c, f, h, w))

            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if state is not None:
                denoised = apply_denoise_mask(denoised, state.clean_latent.astype(mx.float32), state.denoise_mask)

            # Euler step in float32 (latents stay in float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
            else:
                latents = denoised

            mx.eval(latents)
            progress.advance(task)

    return latents.astype(dtype)


def denoise_dev_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for dev pipeline with CFG/APG and audio.

    Args:
        cfg_rescale: Rescale factor for CFG (0.0-1.0). Higher values blend the CFG result
                     towards the positive-only prediction, helping reduce artifacts.
                     Default 0.0 means no rescaling (standard CFG).
        use_apg: Use Adaptive Projected Guidance instead of standard CFG for video.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
    """
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    # Keep latents in float32 throughout the denoising loop to avoid
    # bfloat16 quantization noise accumulation over many steps.
    # PyTorch keeps latents in float32; model input is cast to model dtype.
    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    num_steps = len(sigmas_list) - 1

    # Precompute video RoPE
    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )

    # Precompute audio RoPE
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Denoising A/V (CFG)[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]

            # Flatten video latents (cast to model dtype for transformer input)
            b, c, f, h, w = video_latents.shape
            num_video_tokens = f * h * w
            video_flat = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1)).astype(dtype)

            # Flatten audio latents (cast to model dtype for transformer input)
            ab, ac, at, af = audio_latents.shape
            audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

            # Compute timesteps
            if video_state is not None:
                denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
                video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

            audio_timesteps = mx.full((ab, at), sigma, dtype=dtype)

            # Positive conditioning pass
            video_modality_pos = Modality(
                latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                context=video_embeddings_pos, context_mask=None, enabled=True,
                positional_embeddings=precomputed_video_rope,
            )
            audio_modality_pos = Modality(
                latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                context=audio_embeddings_pos, context_mask=None, enabled=True,
                positional_embeddings=precomputed_audio_rope,
            )
            video_vel_pos, audio_vel_pos = transformer(video=video_modality_pos, audio=audio_modality_pos)
            mx.eval(video_vel_pos, audio_vel_pos)

            # Convert velocity to denoised (x0) using per-token timesteps
            # This matches PyTorch's X0ModelWrapper: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (velocity is irrelevant)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            # Use the float32 latents (not the bfloat16 model input) for precision
            video_flat_f32 = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))
            audio_flat_f32 = mx.reshape(mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af))
            video_timesteps_f32 = mx.expand_dims(video_timesteps.astype(mx.float32), axis=-1)
            audio_timesteps_f32 = mx.expand_dims(audio_timesteps.astype(mx.float32), axis=-1)

            video_x0_pos_f32 = video_flat_f32 - video_timesteps_f32 * video_vel_pos.astype(mx.float32)
            audio_x0_pos_f32 = audio_flat_f32 - audio_timesteps_f32 * audio_vel_pos.astype(mx.float32)

            if use_cfg:
                # Negative conditioning pass
                video_modality_neg = Modality(
                    latent=video_flat, timesteps=video_timesteps, positions=video_positions,
                    context=video_embeddings_neg, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_video_rope,
                )
                audio_modality_neg = Modality(
                    latent=audio_flat, timesteps=audio_timesteps, positions=audio_positions,
                    context=audio_embeddings_neg, context_mask=None, enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                )
                video_vel_neg, audio_vel_neg = transformer(video=video_modality_neg, audio=audio_modality_neg)
                mx.eval(video_vel_neg, audio_vel_neg)

                # Convert negative velocity to x0 using per-token timesteps
                video_x0_neg_f32 = video_flat_f32 - video_timesteps_f32 * video_vel_neg.astype(mx.float32)
                audio_x0_neg_f32 = audio_flat_f32 - audio_timesteps_f32 * audio_vel_neg.astype(mx.float32)

                # Apply guidance to x0 (denoised) predictions
                # For conditioned tokens: x0_pos = x0_neg = latent, so delta = 0 (no effect)
                if use_apg:
                    # APG for video (more stable for I2V), standard CFG for audio
                    video_x0_guided_f32 = video_x0_pos_f32 + apg_delta(
                        video_x0_pos_f32, video_x0_neg_f32, cfg_scale,
                        eta=apg_eta, norm_threshold=apg_norm_threshold
                    )
                else:
                    video_x0_guided_f32 = video_x0_pos_f32 + (cfg_scale - 1.0) * (video_x0_pos_f32 - video_x0_neg_f32)
                # Always use standard CFG for audio
                audio_x0_guided_f32 = audio_x0_pos_f32 + (cfg_scale - 1.0) * (audio_x0_pos_f32 - audio_x0_neg_f32)

                # Apply CFG rescale if enabled
                if cfg_rescale > 0.0:
                    video_x0_guided_f32 = cfg_rescale * video_x0_pos_f32 + (1.0 - cfg_rescale) * video_x0_guided_f32
                    audio_x0_guided_f32 = cfg_rescale * audio_x0_pos_f32 + (1.0 - cfg_rescale) * audio_x0_guided_f32
            else:
                video_x0_guided_f32 = video_x0_pos_f32
                audio_x0_guided_f32 = audio_x0_pos_f32

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            video_denoised_f32 = mx.reshape(mx.transpose(video_x0_guided_f32, (0, 2, 1)), (b, c, f, h, w))
            audio_denoised_f32 = mx.reshape(audio_x0_guided_f32, (ab, at, ac, af))
            audio_denoised_f32 = mx.transpose(audio_denoised_f32, (0, 2, 1, 3))

            # Post-process: blend denoised with clean latent using mask
            # Matches PyTorch's post_process_latent: denoised * mask + clean * (1 - mask)
            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if video_state is not None:
                clean_f32 = video_state.clean_latent.astype(mx.float32)
                mask_f32 = video_state.denoise_mask.astype(mx.float32)
                video_denoised_f32 = video_denoised_f32 * mask_f32 + clean_f32 * (1.0 - mask_f32)

            mx.eval(video_denoised_f32, audio_denoised_f32)

            # Euler step matching PyTorch: sample + velocity * dt
            # Latents stay in float32 throughout (matching PyTorch behavior)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                dt_f32 = sigma_next_f32 - sigma_f32

                video_velocity_f32 = (video_latents - video_denoised_f32) / sigma_f32
                video_latents = video_latents + video_velocity_f32 * dt_f32

                audio_velocity_f32 = (audio_latents - audio_denoised_f32) / sigma_f32
                audio_latents = audio_latents + audio_velocity_f32 * dt_f32
            else:
                video_latents = video_denoised_f32
                audio_latents = audio_denoised_f32

            mx.eval(video_latents, audio_latents)
            progress.advance(task)

    return video_latents, audio_latents


# =============================================================================
# Audio Loading and Processing
# =============================================================================

def load_audio_decoder(model_path: Path, pipeline: PipelineType):
    """Load audio VAE decoder."""
    from mlx_video.models.ltx.audio_vae import AudioDecoder

    decoder = AudioDecoder.from_pretrained(model_path / "audio_vae")

    return decoder


def load_vocoder(model_path: Path, pipeline: PipelineType):
    """Load vocoder for mel to waveform conversion."""
    from mlx_video.models.ltx.audio_vae import Vocoder

    vocoder = Vocoder.from_pretrained(model_path / "vocoder")

    return vocoder


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV file."""
    import wave

    if audio.ndim == 2:
        audio = audio.T

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/]")
        return False
    except FileNotFoundError:
        console.print("[red]FFmpeg not found. Please install ffmpeg.[/]")
        return False


# =============================================================================
# Unified Generate Function
# =============================================================================

def generate_video(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    pipeline: PipelineType = PipelineType.DISTILLED,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "auto",
    stream: bool = False,
    audio: bool = False,
    output_audio_path: Optional[str] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
):
    """Generate video using LTX-2 models.

    Supports two pipelines:
    - DISTILLED: Two-stage generation with upsampling, fixed sigma schedules, no CFG
    - DEV: Single-stage generation with dynamic sigmas and CFG

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        pipeline: Pipeline type (DISTILLED or DEV)
        negative_prompt: Negative prompt for CFG (dev pipeline only)
        height: Output video height (must be divisible by 32/64)
        width: Output video width (must be divisible by 32/64)
        num_frames: Number of frames (must be 1 + 8*k)
        num_inference_steps: Number of denoising steps (dev pipeline only)
        cfg_scale: Guidance scale for CFG (dev pipeline only)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V
        image_strength: Conditioning strength for I2V
        image_frame_idx: Frame index to condition for I2V
        tiling: Tiling mode for VAE decoding
        stream: Stream frames to output as they're decoded
        audio: Enable synchronized audio generation
        output_audio_path: Path to save audio file
        use_apg: Use Adaptive Projected Guidance instead of CFG (more stable for I2V)
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
    """
    start_time = time.time()

    # Validate dimensions
    divisor = 64 if pipeline == PipelineType.DISTILLED else 32
    assert height % divisor == 0, f"Height must be divisible by {divisor}, got {height}"
    assert width % divisor == 0, f"Width must be divisible by {divisor}, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        console.print(f"[yellow]⚠️  Number of frames must be 1 + 8*k. Using: {adjusted_num_frames}[/]")
        num_frames = adjusted_num_frames

    is_i2v = image is not None
    mode_str = "I2V" if is_i2v else "T2V"
    if audio:
        mode_str += "+Audio"

    pipeline_name = "DEV" if pipeline == PipelineType.DEV else "DISTILLED"
    header = f"[bold cyan]🎬 [{pipeline_name}] [{mode_str}] {width}x{height} • {num_frames} frames[/]"
    console.print(Panel(header, expand=False))
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/]")

    if pipeline == PipelineType.DEV:
        console.print(f"[dim]Steps: {num_inference_steps}, CFG: {cfg_scale}[/]")

    if is_i2v:
        console.print(f"[dim]Image: {image} (strength={image_strength}, frame={image_frame_idx})[/]")

    audio_frames = None
    if audio:
        audio_frames = compute_audio_frames(num_frames, fps)
        console.print(f"[dim]Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz[/]")

    # Get model path
    model_path = get_model_path(model_repo)
    text_encoder_path = model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)

    # Model weight file
    # Calculate latent dimensions
    if pipeline == PipelineType.DISTILLED:
        stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
        stage2_h, stage2_w = height // 32, width // 32
    else:
        latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Read transformer config to detect model version
    import json
    transformer_config_path = model_path / "transformer" / "config.json"
    has_prompt_adaln = False
    if transformer_config_path.exists():
        with open(transformer_config_path) as f:
            has_prompt_adaln = json.load(f).get("has_prompt_adaln", False)

    # Load text encoder
    with console.status("[blue]📝 Loading text encoder...[/]", spinner="dots"):
        from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
        text_encoder = LTX2TextEncoder(has_prompt_adaln=has_prompt_adaln)
        text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
        mx.eval(text_encoder.parameters())
    console.print("[green]✓[/] Text encoder loaded")

    # Optionally enhance the prompt
    if enhance_prompt:
        console.print("[bold magenta]✨ Enhancing prompt[/]")
        prompt = text_encoder.enhance_t2v(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, verbose=verbose)
        console.print(f"[dim]Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}[/]")

    # Encode prompts
    if pipeline == PipelineType.DEV:
        # Dev pipeline needs positive and negative embeddings
        if audio:
            video_embeddings_pos, audio_embeddings_pos = text_encoder(prompt, return_audio_embeddings=True)
            video_embeddings_neg, audio_embeddings_neg = text_encoder(negative_prompt, return_audio_embeddings=True)
            model_dtype = video_embeddings_pos.dtype
            mx.eval(video_embeddings_pos, video_embeddings_neg, audio_embeddings_pos, audio_embeddings_neg)
        else:
            video_embeddings_pos, _ = text_encoder(prompt, return_audio_embeddings=False)
            video_embeddings_neg, _ = text_encoder(negative_prompt, return_audio_embeddings=False)
            audio_embeddings_pos = audio_embeddings_neg = None
            model_dtype = video_embeddings_pos.dtype
            mx.eval(video_embeddings_pos, video_embeddings_neg)
    else:
        # Distilled pipeline - single embedding
        if audio:
            text_embeddings, audio_embeddings = text_encoder(prompt, return_audio_embeddings=True)
            mx.eval(text_embeddings, audio_embeddings)
        else:
            text_embeddings, _ = text_encoder(prompt, return_audio_embeddings=False)
            audio_embeddings = None
            mx.eval(text_embeddings)
        model_dtype = text_embeddings.dtype

    del text_encoder
    mx.clear_cache()

    # Load transformer
    transformer_desc = f"🤖 Loading {pipeline_name.lower()} transformer{' (A/V mode)' if audio else ''}..."
    with console.status(f"[blue]{transformer_desc}[/]", spinner="dots"):
        transformer = LTXModel.from_pretrained(model_path=model_path / "transformer", strict=True)

    console.print("[green]✓[/] Transformer loaded")

    # ==========================================================================
    # Pipeline-specific generation logic
    # ==========================================================================

    if pipeline == PipelineType.DISTILLED:
        # ======================================================================
        # DISTILLED PIPELINE: Two-stage with upsampling
        # ======================================================================

        # Load VAE encoder for I2V
        stage1_image_latent = None
        stage2_image_latent = None
        if is_i2v:
            with console.status("[blue]🖼️  Loading VAE encoder and encoding image...[/]", spinner="dots"):
                vae_encoder = VideoEncoder.from_pretrained(model_path / "vae" / "encoder")

                input_image = load_image(image, height=height // 2, width=width // 2, dtype=model_dtype)
                stage1_image_tensor = prepare_image_for_encoding(input_image, height // 2, width // 2, dtype=model_dtype)
                stage1_image_latent = vae_encoder(stage1_image_tensor)
                mx.eval(stage1_image_latent)

                input_image = load_image(image, height=height, width=width, dtype=model_dtype)
                stage2_image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
                stage2_image_latent = vae_encoder(stage2_image_tensor)
                mx.eval(stage2_image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image encoded")

        # Stage 1
        console.print(f"\n[bold yellow]⚡ Stage 1:[/] Generating at {width//2}x{height//2} (8 steps)")
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = None
        audio_latents = None
        if audio:
            audio_positions = create_audio_position_grid(1, audio_frames)
            audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)).astype(model_dtype)
            mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning
        state1 = None
        if is_i2v and stage1_image_latent is not None:
            latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
            state1 = LatentState(
                latent=mx.zeros(latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(latent=stage1_image_latent, frame_idx=image_frame_idx, strength=image_strength)
            state1 = apply_conditioning(state1, [conditioning])

            noise = mx.random.normal(latent_shape, dtype=model_dtype)
            noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal((1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype)
            mx.eval(latents)

        latents, audio_latents = denoise_distilled(
            latents, positions, text_embeddings, transformer, STAGE_1_SIGMAS,
            verbose=verbose, state=state1,
            audio_latents=audio_latents, audio_positions=audio_positions, audio_embeddings=audio_embeddings,
        )

        # Upsample latents
        with console.status("[magenta]🔍 Upsampling latents 2x...[/]", spinner="dots"):
            upscaler_files = sorted(model_path.glob("*spatial-upscaler-x2*.safetensors"))
            if not upscaler_files:
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler = load_upsampler(str(upscaler_files[0]))
            mx.eval(upsampler.parameters())

            vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

            latents = upsample_latents(latents, upsampler, vae_decoder.per_channel_statistics.mean, vae_decoder.per_channel_statistics.std)
            mx.eval(latents)

            del upsampler
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")

        # Stage 2
        console.print(f"\n[bold yellow]⚡ Stage 2:[/] Refining at {width}x{height} (3 steps)")
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and stage2_image_latent is not None:
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(latent=stage2_image_latent, frame_idx=image_frame_idx, strength=image_strength)
            state2 = apply_conditioning(state2, [conditioning])

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)

            if audio and audio_latents is not None:
                audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
                one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
                audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
                mx.eval(audio_latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

            if audio and audio_latents is not None:
                audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
                audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
                mx.eval(audio_latents)

        latents, audio_latents = denoise_distilled(
            latents, positions, text_embeddings, transformer, STAGE_2_SIGMAS,
            verbose=verbose, state=state2,
            audio_latents=audio_latents, audio_positions=audio_positions, audio_embeddings=audio_embeddings,
        )

    else:
        # ======================================================================
        # DEV PIPELINE: Single-stage with CFG
        # ======================================================================

        # Load VAE encoder for I2V
        image_latent = None
        if is_i2v:
            with console.status("[blue]🖼️  Loading VAE encoder and encoding image...[/]", spinner="dots"):
                vae_encoder = VideoEncoder.from_pretrained(model_path / "vae" / "encoder")

                input_image = load_image(image, height=height, width=width, dtype=model_dtype)
                image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
                image_latent = vae_encoder(image_tensor)
                mx.eval(image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image encoded")

        # Generate sigma schedule with token-count-dependent shifting
        num_tokens = latent_frames * latent_h * latent_w
        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        console.print(f"[dim]Sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]")

        console.print(f"\n[bold yellow]⚡ Generating:[/] {width}x{height} ({num_inference_steps} steps, CFG={cfg_scale})")
        mx.random.seed(seed)

        video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
        mx.eval(video_positions)

        audio_positions = None
        audio_latents = None
        if audio:
            audio_positions = create_audio_position_grid(1, audio_frames)
            audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS), dtype=model_dtype)
            mx.eval(audio_positions, audio_latents)

        # Initialize latents with optional I2V conditioning
        video_state = None
        video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
        if is_i2v and image_latent is not None:
            video_state = LatentState(
                latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(latent=image_latent, frame_idx=image_frame_idx, strength=image_strength)
            video_state = apply_conditioning(video_state, [conditioning])

            noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = video_state.denoise_mask * noise_scale
            video_state = LatentState(
                latent=noise * scaled_mask + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=video_state.clean_latent,
                denoise_mask=video_state.denoise_mask,
            )
            latents = video_state.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
            mx.eval(latents)

        # Denoise with CFG/APG
        if audio:
            latents, audio_latents = denoise_dev_av(
                latents, audio_latents,
                video_positions, audio_positions,
                video_embeddings_pos, video_embeddings_neg,
                audio_embeddings_pos, audio_embeddings_neg,
                transformer, sigmas, cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale, verbose=verbose, video_state=video_state,
                use_apg=use_apg, apg_eta=apg_eta, apg_norm_threshold=apg_norm_threshold
            )
        else:
            # Use original denoise_dev with computed sigmas
            latents = denoise_dev(
                latents, video_positions,
                video_embeddings_pos, video_embeddings_neg,
                transformer, sigmas, cfg_scale=cfg_scale,
                verbose=verbose, state=video_state,
                use_apg=use_apg, apg_eta=apg_eta, apg_norm_threshold=apg_norm_threshold
            )

        # Load VAE decoder (for dev pipeline, loaded here instead of during upsampling)
        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

    del transformer
    mx.clear_cache()

    # ==========================================================================
    # Decode and save outputs (common to both pipelines)
    # ==========================================================================

    console.print("\n[blue]🎞️  Decoding video...[/]")

    # Select tiling configuration
    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        console.print(f"[yellow]  Unknown tiling mode '{tiling}', using auto[/]")
        tiling_config = TilingConfig.auto(height, width, num_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream mode
    video_writer = None
    stream_progress = None

    if stream and tiling_config is not None:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        stream_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        stream_progress.start()
        stream_task = stream_progress.add_task("[cyan]Streaming frames[/]", total=num_frames)

        def on_frames_ready(frames: mx.array, _start_idx: int):
            frames = mx.squeeze(frames, axis=0)
            frames = mx.transpose(frames, (1, 2, 3, 0))
            frames = mx.clip((frames + 1.0) / 2.0, 0.0, 1.0)
            frames = (frames * 255).astype(mx.uint8)
            frames_np = np.array(frames)

            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                stream_progress.advance(stream_task)
    else:
        on_frames_ready = None

    if tiling_config is not None:
        spatial_info = f"{tiling_config.spatial_config.tile_size_in_pixels}px" if tiling_config.spatial_config else "none"
        temporal_info = f"{tiling_config.temporal_config.tile_size_in_frames}f" if tiling_config.temporal_config else "none"
        console.print(f"[dim]  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}[/]")
        video = vae_decoder.decode_tiled(latents, tiling_config=tiling_config, tiling_mode=tiling, debug=verbose, on_frames_ready=on_frames_ready)
    else:
        console.print("[dim]  Tiling: disabled[/]")
        video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()

    # Close stream writer
    if video_writer is not None:
        video_writer.release()
        if stream_progress is not None:
            stream_progress.stop()
        console.print(f"[green]✅ Streamed video to[/] {output_path}")
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)
    else:
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)

        if audio:
            temp_video_path = output_path.with_suffix('.temp.mp4')
            save_path = temp_video_path
        else:
            save_path = output_path

        try:
            import cv2
            h, w = video_np.shape[1], video_np.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
            for frame in video_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            if not audio:
                console.print(f"[green]✅ Saved video to[/] {output_path}")
        except Exception as e:
            console.print(f"[red]❌ Could not save video: {e}[/]")

    # Decode and save audio if enabled
    audio_np = None
    if audio and audio_latents is not None:
        with console.status("[blue]🔊 Decoding audio...[/]", spinner="dots"):
            audio_decoder = load_audio_decoder(model_path, pipeline)
            vocoder = load_vocoder(model_path, pipeline)
            mx.eval(audio_decoder.parameters(), vocoder.parameters())

            mel_spectrogram = audio_decoder(audio_latents)
            mx.eval(mel_spectrogram)

            audio_waveform = vocoder(mel_spectrogram)
            mx.eval(audio_waveform)

            audio_np = np.array(audio_waveform.astype(mx.float32))
            if audio_np.ndim == 3:
                audio_np = audio_np[0]

            del audio_decoder, vocoder
            mx.clear_cache()
        console.print("[green]✓[/] Audio decoded")

        audio_path = Path(output_audio_path) if output_audio_path else output_path.with_suffix('.wav')
        save_audio(audio_np, audio_path, AUDIO_SAMPLE_RATE)
        console.print(f"[green]✅ Saved audio to[/] {audio_path}")

        with console.status("[blue]🎬 Combining video and audio...[/]", spinner="dots"):
            temp_video_path = output_path.with_suffix('.temp.mp4')
            success = mux_video_audio(temp_video_path, audio_path, output_path)
        if success:
            console.print(f"[green]✅ Saved video with audio to[/] {output_path}")
            temp_video_path.unlink()
        else:
            temp_video_path.rename(output_path)
            console.print(f"[yellow]⚠️  Saved video without audio to[/] {output_path}")

    del vae_decoder
    mx.clear_cache()

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        console.print(f"[green]✅ Saved {len(video_np)} frames to {frames_dir}[/]")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    console.print(Panel(
        f"[bold green]🎉 Done![/] Generated in {time_str} ({elapsed/num_frames:.2f}s/frame)\n"
        f"[bold green]✨ Peak memory:[/] {mx.get_peak_memory() / (1024 ** 3):.2f}GB",
        expand=False
    ))

    if audio:
        return video_np, audio_np
    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 (Distilled or Dev pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distilled pipeline (two-stage, fast, no CFG)
  python -m mlx_video.generate --prompt "A cat walking on grass"
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline distilled

  # Dev pipeline (single-stage, CFG, higher quality)
  python -m mlx_video.generate --prompt "A cat walking" --pipeline dev --cfg-scale 4.0
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline dev --steps 50

  # Image-to-Video (works with both pipelines)
  python -m mlx_video.generate --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate --prompt "Waves crashing" --image beach.png --pipeline dev

  # With Audio (works with both pipelines)
  python -m mlx_video.generate --prompt "Ocean waves crashing" --audio
  python -m mlx_video.generate --prompt "A jazz band playing" --audio --pipeline dev
        """
    )

    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text description of the video to generate")
    parser.add_argument("--pipeline", type=str, default="distilled", choices=["distilled", "dev"],
                        help="Pipeline type: distilled (two-stage, fast) or dev (single-stage, CFG)")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt for CFG (dev pipeline only)")
    parser.add_argument("--height", "-H", type=int, default=512, help="Output video height")
    parser.add_argument("--width", "-W", type=int, default=512, help="Output video width")
    parser.add_argument("--num-frames", "-n", type=int, default=33, help="Number of frames")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps (dev pipeline only)")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG guidance scale (dev pipeline only)")
    parser.add_argument("--cfg-rescale", type=float, default=0.0, help="CFG rescale factor (0.0-1.0). Higher values reduce artifacts by blending towards positive-only prediction (dev pipeline only)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--output-path", "-o", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames as images")
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2", help="Model repository")
    parser.add_argument("--text-encoder-repo", type=str, default=None, help="Text encoder repository")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--enhance-prompt", action="store_true", help="Enhance the prompt using Gemma")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for prompt enhancement")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to conditioning image for I2V")
    parser.add_argument("--image-strength", type=float, default=1.0, help="Conditioning strength for I2V")
    parser.add_argument("--image-frame-idx", type=int, default=0, help="Frame index to condition for I2V")
    parser.add_argument("--tiling", type=str, default="auto",
                        choices=["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"],
                        help="Tiling mode for VAE decoding")
    parser.add_argument("--stream", action="store_true", help="Stream frames to output as they're decoded")
    parser.add_argument("--audio", "-a", action="store_true", help="Enable synchronized audio generation")
    parser.add_argument("--output-audio", type=str, default=None, help="Output audio path")
    parser.add_argument("--apg", action="store_true", help="Use Adaptive Projected Guidance instead of CFG (more stable for I2V)")
    parser.add_argument("--apg-eta", type=float, default=1.0, help="APG parallel component weight (1.0 = keep full parallel)")
    parser.add_argument("--apg-norm-threshold", type=float, default=0.0, help="APG guidance norm clamp (0 = no clamping)")
    args = parser.parse_args()

    pipeline = PipelineType.DEV if args.pipeline == "dev" else PipelineType.DISTILLED

    generate_video(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        pipeline=pipeline,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        cfg_rescale=args.cfg_rescale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        save_frames=args.save_frames,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        tiling=args.tiling,
        stream=args.stream,
        audio=args.audio,
        output_audio_path=args.output_audio,
        use_apg=args.apg,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
    )


if __name__ == "__main__":
    main()
