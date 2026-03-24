"""IC-LoRA conditioning utilities for MLX.

Implements reference video token preparation and attention mask construction
for In-Context LoRA (IC-LoRA) inference, ported from the official LTX-2
ltx_core/conditioning/types/reference_video_cond.py and mask_utils.py.

Key design decision: ref tokens are concatenated IN the denoise loop after
latents are already flattened to (B, N, C). The transformer itself is
unchanged – self-attention mask is passed via Modality.self_attention_mask.
"""

from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Position grid helpers
# ---------------------------------------------------------------------------

def create_ref_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    downscale_factor: int = 1,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Build pixel-space positions for a reference video latent.

    Mirrors ``create_position_grid`` in generate.py but also applies
    ``downscale_factor`` to the spatial axes so that a lower-resolution
    reference maps to the same coordinate space as the full-resolution target.

    Args:
        batch_size: Batch size B.
        num_frames: Number of latent frames F.
        height: Latent height H.
        width: Latent width W.
        downscale_factor: Target/reference resolution ratio stored in LoRA
            metadata (1 = same resolution as target, 2 = half resolution, …).
            Spatial positions are multiplied by this factor so the reference
            lands in the same coordinate system as the target.
        temporal_scale: VAE temporal scale factor (default 8).
        spatial_scale: VAE spatial scale factor (default 32).
        fps: Frames per second used for temporal coordinate normalisation.
        causal_fix: Apply causal offset to temporal coords (default True).

    Returns:
        Position array of shape ``(B, 3, N_ref, 2)`` in pixel space,
        consistent with ``create_position_grid`` output format.
    """
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
        3, 1, 1, 1
    )
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None
        )

    # Temporal coords → divide by fps
    pixel_coords[:, 0, :, :] /= fps

    # Apply downscale_factor to spatial axes so the reference's lower-res
    # positions map to the target-resolution coordinate space.
    if downscale_factor != 1:
        pixel_coords[:, 1, :, :] *= downscale_factor  # height axis
        pixel_coords[:, 2, :, :] *= downscale_factor  # width axis

    # Cast through bfloat16 to match PyTorch precision behaviour (same as
    # create_position_grid in generate.py).
    positions_bf16 = mx.array(pixel_coords, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


# ---------------------------------------------------------------------------
# Attention mask construction
# ---------------------------------------------------------------------------

def build_ic_attention_mask(
    n_target: int,
    n_ref: int,
    batch_size: int,
    cross_strength: float = 1.0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build 2-D self-attention mask for IC-LoRA token sequence.

    The full sequence is ``[target_tokens | ref_tokens]``.  The mask
    replicates the block structure from ``mask_utils.build_attention_mask``
    in the official repo:

    ::

                 target (N_t)   ref (N_r)
        target  [    1           cross   ]
        ref     [   cross         1      ]

    Where ``cross = cross_strength ∈ [0, 1]``.

    Note: this is the *single-conditioning* simplified version (no prev_ref
    block) which matches the common inference case where only one reference
    video is provided.

    Args:
        n_target: Number of target (noisy) tokens N_t.
        n_ref: Number of reference tokens N_r.
        batch_size: Batch size B.
        cross_strength: Attention weight between target and ref tokens.
            0.0 = ignore conditioning, 1.0 = full conditioning influence.
        dtype: Output dtype (float32 recommended for stability).

    Returns:
        Mask array of shape ``(B, N_t + N_r, N_t + N_r)`` with values in
        ``[0, 1]``.  Will be converted to additive log-space mask by
        ``_prepare_attention_mask`` inside ``TransformerArgsPreprocessor``.
    """
    total = n_target + n_ref

    # Start with zeros
    mask_np = np.zeros((batch_size, total, total), dtype=np.float32)

    # Target ↔ target: full attention
    mask_np[:, :n_target, :n_target] = 1.0

    # Ref ↔ ref: full attention
    mask_np[:, n_target:, n_target:] = 1.0

    # Target ↔ ref: cross_strength (symmetric)
    mask_np[:, :n_target, n_target:] = cross_strength
    mask_np[:, n_target:, :n_target] = cross_strength

    return mx.array(mask_np, dtype=dtype)


# ---------------------------------------------------------------------------
# Ref token preparation (the main public entry point)
# ---------------------------------------------------------------------------

def prepare_ref_tokens(
    ref_latent: mx.array,
    n_target: int,
    fps: float,
    downscale_factor: int = 1,
    strength: float = 1.0,
    conditioning_attention_strength: float = 1.0,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    causal_fix: bool = True,
    dtype: mx.Dtype = mx.bfloat16,
) -> dict:
    """Prepare reference video tokens for IC-LoRA conditioning.

    Given a VAE-encoded reference latent, produces everything needed to
    concatenate the reference tokens into the denoising loop:

    * ``ref_tokens`` – flattened token sequence ``(B, N_ref, C)``.
    * ``ref_positions`` – pixel-space positions ``(B, 3, N_ref, 2)``.
    * ``ref_timesteps`` – per-token timestep for ref tokens
      ``(B, N_ref)``; will be multiplied by ``sigma`` in the loop to give
      the effective noise level (0 for strength=1, i.e. keep clean).
    * ``attention_mask`` – full ``(B, N_t+N_r, N_t+N_r)`` self-attention mask.

    Args:
        ref_latent: Reference video latent of shape ``(B, C, F, H, W)``.
            Must already be VAE-encoded and normalised (same pipeline as target
            latent).
        n_target: Number of target tokens N_t (= F_t * H_t * W_t).
        fps: Frames per second (for temporal position normalisation).
        downscale_factor: Position scaling factor (from LoRA metadata).
        strength: Conditioning strength.  1.0 = keep reference clean
            (denoise_mask = 0), 0.0 = allow reference to denoise freely.
        conditioning_attention_strength: Cross-attention strength in [0, 1].
        temporal_scale: VAE temporal scale (default 8).
        spatial_scale: VAE spatial scale (default 32).
        causal_fix: Apply causal temporal fix (default True).
        dtype: Compute dtype.

    Returns:
        Dict with keys:
            ``ref_tokens`` (B, N_ref, C),
            ``ref_positions`` (B, 3, N_ref, 2),
            ``ref_timestep_scale`` (B, N_ref) – values are ``1 - strength``,
                                  multiply by sigma to get effective timestep,
            ``attention_mask`` (B, N_t+N_r, N_t+N_r).
    """
    b, c, f, h, w = ref_latent.shape

    # 1. Flatten latent → tokens, same as denoise_distilled does for the target
    ref_tokens = mx.transpose(
        mx.reshape(ref_latent, (b, c, -1)), (0, 2, 1)
    ).astype(dtype)  # (B, N_ref, C)

    n_ref = ref_tokens.shape[1]

    # 2. Positions for the reference video
    ref_positions = create_ref_position_grid(
        batch_size=b,
        num_frames=f,
        height=h,
        width=w,
        downscale_factor=downscale_factor,
        temporal_scale=temporal_scale,
        spatial_scale=spatial_scale,
        fps=fps,
        causal_fix=causal_fix,
    )  # (B, 3, N_ref, 2)

    # 3. Per-token timestep scale: (1 - strength) * sigma gives effective
    #    noise level.  strength=1.0 → scale=0.0 → timestep=0 → no denoising.
    ref_timestep_scale = mx.full(
        (b, n_ref), float(1.0 - strength), dtype=dtype
    )  # (B, N_ref)

    # 4. Attention mask
    attention_mask = build_ic_attention_mask(
        n_target=n_target,
        n_ref=n_ref,
        batch_size=b,
        cross_strength=conditioning_attention_strength,
        dtype=dtype,
    )  # (B, N_t+N_r, N_t+N_r)

    mx.eval(ref_tokens, ref_positions, ref_timestep_scale, attention_mask)

    return {
        "ref_tokens": ref_tokens,
        "ref_positions": ref_positions,
        "ref_timestep_scale": ref_timestep_scale,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# LoRA metadata helper
# ---------------------------------------------------------------------------

def read_lora_downscale_factor(lora_path: str) -> int:
    """Read ``reference_downscale_factor`` from LoRA safetensors metadata.

    IC-LoRAs trained with lower-resolution reference videos store this factor
    so inference can resize reference videos accordingly.

    Args:
        lora_path: Path to the LoRA ``.safetensors`` file.

    Returns:
        ``reference_downscale_factor`` as an int (1 if absent or unreadable).
    """
    import logging

    try:
        import safetensors
        with safetensors.safe_open(lora_path, framework="numpy") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logging.warning(
            f"[IC-LoRA] Failed to read metadata from '{lora_path}': {e}. "
            "Defaulting reference_downscale_factor=1."
        )
        return 1
