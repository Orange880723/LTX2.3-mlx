"""Keyframe-based conditioning for video interpolation.

This module provides conditioning that adds guiding latents (virtual frames)
to the video generation process, enabling smooth interpolation between keyframes.

Key difference from latent.py (I2V):
- latent.py: Replaces latents at specific frames (denoise_mask=0)
- keyframe.py: Adds guiding latents as virtual frames (denoise_mask=1-strength)
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import numpy as np

from .latent import LatentState


@dataclass
class VideoConditionByKeyframeIndex:
    """Condition video generation by adding guiding latents (virtual frames).

    Unlike VideoConditionByLatentIndex which replaces latents, this class
    appends keyframe latents as virtual frames that guide generation through
    attention mechanisms.

    Args:
        keyframe_latent: Encoded keyframe latent of shape (B, C, 1, H, W)
        frame_idx: Frame index this keyframe represents (for position encoding)
        strength: Guidance strength (1.0 = full guidance, 0.0 = no guidance)
    """

    keyframe_latent: mx.array
    frame_idx: int
    strength: float = 1.0

    def get_num_latent_frames(self) -> int:
        """Get number of latent frames in the conditioning."""
        return self.keyframe_latent.shape[2]


def apply_keyframe_conditioning(
    state: LatentState,
    conditionings: List[VideoConditionByKeyframeIndex],
) -> LatentState:
    """Apply keyframe conditioning by appending virtual frames.

    This function appends keyframe latents as virtual frames to the latent state.
    These virtual frames guide the generation through attention mechanisms rather
    than replacing existing frames.

    Args:
        state: Current latent state
        conditionings: List of keyframe conditioning items to apply

    Returns:
        Updated LatentState with virtual frames appended
    """
    if not conditionings:
        return state

    state = state.clone()
    dtype = state.latent.dtype
    b, c, f, h, w = state.latent.shape

    # Collect virtual frames to append
    virtual_latents = []
    virtual_clean_latents = []
    virtual_masks = []

    for cond in conditionings:
        keyframe_latent = cond.keyframe_latent
        strength = cond.strength

        # Validate shapes
        _, cond_c, cond_f, cond_h, cond_w = keyframe_latent.shape
        if (cond_c, cond_h, cond_w) != (c, h, w):
            raise ValueError(
                f"Keyframe latent spatial shape ({cond_c}, {cond_h}, {cond_w}) "
                f"does not match target shape ({c}, {h}, {w})"
            )

        # Append virtual frame
        virtual_latents.append(keyframe_latent)
        virtual_clean_latents.append(keyframe_latent)

        # Set denoise mask: 1.0 - strength
        # strength=1.0 -> mask=0.0 (no denoising, keep clean)
        # strength=0.0 -> mask=1.0 (full denoising, no guidance)
        virtual_masks.append(
            mx.full((b, 1, cond_f, 1, 1), 1.0 - strength, dtype=dtype)
        )

    # Concatenate virtual frames to the end
    if virtual_latents:
        state.latent = mx.concatenate(
            [state.latent] + virtual_latents, axis=2
        )
        state.clean_latent = mx.concatenate(
            [state.clean_latent] + virtual_clean_latents, axis=2
        )
        state.denoise_mask = mx.concatenate(
            [state.denoise_mask] + virtual_masks, axis=2
        )

    return state


def add_keyframe_positions(
    positions: mx.array,
    keyframe_indices: List[int],
    height: int,
    width: int,
    fps: float = 24.0,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
) -> mx.array:
    """Add position encodings for keyframe virtual frames.

    This function appends position encodings for virtual frames to the existing
    position grid. Virtual frames use the same spatial positions as regular frames
    but have temporal positions corresponding to their frame indices.

    Args:
        positions: Existing position grid (B, 3, num_patches, 2)
        keyframe_indices: List of frame indices for keyframes
        height: Latent height
        width: Latent width
        fps: Frames per second
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)

    Returns:
        Updated position grid with virtual frame positions appended
    """
    if not keyframe_indices:
        return positions

    batch_size = positions.shape[0]
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    # Create spatial grid (same for all virtual frames)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)
    h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing="ij")

    virtual_positions = []

    for frame_idx in keyframe_indices:
        # Create temporal grid for this keyframe
        t_grid = np.full_like(h_grid, frame_idx)

        # Stack coordinates
        patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

        # Add patch size to get ends
        patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
            3, 1, 1
        )
        patch_ends = patch_starts + patch_size_delta

        # Stack starts and ends
        latent_coords = np.stack([patch_starts, patch_ends], axis=-1)

        # Reshape to (3, num_patches, 2)
        num_patches = height * width
        latent_coords = latent_coords.reshape(3, num_patches, 2)

        # Tile for batch size
        latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

        # Scale to pixel space
        scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
            1, 3, 1, 1
        )
        pixel_coords = (latent_coords * scale_factors).astype(np.float32)

        # Divide temporal coords by fps
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

        # Cast through bfloat16 to match PyTorch behavior
        positions_bf16 = mx.array(pixel_coords, dtype=mx.bfloat16)
        mx.eval(positions_bf16)
        virtual_pos = positions_bf16.astype(mx.float32)

        virtual_positions.append(virtual_pos)

    # Concatenate virtual positions to the end
    if virtual_positions:
        positions = mx.concatenate(
            [positions] + virtual_positions, axis=2
        )

    return positions


def remove_virtual_frames(
    state: LatentState,
    num_real_frames: int,
) -> LatentState:
    """Remove virtual frames from latent state after denoising.

    Args:
        state: Latent state with virtual frames
        num_real_frames: Number of real frames (excluding virtual frames)

    Returns:
        LatentState with only real frames
    """
    state = state.clone()

    state.latent = state.latent[:, :, :num_real_frames]
    state.clean_latent = state.clean_latent[:, :, :num_real_frames]
    state.denoise_mask = state.denoise_mask[:, :, :num_real_frames]

    return state
