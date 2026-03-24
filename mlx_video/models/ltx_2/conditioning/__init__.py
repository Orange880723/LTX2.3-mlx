"""Conditioning modules for LTX-2 video generation."""

from mlx_video.models.ltx_2.conditioning.keyframe import (
    VideoConditionByKeyframeIndex,
    add_keyframe_positions,
    apply_keyframe_conditioning,
    remove_virtual_frames,
)
from mlx_video.models.ltx_2.conditioning.latent import (
    LatentState,
    VideoConditionByLatentIndex,
    add_noise_with_state,
    apply_conditioning,
    apply_denoise_mask,
    create_initial_state,
)

__all__ = [
    # Latent conditioning (I2V - replacing latents)
    "LatentState",
    "VideoConditionByLatentIndex",
    "create_initial_state",
    "apply_conditioning",
    "apply_denoise_mask",
    "add_noise_with_state",
    # Keyframe conditioning (interpolation - guiding latents)
    "VideoConditionByKeyframeIndex",
    "apply_keyframe_conditioning",
    "add_keyframe_positions",
    "remove_virtual_frames",
]
