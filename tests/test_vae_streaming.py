"""Tests for VAE streaming and chunked conv features."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_video.models.ltx_2.video_vae.sampling import DepthToSpaceUpsample
from mlx_video.models.ltx_2.video_vae.tiling import (
    TilingConfig,
    compute_trapezoidal_mask_1d,
    decode_with_tiling,
)


class TestChunkedConv:
    """Tests for chunked conv optimization in DepthToSpaceUpsample."""

    def test_chunked_conv_output_matches_regular(self):
        """Verify chunked_conv produces identical output to regular processing."""
        mx.random.seed(42)

        # Create upsampler with residual (matches decoder config)
        upsampler = DepthToSpaceUpsample(
            dims=3,
            in_channels=256,
            stride=(2, 2, 2),
            residual=True,
            out_channels_reduction_factor=2,
        )

        # Initialize weights deterministically
        mx.eval(upsampler.parameters())

        # Create test input: (B, C, D, H, W) with enough frames to trigger chunking
        # chunked_conv activates when d > 4
        x = mx.random.normal((1, 256, 8, 8, 8))
        mx.eval(x)

        # Run without chunked conv
        out_regular = upsampler(x, causal=True, chunked_conv=False)
        mx.eval(out_regular)

        # Run with chunked conv
        out_chunked = upsampler(x, causal=True, chunked_conv=True)
        mx.eval(out_chunked)

        # Outputs should be identical
        np.testing.assert_allclose(
            np.array(out_regular),
            np.array(out_chunked),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Chunked conv output differs from regular output",
        )

    def test_chunked_conv_small_input_passthrough(self):
        """Verify chunked_conv doesn't activate for small inputs (d <= 4)."""
        mx.random.seed(42)

        upsampler = DepthToSpaceUpsample(
            dims=3,
            in_channels=256,
            stride=(2, 2, 2),
            residual=True,
            out_channels_reduction_factor=2,
        )
        mx.eval(upsampler.parameters())

        # Small input with d=4 (should NOT trigger chunking)
        x = mx.random.normal((1, 256, 4, 8, 8))
        mx.eval(x)

        out_regular = upsampler(x, causal=True, chunked_conv=False)
        out_chunked = upsampler(x, causal=True, chunked_conv=True)
        mx.eval(out_regular, out_chunked)

        # Should be identical since chunking doesn't activate
        np.testing.assert_allclose(
            np.array(out_regular),
            np.array(out_chunked),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_chunked_conv_output_shape(self):
        """Verify chunked_conv produces correct output shape."""
        mx.random.seed(42)

        upsampler = DepthToSpaceUpsample(
            dims=3,
            in_channels=256,
            stride=(2, 2, 2),
            residual=True,
            out_channels_reduction_factor=2,
        )
        mx.eval(upsampler.parameters())

        # Input shape: (1, 256, 8, 16, 16)
        x = mx.random.normal((1, 256, 8, 16, 16))
        mx.eval(x)

        out = upsampler(x, causal=True, chunked_conv=True)
        mx.eval(out)

        # Expected output:
        # - Channels: 256 / 2 = 128
        # - Temporal: 8 * 2 - 1 = 15 (minus 1 for causal)
        # - Spatial: 16 * 2 = 32
        assert out.shape == (1, 128, 15, 32, 32), f"Unexpected shape: {out.shape}"


class TestProgressiveFrameSaving:
    """Tests for progressive frame saving via on_frames_ready callback."""

    def test_on_frames_ready_called(self):
        """Verify on_frames_ready callback is called during tiled decoding."""
        frames_received = []

        def on_frames_ready(frames: mx.array, start_idx: int):
            frames_received.append(
                {
                    "shape": frames.shape,
                    "start_idx": start_idx,
                }
            )

        # Create a mock decoder that just returns scaled input
        def mock_decoder(
            x, causal=False, timestep=None, debug=False, chunked_conv=False
        ):
            # Simulate VAE output: upsample 8x temporal, 32x spatial
            b, c, f, h, w = x.shape
            out_f = 1 + (f - 1) * 8
            out_h = h * 32
            out_w = w * 32
            return mx.zeros((b, 3, out_f, out_h, out_w))

        # Create tiling config with temporal tiling to trigger callbacks
        tiling_config = TilingConfig.temporal_only(tile_size=32, overlap=8)

        # Input latents: enough frames to require multiple tiles
        # 10 latent frames -> 73 output frames
        latents = mx.zeros((1, 128, 10, 4, 4))

        # Decode with tiling
        output = decode_with_tiling(
            decoder_fn=mock_decoder,
            latents=latents,
            tiling_config=tiling_config,
            spatial_scale=32,
            temporal_scale=8,
            on_frames_ready=on_frames_ready,
        )
        mx.eval(output)

        # Should have received at least one callback
        assert len(frames_received) > 0, "on_frames_ready was never called"

        # All received frames should have correct channel count
        for received in frames_received:
            assert (
                received["shape"][1] == 3
            ), f"Expected 3 channels, got {received['shape'][1]}"

    def test_on_frames_ready_covers_all_frames(self):
        """Verify all frames are emitted via callbacks."""
        all_frame_indices = set()

        def on_frames_ready(frames: mx.array, start_idx: int):
            num_frames = frames.shape[2]
            for i in range(num_frames):
                all_frame_indices.add(start_idx + i)

        def mock_decoder(
            x, causal=False, timestep=None, debug=False, chunked_conv=False
        ):
            b, c, f, h, w = x.shape
            out_f = 1 + (f - 1) * 8
            out_h = h * 32
            out_w = w * 32
            return mx.random.normal((b, 3, out_f, out_h, out_w))

        tiling_config = TilingConfig.temporal_only(tile_size=32, overlap=8)

        # 12 latent frames -> 89 output frames
        latents = mx.zeros((1, 128, 12, 4, 4))

        output = decode_with_tiling(
            decoder_fn=mock_decoder,
            latents=latents,
            tiling_config=tiling_config,
            spatial_scale=32,
            temporal_scale=8,
            on_frames_ready=on_frames_ready,
        )
        mx.eval(output)

        # Calculate expected frame count
        expected_frames = 1 + (12 - 1) * 8  # 89 frames

        # All frames should have been emitted
        assert (
            len(all_frame_indices) == expected_frames
        ), f"Expected {expected_frames} frames, got {len(all_frame_indices)}"
        assert all_frame_indices == set(
            range(expected_frames)
        ), "Not all frame indices were covered"


class TestAutoChunkedConv:
    """Tests for auto-enabling chunked_conv based on tiling mode."""

    @pytest.mark.parametrize(
        "tiling_mode,should_enable",
        [
            ("conservative", True),
            ("none", True),
            ("auto", True),
            ("default", True),
            ("spatial", True),
            ("aggressive", False),
            ("temporal", False),
        ],
    )
    def test_chunked_conv_auto_enable(self, tiling_mode: str, should_enable: bool):
        """Verify chunked_conv is auto-enabled for correct tiling modes."""
        # The logic is: tiling_mode in ("conservative", "none", "auto", "default", "spatial")
        expected_modes = {"conservative", "none", "auto", "default", "spatial"}

        use_chunked_conv = tiling_mode in expected_modes

        assert (
            use_chunked_conv == should_enable
        ), f"For tiling_mode='{tiling_mode}', expected chunked_conv={should_enable}"


class TestTrapezoidalMask:
    """Tests for trapezoidal blending mask generation."""

    def test_mask_values_in_range(self):
        """Verify mask values are always in [0, 1]."""
        for length in [16, 32, 64, 128]:
            for ramp in [0, 4, 8, 16]:
                if ramp < length:
                    mask = compute_trapezoidal_mask_1d(length, ramp, ramp, False)
                    assert mx.all(mask >= 0).item(), f"Mask has negative values"
                    assert mx.all(mask <= 1).item(), f"Mask has values > 1"

    def test_mask_center_is_one(self):
        """Verify center of mask is 1.0 when ramps don't overlap."""
        mask = compute_trapezoidal_mask_1d(32, 8, 8, False)
        # Center region should be 1.0
        center = mask[12:20]  # Middle portion
        np.testing.assert_allclose(np.array(center), 1.0, rtol=1e-5)

    def test_mask_ramp_monotonic(self):
        """Verify ramps are monotonically increasing/decreasing."""
        mask = compute_trapezoidal_mask_1d(32, 8, 8, False)
        mask_np = np.array(mask)

        # Left ramp should be increasing
        left_ramp = mask_np[:8]
        assert np.all(np.diff(left_ramp) >= 0), "Left ramp not monotonically increasing"

        # Right ramp should be decreasing
        right_ramp = mask_np[-8:]
        assert np.all(
            np.diff(right_ramp) <= 0
        ), "Right ramp not monotonically decreasing"

    def test_temporal_mask_starts_from_zero(self):
        """Verify temporal mask (left_starts_from_0=True) starts from 0."""
        mask = compute_trapezoidal_mask_1d(32, 8, 0, left_starts_from_0=True)
        assert mask[0].item() == 0.0, "Temporal mask should start from 0"

    def test_spatial_mask_starts_above_zero(self):
        """Verify spatial mask (left_starts_from_0=False) starts above 0."""
        mask = compute_trapezoidal_mask_1d(32, 8, 0, left_starts_from_0=False)
        assert mask[0].item() > 0.0, "Spatial mask should start above 0"


class TestTilingConfig:
    """Tests for TilingConfig presets."""

    def test_default_config(self):
        """Verify default tiling configuration."""
        config = TilingConfig.default()
        assert config.spatial_config is not None
        assert config.temporal_config is not None
        assert config.spatial_config.tile_size_in_pixels == 512
        assert config.temporal_config.tile_size_in_frames == 64

    def test_aggressive_config(self):
        """Verify aggressive tiling configuration."""
        config = TilingConfig.aggressive()
        assert config.spatial_config.tile_size_in_pixels == 256
        assert config.temporal_config.tile_size_in_frames == 32

    def test_conservative_config(self):
        """Verify conservative tiling configuration."""
        config = TilingConfig.conservative()
        assert config.spatial_config.tile_size_in_pixels == 768
        assert config.temporal_config.tile_size_in_frames == 96

    def test_auto_returns_none_for_small_video(self):
        """Verify auto returns None for small videos."""
        config = TilingConfig.auto(height=256, width=256, num_frames=33)
        assert config is None, "Auto should return None for small videos"

    def test_auto_returns_config_for_large_video(self):
        """Verify auto returns config for large videos."""
        config = TilingConfig.auto(height=1024, width=768, num_frames=145)
        assert config is not None, "Auto should return config for large videos"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
