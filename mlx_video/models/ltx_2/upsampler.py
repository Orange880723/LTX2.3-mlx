from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class Conv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape: (C_out, KD, KH, KW, C_in)
        scale = (
            1.0
            / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** 0.5
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(
                out_channels,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                in_channels,
            ),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, D, H, W, C_in)

        Returns:
            Output tensor of shape (N, D', H', W', C_out)
        """
        y = mx.conv3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class GroupNorm3d(nn.Module):

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C)
        n, d, h, w, c = x.shape
        input_dtype = x.dtype

        x = x.astype(mx.float32)

        # Reshape to (N, D*H*W, num_groups, C//num_groups)
        x = mx.reshape(x, (n, d * h * w, self.num_groups, c // self.num_groups))

        # Compute mean and var over spatial and channel group dims
        mean = mx.mean(x, axis=(1, 3), keepdims=True)
        var = mx.var(x, axis=(1, 3), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = mx.reshape(x, (n, d, h, w, c))

        # Apply weight and bias
        weight = self.weight.astype(mx.float32)
        bias = self.bias.astype(mx.float32)
        x = x * weight + bias

        # Convert back to input dtype
        x = x.astype(input_dtype)

        return x


class PixelShuffle2D(nn.Module):
    """Pixel shuffle for 2D spatial upsampling with per-axis factors."""

    def __init__(self, upscale_factor_h: int = 2, upscale_factor_w: int = 2):
        super().__init__()
        self.rh = upscale_factor_h
        self.rw = upscale_factor_w

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, H, W, C) where C = out_channels * rh * rw
        n, h, w, c = x.shape
        rh, rw = self.rh, self.rw
        out_c = c // (rh * rw)

        # Reshape: (N, H, W, out_c, rh, rw)
        x = mx.reshape(x, (n, h, w, out_c, rh, rw))

        # Permute: (N, H, rh, W, rw, out_c)
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

        # Reshape: (N, H*rh, W*rw, out_c)
        x = mx.reshape(x, (n, h * rh, w * rw, out_c))

        return x


class BlurDownsample(nn.Module):
    """Anti-aliased downsampling with a fixed 5x5 binomial blur kernel.

    PyTorch source uses a depthwise conv with the binomial kernel.
    The kernel weight is stored as (1, 1, 5, 5) and loaded via safetensors.
    """

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride
        # 5x5 binomial (1,4,6,4,1) kernel, normalized
        # This will be overwritten by loaded weights if available
        k = mx.array([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel_2d = mx.outer(k, k)
        kernel_2d = kernel_2d / kernel_2d.sum()
        # MLX conv2d weight: (O, H, W, I) — we use (1, 5, 5, 1) for per-channel
        self.kernel = kernel_2d.reshape(1, 5, 5, 1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, H, W, C) channels-last
        n, h, w, c = x.shape

        # Pad with edge replication (2 on each side for 5x5 kernel)
        x = mx.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)], mode="edge")

        # Apply blur per-channel: reshape so each channel is a separate "batch"
        # (N, H+4, W+4, C) -> (N*C, H+4, W+4, 1)
        x = mx.transpose(x, (0, 3, 1, 2))  # (N, C, H+4, W+4)
        x = mx.reshape(x, (n * c, h + 4, w + 4, 1))

        # Depthwise conv: (N*C, H+4, W+4, 1) * (1, 5, 5, 1) -> (N*C, H_out, W_out, 1)
        x = mx.conv2d(x, self.kernel, stride=(self.stride, self.stride))

        _, h_out, w_out, _ = x.shape
        # Reshape back: (N*C, H_out, W_out, 1) -> (N, C, H_out, W_out) -> (N, H_out, W_out, C)
        x = mx.reshape(x, (n, c, h_out, w_out))
        x = mx.transpose(x, (0, 2, 3, 1))

        return x


class SpatialUpsampler2x(nn.Module):
    """Standard 2x spatial upsampler: Conv2d + PixelShuffle(2)."""

    def __init__(self, mid_channels: int = 1024):
        super().__init__()
        self.scale = 2.0
        # Sequential: conv (index 0) + pixel shuffle
        # Weight key: upsampler.0.weight -> mapped to upsampler.conv.weight in sanitize
        self.conv = nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle2D(2, 2)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C)
        n, d, h, w, c = x.shape
        x = mx.reshape(x, (n * d, h, w, c))
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = mx.reshape(x, (n, d, h * 2, w * 2, c))
        return x


class SpatialRationalResampler(nn.Module):
    """Rational spatial resampler for non-integer scale factors (e.g., 1.5x).

    For scale=1.5: upsample 3x via PixelShuffle, then downsample 2x via BlurDownsample.
    Rational fraction: 1.5 = 3/2.
    """

    def __init__(self, mid_channels: int = 1024, scale: float = 1.5):
        super().__init__()
        self.scale = scale

        # Rational fraction for 1.5: numerator=3, denominator=2
        num, den = _rational_for_scale(scale)
        self.num = num
        self.den = den

        # Conv2d: mid_channels -> num^2 * mid_channels for PixelShuffle(num)
        self.conv = nn.Conv2d(
            mid_channels, num * num * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffle2D(num, num)
        self.blur_down = BlurDownsample(stride=den)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C)
        n, d, h, w, c = x.shape
        x = mx.reshape(x, (n * d, h, w, c))

        x = self.conv(x)
        x = self.pixel_shuffle(x)  # H*num, W*num
        x = self.blur_down(x)  # H*num/den, W*num/den

        _, h_out, w_out, _ = x.shape
        x = mx.reshape(x, (n, d, h_out, w_out, c))
        return x


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    """Convert a float scale to a rational fraction (numerator, denominator)."""
    from fractions import Fraction

    frac = Fraction(scale).limit_denominator(10)
    return frac.numerator, frac.denominator


class ResBlock3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm3d(32, channels)
        self.conv2 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm3d(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Activation AFTER residual addition
        x = nn.silu(x + residual)

        return x


class LatentUpsampler(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.spatial_scale = spatial_scale

        # Initial projection
        self.initial_conv = Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm3d(32, mid_channels)

        # Pre-upsample ResBlocks - use dict with int keys for MLX parameter tracking
        self.res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        # Upsampler: 2D spatial upsampling (frame-by-frame)
        if rational_resampler:
            self.upsampler = SpatialRationalResampler(
                mid_channels=mid_channels, scale=spatial_scale
            )
        else:
            self.upsampler = SpatialUpsampler2x(mid_channels=mid_channels)

        # Post-upsample ResBlocks - use dict with int keys for MLX parameter tracking
        self.post_upsample_res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        # Final projection
        self.final_conv = Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array, debug: bool = False) -> mx.array:
        """Upsample latents spatially.

        Args:
            latent: Input tensor of shape (B, C, F, H, W) - channels first
            debug: If True, print intermediate values for debugging

        Returns:
            Upsampled tensor of shape (B, C, F, H*scale, W*scale) - channels first
        """

        def debug_stats(name, t):
            if debug:
                mx.eval(t)
                print(
                    f"    {name}: shape={t.shape}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}"
                )

        if debug:
            print("  [DEBUG] LatentUpsampler forward pass:")
            debug_stats("Input (channels first)", latent)

        # Convert from channels first (B, C, F, H, W) to channels last (B, F, H, W, C)
        x = mx.transpose(latent, (0, 2, 3, 4, 1))

        # Initial conv
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        # Pre-upsample blocks
        for i in sorted(self.res_blocks.keys()):
            x = self.res_blocks[i](x)

        # Upsample (2D spatial, frame-by-frame)
        x = self.upsampler(x)
        if debug:
            debug_stats(f"After upsampler (spatial {self.spatial_scale}x)", x)

        # Post-upsample blocks
        for i in sorted(self.post_upsample_res_blocks.keys()):
            x = self.post_upsample_res_blocks[i](x)

        # Final conv
        x = self.final_conv(x)

        # Convert back to channels first (B, C, F, H, W)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        if debug:
            debug_stats("Output (channels first)", x)

        return x


def upsample_latents(
    latent: mx.array,
    upsampler: LatentUpsampler,
    latent_mean: mx.array,
    latent_std: mx.array,
    debug: bool = False,
) -> mx.array:
    # Un-normalize: latent * std + mean
    latent_mean = latent_mean.reshape(1, -1, 1, 1, 1)
    latent_std = latent_std.reshape(1, -1, 1, 1, 1)
    latent = latent * latent_std + latent_mean

    # Upsample
    latent = upsampler(latent, debug=debug)

    # Re-normalize: (latent - mean) / std
    latent = (latent - latent_mean) / latent_std

    return latent


def load_upsampler(weights_path: str) -> Tuple[LatentUpsampler, float]:
    """Load upsampler from safetensors weights.

    Auto-detects whether the weights are for x2 or x1.5 upscaling based on
    the upsampler conv output channels:
      - x2: upsampler.0.weight shape [4*mid, mid, 3, 3] (4096 out channels)
      - x1.5: upsampler.conv.weight shape [9*mid, mid, 3, 3] (9216 out channels)

    Args:
        weights_path: Path to upsampler weights file

    Returns:
        Tuple of (LatentUpsampler model, spatial_scale)
    """
    print(f"Loading spatial upsampler from {weights_path}...")
    raw_weights = mx.load(weights_path)

    # Detect mid_channels from res_blocks
    sample_key = "res_blocks.0.conv1.weight"
    if sample_key in raw_weights:
        mid_channels = raw_weights[sample_key].shape[0]
    else:
        mid_channels = 1024

    # Detect upsampler type from conv output channels
    # x2: conv out = 4 * mid (2^2 * mid for PixelShuffle(2))
    # x1.5: conv out = 9 * mid (3^2 * mid for PixelShuffle(3)) + blur downsample
    # Both formats may have upsampler.blur_down.kernel, so use channel count
    conv_key = (
        "upsampler.conv.weight"
        if "upsampler.conv.weight" in raw_weights
        else "upsampler.0.weight"
    )
    if conv_key in raw_weights:
        out_channels = raw_weights[conv_key].shape[0]
        ratio = out_channels // mid_channels
        rational_resampler = ratio == 9  # 3^2 for PixelShuffle(3) + blur downsample
        spatial_scale = 1.5 if rational_resampler else 2.0
    else:
        rational_resampler = False
        spatial_scale = 2.0

    print(
        f"  Detected: mid_channels={mid_channels}, scale={spatial_scale}x, rational={rational_resampler}"
    )

    # Create model
    upsampler = LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        spatial_scale=spatial_scale,
        rational_resampler=rational_resampler,
    )

    # Sanitize weights - convert from PyTorch to MLX format
    sanitized = {}
    for key, value in raw_weights.items():
        new_key = key

        # x2 upsampler uses sequential indexing: upsampler.0.* -> upsampler.conv.*
        if key.startswith("upsampler.0."):
            new_key = key.replace("upsampler.0.", "upsampler.conv.")

        # Conv3d weights: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Conv2d weights: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if ("weight" in new_key or "kernel" in new_key) and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    # Load weights
    upsampler.load_weights(list(sanitized.items()), strict=False)

    print(f"  Loaded {len(sanitized)} weights")

    return upsampler, spatial_scale
