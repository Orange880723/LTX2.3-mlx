"""Vocoder for converting mel spectrograms to audio waveforms."""

import math
from typing import Dict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import check_array_shape
from ..config import VocoderModelConfig
from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, leaky_relu


class Vocoder(nn.Module):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    Based on HiFi-GAN architecture.

    Args:
        resblock_kernel_sizes: List of kernel sizes for the residual blocks
        upsample_rates: List of upsampling rates
        upsample_kernel_sizes: List of kernel sizes for the upsampling layers
        resblock_dilation_sizes: List of dilation sizes for the residual blocks
        upsample_initial_channel: Initial number of channels for upsampling
        stereo: Whether to use stereo output
        resblock: Type of residual block to use ("1" or "2")
        output_sample_rate: Waveform sample rate
    """

    def __init__(
        self,
        config: VocoderModelConfig
    ):
        super().__init__()


        
        self.output_sample_rate = config.output_sample_rate
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.upsample_rates = config.upsample_rates
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.upsample_initial_channel = config.upsample_initial_channel

        in_channels = 128 if config.stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3)

        resblock_class = ResBlock1 if config.resblock == "1" else ResBlock2

        # Upsampling layers using ConvTranspose1d
        self.ups = {}
        for i, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            in_ch = config.upsample_initial_channel // (2**i)
            out_ch = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups[i] = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )

        # Residual blocks
        self.resblocks = {}
        block_idx = 0
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks[block_idx] = resblock_class(ch, kernel_size, tuple(dilations))
                block_idx += 1

        out_channels = 2 if config.stereo else 1
        final_channels = config.upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.upsample_factor = math.prod(config.upsample_rates)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}

        if "vocoder." not in weights:
            return weights

        for key, value in weights.items():
            new_key = key

            # Handle vocoder weights
            if key.startswith("vocoder."):
                new_key = key.replace("vocoder.", "")

                # Handle ModuleList indices -> dict keys
                # PyTorch: ups.0, ups.1, ... -> ups.0, ups.1, ...
                # PyTorch: resblocks.0, resblocks.1, ... -> resblocks.0, resblocks.1, ...

                # Handle Conv1d weight shape conversion
                # PyTorch: (out_channels, in_channels, kernel)
                # MLX: (out_channels, kernel, in_channels)
                if "weight" in new_key and value.ndim == 3:
                    if "ups" in new_key:
                        # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                        value = value if check_array_shape(value) else mx.transpose(value, (1, 2, 0))
                    else:
                        # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                        value = value if check_array_shape(value) else mx.transpose(value, (0, 2, 1))

                sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path, strict: bool = True) -> "Vocoder":
        """Load vocoder from pretrained model."""
        from mlx_video.models.ltx.config import VocoderModelConfig
        import json

        config_dict = {}
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)

        config = VocoderModelConfig.from_dict(config_dict)
        model = cls(config)
        weights = mx.load(str(model_path / "model.safetensors"))

        # weights = vocoder.sanitize(weights)
        model.load_weights(list(weights.items()), strict=strict)
        return model


    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono - MLX format (N, L, C)
               - 4D: (batch_size, 2, time, mel_bins) for stereo - PyTorch format (N, C, H, W)
        Returns:
            Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        # Input: (batch, channels, time, mel_bins) from audio decoder
        # Transpose to (batch, channels, mel_bins, time)
        x = mx.transpose(x, (0, 1, 3, 2))

        if x.ndim == 4:  # stereo
            # x shape: (batch, 2, mel_bins, time)
            # Rearrange to (batch, 2*mel_bins, time)
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        # MLX Conv1d expects (N, L, C), so transpose
        # Current: (batch, channels, time) -> (batch, time, channels)
        x = mx.transpose(x, (0, 2, 1))

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            start = i * self.num_kernels
            end = start + self.num_kernels

            # Apply residual blocks and average their outputs
            block_outputs = []
            for idx in range(start, end):
                block_outputs.append(self.resblocks[idx](x))

            # Stack and mean
            x = mx.stack(block_outputs, axis=0)
            x = mx.mean(x, axis=0)

        # IMPORTANT: Use default leaky_relu slope (0.01), NOT LRELU_SLOPE (0.1)
        # PyTorch uses F.leaky_relu(x) which defaults to 0.01
        x = nn.leaky_relu(x)  # Default negative_slope=0.01
        x = self.conv_post(x)
        x = mx.tanh(x)

        # Transpose back to (batch, channels, time)
        x = mx.transpose(x, (0, 2, 1))

        return x
