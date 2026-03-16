"""Audio VAE module for LTX-2 audio generation."""

from .attention import AttentionType, AttnBlock, make_attn
from .audio_vae import AudioDecoder, AudioEncoder, decode_audio
from .audio_processor import load_audio, ensure_stereo, waveform_to_mel
from .causal_conv_2d import CausalConv2d, make_conv2d
from ..config import CausalityAxis
from .downsample import Downsample, build_downsampling_path
from .normalization import NormType, PixelNorm, build_normalization_layer
from .ops import AudioLatentShape, AudioPatchifier, PerChannelStatistics
from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, ResnetBlock
from .upsample import Upsample, build_upsampling_path
from .vocoder import Vocoder, load_vocoder

__all__ = [
    # Main components
    "AudioEncoder",
    "AudioDecoder",
    "Vocoder",
    "load_vocoder",
    "decode_audio",
    # Audio processing
    "load_audio",
    "ensure_stereo",
    "waveform_to_mel",
    # Ops
    "AudioLatentShape",
    "AudioPatchifier",
    "PerChannelStatistics",
    # Building blocks
    "AttentionType",
    "AttnBlock",
    "make_attn",
    "CausalConv2d",
    "make_conv2d",
    "CausalityAxis",
    "Downsample",
    "build_downsampling_path",
    "NormType",
    "PixelNorm",
    "build_normalization_layer",
    "ResBlock1",
    "ResBlock2",
    "ResnetBlock",
    "LRELU_SLOPE",
    "Upsample",
    "build_upsampling_path",
]
