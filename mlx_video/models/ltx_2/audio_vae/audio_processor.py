"""Audio processing utilities for loading audio files and computing mel-spectrograms.

Matches the PyTorch AudioProcessor from LTX-2 (torchaudio.transforms.MelSpectrogram)
using librosa for macOS/MLX compatibility.
"""


import mlx.core as mx
import numpy as np


def load_audio(
    path: str,
    target_sr: int = 16000,
    start_time: float = 0.0,
    max_duration: float | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio file, resample to target sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, OGG, or video with audio track).
        target_sr: Target sample rate (default 16000 Hz).
        start_time: Start time in seconds.
        max_duration: Maximum duration in seconds. None = read to end.
        mono: If True, convert to mono. Default False (preserve channels).

    Returns:
        (waveform, sample_rate) where waveform is (channels, samples) float32 numpy array.
    """
    import librosa

    # librosa.load returns mono by default; we want to preserve stereo
    y, sr = librosa.load(
        path,
        sr=target_sr,
        mono=mono,
        offset=start_time,
        duration=max_duration,
    )

    # Ensure 2D: (channels, samples)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, samples)

    return y.astype(np.float32), sr


def ensure_stereo(waveform: np.ndarray) -> np.ndarray:
    """Ensure waveform is stereo (2, samples). Duplicates mono if needed."""
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.shape[0] == 1:
        waveform = np.concatenate([waveform, waveform], axis=0)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def waveform_to_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 1024,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> mx.array:
    """Convert waveform to log-mel spectrogram matching PyTorch MelSpectrogram.

    PyTorch reference:
        MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
                       f_min=0.0, f_max=8000.0, n_mels=64, power=1.0,
                       mel_scale="slaney", norm="slaney", center=True, pad_mode="reflect")

    Args:
        waveform: (channels, samples) float32 numpy array.
        sample_rate: Sample rate of the waveform.
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length.
        n_mels: Number of mel bins.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.

    Returns:
        Log-mel spectrogram as mx.array of shape (1, channels, time, n_mels).
    """
    import librosa

    # Ensure 2D
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    channels = waveform.shape[0]
    mels = []

    for ch in range(channels):
        # Magnitude spectrogram (power=1.0)
        S = np.abs(
            librosa.stft(
                waveform[ch],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=True,
                pad_mode="reflect",
            )
        )

        # Mel filterbank with slaney normalization
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm="slaney",
        )
        mel = mel_basis @ S

        # Log scale
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

        # Transpose: (n_mels, time) -> (time, n_mels)
        mel = mel.T
        mels.append(mel)

    # Stack channels: (channels, time, n_mels)
    mel_spec = np.stack(mels, axis=0)

    # Add batch dim: (1, channels, time, n_mels)
    mel_spec = mel_spec[np.newaxis, ...]

    return mx.array(mel_spec, dtype=mx.float32)
