"""Second-order res_2s sampler for diffusion models.

Implements the exponential Rosenbrock-type Runge-Kutta integrator with SDE
noise injection, ported from the LTX-2 PyTorch implementation.
"""

import math
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Phi functions and RK coefficients (pure Python math, no MLX needed)
# ---------------------------------------------------------------------------

def phi(j: int, neg_h: float) -> float:
    """Compute phi_j(z) where z = -h (negative step size in log-space).

    phi_1(z) = (e^z - 1) / z
    phi_2(z) = (e^z - 1 - z) / z^2
    phi_j(z) = (e^z - sum_{k=0}^{j-1} z^k/k!) / z^j
    """
    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)

    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(
    h: float,
    phi_cache: dict,
    c2: float = 0.5,
) -> tuple[float, float, float]:
    """Compute res_2s Runge-Kutta coefficients for a given step size.

    Args:
        h: Step size in log-space = log(sigma / sigma_next)
        phi_cache: Dictionary to cache phi function results.
        c2: Substep position (default 0.5 = midpoint)

    Returns:
        (a21, b1, b2): RK coefficients.
    """
    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    neg_h_c2 = -h * c2
    phi_1_c2 = get_phi(1, neg_h_c2)
    a21 = c2 * phi_1_c2

    neg_h_full = -h
    phi_2_full = get_phi(2, neg_h_full)
    b2 = phi_2_full / c2

    phi_1_full = get_phi(1, neg_h_full)
    b1 = phi_1_full - b2

    return a21, b1, b2


# ---------------------------------------------------------------------------
# SDE noise injection
# ---------------------------------------------------------------------------

def get_sde_coeff(
    sigma_next: float,
) -> tuple[float, float, float]:
    """Compute SDE coefficients for variance-preserving noise injection.

    Uses sigma_up = sigma_next * 0.5 (hardcoded in PyTorch Res2sDiffusionStep).

    Returns:
        (alpha_ratio, sigma_down, sigma_up)
    """
    sigma_up = sigma_next * 0.5
    # Clamp sigma_up to avoid sqrt(negative)
    sigma_up = min(sigma_up, sigma_next * 0.9999)

    sigma_signal = 1.0 - sigma_next  # sigma_max=1
    sigma_residual = math.sqrt(max(sigma_next**2 - sigma_up**2, 0.0))
    alpha_ratio = sigma_signal + sigma_residual

    if alpha_ratio == 0:
        sigma_down = sigma_next
    else:
        sigma_down = sigma_residual / alpha_ratio

    # Handle NaN edge cases
    if math.isnan(sigma_up):
        sigma_up = 0.0
    if math.isnan(sigma_down):
        sigma_down = sigma_next
    if math.isnan(alpha_ratio):
        alpha_ratio = 1.0

    return alpha_ratio, sigma_down, sigma_up


def sde_noise_step(
    sample: mx.array,
    denoised_sample: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array,
) -> mx.array:
    """Apply SDE noise injection step.

    Advances sample from sigma to sigma_next with stochastic noise injection.

    Args:
        sample: Current sample (anchor point)
        denoised_sample: Denoised prediction at this step
        sigma: Current noise level
        sigma_next: Next noise level
        noise: Pre-generated noise tensor (channel-wise normalized)

    Returns:
        Noised sample at sigma_next
    """
    alpha_ratio, sigma_down, sigma_up = get_sde_coeff(sigma_next)

    if sigma_up == 0 or sigma_next == 0:
        return denoised_sample

    # Float32 arithmetic
    sample_f32 = sample.astype(mx.float32)
    denoised_f32 = denoised_sample.astype(mx.float32)
    noise_f32 = noise.astype(mx.float32)

    # Extract epsilon prediction
    eps_next = (sample_f32 - denoised_f32) / (sigma - sigma_next)
    denoised_next = sample_f32 - sigma * eps_next

    # Mix deterministic and stochastic components
    x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise_f32

    return x_noised


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------

def channelwise_normalize(x: mx.array) -> mx.array:
    """Normalize each channel to zero mean and unit variance over spatial dims.

    Operates on the last 2 dimensions (spatial H, W or time, freq).
    """
    mean = mx.mean(x, axis=(-2, -1), keepdims=True)
    x = x - mean
    std = mx.sqrt(mx.mean(x * x, axis=(-2, -1), keepdims=True) + 1e-8)
    x = x / std
    return x


def get_new_noise(shape: tuple, key: mx.array) -> mx.array:
    """Generate channel-wise normalized Gaussian noise.

    PyTorch uses float64; we use float32 (MLX doesn't support float64).
    The channel-wise normalization is the key quality-affecting step.

    Args:
        shape: Shape of the noise tensor
        key: MLX random key for deterministic generation

    Returns:
        Channel-wise normalized noise in float32
    """
    noise = mx.random.normal(shape, dtype=mx.float32, key=key)
    # Global normalization
    noise = (noise - mx.mean(noise)) / (mx.sqrt(mx.mean(noise * noise)) + 1e-8)
    # Channel-wise normalization
    noise = channelwise_normalize(noise)
    return noise
