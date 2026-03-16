"""Video VAE Encoder for LTX-2 Image-to-Video.

The encoder compresses input images/videos to latent representations.
Used for I2V (image-to-video) conditioning by encoding the input image
to latent space, which can then be used to condition video generation.
"""

import mlx.core as mx
from mlx_video.models.ltx_2.video_vae.video_vae import VideoEncoder



def encode_image(
    image: mx.array,
    encoder: VideoEncoder,
) -> mx.array:
    """Encode a single image to latent space.

    Args:
        image: Image tensor of shape (H, W, 3) in range [0, 1] or (B, H, W, 3)
        encoder: Loaded VAE encoder

    Returns:
        Latent tensor of shape (1, 128, 1, H//32, W//32)
    """
    # Add batch dimension if needed
    if image.ndim == 3:
        image = mx.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Convert from (B, H, W, C) to (B, C, H, W)
    image = mx.transpose(image, (0, 3, 1, 2))  # (B, 3, H, W)

    # Normalize to [-1, 1]
    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2.0 - 1.0

    # Add temporal dimension: (B, C, H, W) -> (B, C, 1, H, W)
    image = mx.expand_dims(image, axis=2)  # (B, 3, 1, H, W)

    # Encode
    latent = encoder(image)

    return latent
