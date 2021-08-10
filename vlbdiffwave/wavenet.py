from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from .weightnorm import WNDilatedConv, WNDense


class WaveNetBlock(nn.Module):
    """WaveNet block.
    """
    channels: int
    kernels: int
    dilations: int

    def setup(self):
        """Setup modules.
        """
        self.proj_embed = WNDense(channels=self.channels)
        self.proj_mel = WNDense(channels=self.channels * 2)
        self.conv = WNDilatedConv(
            channels=self.channels * 2,
            kernels=self.kernels,
            rhs_dilations=self.dilations)
        self.proj_res = WNDense(channels=self.channels)
        self.proj_skip = WNDense(channels=self.channels)

    def __call__(self,
                 inputs: jnp.ndarray,
                 embedding: jnp.ndarray,
                 mel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Pass to wavenet block.
        Args:
            inputs: [float32; [B, T, C]], input tensor.
            embedding: [float32; [B, E]], embedding tensor.
            mel: [float32; [B, T, M]], expanded mel-spectrogram..
        Returns:
            residual: [float32; [B, T, C]], residually connected.
            skip: [float32; [B, T, C]], for skip connection.
        """
        # [B, T, C]
        x = inputs + self.proj_embed(embedding)[:, None]
        # [B, T, C + C]
        x = self.conv(x) + self.proj_mel(mel)
        # [B, T, C]
        x = jnp.tanh(x[..., :self.channels]) * nn.sigmoid(x[..., self.channels:])
        # [B, T, C]
        res = self.proj_res(x) + inputs
        return res, self.proj_skip(x)
