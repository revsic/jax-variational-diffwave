import flax.linen as nn
import jax.numpy as jnp

from .config import Config
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
                 mel: jnp.ndarray) -> jnp.ndarray:
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


class WaveNet(nn.Module):
    """WaveNet: A Generative Model for Raw Audio.
    """
    config: Config

    def setup(self):
        """Setup modules.
        """
        config = self.config
        # signal proj
        self.proj = nn.Dense(config.channels)
        # embedding
        self.proj_embed = [
            nn.Dense(config.embedding_proj)
            for _ in range(config.embedding_layers)]
        # mel-upsampler
        self.upsample = [
            nn.ConvTranspose(
                1,
                config.upsample_kernels,
                config.upsample_strides,
                padding='SAME')
            for _ in range(config.upsample_layers)]
        # wavenet blocks
        self.blocks = [
            WaveNetBlock(
                channels=config.channels,
                kernels=config.kernels,
                dilations=config.dilations ** i)
            for _ in range(config.num_cycles)
            for i in range(config.num_layers)]
        self.scale = config.num_layers ** -0.5
        # output projection
        self.proj_context = nn.Dense(config.channels)
        self.proj_out = nn.Dense(1)
    
    def __call__(self,
                 signal: jnp.ndarray,
                 snr: jnp.ndarray,
                 mel: jnp.ndarray) -> jnp.ndarray:
        """Estimate noise from signal with respect to given snr and mel-spectrogram.
        Args:
            signal: [float32; [B, T]], noised signal.
            snr: [float32; [B]], normalized signal-to-noise ratio.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:
            [float32; [B, T]], denoised signal.
        """
        # [B, T, C]
        x = nn.swish(self.proj(signal[..., None]))
        # [B, E']
        embed = self.embedding(snr)
        # [B, E]
        for proj in self.proj_embed:
            embed = nn.swish(proj(embed))
        # [B, T // H, M, 1]
        mel = mel[..., None]
        for upsample in self.upsample:
            mel = nn.swish(upsample(mel))
        # [B, T, M]
        mel = mel.squeeze(-1)
        # WaveNet
        context = 0.
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            x, skip = block(x, embed, mel)
            context = context + skip
        # [B, T, C]
        context = nn.swish(self.proj_context(context * self.scale))
        # [B, T]
        return nn.tanh(self.proj_out(context)).squeeze(-1)

    def embedding(self, snr: jnp.ndarray) -> jnp.ndarray:
        """Generate embedding.
        Args:
            snr: [float32; [B]], unit normalized signal-to-noise ratio.
        Returns:
            [float32; [B, E]], embeddings.
        """
        # [E // 2]
        i = jnp.arange(0, self.config.embedding_size, 2)
        # [E // 2]
        denom = jnp.exp(-jnp.log(10000) * i / self.config.embedding_size)
        # [B, E // 2]
        context = snr[:, None] * denom[None] * self.config.embedding_factor
        # [B, E // 2, 2]
        pe = jnp.stack([jnp.sin(context), jnp.cos(context)], axis=-1)
        # [B, E]
        return pe.reshape(-1, self.config.embedding_size)
