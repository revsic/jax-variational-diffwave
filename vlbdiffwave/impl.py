import flax.linen as nn
import jax.numpy as jnp

from .config import Config
from .diffwave import DiffWave
from .logsnr import LogSNR


class Denoiser(nn.Module):
    """Model definition
    """
    config: Config

    def setup(self):
        """Initialize models.
        """
        self.diffwave = DiffWave(config=self.config)
        self.logsnr = LogSNR(internal=self.config.internal)
    
    def __call__(self,
                 signal: jnp.ndarray,
                 time: jnp.ndarray,
                 mel: jnp.ndarray) -> jnp.ndarray:
        """Denoise signal w.r.t timestep on mel-condition.
        Args:
            signal: [float32; [B, T]], noised signal.
            timestep: [float32; [B]], current timestep.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:
            [float32; [B, T]], noise and denoised signal.
        """
        # [B], [B]
        logsnr, norm_nlogsnr = self.logsnr(time)
        # [B, T]
        noise = self.diffwave(signal, norm_nlogsnr, mel)
        # [B]
        alpha = jnp.sqrt(jnp.maximum(nn.sigmoid(logsnr), 1e-5))
        sigma = jnp.sqrt(jnp.maximum(nn.sigmoid(-logsnr), 1e-5))
        # [B, T]
        denoised = (signal - sigma * noise) / alpha        
        return noise, denoised
