from typing import Tuple

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
    
    def snr(self, time: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute SNR and alpha, sigma.
        Args:
            time: [float32; [B]], current timestep.
        Returns:
            [float32; [B]], logSNR, normalized -logSNR, alpha and sigma.
        """
        # [B], [B]
        logsnr, norm_nlogsnr = self.logsnr(time)
        # [B]
        alpha = jnp.sqrt(jnp.maximum(nn.sigmoid(logsnr), 1e-5))
        sigma = jnp.sqrt(jnp.maximum(nn.sigmoid(-logsnr), 1e-5))
        return logsnr, norm_nlogsnr, alpha, sigma

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
        # [B] x 4
        _, norm_nlogsnr, alpha, sigma = self.snr(time)
        # [B, T]
        noise = self.diffwave(signal, norm_nlogsnr, mel)
        # [B, T]
        denoised = (signal - sigma * noise) / alpha        
        return noise, denoised

    def diffusion(self,
                  signal: jnp.ndarray,
                  noise: jnp.ndarray,
                  s: jnp.ndarray,
                  t: jnp.ndarray) -> jnp.ndarray:
        """Add noise to the signal.
        Args:
            signal: [float32; [B, T]], input signal.
            noise: [float32; [B, T]], gaussian noise.
            s: [float32; [B]], start time in range[0, 1].
            t: [float32; [B]], target time in range[0, 1], s < t.
        """
        # B
        bsize = s.shape[0]
        # [B + B]
        time = jnp.concatenate([s, t], axis=0)
        # [B + B] x 4
        _, _, alpha, sigma = self.denoiser.logsnr.snr(time)
        # [B]
        alpha_s, alpha_t = alpha[:bsize], alpha[bsize:]
        sigma_s, sigma_t = sigma[:bsize], sigma[bsize:]
        # [B]
        alpha_tbars = alpha_t / alpha_s
        sigma_tbars = jnp.sqrt(sigma_t ** 2 - alpha_tbars * sigma_s ** 2)
        # [B, T]
        return alpha_tbars * signal + sigma_tbars * noise
