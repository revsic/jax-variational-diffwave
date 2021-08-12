from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax.numpy as jnp

from .config import Config
from .diffwave import DiffWave
from .hook import hooked_logsnr
from .logsnr import LogSNR


class VLBDiffWave:
    """Model definition of VLB-Diffwave.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configuration.
        """
        self.diffwave = DiffWave(config=config)
        self.logsnr = LogSNR(internal=config.internal)

    def init(self,
             key: jnp.ndarray,
             signal: jnp.ndarray,
             aux: jnp.ndarray,
             mel: jnp.ndarray) -> flax.core.frozen_dict.FrozenDict:
        """Initialize model parameters.
        Args:
            signal: [float32; [B, T]], noise signal.
            aux: [float32; [B]], timestep for logsnr, logSNR for diffwave.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:
            model parameters.
        """
        lparam = self.logsnr.init(key, aux)
        dparam = self.diffwave.init(key, signal, aux, mel)
        return flax.core.freeze({'diffwave': dparam, 'logsnr': lparam})

    def snr(self, param: flax.core.frozen_dict.FrozenDict, time: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute SNR and alpha, sigma.
        Args:
            param: parameters of LogSNR.
            time: [float32; [B]], current timestep.
        Returns:
            [float32; [B]], logSNR, normalized -logSNR, alpha and sigma.
        """
        # [B], [B]
        logsnr, norm_nlogsnr = hooked_logsnr(self.logsnr, param, time)
        # [B]
        alpha = jnp.sqrt(jnp.maximum(nn.sigmoid(logsnr), 1e-5))
        sigma = jnp.sqrt(jnp.maximum(nn.sigmoid(-logsnr), 1e-5))
        return logsnr, norm_nlogsnr, alpha, sigma

    def apply(self,
                 param: flax.core.frozen_dict.FrozenDict,
                 signal: jnp.ndarray,
                 time: jnp.ndarray,
                 mel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Denoise signal w.r.t timestep on mel-condition.
        Args:
            param: model parameters.
            signal: [float32; [B, T]], noised signal.
            timestep: [float32; [B]], current timestep.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:
            [float32; [B, T]], noise and denoised signal.
        """
        # [B] x 4
        _, norm_nlogsnr, alpha, sigma = self.snr(param['logsnr'], time)
        # [B, T]
        noise = self.diffwave.apply(param['diffwave'], signal, norm_nlogsnr, mel)
        # [B, T]
        denoised = (signal - sigma * noise) / alpha        
        return noise, denoised

    def diffusion(self,
                  param: flax.core.frozen_dict.FrozenDict,
                  signal: jnp.ndarray,
                  noise: jnp.ndarray,
                  s: jnp.ndarray,
                  t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Add noise to signal.
        Args:
            param: model parameters.
            signal: [float32; [B, T]], input signal.
            noise: [float32; [B, T]], gaussian noise.
            s: [float32; [B]], start time in range[0, 1].
            t: [float32; [B]], target time in range[0, 1], s < t.
                if t is None, compute q(z_t|x), otherwise, q(z_t|z_s).
        Returns:
            [float32; [B, T]], noised signal.
        """
        # B
        bsize = s.shape[0]
        # [B']
        time = s if t is None else jnp.concatenate([s, t], axis=0)
        # [B'] x 4
        _, _, alpha, sigma = self.snr(param['logsnr'], time)
        if t is None:
            # [B]
            alpha_s, alpha_t = alpha[:bsize], alpha[bsize:]
            sigma_s, sigma_t = sigma[:bsize], sigma[bsize:]
            # [B]
            alpha_tbars = alpha_t / alpha_s
            sigma_tbars = jnp.sqrt(sigma_t ** 2 - alpha_tbars * sigma_s ** 2)
            # [B]
            alpha, sigma = alpha_tbars, sigma_tbars
        # [B, T]
        return alpha * signal + sigma * noise
