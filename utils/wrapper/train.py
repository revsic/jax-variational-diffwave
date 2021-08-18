from typing import Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from vlbdiffwave.hook import hooked_logsnr
from vlbdiffwave.impl import VLBDiffWave


class TrainWrapper:
    """Train-wrapper for vlb-diffwave.
    """
    def __init__(self, diffwave: VLBDiffWave):
        """Initializer.
        Args:
            diffwave: Target model.
        """
        self.model = diffwave
        self.gradient = jax.jit(jax.value_and_grad(self.compute_loss, has_aux=True))

    def reinit(self):
        """Reinitialize.
        """
        self.model.logsnr.pipeline.memory = None

    def compute_loss(self,
                     params: flax.core.frozen_dict.FrozenDict,
                     signal: jnp.ndarray,
                     noise: jnp.ndarray,
                     mel: jnp.ndarray,
                     timestep: jnp.ndarray,
                     hook: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute VDM loss.
        Args:
            params: model prameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
            timestep: [float32; [B]], input timestep.
            hook: [bool, []], whether hook the logsnr or not.
        Returns:
            [float32; []], loss value.
        """
        # [B, T]
        _, _, z0 = self.model.diffusion(params, signal, noise, jnp.zeros(timestep.shape))
        # [B], [B], [B, T]
        alpha1, sigma1, z1 = self.model.diffusion(params, signal, noise, jnp.ones(timestep.shape))
        # []
        prior_ll = self.likelihood(z1, jnp.zeros(1), jnp.ones(1)).mean()
        # []
        prior_entropy = -self.likelihood(z1, alpha1[:, None] * signal, sigma1).mean()
        # []
        reconst = -self.likelihood(z0, signal).mean()
        # []
        diffusion_loss = self.diffusion_loss(params, signal, noise, mel, timestep, hook)
        # []
        # , minimize: reconstruction, diffusion loss
        # , maximize: prior likelihood, prior entropy
        loss = reconst + diffusion_loss - prior_ll - prior_entropy
        return loss, {
            'loss': loss,
            'reconst': reconst, 'diffusion': diffusion_loss,
            'prior-ll': prior_ll, 'prior-entropy': prior_entropy}

    def likelihood(self, sample: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        """Compute point-wise gaussian likelihood.
        Args:
            sample: [float32; [...]], data sample.
            mean: [float32; [...]], gaussian mean.
            std: [float32; [...]], gaussian standard deviation, positive real.
        """
        # [...]
        logstd = jnp.log(jnp.maximum(std, 1e-5))
        # [...]
        return -0.5 * (jnp.log(2 * jnp.pi) + 2 * logstd + std ** -2 * (sample - mean) ** 2)

    def diffusion_loss(self,
                       params: flax.core.frozen_dict.FrozenDict,
                       signal: jnp.ndarray,
                       noise: jnp.ndarray,
                       mel: jnp.ndarray,
                       timestep: jnp.ndarray,
                       hook: bool = True) -> jnp.ndarray:
        """Compute noise estimation loss.
        Args:
            params: model prameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
            timestep: [float32; [B]], input timestep.
            hook: [bool, []], whether hook the logsnr or not.
        Returns:
            [float32; []], loss value.
        """
        # [B, T]
        _, _, diffusion = self.model.diffusion(params, signal, noise, timestep)
        # [B, T]
        estim, _ = self.model.apply(params, diffusion, mel, timestep)
        # [B]
        mse = jnp.square(noise - estim).mean(axis=-1)
        # for derivatives of log-SNR
        def logsnr(time: jnp.ndarray):
            # condition on hook for dlogsnr tensor trace error
            logsnr, _ = hooked_logsnr(self.model.logsnr, params['logsnr'], time) \
                if hook else self.model.logsnr.apply(params['logsnr'], time)
            return logsnr.sum()
        # [B], dlog-SNR/dt
        dlogsnr = jax.grad(logsnr)(timestep)
        # [B]
        loss = -0.5 * dlogsnr * mse
        # []
        loss = loss.mean()
        # set loss to the memory
        self.model.logsnr.pipeline.memory = loss
        return loss
