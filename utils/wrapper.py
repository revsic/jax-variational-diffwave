from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from vlbdiffwave import VLBDiffWave


class TrainWrapper:
    """Train-wrapper for vlb-diffwave.
    """
    def __init__(self, diffwave: VLBDiffWave):
        """Initializer.
        Args:
            diffwave: Target model.
        """
        self.diffwave = diffwave

    def compute_loss(self,
                     params: flax.core.frozen_dict.FrozenDict,
                     signal: jnp.ndarray,
                     noise: jnp.ndarray,
                     timestep: jnp.ndarray,
                     mel: jnp.ndarray) -> jnp.ndarray:
        """Compute noise estimation loss.
        Args:
            params: model prameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            timestep: [float32; [B]], input timestep.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:
            [float32; []], loss value.
        """
        model = self.diffwave.model
        # [B, T]
        diffusion = model.apply(
            params, signal, noise, timestep, method=model.diffusion)
        # [B, T]
        estim, _ = model.apply(params, diffusion, timestep, mel)
        # [B]
        mse = jnp.square(noise - estim).sum(axis=-1)
        # [B]
        dlogsnr, _ = jax.grad(
            # lifting
            lambda t: model.logsnr.apply(params['logsnr'], t),
            # compute gradient only on log-SNR
            has_aux=True)(timestep)
        # [B]
        loss = -0.5 * dlogsnr * mse
        # []
        return loss.mean()

    @jax.jit
    def gradient(self,
                 params: flax.core.frozen_dict.FrozenDict,
                 signal: jnp.ndarray,
                 noise: jnp.ndarray,
                 timestep: jnp.ndarray,
                 mel: jnp.ndarray) -> \
            Tuple[jnp.ndarray, flax.core.frozen_dict.FrozenDict]:
        """Compute gradients.
        Args:
            params: diffwave model parameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            timestep: [float32; [B]], input timestep.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
        Returns:

        """
        # [], FrozenDict
        loss, grads = jax.value_and_grad(self.compute_loss)(
            params, signal, noise, timestep, mel)
        return loss, grads
        # regularizer

