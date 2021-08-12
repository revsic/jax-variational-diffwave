from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from vlbdiffwave import VLBDiffWaveApp


class TrainWrapper:
    """Train-wrapper for vlb-diffwave.
    """
    def __init__(self, diffwave: VLBDiffWaveApp):
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
        diffusion = model.diffusion(params, signal, noise, timestep)
        # [B, T]
        estim, _ = model.apply(params, diffusion, timestep, mel)
        # [B]
        mse = jnp.square(noise - estim).sum(axis=-1)
        # [B], dlog-SNR/dt
        dlogsnr, _ = jax.grad(
            # lifting
            lambda t: model.logsnr.apply(params['logsnr'], t),
            # compute gradient only on log-SNR w.r.t. timestep
            argnums=1, has_aux=True)(timestep)
        # [B]
        loss = -0.5 * dlogsnr * mse
        # []
        loss = loss.mean()
        # set loss to the memory
        _, updated_state = model.logsnr.apply(
            params['logsnr'], loss,
            method=model.logsnr.put, mutable=['memory'])
        return loss, updated_state

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
            loss: loss value.
            grads: gradients for each parameters.
        """
        # [], FrozenDict
        (loss, updated_state), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True)(params, signal, noise, timestep, mel)
        return loss, updated_state, grads
