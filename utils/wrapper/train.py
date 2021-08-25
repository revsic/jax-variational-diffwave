from typing import Dict, Tuple

import flax
import jax
import jax.numpy as jnp

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
        self.gradient_fn = jax.jit(jax.value_and_grad(self.compute_loss, has_aux=True))

    def gradient(self,
                 params: flax.core.frozen_dict.FrozenDict,
                 signal: jnp.ndarray,
                 noise: jnp.ndarray,
                 mel: jnp.ndarray,
                 timestep: jnp.ndarray) -> \
            Tuple[
                Tuple[jnp.ndarray, Dict[str, jnp.ndarray]],
                flax.core.frozen_dict.FrozenDict]:
        """Compute gradient with MC-variance regularizing loss.
        Args:
            param: model parameters.
            speech: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], sampled noise.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
            timestep: [float32; [B]], timesteps.
        Returns:
            loss: [float32; []], total loss.
            losses: [float32; []], loss values for summary.
            grads: gradients for each parameters.
        """
        # [], FrozenDict, FrozenDict
        (loss, losses), grads = self.gradient_fn(params, signal, noise, mel, timestep)
        # compute squared loss for snr interpolation parameters
        interp = {
            key: val
            for key, val in grads['logsnr']['params'].items()
            if not key.startswith('gamma')}
        interp = jax.tree_map(lambda p: 2 * loss * p, interp)
        # udpate gradients
        grads = flax.core.freeze({
            'diffwave': grads['diffwave'],
            'logsnr': {'params': {**grads['logsnr']['params'], **interp}}})
        return (loss, losses), grads

    def compute_loss(self,
                     params: flax.core.frozen_dict.FrozenDict,
                     signal: jnp.ndarray,
                     noise: jnp.ndarray,
                     mel: jnp.ndarray,
                     timestep: jnp.ndarray) -> \
            Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute VDM loss.
        Args:
            params: model prameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
            timestep: [float32; [B]], input timestep.
        Returns:
            [float32; []], loss value.
        """
        # [B, T]
        _, _, z0 = self.model.diffusion(params, signal, noise, jnp.zeros(timestep.shape))
        # [B], [B], [B, T]
        alpha1, sigma1, z1 = self.model.diffusion(params, signal, noise, jnp.ones(timestep.shape))
        # [], standard gaussian negative log-likelihood
        prior_loss = jnp.square(z1).mean()
        # []
        prior_entropy = self.nll(
            z1, alpha1[:, None] * signal, sigma1[:, None]).mean()
        # []
        reconst = jnp.square(z0 - signal).mean()
        # []
        diffusion_loss = self.diffusion_loss(params, signal, noise, mel, timestep)
        # []
        loss = reconst + diffusion_loss + prior_loss - prior_entropy
        return loss, {
            'loss': loss,
            'reconst': reconst, 'diffusion': diffusion_loss,
            'prior': prior_loss, 'prior-entropy': prior_entropy}

    def nll(self, sample: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        """Compute point-wise gaussian negative log-likelihood.
        Args:
            sample: [float32; [...]], data sample.
            mean: [float32; [...]], gaussian mean.
            std: [float32; [...]], gaussian standard deviation, positive real.
        Returns:
            [float32; [...]], nll.
        """
        # [...]
        logstd = jnp.log(jnp.maximum(std, 1e-5))
        # [...]
        return 2 * logstd + std ** -2 * (sample - mean) ** 2

    def diffusion_loss(self,
                       params: flax.core.frozen_dict.FrozenDict,
                       signal: jnp.ndarray,
                       noise: jnp.ndarray,
                       mel: jnp.ndarray,
                       timestep: jnp.ndarray) -> jnp.ndarray:
        """Compute noise estimation loss.
        Args:
            params: model prameters.
            signal: [float32; [B, T]], speech signal.
            noise: [float32; [B, T]], noise signal.
            mel: [float32; [B, T // H, M]], mel-spectrogram.
            timestep: [float32; [B]], input timestep.
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
            logsnr, _ = self.model.logsnr.apply(params['logsnr'], time)
            return logsnr.sum()
        # [B], dlog-SNR/dt
        dlogsnr = jax.grad(logsnr)(timestep)
        # [B]
        loss = -0.5 * dlogsnr * mse
        # []
        loss = loss.mean()
        return loss
