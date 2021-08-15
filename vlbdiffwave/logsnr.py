from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp

from .weightnorm import constant


class PosDense(nn.Module):
    """Dense-layer with positive weights
    """
    channels: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Project inputs with positive weights.
        Args:
            inputs: [float32; [..., C]], input tensor.
        Returns:
            [float32; [..., C']], projected.
        """
        # C
        in_channels = inputs.shape[-1]
        # [C, C']
        kernel = self.param('kernel', self.kernel_init,
                            [in_channels, self.channels])
        # [..., C']
        x = inputs @ nn.softplus(kernel)
        if self.use_bias:
            # [C']
            bias = self.param('bias', self.bias_init, [self.channels])
            # [..., C']
            x = x + nn.softplus(bias)
        return x


class LogSNR(nn.Module):
    """Learnable noise scheduler: logSNR.
    """
    internal: int
    # initialize in range [-10, 10]
    # reference from Variational Diffusion Models, range of learned log-SNR.
    initial_gamma_min: float = -10.
    initial_gamma_gap: float = 20.

    def setup(self):
        """Setup layers.
        """
        # boundary
        self.gamma_min = self.param(
            'gamma_min', constant(self.initial_gamma_min), [])
        # force positive
        self.gamma_gap = nn.softplus(
            self.param('gamma_gap', constant(self.initial_gamma_gap), []))
        # projector
        self.proj1 = PosDense(channels=1)
        self.proj2 = PosDense(channels=self.internal)
        self.proj3 = PosDense(channels=1)
        # memory
        self.memory = None

    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute logSNR from continuous timesteps.
        Args:
            inputs: [float32; [B]], timesteps in [0, 1].
        Returns:
            [float32; [B]], logSNR and normalized -logSNR.
        """
        # [B + 2], add terminal point
        x = jnp.concatenate([jnp.array([0., 1.]), inputs], axis=0) 
        # [B + 2, 1]
        l1 = self.proj1(x[:, None])
        # [B + 2, C]
        l2 = nn.sigmoid(self.proj2(l1))
        # [B + 2], learned scheduler
        sched = jnp.squeeze(l1 + self.proj3(l2), axis=-1)
        # [], [], [B]
        s0, s1, sched = sched[0], sched[1], sched[2:]
        # [B], normalized -logSNR
        norm_nlogsnr = (sched - s0) / (s1 - s0)        
        # [B], boundary matching
        nlogsnr = self.gamma_min + self.gamma_gap * norm_nlogsnr
        return -nlogsnr, norm_nlogsnr
