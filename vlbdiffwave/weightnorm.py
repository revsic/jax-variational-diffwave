from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


def constant(value: float):
    """Constant initializer.
    """
    def init(_, shape, dtype=jnp.float32):
        # _ = key
        return jnp.full(shape, value, dtype)
    return init


class WNDilatedConv(nn.Module):
    """Weight-normalized dilated convolution.
    """
    channels: int
    kernels: int
    lhs_dilations: int = 1
    rhs_dilations: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        """Run wn-dilated convolution.
        Args:
            inputs: [float32; [B, T, C]], input tensor.
        Returns:
            [float32; [B, T, C']], convolved.
        """
        # C
        in_channels = inputs.shape[-1]
        # [K, C, C']
        kernel = self.param('kernel', self.kernel_init,
                            [self.kernels, in_channels, self.channels])
        # []
        kernel_norm = jnp.linalg.norm(kernel)
        # []
        norm = self.param('norm', constant(kernel_norm), [])
        # [B, T, C']
        x = jax.lax.conv_general_dilated(
            inputs,
            norm * kernel / kernel_norm,
            window_strides=[1],
            padding='SAME',
            lhs_dilation=[self.lhs_dilations],  # = strides of transposed conv
            rhs_dilation=[self.rhs_dilations],  # = atrous conv = dilated conv
            dimension_numbers=nn.linear._conv_dimension_numbers(inputs.shape))
        if self.use_bias:
            bias = self.param('bias', self.bias_init, [self.channels])
            # [B, T, C']
            x = x + bias
        return x


class WNDense(nn.Module):
    """Weight-normalized Dense layer.
    """
    channels: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Run weight-normalized dense layer.
        Args:
            inputs: [float32; [..., C]], input tensor.
        Returns:
            [float32; [..., C']], projected tensor.
        """
        # [C, C']
        kernel = self.param(
            'kernel', self.kernel_init, (inputs.shape[-1], self.channels))
        # []
        kernel_norm = jnp.linalg.norm(kernel)
        # []
        norm = self.param('norm', constant(kernel_norm), [])
        # [C, C']
        normed_kernel = norm * kernel / kernel_norm
        # [..., C]
        x = inputs @ normed_kernel
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.channels,))
            x = x + bias
        return x
