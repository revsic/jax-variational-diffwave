import functools
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp

from .logsnr import LogSNR


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def hooked_logsnr(logsnr: LogSNR,
                  param: flax.core.frozen_dict.FrozenDict,
                  time: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Hook log-SNR for efficient gradient computation of MC variance regularizer.
    Args:
        logsnr: model of LogSNR.
        param: model parameters.
        time: [float32; [B]], current timestep.
    Returns:
        [float32; [B]], log-SNR and normalized -logSNR.
    """
    return logsnr.apply(param, time)


def fwd_logsnr(logsnr: LogSNR,
               param: flax.core.frozen_dict.FrozenDict,
               time: jnp.ndarray) -> \
        Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable]:
    """Forward function of hooked-logsnr.
    Args:
        logsnr: model of LogSNR.
        param: model parameters.
        time: [float32; [B]], current timestep.
    Returns:
        out: [float32; [B]], log-SNR and normalized -logSNR.
        vjp: vjp function of `logsnr.apply`.
    """
    # ([B], [B]), Callable
    return jax.vjp(logsnr.apply, param, time)


def bwd_logsnr(logsnr: LogSNR,
               vjp: Callable,
               cot: Tuple[jnp.ndarray, jnp.ndarray]) -> \
        Tuple[flax.core.frozen_dict.FrozenDict, jnp.ndarray]:
    """Backward pass.
    Args:
        _: model of LogSNR.
        vjp: vjp function.
        cot: cotangent value of two outputs of `logsnr.apply`.
    """
    # flax.core.frozen_dict.FrozenDict, jnp.ndarray
    cot_param, cot_time = vjp(cot)
    # get loss
    loss = 0.5 if logsnr.pipeline.memory is None \
        else jax.lax.stop_gradient(logsnr.pipeline.memory)
    # split parameters
    endp, interp = {}, {}
    for key, val in cot_param['params'].items():
        if key.startswith('gamma'):
            endp[key] = val
        else:
            interp[key] = val
    # add MC variance regularizer
    interp = jax.tree_map(lambda p: 2 * loss * p, interp)
    # do not touch contangent of time
    return flax.core.freeze({'params': {**endp, **interp}}), cot_time

# define vjp
hooked_logsnr.defvjp(fwd_logsnr, bwd_logsnr)
