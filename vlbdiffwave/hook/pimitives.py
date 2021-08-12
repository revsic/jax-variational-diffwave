import functools
from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp

from ..logsnr import LogSNR
from ..pipeline import Pipeline


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def hooked_logsnr(logsnr: LogSNR,
                  _: Pipeline,
                  param: flax.core.frozen_dict.FrozenDict,
                  time: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Hook log-SNR for efficient gradient computation of MC variance regularizer.
    Args:
        logsnr: model of LogSNR.
        _: residual pipeline.
        param: model parameters.
        time: [float32; [B]], current timestep.
    Returns:
        [float32; [B]], log-SNR and normalized -logSNR.
    """
    return logsnr.apply(param, time)


def fwd_logsnr(logsnr: LogSNR,
               _: Pipeline,
               param: flax.core.frozen_dict.FrozenDict,
               time: jnp.ndarray) -> \
        Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable]:
    """Forward function of hooked-logsnr.
    Args:
        logsnr: model of LogSNR.
        _: residual pipeline.
        param: model parameters.
        time: [float32; [B]], current timestep.
    Returns:
        out: [float32; [B]], log-SNR and normalized -logSNR.
        vjp: vjp function of `logsnr.apply`.
    """
    # ([B], [B]), Callable
    return jax.vjp(logsnr.apply, param, time)


def bwd_logsnr(_: LogSNR,
               pipeline: Pipeline,
               vjp: Callable,
               cot: Tuple[jnp.ndarray, jnp.ndarray]) -> \
        Tuple[flax.core.frozen_dict.FrozenDict, jnp.ndarray]:
    """Backward pass.
    Args:
        _: model of LogSNR.
        pipeline: residual pipeline.
        vjp: vjp function (residual set).
        cot: cotangent value of two outputs of `logsnr.apply`.
    """
    # flax.core.frozen_dict.FrozenDict, jnp.ndarray
    cot_param, cot_time = vjp(cot)
    # loss value, []
    loss = pipeline.get()
    # add MC variance regularizer
    cot_param = jax.tree_map(lambda p: 2 * loss * p, cot_param)
    # do not touch contangent of time
    return cot_param, cot_time

# define vjp
hooked_logsnr.defvjp(fwd_logsnr, bwd_logsnr)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def hook_pipeline(fn: Callable, pipeline: Pipeline, *args) -> Any:
    """Save result of `fn` to the pipeline.
    Args:
        fn: target function.
        pipeline: residual pipeline.
        args: required arguments.
    Returns:
        outputs of fn.
    """
    result = fn(*args)
    pipeline.put(result)
    return result


def fwd_pipeline(fn: Callable, pipeline: Pipeline, *args) -> Tuple[Any, Callable]:
    """Save result of `fn` to the pipeline.
    Args:
        fn: target function.
        pipeline: residual pipeline.
        args: required arguments.
    Returns:
        outputs of fn and vjp function.
    """
    primals, vjp = jax.vjp(fn, *args)
    pipeline.put(primals)
    return primals, vjp


def bwd_pipeline(_: Callable, _: Pipeline, vjp: Callable, cot: jnp.ndarray) -> Any:
    """Backward.
    Args:
        fn: target function.
        pipeline: residual pipeline.
        vjp: vjp function.
        cot: cotangent.
    Returns:
        cotangents.
    """
    return vjp(cot)

# define vjp
hook_pipeline.defvjp(fwd_pipeline, bwd_pipeline)
