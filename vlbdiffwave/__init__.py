import os
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax
import numpy as np

from .config import Config
from .impl import VLBDiffWave


class VLBDiffWaveApp:
    """Application for vlb-diffwave.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configuration.
        """
        self.config = config
        self.model = VLBDiffWave(config)
        self.param = None

    def __call__(self,
                 mel: jnp.ndarray,
                 timesteps: Union[int, jnp.ndarray] = 10,
                 key: Optional[jnp.ndarray] = None,
                 noise: Optional[jnp.ndarray] = None) -> \
            Tuple[jnp.ndarray, List[np.ndarray]]:
        """Generate audio from mel-spectrogram.
        Args:
            mel: [float32; [B, T // H, M]], condition mel-spectrogram.
            timesteps: [float32; [S]], time steps, from one to zero, including endpoint.
            key: jax random prng key.
            noise: [float32; [B, T]], starting noise.
                neither key nor noise should be None.
        Returns:
            [float32; [B, T]], generated audio and intermediate representations.
        """
        # assertion
        assert key is not None or noise is not None
        if noise is None:
            # B, T // H, _
            bsize, mellen, _ = mel.shape
            # [B, T]
            noise = jax.random.normal(key, shape=(bsize, mellen * self.config.hop))
        if isinstance(timesteps, int):
            # [S]
            timesteps = jnp.linspace(1., 0., timesteps + 1)
        # scanning, outputs and intermediate representations
        return self.inference(mel, timesteps, noise)

    def inference(self, mel: jnp.ndarray, timesteps: jnp.ndarray, noise: jnp.ndarray) -> \
            Tuple[jnp.ndarray, List[np.ndarray]]:
        """Generate audio, just-in-time compiled.
        Args:
            mel: [float32; [B, T // H, M]], condition mel-spectrogram.
            timesteps: [float32; [S + 1]], time steps.
            noise: [float32; [B, T]], starting noise.
                neither key nor noise should be None.
        Returns:
            [float32; [B, T]], generated audio and intermdeidate representations.
        """
        def scanner(signal: jnp.ndarray, time: jnp.ndarray) -> \
                Tuple[jnp.ndarray, jnp.ndarray]:
            """Scanner for iterating timesteps and gradual denoising.
            Args:
                signal: [float32; [B, T]], speech signal.
                time: [float32; [2]], current and next timesteps.
            Returns:
                [float32; [B, T]], denoised signal for both carry and outputs.
            """
            # [], []
            time_t, time_s = time
            # [B, T]
            denoised = self.model.denoise(
                self.param, signal, mel, time_t[None], time_s[None])
            # stop trace ir and move to cpu memory for preventing oom
            return denoised, np.asarray(jax.lax.stop_gradient(denoised))
        # [S, 2], 
        timesteps = jnp.stack([timesteps[:-1], timesteps[1:]], axis=1)
        # scan
        return jax.lax.scan(scanner, noise, timesteps)

    def init(self, key: np.ndarray):
        """Initialize model parameters.
        Args:
            key: jax prng keys.
        Returns:
            model parameters.
        """
        # placeholders
        signal = jnp.zeros([1, self.config.hop])
        time = jnp.zeros([1])
        mel = jnp.zeros([1, 1, 80])
        # initialize
        self.param = self.model.init(key, signal, time, mel)

    def write(self, path: str, optim: Optional[Any] = None):
        """Write model checkpoints.
        Args:
            path: path to the checkpoint.
            optim: optimizer state.
        """
        with open(path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.param))
        if optim is not None:
            name, ext = os.path.splitext(path)
            with open(f'{name}_optim{ext}', 'wb') as f:
                f.write(flax.serialization.to_bytes(optim))

    def restore(self, path: str, optim: Optional[Any] = None):
        """Restore model parameters from `path` checkpoint.
        Args:
            path: path to the checkpoint.
            optim: optimizer state, if provided.
        """
        with open(path, 'rb') as f:
            binary = f.read()
        if self.param is None:
            # initialize parameters with dummy key.
            self.init(jax.random.PRNGKey(0))
        # restore
        flax.serialization.from_bytes(self.param, binary)
        # auxiliary restoration
        if optim is not None:
            name, ext = os.path.splitext(path)
            with open(f'{name}_optim{ext}', 'rb') as f:
                binary = f.read()
            # restore
            flax.serialization.from_bytes(optim, binary)
