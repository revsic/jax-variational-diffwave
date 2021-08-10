from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax
import numpy as np

from .config import Config
from .impl import Model


class VLBDiffWave:
    """VLB-DiffWave: Variational Diffusion for Audio Synthesis.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configuration.
        """
        self.config = config
        self.model = Model(config=config)
        self.param = None

    def __call__(self,
                 mel: jnp.ndarray,
                 timesteps: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None,
                 noise: Optional[jnp.ndarray] = None) -> \
            Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Generate audio from mel-spectrogram.
        Args:
            mel: [float32; [B, T // H, M]], condition mel-spectrogram.
            timesteps: [float32; [S]], time steps.
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
            noise = jnp.random.normal(key, shape=(bsize, mellen * self.config.hop))
        # initilize
        ir, signal = [noise], noise
        # []
        for time in timesteps:
            # [B, T]
            _, signal = self.model.apply(self.param, signal, time, mel)
            # save intermediate representations
            ir.append(signal)
        # [B, T]
        return signal, ir

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

    def write(self, path: str):
        """Write model checkpoints.
        Args:
            path: path to the checkpoint.
        """
        with open(path + '.ckpt', 'wb') as f:
            f.write(flax.serialization.to_bytes(self.param))

    def restore(self, path: str):
        """Restore model parameters from `path` checkpoint.
        Args:
            path: path to the checkpoint.
            params: model parameters, for restore.
        """
        with open(path, 'rb') as f:
            binary = f.read()
        if self.param is None:
            # initialize parameters with dummy key.
            self.init(jax.random.PRNGKey(0))
        # restore
        flax.serialization.from_bytes(self.param, binary)
