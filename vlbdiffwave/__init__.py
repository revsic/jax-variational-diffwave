import jax
import jax.numpy as jnp
import flax
import numpy as np

from .config import Config
from .impl import Denoiser


class VLBDiffWave:
    """VLB-DiffWave: Variational Diffusion for Audio Synthesis.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configuration.
        """
        self.config = config
        self.denoiser = Denoiser(config=config)
        self.param = None

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
        self.param = self.denoiser.init(key, signal, time, mel)

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
