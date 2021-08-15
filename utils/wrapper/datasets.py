from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from speechset import VocoderDataset


class DatasetWrapper:
    """Vocoder datasets with random segment.
    """
    def __init__(self, vocdata: VocoderDataset, segsize: int, hop: int):
        """Initializer.
        Args:
            vocdata: vocoder datasets.
            segsize: size of the segment.
            hop: size of the stft hop.
        """
        self.dataset = vocdata
        self.segsize = segsize
        self.hop = hop
        self.length = tf.data.experimental.cardinality(
            self.dataset).numpy().item()

    def __len__(self) -> int:
        """Return lengths.
        """
        return self.length

    def __iter__(self) -> Iterable:
        """Generate wrapped iterator.
        Returns:
            Datasets.Iterator, iterator wrapper for audio random segmentation.
        """
        return DatasetWrapper.Iterator(
            self.dataset.as_numpy_iterator(), self.segsize, self.hop)

    class Iterator:
        """Iterator wrapper for audio random segmentation.
        """
        def __init__(self, iterator: Iterable, segsize: int, hop: int):
            """Initializer.
            Args:
                iterator: iterable object.
                segsize: size of the segment.
                hop: size of the stft hop.
            """
            self.iterator = iterator
            self.segsize = segsize
            self.hop = hop

        def __next__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Get next elements.
            Returns:
                mel: [float32; [B, S // H, M]], randomly segmented mel spectrogram.
                speech: [float32; [B, S]], segmented audio w.r.t. mel.
            """
            melseg = self.segsize // self.hop
            # np.ndarray: [B, T // H, M], [B, T], [B], [B]
            mel, speech, mellen, speechlen = next(self.iterator)
            # start position
            pos = np.random.uniform(0, mellen - melseg).astype(np.long)
            # [B, S // H, M], segmentation
            mel = np.stack([m[p:p + melseg] for m, p in zip(mel, pos)])
            # [B, S]
            speech = np.stack([s[p:p + self.segsize] for s, p in zip(speech, pos * self.hop)])
            return jnp.asarray(mel), jnp.asarray(speech)
