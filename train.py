import argparse
import json
import os
from typing import List, Tuple

import librosa

import git
import jax
import jax.numpy as jnp
import librosa
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tqdm

from config import Config
from speechset import VocoderDataset
from speechset.datasets import LJSpeech
from utils.wrapper import DatasetWrapper, TrainWrapper
from vlbdiffwave import VLBDiffWaveApp


class Trainer:
    """DiffWave trainer.
    """
    def __init__(self, app: VLBDiffWaveApp, vocdata: VocoderDataset, config: Config):
        """Initializer.
        Args:
            app: diffwave model.
            vocdata: vocoder datasets.
            config: Config, unified configurations.
        """
        self.app = app
        self.vocdata = vocdata
        self.config = config

        self.wrapper = TrainWrapper(self.app.model)

        trainset, testset = self.vocdata.dataset(config.train.split)
        self.trainset = DatasetWrapper(trainset
                .shuffle(config.train.bufsiz)
                .prefetch(tf.data.experimental.AUTOTUNE),
            self.config.train.segsize,
            self.config.data.hop)
        self.testset = testset.prefetch(tf.data.experimental.AUTOTUNE)

        self.optim = optax.adam(
            config.train.learning_rate,
            config.train.beta1,
            config.train.beta2,
            config.train.eps)
        self.optim_state = self.optim.init(self.model.param)

        self.eval_intval = config.train.eval_intval // config.data.batch
        self.ckpt_intval = config.train.ckpt_intval // config.data.batch

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.cmap = plt.get_cmap('viridis').colors
        self.melfilter = librosa.filters.mel(
            config.sr, config.fft, config.mel, config.fmin, config.fmax).T

    def train(self, key: jnp.ndarray, epoch: int = 0, timesteps: int = 10):
        """Train wavegrad.
        Args:
            epoch: int, starting step.
        """
        step = epoch * self.trainsize
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=self.trainsize, leave=False) as pbar:
                for it, (mel, speech) in enumerate(self.trainset):
                    # split key
                    key, s1, s2 = jax.random.split(key, num=3)
                    # [B, T]
                    noise = jax.random.normal(s1, speech.shape)
                    # [B]
                    time = jax.random.uniform(s2, (speech.shape[0],))
                    # [], FrozenDict, ForzenDict
                    loss, grads = self.wrapper.gradient(
                        self.app.param, speech, noise, time, mel)
                    # optimizer update
                    updates, self.optim_state = self.optim.update(grads, self.optim_state)
                    # gradient update
                    self.app.param = optax.apply_update(self.app.param, updates)

                    norm = jnp.mean([jnp.norm(x) for x in jax.tree_utils.tree_leaves(grads)])
                    del grads

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy().item(),
                         'step': step,
                         'grad': norm.numpy().item()})

                    with self.train_log.as_default():
                        tf.summary.scalar('loss', loss, step)
                        tf.summary.scalar('grad norm', norm, step)

                        if (it + 1) % (len(self.trainset) // 10) == 0:
                            key, sub = jax.random.split(key)
                            pred, _ = self.app(mel, jnp.linspace(0., 1., timesteps), key=sub)
                            tf.summary.audio(
                                'train', pred[..., None], self.config.data.sr, step)
                            tf.summary.image(
                                'train mel', self.mel_img(pred), step)
                            del pred

            self.app.write(
                '{}_{}.ckpt'.format(self.ckpt_path, epoch), self.optim_state)

            # evaluation loss
            loss = [
                self.wrapper.compute_loss(
                    self.app.param, speech, noise, time, mel).item()
                for mel, speech in DatasetWrapper(
                    self.testset, self.config.train.segsize, self.config.data.hop)]
            loss = sum(loss) / len(loss)
            # test log
            with self.test_log.as_default():
                tf.summary.scalar('loss', loss, step)

                gt, pred, ir = self.eval(timesteps)
                tf.summary.audio(
                    'gt', gt[None, :, None], self.config.data.sr, step)
                tf.summary.audio(
                    'eval', pred[None, :, None], self.config.data.sr, step)

                tf.summary.image(
                    'gt mel', self.mel_img(gt[None]), step)
                tf.summary.image(
                    'eval mel', self.mel_img(pred[None]), step)

                for i in ir:
                    tf.summary.audio(
                        'ir_{}'.format(i),
                        np.clip(ir[i][None, :, None], -1., 1.),
                        self.config.data.sr, step)
                
                del gt, pred, ir

    def mel_img(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [float32; [B, T]], speech signal.
        Returns:
            [float32; [B, M, T // H, 3]], mel-spectrogram in viridis color map.
        """
        # [B, M, T // H]
        mel = self.mel_fn(signal).transpose(0, 2, 1)
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min())
        # in range(0, 255)
        mel = (mel * 255).astype(jnp.long)
        # [B, M, T // H, 3]
        mel = self.cmap[mel]
        # make origin lower
        mel = jnp.flip(mel, axis=1)
        return mel

    def eval(self, timesteps: int = 10) -> \
            Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
        """Generate evaluation purpose audio.
        Args:
            timesteps: the number of the time steps. 
        Returns:
            speech: [float32; [T]], ground truth.
            pred: [float32; [T]], predicted.
            ir: List[jnp.ndarray], steps x [float32; [T]],
                intermediate represnetations.
        """
        # [B, T // H, M], [B, T], [B], [B]
        mel, speech, mellen, speechlen = next(self.testset.as_numpy_iterator())
        # [1, T], steps x [1, T]
        pred, ir = self.app(
            mel[0:1, :mellen[0]],
            jnp.linspace(0., 1., timesteps),
            key=jax.random.PRNGKey(0))
        # [T]
        pred = pred.squeeze(axis=0)
        # config.model.iter x [T]
        ir = [i.squeeze(axis=0) for i in ir]
        return speech[0, :speechlen[0]], pred, ir
    
    def mel_fn(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Convert signal to the mel-spectrogram.
        Args:
            signal: [float32; [B, T]], input signal.
        Returns:
            [float32; [B, T // H, M]], mel-spectrogram.
        """
        # [B, T // H, fft // 2 + 1]
        stft = librosa.stft(
            np.asarray(signal),
            self.config.data.fft,
            self.config.data.hop,
            self.config.data.win,
            self.config.data.win_fn,
            center=True, pad_mode='reflect')
        # [B, T // H, M]
        mel = np.abs(stft) @ self.melfilter
        # [B, T // H, M]
        return jnp.array(np.log(np.maximum(mel, self.config.data.eps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--download', default=False, action='store_true')
    parser.add_argument('--from-raw', default=False, action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    args = parser.parse_args()

    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # prepare datasets
    lj = LJSpeech(args.data_dir, args.download, not args.from_raw)
    vocdata = VocoderDataset(lj, config.data)

    # randomness control
    key, sub = jax.random.split(jax.random.PRNGKey(args.seed), num=2)

    # model definition
    diffwave = VLBDiffWaveApp(config.model)
    diffwave.init(sub)
    trainer = Trainer(diffwave, vocdata, config)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        diffwave.restore(ckpt_path, trainer.optim_state)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(key, args.load_epoch)
