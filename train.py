import argparse
import json
import os
from typing import Any, Callable, Dict, List, Tuple

import librosa

import flax
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

IR_INTERVAL = 5


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
        self.testset = DatasetWrapper(
            testset.prefetch(tf.data.experimental.AUTOTUNE),
            self.config.train.segsize,
            self.config.data.hop)

        self.optim = optax.adam(
            config.train.learning_rate,
            config.train.beta1,
            config.train.beta2,
            config.train.eps)
        self.optim_state = self.optim.init(self.app.param)

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.cmap = np.array(plt.get_cmap('viridis').colors)
        self.melfilter = librosa.filters.mel(
            config.data.sr, config.data.fft, config.data.mel,
            config.data.fmin, config.data.fmax)
        
        self.loss_fn = jax.jit(self.wrapper.compute_loss)
        self.update_fn = self.jaxjit_update()

    def jaxjit_update(self) -> Callable:
        """Just-in-time compiled update function.
        """
        def update(param: flax.core.frozen_dict.FrozenDict,
                   optim_state: Any,
                   speech: jnp.ndarray,
                   noise: jnp.ndarray,
                   mel: jnp.ndarray,
                   timestep: jnp.ndarray) -> \
                Tuple[
                    Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],
                    flax.core.frozen_dict.FrozenDict,
                    Any]:
            """Update function.
            Args:
                param: model parameters.
                optim_state: optimizer states.
                speech: [float32; [B, T]], speech signal.
                noise: [float32; [B, T]], sampled noise.
                time: [float32; [B]], timesteps.
                mel: [float32; [B, T // H, M]], mel-spectrogram.
            Returns:
                udpated parameters and optimizer states.
            """
            # [], FrozenDict
            (loss, losses), grads = self.wrapper.gradient(param, speech, noise, mel, timestep)
            # optimizer update
            updates, optim_state = self.optim.update(grads, optim_state)
            # gradient update
            param = optax.apply_updates(param, updates)
            # gradient norm
            gradnorm = jnp.array(
                [jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(grads)]).mean()
            return (loss, losses, gradnorm), param, optim_state
        # jit
        return jax.jit(update)

    def train(self, key: jnp.ndarray, epoch: int = 0, timesteps: int = 10):
        """Train wavegrad.
        Args:
            key: initial random prng key.
            epoch: starting step.
            timesteps: sampling steps.
        """
        step = epoch * len(self.trainset)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.trainset), leave=False) as pbar:
                for it, (mel, speech) in enumerate(self.trainset):
                    # split key
                    key, s1, s2 = jax.random.split(key, num=3)
                    # [B, T]
                    noise = jax.random.normal(s1, speech.shape)
                    # [B], sample uniformly
                    time = jnp.linspace(0., 1., speech.shape[0], endpoint=False)
                    # add start point
                    time = jnp.fmod(jax.random.uniform(s2) + time, 1.)
                    # ([], [], []), FrozenDict, State
                    (loss, losses, grad_norm), self.app.param, self.optim_state = \
                        self.update_fn(self.app.param, self.optim_state,
                                       speech, noise, mel, time)

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss.item(), 'step': step})

                    with self.train_log.as_default():
                        for name, loss in losses.items():
                            tf.summary.scalar(f'common/{name}', loss.item(), step)

                        tf.summary.scalar('common/grad-norm', grad_norm.item(), step)

                        param_norm = np.mean(
                            [jnp.linalg.norm(x)
                             for x in jax.tree_util.tree_leaves(self.app.param)])
                        tf.summary.scalar('common/param-norm', param_norm.item(), step)

                        if (it + 1) % (len(self.trainset) // 10) == 0:
                            key, sub = jax.random.split(key)
                            # [1, T]
                            pred, _ = self.app(mel[0:1], timesteps, key=sub, use_tqdm=True)
                            # [T]
                            pred = np.asarray(pred).squeeze(0)
                            tf.summary.audio(
                                'train/audio', pred[None, :, None], self.config.data.sr, step)
                            tf.summary.image(
                                'train/mel', self.mel_img(pred)[None], step)
                            del pred
                    
                    del mel, speech, noise, time, loss, losses, grad_norm, param_norm

            self.app.write(
                '{}_{}.ckpt'.format(self.ckpt_path, epoch), self.optim_state)

            # test loss
            losses = {}
            for mel, speech in tqdm.tqdm(self.testset, leave=False):
                key, s1, s2 = jax.random.split(key, num=3)
                # [B, T]
                noise = jax.random.normal(s1, speech.shape)
                # [B], sample uniformly
                time = jnp.linspace(0., 1., speech.shape[0], endpoint=False)
                # add start point
                time = jnp.fmod(jax.random.uniform(s2) + time, 1.)
                # []
                _, lossdict = self.loss_fn(self.app.param, speech, noise, mel, time)
                # update dict
                for name, loss in lossdict.items():
                    if name not in losses:
                        losses[name] = []
                    losses[name].append(loss.item())
                # remove 
                del mel, speech, noise, time, loss, lossdict
            # test log
            with self.test_log.as_default():
                for name, loss in losses.items():
                    tf.summary.scalar(f'common/{name}', np.mean(loss), step)

                gt, pred, ir = self.eval(timesteps)
                tf.summary.audio(
                    'eval/gt', gt[None, :, None], self.config.data.sr, step)
                tf.summary.audio(
                    'eval/audio', pred[None, :, None], self.config.data.sr, step)

                tf.summary.image(
                    'eval/gt', self.mel_img(gt)[None], step)
                tf.summary.image(
                    'eval/mel', self.mel_img(pred)[None], step)

                interval = 1 if timesteps < IR_INTERVAL else timesteps // IR_INTERVAL
                for i in range(0, timesteps, interval):
                    tf.summary.image(f'eval/ir{i}', self.mel_img(ir[i])[None], step)

                del gt, pred, ir

    def eval(self, timesteps: int = 10) -> \
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Generate evaluation purpose audio.
        Args:
            timesteps: the number of the time steps. 
        Returns:
            speech: [float32; [T]], ground truth.
            pred: [float32; [T]], predicted.
            ir: List[np.ndarray], steps x [float32; [T]],
                intermediate represnetations.
        """
        MAX_MELLEN = 160
        # [B, T // H, M], [B, T], [B], [B]
        mel, speech, mellen, speechlen = next(self.testset.dataset.as_numpy_iterator())
        # [1, T], steps x [1, T]
        pred, ir = self.app(
            mel[0:1, :min(mellen[0], MAX_MELLEN)], timesteps,
            key=jax.random.PRNGKey(0), use_tqdm=True)
        # [T]
        pred = np.asarray(pred.squeeze(axis=0))
        # config.model.iter x [T]
        ir = [i.squeeze(axis=0) for i in ir]
        return speech[0, :speechlen[0]], pred, ir
    
    def mel_fn(self, signal: np.ndarray) -> np.ndarray:
        """Convert signal to the mel-spectrogram.
        Args:
            signal: [float32; [T]], input signal.
        Returns:
            [float32; [M, T // H]], mel-spectrogram.
        """
        # [fft // 2 + 1, T // H]
        stft = librosa.stft(
            signal,
            self.config.data.fft,
            self.config.data.hop,
            self.config.data.win,
            self.config.data.win_fn,
            center=True, pad_mode='reflect')
        # [M, T // H]
        mel = self.melfilter @ np.abs(stft)
        # [M, T // H]
        return np.log(np.maximum(mel, self.config.data.eps))

    def mel_img(self, signal: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [float32; [T]], speech signal.
        Returns:
            [float32; [M, T // H, 3]], mel-spectrogram in viridis color map.
        """
        # [M, T // H]
        mel = self.mel_fn(signal)
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min())
        # in range(0, 255)
        mel = (mel * 255).astype(np.long)
        # [M, T // H, 3]
        mel = self.cmap[mel]
        # make origin lower
        mel = np.flip(mel, axis=0)
        return mel


if __name__ == '__main__':
    # disable tensorflow gpu
    tf.config.set_visible_devices([], 'GPU')
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--timesteps', default=10, type=int)
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
        trainer.optim_state = diffwave.restore(ckpt_path, trainer.optim_state)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(key, args.load_epoch, args.timesteps)
