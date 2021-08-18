# jax-variational-diffwave

Jax/Flax implementation of Variational-DiffWave. (Zhifeng Kong et al., 2020, Diederik P. Kingma et al., 2021.)

- DiffWave with Continuous-time Variational Diffusion Models.
- DiffWave: A Versatile Diffusion Model for Audio Synthesis, Zhifeng Kong et al., 2020. [[arXiv:2009.09761]](https://arxiv.org/abs/2009.09761)
- Variational Diffusion Models, Diederik P. Kingma et al., 2021. [[arXiv:2107.00630]](https://arxiv.org/abs/2107.00630)


## Requirements

Tested in python 3.7.9 conda environment, [requirements.txt](./requirements.txt)

## Usage

To train model, run [train.py](./train.py). \
Checkpoint will be written on `TrainConfig.ckpt`, tensorboard summary on `TrainConfig.log`.

```sh
python train.py --data-dir /datasets/ljspeech --from-raw
tensorboard --logdir ./log/
```

To start to train from previous checkpoint, `--load-step` is available.

```sh
python train.py --load-epoch 10 --config ./ckpt/l1.json
```

[WIP] To synthesize test set, run `synth.py`.

```sh
python synth.py
```

[WIP] Pretrained checkpoints are relased on releases.

To use pretrained model, download files and unzip it. \
Checkout git repository to proper commit tags and following is sample script.

```python
with open('l1.json') as f:
    config = Config.load(json.load(f))

diffwave = VLBDiffWaveApp(config.model)
diffwave.restore('./l1/l1_99.ckpt')

# mel: [B, T, mel]
audio, _ = diffwave(mel, timesteps=50, key=jax.random.PRNGKey(0))
```
