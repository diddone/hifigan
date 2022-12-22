import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from datetime import datetime
import numpy as np
import wandb
import torchaudio


def set_requires_grad(models, is_req_grad):
    for model in models:
        for param in model.parameters():
            param.requires_grad = is_req_grad

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_activation(params):
    activation_name = params['activation']['name'].lower()
    if activation_name == 'lrelu':
        return nn.LeakyReLU(negative_slope=params['activation']['slope'])
    elif activation_name == 'elu':
        return nn.ELU(alpha=params['activation']['slope'])
    else:
        raise NotImplementedError('Not implemented type of activation function')


def load_mels_val_batch(mel_spec, device):
    paths = [
        'data/datasets/ljspeech/train/LJ001-0004.wav',
        'data/datasets/ljspeech/train/LJ001-0006.wav'
    ]

    wavs_list = []
    max_len = 0
    for path in paths:
        wav, _ = torchaudio.load(path)
        wavs_list.append(wav)
        max_len = max(max_len, wav.shape[-1])

    wav_batch = torch.zeros([len(wavs_list), 1, max_len]).to(device)

    for i, wav in enumerate(wavs_list):
        wav_batch[i, :, :wav.shape[-1]] = wav

    return mel_spec(wav_batch)


class WanDBWriter:
    def __init__(self, params):
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if 'wandb_project' not in params:
            raise ValueError("please specify project name for wandb")

        wandb.init(
            project=params['wandb_project'],
            entity=params['wandb_entity'],
            config=params,
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{self.mode}/{scalar_name}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        if isinstance(audio, (torch.Tensor, np.ndarray)):
            audio = audio.T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()