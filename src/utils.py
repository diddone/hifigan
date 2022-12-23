import os
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from datetime import datetime
import numpy as np
import wandb
import torchaudio

from melspecs import MelSpectrogram, MelSpectrogramConfig
import json



def load_config(json_path):
    with open(json_path) as f:
        json_config = f.read()
        config = json.loads(json_config)

    return config


def resume_models_from_ckpt(ckpt_path, gen, mpd, msd, device='cpu'):

    load_dict = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(load_dict['gen_state'])
    mpd.load_state_dict(load_dict['mpd_state'])
    msd.load_state_dict(load_dict['msd_state'])

    gen.to(device)
    mpd.to(device)
    msd.to(device)

def resume_optims_from_ckpt(ckpt_path, optim_g, optim_d, device='cpu'):

    load_dict = torch.load(ckpt_path, map_location=device)
    optim_g.load_state_dict(load_dict['gen_opt_state'])
    optim_d.load_state_dict(load_dict['disc_opt_state'])

def set_lr_to_optim(lr, optim):
    for g in optim.param_groups:
        g['lr'] = lr

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


def get_mel_spec():
    return MelSpectrogram(MelSpectrogramConfig())



def load_mels_batch(path, mel_spec, device):

    wavs_list = []
    max_len = 0
    for file_name in filter(lambda x: '.wav' in x, sorted(os.listdir(path))):
        file_path = os.path.join(path, file_name)
        wav, _ = torchaudio.load(file_path)
        wavs_list.append(wav)
        max_len = max(max_len, wav.shape[-1])

    wav_batch = torch.zeros([len(wavs_list), 1, max_len]).to(device)

    for i, wav in enumerate(wavs_list):
        wav_batch[i, :, :wav.shape[-1]] = wav

    return mel_spec(wav_batch)


def save_wav_batch(save_path, gen_wavs, sampling_rate):
    for i in range(gen_wavs.shape[0]):
        cur_path = str(os.path.join(save_path, f'val_sample_{i}.wav'))
        torchaudio.save(cur_path, gen_wavs[i], sampling_rate)

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