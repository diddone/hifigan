import os
import json
from torch import nn

from datasets_and_loaders import get_training_loader
from datasets_and_loaders import LJspeechDataset
from models import Generator, MPDiscriminator, MSDiscriminator
from utils import set_random_seed, load_config, resume_models_from_ckpt
from train import train
from melspecs import MelSpectrogram, MelSpectrogramConfig

from datetime import datetime
import itertools


def main(config):

    set_random_seed(config['seed'])
    config['save_dir'] = f'save/{datetime.now().strftime(r"%m%d_%H%M%S")}/'
    config['save_path'] = str(os.path.join(config['save_dir'], 'ckpt.tar'))


    ds = LJspeechDataset('train', config['segment_size'])
    training_loader = get_training_loader(ds, config)

    gen = Generator(config)
    mpd = MPDiscriminator(config)
    msd = MSDiscriminator(config)
    if 'ckpt_path' in params.keys():
        resume_models_from_ckpt(params['ckpt_path'], gen, mpd, msd)


    optim_g = torch.optim.AdamW(gen.parameters(), config['lr'], betas=[config['adam_b1'], config['adam_b2']])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                config['lr'], betas=[config['adam_b1'], config['adam_b2']])

    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config['lr_decay'], last_epoch=-1)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config['lr_decay'], last_epoch=-1)

    mel_spec = MelSpectrogram(MelSpectrogramConfig())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    train(training_loader, gen, mpd, msd, optim_g, optim_d, sched_g, sched_d, config, mel_spec, device)


if __name__ == "__main__":

    config = load_config('config.json')
    main(config)

