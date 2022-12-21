import os
import torch
import torch.nn.functional as F
from models import FeatureLoss, MelLoss, GeneratorAdvLoss, DiscriminatorAdvLoss
from utils import WanDBWriter, set_requires_grad

def train(
    training_loader, gen, mpd, msd,
    optim_g, optim_d, sched_g, sched_d,
    params, mel_spec, device):

    wandb_writer = WanDBWriter(params)
    gen.train()
    mpd.train()
    msd.train()

    gen = gen.to(device)
    mpd = mpd.to(device)
    msd = msd.to(device)
    mel_spec.to(device)

    n_epochs = params['n_epochs']

    mel_loss_coef = params['mel_loss_coef']
    feat_loss_coef = params['feat_loss_coef']
    discr_criterion = DiscriminatorAdvLoss(device)
    gen_criterion = GeneratorAdvLoss(device)
    feat_criterion = FeatureLoss()
    mel_criterion = MelLoss()

    step = 0
    for epoch in range(n_epochs):
        for i, batch in enumerate(training_loader):

            real_mels = batch['spectrogram'].to(device, non_blocking=True)
            real_wavs = batch['audio'].to(device, non_blocking=True)

            #mpd and msd losses
            optim_d.zero_grad()
            gen_wavs = gen(real_mels)
            gen_mels = mel_spec(gen_wavs)
            real_wavs = F.pad(real_wavs, (0, gen_wavs.shape[-1] - real_wavs.shape[-1]))

            #mpd
            ys_real_p, _ = mpd(real_wavs)
            ys_gen_p, _  = mpd(gen_wavs.detach())
            disc_p_loss = discr_criterion(ys_real_p, ys_gen_p)
            #msd
            ys_real_s, _ = msd(real_wavs)
            ys_gen_s, _  = msd(gen_wavs.detach())
            disc_s_loss = discr_criterion(ys_real_s, ys_gen_s)

            disc_loss = disc_p_loss + disc_s_loss
            disc_loss.backward()
            optim_d.step()

            #gen losses
            optim_g.zero_grad()

            #mel loss
            pad_value = mel_spec.pad_value
            pad_size = gen_mels.shape[-1] - real_mels.shape[-1]
            mel_loss = mel_criterion(gen_mels, F.pad(real_mels, pad=(0,pad_size), value=pad_value))

            set_requires_grad([mpd, msd], False)
            print('Gen and real shapes', gen_wavs.shape, real_wavs.shape)
            ys_gen_p, fs_gen_p = mpd(gen_wavs)
            ys_gen_s, fs_gen_s = msd(gen_wavs)
            ys_real_p, fs_real_p = mpd(real_wavs)
            ys_real_s, fs_real_s = msd(real_wavs)

            for l1, l2 in zip(fs_gen_p, fs_real_p):
                for x, y in zip(l1, l2):
                    print(x.shape, y.shape)
                    assert x.shape == y.shape
            print('done')
            feat_p_loss = feat_criterion(fs_gen_p, fs_real_p)
            feat_s_loss = feat_criterion(fs_gen_s, fs_real_s)
            gen_p_loss = gen_criterion(ys_gen_p)
            gen_s_loss = gen_criterion(ys_gen_s)

            gen_loss = mel_loss_coef * mel_loss + feat_loss_coef * (feat_p_loss + feat_s_loss) + gen_p_loss + gen_s_loss
            gen_loss.backward()
            optim_g.step()

            set_requires_grad([mpd, msd], True)
            sched_d.step()
            sched_g.step()

            wandb_writer.add_scalar('gen_loss', gen_loss.item())
            wandb_writer.add_scalar('disc_loss', disc_loss.item())

            step += 1
            wandb_writer.set_step(step)

        if i % 5 == 4:
            wandb_writer.add_audio('gen_audio', gen_wavs)

            if not os.path.isdir(config['save_dir']):
                os.makedirs(config['save_dir'], exist_ok=True)

            torch.save({
                'gen_state': gen.state_dict(),
                'gen_opt_state': optim_g.state_dict(),
                'mpd_state': mpd.state_dict(),
                'msd_state': msd.state_dict(),
                'disc_opt_state': optim_d.state_dict(),
                'params': params
            }, params['save_path'])




