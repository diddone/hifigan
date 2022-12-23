import os
import json
import torch
from torch import nn

from src.models import Generator, MPDiscriminator, MSDiscriminator
import src.utils

import argparse
from pathlib import Path

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    load_dict = torch.load(pargs.ckpt_path, map_location='cpu')

    config = load_dict['params'] if pargs.config is None else load_json(pargs.config)
    gen = Generator(config)
    gen.load_state_dict(load_dict['gen_state'])
    mel_spec = utils.get_mel_spec()

    gen.to(device)
    mel_spec.to(device)

    mel_spec_batch = utils.load_mels_batch(pargs.test_data_dir, mel_spec, device)

    with torch.no_grad():
        gen_wavs = gen(mel_spec_batch).detach().cpu()

    utils.save_wav_batch(pargs.output_dir, gen_wavs, config['sampling_rate'])
    print(f'Audios have been saved to "{pargs.output_dir}"')

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="hifigan testing")

    args.add_argument(
        "--ckpt-path",
        type=Path,
        required=True,
        help="path to ckpt",
    )
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=Path,
        help="path to json file, if not given it will be taken from ckpt",
    )
    args.add_argument(
        "-t",
        "--test-data-dir",
        default="data/test_audio/",
        type=Path,
        help="path to test data directory with test samples",
    )
    args.add_argument(
        "-o",
        "--output-dir",
        default="results/",
        type=Path,
        help="path to output dir",
    )

    pargs = args.parse_args()
    main(pargs)

