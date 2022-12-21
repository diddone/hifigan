import logging
from typing import List
import torch
import torchaudio
from melspecs import MelSpectrogramConfig

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spec_lengths = []
    audio_lengths = []
    for item in dataset_items:
        spec_lengths.append(item['spectrogram'].shape[-1])
        audio_lengths.append(item['audio'].shape[-1])

    batch_size = len(spec_lengths)
    max_specs_length = max(spec_lengths)
    max_audio_length = max(audio_lengths)

    spec_pad_value = MelSpectrogramConfig.pad_value
    spec_batch = torch.full([batch_size, item['spectrogram'].shape[1], max_specs_length], spec_pad_value)
    audio_batch = torch.zeros([batch_size, 1, max_audio_length])

    for i, item in enumerate(dataset_items):
        spec_batch[i, :, :spec_lengths[i]] = item['spectrogram']
        audio_batch[i, :, :audio_lengths[i]] = item['audio']

    return {
        'spectrogram': spec_batch,
        'audio': audio_batch,
        'spectrogram_length': torch.tensor(spec_lengths).long(),
    }