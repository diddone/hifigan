import logging
from typing import List
import torch
import torchaudio

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    spec_lengths = []

    text = []
    for item in dataset_items:
        spec_lengths.append(item['spectrogram'].shape[-1])
        text.append(item['text'])

    batch_size = len(spec_lengths)
    max_specs_length = max(spec_lengths)

    spec_batch = torch.zeros(batch_size, item['spectrogram'].shape[1], max_specs_length)
    for i, item in enumerate(dataset_items):
        spec_batch[i, :, :spec_lengths[i]] = item['spectrogram']

    return {
        'spectrogram': spec_batch,
        'spectrogram_length': torch.tensor(spec_lengths).long(),
        'text': text,
    }