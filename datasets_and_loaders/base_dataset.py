import os
import logging
import random
from typing import List, Union

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

import re
from melspecs import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


class BaseTextEncoder:
    def encode(self, text) -> Tensor:
        raise NotImplementedError()

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item: int) -> str:
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            sampling_rate,
            segment_size,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
    ):

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.text_encoder = BaseTextEncoder()
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]

        audio_wave = self.load_audio(audio_path)
        if self.segment_size <= audio_wave.shape[1]:
            audio_start = random.randint(0,  audio_wave.shape[1] - self.segment_size)
            audio_wave = audio_wave[:, audio_start:audio_start+self.segment_size]
        else:
            audio_wave = F.pad(audio_wave, (0, self.segment_size - audio_wave.shape[1]))

        audio_spec = self.mel_spec(audio_wave)
        
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.sampling_rate,
            "audio_path": audio_path,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.sampling_rate
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                    np.array(
                        [len(BaseTextEncoder.normalize_text(el["text"])) for el in index]
                    )
                    >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )