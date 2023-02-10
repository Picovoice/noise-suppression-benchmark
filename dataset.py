import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class DataPair:
    clean_path: str
    noisy_path: str
    name: str


class Datasets(Enum):
    INTERSPEECH_2020 = 'Interspeech2020'


class Dataset(object):
    def size(self) -> int:
        raise NotImplementedError()

    def get(self, index: int) -> DataPair:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def load(ds: Datasets, folder: str) -> 'Dataset':
        if ds is Datasets.INTERSPEECH_2020:
            return Interspeech2020Dataset(folder)
        else:
            raise ValueError(f"No implementation available for dataset `{ds}`")


class Interspeech2020Dataset(Dataset):
    NOISY_FILENAME_PATTERN = re.compile(r'clnsp\d+_(.+)_snr\d+_tl-\d+_fileid_(\d+)\.wav')
    CLEAN_FILENAME_FORMAT = 'clean_fileid_{}.wav'

    def __init__(self, folder: str) -> None:
        self._folder = folder

        noisy_dir = os.path.join(folder, 'noisy')
        if not os.path.exists(noisy_dir):
            raise FileNotFoundError(f'Subdirectory `{noisy_dir}` does not exist')
        clean_dir = os.path.join(folder, 'clean')
        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f'Subdirectory `{clean_dir}` does not exist')

        self._data: List[DataPair] = []
        for noisy_filename in os.listdir(noisy_dir):
            match = self.NOISY_FILENAME_PATTERN.fullmatch(noisy_filename)
            if not match:
                continue

            noise_name = match.group(1)
            fileid = match.group(2)
            name = f'{fileid}_{noise_name}'
            noisy_path = os.path.join(noisy_dir, noisy_filename)
            clean_path = os.path.join(clean_dir, self.CLEAN_FILENAME_FORMAT.format(fileid))
            self._data.append(DataPair(clean_path, noisy_path, name))

    def size(self) -> int:
        return len(self._data)

    def get(self, index: int) -> DataPair:
        return self._data[index]

    def __str__(self) -> str:
        return Datasets.INTERSPEECH_2020.value


__all__ = [
    'DataPair',
    'Dataset',
    'Datasets',
]
