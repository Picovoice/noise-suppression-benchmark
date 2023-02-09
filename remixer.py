import os

import numpy as np
import soundfile

from dataset import DataPair, Dataset


class RemixedDataset(Dataset):
    ENERGY_FRAME_LENGTH = 512

    def __init__(self, raw_dataset: Dataset, snr_db: float, cache_folder: str) -> None:
        self._raw_dataset = raw_dataset
        self._snr_db = snr_db
        self._cache_folder = cache_folder

    def size(self) -> int:
        return self._raw_dataset.size()

    def get(self, index: int) -> DataPair:
        data = self._raw_dataset.get(index)

        noisy_name = os.path.basename(data.noisy_path)
        remix_folder = os.path.join(self._cache_folder, str(self))
        os.makedirs(remix_folder, exist_ok=True)
        remix_path = os.path.join(remix_folder, noisy_name)

        if not os.path.exists(remix_path):
            clean_pcm, clean_sr = soundfile.read(data.clean_path)
            clean_energy = self._energy(clean_pcm)

            pure_noise_folder = os.path.join(self._cache_folder, f'{str(self._raw_dataset)}_pure_noise')
            os.makedirs(pure_noise_folder, exist_ok=True)
            pure_noise_path = os.path.join(pure_noise_folder, noisy_name)

            if os.path.exists(pure_noise_path):
                pure_noise_pcm, pure_noise_sr = soundfile
                if pure_noise_sr != clean_sr or clean_pcm.size != pure_noise_pcm.size:
                    raise ValueError(
                        f"Cannot mix `{data.clean_path}` with `{pure_noise_path}`: samplerate or length mismatch")
            else:
                noisy_pcm, noisy_sr = soundfile.read(data.noisy_path)
                if noisy_sr != clean_sr or clean_pcm.size != noisy_pcm.size:
                    raise ValueError(
                        f"Cannot subtract `{data.clean_path}` from `{data.noisy_path}`: samplerate or length mismatch")

                pure_noise_pcm = noisy_pcm - clean_pcm
                soundfile.write(pure_noise_path, data=pure_noise_pcm, samplerate=clean_sr)

            noise_energy = self._energy(pure_noise_pcm)
            noise_scale = np.sqrt(clean_energy / (noise_energy * (10 ** (0.1 * self._snr_db))))
            mixed_pcm = clean_pcm + noise_scale * pure_noise_pcm
            soundfile.write(remix_path, data=mixed_pcm, samplerate=clean_sr)

        return DataPair(data.clean_path, remix_path, f'{data.name}_remixed{self._snr_db:g}db')

    def __str__(self) -> str:
        return f'{str(self._raw_dataset)}_remixed{self._snr_db:g}db'

    @classmethod
    def _energy(cls, pcm: np.ndarray) -> float:
        num_frames = pcm.size // cls.ENERGY_FRAME_LENGTH
        frames = pcm[:num_frames * cls.ENERGY_FRAME_LENGTH].reshape((num_frames, cls.ENERGY_FRAME_LENGTH))
        return (frames ** 2).sum(axis=1).max()


__all__ = [
    'RemixedDataset',
]
