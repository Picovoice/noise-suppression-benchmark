
import numpy as np
import pystoi


def eval_stoi(clean_pcm: np.ndarray, denoised_pcm: np.ndarray, sample_rate: float) -> float:
    return pystoi.stoi(
        x=clean_pcm,
        y=denoised_pcm,
        fs_sig=sample_rate,
        extended=False)


__all__ = [
    'eval_stoi',
]
