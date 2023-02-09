import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any

import numpy as np
import soundfile
import pvkoala


@dataclass
class EngineResult:
    denoised_pcm: np.ndarray
    runtime: float


class Engines(Enum):
    MOZILLA_RNNOISE = 'mozilla_rnnoise'
    PICOVOICE_KOALA = 'picovoice_koala'


class Engine(object):
    def process(self, path: str) -> EngineResult:
        raise NotImplementedError()

    def cleanup(self) -> None:
        pass

    def __str__(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def create(engine: Engines, **kwargs: Any) -> 'Engine':
        if engine is Engines.MOZILLA_RNNOISE:
            return RNNoiseEngine(**kwargs)
        elif engine is Engines.PICOVOICE_KOALA:
            return KoalaEngine(**kwargs)
        else:
            raise ValueError(f"No implementation available for engine `{engine}`")


class KoalaEngine(Engine):
    def __init__(self, access_key: str) -> None:
        self._koala = pvkoala.create(access_key)

    def process(self, path: str) -> EngineResult:
        pcm, sample_rate = soundfile.read(path, dtype=np.int16)
        if sample_rate != self._koala.sample_rate:
            raise ValueError(f"{self} requires sample rate {self._koala.sample_rate}, got {sample_rate}")

        frame_length = self._koala.frame_length
        delay_sample = self._koala.delay_sample
        length_sample = pcm.size
        num_frames = 1 + (length_sample + delay_sample - 1) // frame_length
        pcm_frames = np.pad(pcm, (0, num_frames * frame_length - length_sample)).reshape((num_frames, frame_length))

        start_time = perf_counter()
        enhanced_frames = [self._koala.process(frame) for frame in pcm_frames]
        runtime = perf_counter() - start_time

        enhanced_pcm = np.concatenate(enhanced_frames, dtype=np.int16)
        enhanced_pcm = enhanced_pcm[delay_sample:delay_sample + length_sample]
        return EngineResult(enhanced_pcm, runtime)

    def cleanup(self) -> None:
        self._koala.delete()

    def __str__(self) -> str:
        return Engines.PICOVOICE_KOALA.value


class RNNoiseEngine(Engine):
    REQUIRED_SAMPLE_RATE = 48000
    PADDING_LENGTH_SAMPLE = 2 * 480 - 1
    FFMPEG_BASE = ["ffmpeg", "-y", "-loglevel", "error", "-hide_banner", "-channel_layout", "mono"]

    def __init__(self, demo_path: str) -> None:
        self._demo_path = demo_path

    def process(self, path: str) -> EngineResult:
        input_info = soundfile.info(path)
        length_sample = input_info.frames
        sample_rate = input_info.samplerate

        tmpdir = tempfile.gettempdir()
        resampled_input_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.pcm")
        original_output_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.pcm")
        resampled_output_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.wav")

        try:
            subprocess.check_call([
                *self.FFMPEG_BASE,
                "-i", path,
                "-f", "s16le",
                "-ar", str(self.REQUIRED_SAMPLE_RATE),
                "-acodec", "pcm_s16le",
                "-af", f"apad=pad_len={self.PADDING_LENGTH_SAMPLE}",
                resampled_input_path])

            start_time = perf_counter()
            subprocess.check_call([self._demo_path, resampled_input_path, original_output_path])
            runtime = perf_counter() - start_time

            subprocess.check_call([
                *self.FFMPEG_BASE,
                "-f", "s16le",
                "-ar", str(self.REQUIRED_SAMPLE_RATE),
                "-i", original_output_path,
                "-ar", str(sample_rate),
                resampled_output_path])

            enhanced_pcm, _ = soundfile.read(resampled_output_path)
            enhanced_pcm = enhanced_pcm[:length_sample]
            return EngineResult(enhanced_pcm, runtime)

        finally:
            if os.path.exists(resampled_input_path):
                os.remove(resampled_input_path)
            if os.path.exists(original_output_path):
                os.remove(original_output_path)
            if os.path.exists(resampled_output_path):
                os.remove(resampled_output_path)

    def __str__(self) -> str:
        return Engines.MOZILLA_RNNOISE.value


__all__ = [
    'Engine',
    'EngineResult',
    'Engines',
]
