from __future__ import annotations

from pathlib import Path

import numpy as np


CLAP_SAMPLE_RATE = 48_000


def load_audio_mono(path: str | Path, target_sample_rate: int = CLAP_SAMPLE_RATE) -> np.ndarray:
    try:
        import librosa
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "audio loading requires soundfile and librosa. Install project requirements first."
        ) from exc

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    return np.asarray(audio, dtype=np.float32)


def rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return round(float(np.sqrt(np.mean(np.square(audio)))), 6)
