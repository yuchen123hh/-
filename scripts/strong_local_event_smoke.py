from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from scripts.audio_test_page import analyze_environment_audio, run_local_strong_model


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "strong_local_event_smoke.json"
OUT_DIR = ROOT / "reports" / "strong_local_event_smoke_audio"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def impulse(sample_rate: int, offsets: list[float], amp: float = 0.9) -> np.ndarray:
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    width = int(sample_rate * 0.008)
    window = np.hanning(width).astype(np.float32)
    for offset in offsets:
        start = int(offset * sample_rate)
        audio[start : start + width] += window * amp
    return audio


def alarm(sample_rate: int) -> np.ndarray:
    t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    sweep = np.sin(2 * math.pi * (650 + 250 * np.sin(2 * math.pi * 3 * t)) * t)
    gate = (np.sin(2 * math.pi * 4 * t) > 0).astype(np.float32)
    return (sweep * gate * 0.45).astype(np.float32)


def cough(sample_rate: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    burst = rng.normal(0, 0.22, int(sample_rate * 0.28)).astype(np.float32)
    env = np.hanning(burst.size).astype(np.float32)
    audio[int(0.45 * sample_rate) : int(0.45 * sample_rate) + burst.size] = burst * env
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default="models/faster-whisper-small-ct2")
    args = parser.parse_args()

    sample_rate = 16000
    fixtures = {
        "knock": impulse(sample_rate, [0.45, 0.72], 0.85),
        "clap": impulse(sample_rate, [0.55], 0.98),
        "alarm": alarm(sample_rate),
        "cough": cough(sample_rate),
    }
    rows = []
    for name, audio in fixtures.items():
        wav_path = OUT_DIR / f"{name}.wav"
        write_wav(wav_path, audio, sample_rate)
        signal = analyze_environment_audio(wav_path)
        local = run_local_strong_model(wav_path, signal, args.local_model)
        rows.append({"fixture": name, "wav_file": str(wav_path), "signal": signal, "local": local})

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "note_zh": "合成事件 smoke 只证明链路和标签映射，不等于真实环境精度。",
        "fixtures": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(REPORT_PATH))


if __name__ == "__main__":
    main()
