from __future__ import annotations

from pathlib import Path


def record_to_file(
    output_path: str | Path,
    *,
    duration_seconds: float = 2.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    device: int | str | None = None,
) -> Path:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "recording requires sounddevice and soundfile. On Linux you can also use arecord manually."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration_seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32", device=device)
    sd.wait()
    sf.write(str(output), audio, sample_rate)
    return output
