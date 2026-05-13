from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (SRC, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from audio_event_poc.recording import record_to_file
from classify_audio import classify_audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one short sample and classify it.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "events.yaml")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-audio", type=Path, default=None)
    args = parser.parse_args()

    if args.keep_audio:
        audio_path = args.keep_audio
    else:
        tempdir = Path(tempfile.mkdtemp(prefix="audio_event_poc_"))
        audio_path = tempdir / "sample.wav"

    record_to_file(
        audio_path,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        channels=args.channels,
        device=args.device,
    )
    result = classify_audio(audio_path, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
