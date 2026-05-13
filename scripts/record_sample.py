from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.recording import record_to_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a short wav sample for the audio event PoC.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output = record_to_file(
        args.output,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        channels=args.channels,
        device=args.device,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
