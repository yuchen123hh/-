from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.clap_backend import ClapAudioClassifier
from audio_event_poc.config import load_event_config
from audio_event_poc.decision import aggregate_prompt_scores, build_result, select_detected_events


def classify_audio(audio_path: Path, config_path: Path, *, max_events: int = 1) -> dict[str, object]:
    events = load_event_config(config_path)
    classifier = ClapAudioClassifier()
    prompt_scores = classifier.score_file(audio_path, events)
    event_scores = aggregate_prompt_scores(prompt_scores, events)
    timestamp = time.time()
    detections = select_detected_events(
        event_scores,
        events,
        timestamp=timestamp,
        max_events=max_events,
    )
    return build_result(
        timestamp=timestamp,
        audio_path=str(audio_path),
        event_scores=event_scores,
        detected_events=detections,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one wav file with CLAP event prompts.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to a wav file.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "events.yaml")
    parser.add_argument("--max-events", type=int, default=1)
    args = parser.parse_args()

    result = classify_audio(args.audio, args.config, max_events=args.max_events)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
