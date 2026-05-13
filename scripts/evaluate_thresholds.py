from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.clap_backend import ClapAudioClassifier
from audio_event_poc.config import load_event_config
from audio_event_poc.decision import aggregate_prompt_scores, build_result, select_detected_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate configured thresholds on a labeled wav manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with columns: audio_path,label")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "events.yaml")
    parser.add_argument("--details-jsonl", type=Path, default=None)
    args = parser.parse_args()

    events = load_event_config(args.config)
    classifier = ClapAudioClassifier()
    labels = [event.key for event in events if not event.suppress]
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    details_file = args.details_jsonl.open("w", encoding="utf-8") if args.details_jsonl else None
    try:
        with args.manifest.open("r", encoding="utf-8", newline="") as manifest_file:
            for row in csv.DictReader(manifest_file):
                audio_path = Path(row["audio_path"])
                expected = row["label"]
                prompt_scores = classifier.score_file(audio_path, events)
                event_scores = aggregate_prompt_scores(prompt_scores, events)
                timestamp = time.time()
                detections = select_detected_events(event_scores, events, timestamp=timestamp)
                predicted = detections[0]["event_key"] if detections else "none"
                confusion[expected][str(predicted)] += 1

                for label in labels:
                    if expected == label and predicted == label:
                        counts[label]["tp"] += 1
                    elif expected != label and predicted == label:
                        counts[label]["fp"] += 1
                    elif expected == label and predicted != label:
                        counts[label]["fn"] += 1

                if details_file:
                    result = build_result(
                        timestamp=timestamp,
                        audio_path=str(audio_path),
                        event_scores=event_scores,
                        detected_events=detections,
                    )
                    result["expected_label"] = expected
                    details_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if details_file:
            details_file.close()

    summary = {
        "per_label": {label: _metrics(raw_counts) for label, raw_counts in counts.items()},
        "confusion": {actual: dict(predicted) for actual, predicted in confusion.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


if __name__ == "__main__":
    raise SystemExit(main())
