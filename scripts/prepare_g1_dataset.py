from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.audioset_manifest import AudioSetClip, build_balanced_manifest


TARGET_EVENTS = ["distress_call", "glass_break", "knock", "cough", "smoke_alarm"]
BACKGROUND_SCENARIOS = [
    "robot_idle_motor",
    "robot_walking_motor",
    "tv_speech",
    "family_conversation",
    "kitchen_utensils",
    "fan_or_air_conditioner",
    "footsteps",
    "door_open_close",
    "water_tap",
    "vacuum_cleaner",
]


def read_metadata(path: Path, audio_root: Path) -> list[AudioSetClip]:
    clips: list[AudioSetClip] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_labels = row.get("audioset_labels") or row.get("labels") or row.get("display_name") or ""
            labels = [item.strip() for item in raw_labels.split("|") if item.strip()] if "|" in raw_labels else [raw_labels.strip()]
            audio_path = row.get("audio_path") or row.get("path") or row.get("file") or ""
            if audio_path:
                candidate = Path(audio_path)
                if not candidate.is_absolute():
                    candidate = audio_root / candidate
                audio_path = str(candidate)
            clips.append(
                AudioSetClip(
                    clip_id=row.get("clip_id") or row.get("YTID") or row.get("id") or Path(audio_path).stem,
                    audio_path=audio_path,
                    labels=labels,
                )
            )
    return clips


def split_rows(rows: list[dict[str, str]], *, val_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for label in sorted(grouped):
        items = grouped[label]
        rng.shuffle(items)
        val_count = int(round(len(items) * val_ratio))
        if len(items) > 1:
            val_count = max(1, min(len(items) - 1, val_count))
        else:
            val_count = 0
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])
    return train_rows, val_rows


def write_rows(path: Path, rows: Iterable[dict[str, str]], *, fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def build_manifest(args: argparse.Namespace) -> int:
    per_label_limit = _parse_limits(args.limit)
    clips = read_metadata(args.metadata_csv, args.audio_root)
    rows = build_balanced_manifest(clips, per_label_limit=per_label_limit)
    if not rows:
        raise RuntimeError("no rows matched the G1 abnormal audio label mapping")
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    write_rows(args.output_dir / "train_manifest.csv", _training_rows(train_rows), fields=["audio_path", "label"])
    write_rows(args.output_dir / "val_manifest.csv", _training_rows(val_rows), fields=["audio_path", "label"])
    write_rows(args.output_dir / "selected_manifest.csv", rows, fields=["clip_id", "audio_path", "label", "audioset_labels"])
    print(f"train={len(train_rows)} val={len(val_rows)} selected={len(rows)}")
    return 0


def write_g1_plan(args: argparse.Namespace) -> int:
    rows: list[dict[str, str]] = []
    for label in TARGET_EVENTS:
        for index in range(1, args.per_event + 1):
            rows.append(
                {
                    "label": label,
                    "sample_id": f"{label}_{index:03d}",
                    "source": "unitree_g1_mic",
                    "scenario": label,
                    "duration_s": str(args.duration_s),
                    "notes": "record on G1 in the target room",
                }
            )
    for index in range(1, args.background + 1):
        scenario = BACKGROUND_SCENARIOS[(index - 1) % len(BACKGROUND_SCENARIOS)]
        rows.append(
            {
                "label": "background",
                "sample_id": f"background_{index:03d}",
                "source": "unitree_g1_mic",
                "scenario": scenario,
                "duration_s": str(args.duration_s),
                "notes": "hard negative for false alarm suppression",
            }
        )
    write_rows(
        args.output,
        rows,
        fields=["label", "sample_id", "source", "scenario", "duration_s", "notes"],
    )
    print(str(args.output))
    return 0


def _training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"audio_path": row["audio_path"], "label": row["label"]} for row in rows]


def _parse_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"limit must use label=count format: {value}")
        label, raw_count = value.split("=", 1)
        limits[label.strip()] = int(raw_count)
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare G1 abnormal audio datasets before paid GPU training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest", help="Build train/val manifests from AudioSet-style metadata.")
    manifest.add_argument("--metadata-csv", type=Path, required=True)
    manifest.add_argument("--audio-root", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--val-ratio", type=float, default=0.15)
    manifest.add_argument("--seed", type=int, default=13)
    manifest.add_argument("--limit", action="append", default=[], help="Optional label=count cap, e.g. cough=1200")
    manifest.set_defaults(func=build_manifest)

    plan = subparsers.add_parser("g1-plan", help="Write a G1 field recording checklist.")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--per-event", type=int, default=50)
    plan.add_argument("--background", type=int, default=150)
    plan.add_argument("--duration-s", type=float, default=3.0)
    plan.set_defaults(func=write_g1_plan)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.func(parsed))
