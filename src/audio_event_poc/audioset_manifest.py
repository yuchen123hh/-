from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


AUDIOSET_TO_G1_LABELS: dict[str, list[str]] = {
    "distress_call": [
        "Screaming",
        "Shout",
        "Yell",
        "Cry",
    ],
    "glass_break": [
        "Glass",
        "Breaking",
        "Shatter",
        "Glass shatter",
    ],
    "knock": [
        "Knock",
        "Door",
        "Doorbell",
        "Tap",
    ],
    "cough": [
        "Cough",
        "Throat clearing",
    ],
    "smoke_alarm": [
        "Smoke detector, smoke alarm",
        "Fire alarm",
        "Alarm",
        "Siren",
        "Beep, bleep",
    ],
    "background": [
        "Domestic sounds, home sounds",
        "Inside, small room",
        "Television",
        "Conversation",
        "Tools",
        "Vacuum cleaner",
        "Air conditioning",
        "Mechanical fan",
        "Typing",
        "Footsteps",
    ],
}

TARGET_PRIORITY = ["distress_call", "glass_break", "knock", "cough", "smoke_alarm", "background"]


@dataclass(frozen=True)
class AudioSetClip:
    clip_id: str
    audio_path: str
    labels: list[str]
    source_type: str = "audioset"


def map_audioset_labels(labels: Iterable[str]) -> str | None:
    normalized = {_normalize(label) for label in labels}
    for target in TARGET_PRIORITY:
        for audioset_label in AUDIOSET_TO_G1_LABELS[target]:
            if _normalize(audioset_label) in normalized:
                return target
    return None


def build_balanced_manifest(
    clips: Iterable[AudioSetClip],
    *,
    per_label_limit: Mapping[str, int] | None = None,
) -> list[dict[str, str]]:
    limits = dict(per_label_limit or {})
    counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    for clip in clips:
        label = map_audioset_labels(clip.labels)
        if label is None:
            continue
        limit = limits.get(label)
        if limit is not None and counts.get(label, 0) >= limit:
            continue
        rows.append(
            {
                "clip_id": clip.clip_id,
                "audio_path": clip.audio_path,
                "label": label,
                "audioset_labels": "|".join(clip.labels),
                "source_type": clip.source_type,
            }
        )
        counts[label] = counts.get(label, 0) + 1
    return rows


def write_manifest(rows: Iterable[Mapping[str, str]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "audio_path", "label", "audioset_labels", "source_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def read_manifest_clips(path: str | Path) -> list[AudioSetClip]:
    rows: list[AudioSetClip] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels = [item for item in (row.get("audioset_labels") or "").split("|") if item]
            rows.append(
                AudioSetClip(
                    clip_id=row.get("clip_id") or row.get("YTID") or "",
                    audio_path=row.get("audio_path") or row.get("path") or "",
                    labels=labels,
                    source_type=row.get("source_type") or "audioset",
                )
            )
    return rows


def _normalize(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").split())
