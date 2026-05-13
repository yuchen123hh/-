from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.audioset_real_data import (
    AUDIOSET_METADATA_URLS,
    build_candidates,
    download_candidates,
    read_candidate_csv,
    write_candidate_csv,
)


def download_metadata(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}
    for name, url in AUDIOSET_METADATA_URLS.items():
        if name == "unbalanced_train" and args.skip_unbalanced:
            continue
        output_path = args.output_dir / Path(url).name
        if output_path.exists() and not args.force:
            downloaded[name] = str(output_path)
            continue
        urllib.request.urlretrieve(url, output_path)  # noqa: S310 - fixed public AudioSet metadata URLs.
        downloaded[name] = str(output_path)
    print(json.dumps(downloaded, ensure_ascii=False, indent=2))
    return 0


def build_candidate_manifest(args: argparse.Namespace) -> int:
    limits = _parse_limits(args.limit)
    candidates = build_candidates(
        args.segments_csv,
        args.class_labels_csv,
        per_label_limit=limits,
    )
    write_candidate_csv(candidates, args.output)
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.label] = counts.get(candidate.label, 0) + 1
    print(json.dumps({"candidates": len(candidates), "label_counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


def download_audio(args: argparse.Namespace) -> int:
    candidates = read_candidate_csv(args.candidates_csv)
    if args.max_clips is not None:
        candidates = candidates[: args.max_clips]
    report = download_candidates(
        candidates,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        failures_path=args.failures,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    return 0 if report["downloaded"] > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real AudioSet audio clips for G1 abnormal sound training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("download-metadata", help="Download official AudioSet CSV metadata.")
    metadata.add_argument("--output-dir", type=Path, required=True)
    metadata.add_argument("--skip-unbalanced", action="store_true", help="Skip the large unbalanced_train_segments.csv file.")
    metadata.add_argument("--force", action="store_true")
    metadata.set_defaults(func=download_metadata)

    candidates = subparsers.add_parser("build-candidates", help="Build G1 candidate rows from official AudioSet segment CSVs.")
    candidates.add_argument("--class-labels-csv", type=Path, required=True)
    candidates.add_argument("--segments-csv", type=Path, action="append", required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--limit", action="append", default=[], help="Optional label=count cap, e.g. cough=200")
    candidates.set_defaults(func=build_candidate_manifest)

    audio = subparsers.add_parser("download-audio", help="Download and crop real YouTube audio clips listed in a candidate CSV.")
    audio.add_argument("--candidates-csv", type=Path, required=True)
    audio.add_argument("--output-dir", type=Path, required=True)
    audio.add_argument("--manifest", type=Path, required=True)
    audio.add_argument("--failures", type=Path, required=True)
    audio.add_argument("--sample-rate", type=int, default=32_000)
    audio.add_argument("--max-clips", type=int, default=None)
    audio.add_argument("--dry-run", action="store_true")
    audio.set_defaults(func=download_audio)
    return parser.parse_args()


def _parse_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"limit must use label=count format: {value}")
        label, raw_count = value.split("=", 1)
        limits[label.strip()] = int(raw_count)
    return limits


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.func(parsed))
