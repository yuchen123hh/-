from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.recording import record_to_file
from audio_event_poc.sample_plan import SampleRequest, build_sample_plan, manifest_rows


DEFAULT_REQUESTS = [
    SampleRequest("knock", 5, "敲门"),
    SampleRequest("cough", 5, "咳嗽"),
    SampleRequest("clap", 5, "拍手"),
    SampleRequest("other", 5, "保持安静或制造键盘/椅子等非目标办公室声音"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided sample collection for knock/cough/clap calibration.")
    parser.add_argument("--sample-root", type=Path, default=ROOT / "samples")
    parser.add_argument("--manifest", type=Path, default=ROOT / "samples" / "manifest.csv")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--count", type=int, default=5, help="Samples per label.")
    parser.add_argument("--labels", nargs="+", default=["knock", "cough", "clap", "other"])
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between samples.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt_by_label = {request.label: request.prompt for request in DEFAULT_REQUESTS}
    requests = [
        SampleRequest(label=label, count=args.count, prompt=prompt_by_label[label])
        for label in args.labels
    ]
    plan = build_sample_plan(args.sample_root, requests)

    if args.dry_run:
        for item in plan:
            print(f"{item.label} #{item.index}: {item.path}")
        return 0

    for item in plan:
        item.path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n准备录制 {item.label} #{item.index}: {item.prompt}")
        for remaining in (3, 2, 1):
            print(f"{remaining}...")
            time.sleep(1)
        print("开始")
        record_to_file(
            item.path,
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            channels=args.channels,
            device=args.device,
        )
        print(f"已保存: {item.path}")
        if args.pause > 0:
            time.sleep(args.pause)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["audio_path", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows(plan))
    print(f"\nmanifest 已保存: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
