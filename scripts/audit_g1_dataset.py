from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.dataset_audit import audit_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit real-world G1 audio manifests before paid GPU training.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--min-duration-s", type=float, default=0.5)
    parser.add_argument("--max-duration-s", type=float, default=15.0)
    parser.add_argument("--min-per-label", type=int, default=1)
    parser.add_argument("--require-all-labels", action="store_true")
    parser.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="Allow missing source_id/clip_id. Use only for legacy local experiments, not paid training.",
    )
    args = parser.parse_args()

    report = audit_manifests(
        [("train", args.train_manifest), ("val", args.val_manifest)],
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        min_per_label=args.min_per_label,
        require_all_labels=args.require_all_labels,
        require_provenance=not args.allow_missing_provenance,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ready_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
