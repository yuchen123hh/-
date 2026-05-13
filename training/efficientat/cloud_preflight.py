from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from train_g1_abnormal import count_manifest_sources, validate_manifest


def check_python_packages() -> dict[str, object]:
    packages: dict[str, object] = {}
    for name in ["torch", "torchaudio", "librosa", "sklearn"]:
        try:
            module = __import__(name)
            packages[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:  # noqa: BLE001
            packages[name] = {"available": False, "error": str(exc)}
    return packages


def check_cuda() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_available": False, "cuda_available": False, "error": str(exc)}

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": len(devices),
        "devices": devices,
    }


def run_preflight(args: argparse.Namespace) -> int:
    hard_failures: list[str] = []
    always_blocking_failures: list[str] = []
    train_counts = {}
    val_counts = {}
    try:
        train_counts = validate_manifest(args.train_manifest)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        hard_failures.append(message)
        always_blocking_failures.append(message)
    try:
        val_counts = validate_manifest(args.val_manifest)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        hard_failures.append(message)
        always_blocking_failures.append(message)
    source_counts = {
        "train": dict(count_manifest_sources(args.train_manifest)),
        "val": dict(count_manifest_sources(args.val_manifest)),
    }
    efficientat_ok = (args.efficientat_root / "models" / "dymn" / "model.py").exists()
    packages = check_python_packages()
    cuda = check_cuda()
    if not efficientat_ok:
        hard_failures.append(f"invalid EfficientAT root: {args.efficientat_root}")
    if not args.allow_cpu and not cuda.get("cuda_available"):
        hard_failures.append("CUDA is not available; do not start paid 4090 training in this environment")
    for name in ["torch", "torchaudio", "librosa", "sklearn"]:
        item = packages[name]
        if isinstance(item, dict) and not item.get("available"):
            hard_failures.append(f"missing package: {name}")

    report = {
        "ready_for_paid_training": not hard_failures,
        "failures": hard_failures,
        "python": sys.version,
        "platform": platform.platform(),
        "efficientat_root": str(args.efficientat_root),
        "efficientat_root_ok": efficientat_ok,
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
        "real_world_source_counts": source_counts,
        "packages": packages,
        "cuda": cuda,
        "budget_guard": "Stop here and ask before starting a paid RTX 4090 instance or paid training run.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if always_blocking_failures:
        return 1
    return 0 if not hard_failures or args.allow_cpu else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check cloud GPU readiness before paid EfficientAT training.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--efficientat-root", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow local CPU preflight to exit 0 for tests.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_preflight(parse_args()))
