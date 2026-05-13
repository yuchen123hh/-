from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


G1_AUDIO_LABELS = ("distress_call", "glass_break", "knock", "cough", "smoke_alarm", "background")
REAL_WORLD_SOURCE_TYPES = ("audioset", "g1_field")
REQUIRED_COLUMNS = ("audio_path", "label", "source_type")


def audit_manifests(
    manifests: Iterable[tuple[str, Path]],
    *,
    min_duration_s: float = 0.5,
    max_duration_s: float = 15.0,
    min_per_label: int = 1,
    require_all_labels: bool = False,
    require_provenance: bool = True,
) -> dict[str, object]:
    manifest_reports = [
        audit_manifest(
            split=split,
            manifest_path=manifest_path,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            require_provenance=require_provenance,
        )
        for split, manifest_path in manifests
    ]
    aggregate = _aggregate_reports(manifest_reports)
    failures = [issue for report in manifest_reports for issue in report["issues"]]
    failures.extend(_duplicate_audio_path_issues(manifest_reports))
    failures.extend(
        _label_coverage_issues(
            label_counts=aggregate["label_counts"],
            min_per_label=min_per_label,
            require_all_labels=require_all_labels,
        )
    )
    report = {
        "ready_for_training": not failures,
        "failures": failures,
        "policy": {
            "labels": list(G1_AUDIO_LABELS),
            "real_world_source_types": list(REAL_WORLD_SOURCE_TYPES),
            "min_duration_s": min_duration_s,
            "max_duration_s": max_duration_s,
            "min_per_label": min_per_label,
            "require_all_labels": require_all_labels,
            "require_provenance": require_provenance,
        },
        "aggregate": aggregate,
        "manifests": [_public_manifest_report(report) for report in manifest_reports],
    }
    return report


def audit_manifest(
    *,
    split: str,
    manifest_path: Path,
    min_duration_s: float,
    max_duration_s: float,
    require_provenance: bool,
) -> dict[str, object]:
    rows, fieldnames = _read_rows(manifest_path)
    issues: list[dict[str, object]] = []
    for column in REQUIRED_COLUMNS:
        if column not in fieldnames:
            issues.append(_issue(split, manifest_path, None, f"missing_{column}_column", f"missing required column: {column}"))

    label_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    duration_by_label: dict[str, list[float]] = defaultdict(list)
    audio_records: list[dict[str, object]] = []

    for row_number, row in enumerate(rows, start=2):
        audio_path = (row.get("audio_path") or "").strip()
        label = (row.get("label") or "").strip()
        source_type = (row.get("source_type") or "").strip().lower()
        source_id = (row.get("source_id") or row.get("clip_id") or "").strip()
        resolved_audio_path = _resolve_audio_path(manifest_path, audio_path) if audio_path else None

        if not audio_path:
            issues.append(_issue(split, manifest_path, row_number, "missing_audio_path", "missing audio_path"))
        if label not in G1_AUDIO_LABELS:
            issues.append(_issue(split, manifest_path, row_number, "unsupported_label", f"unsupported label: {label!r}"))
        else:
            label_counts[label] += 1

        if not source_type:
            issues.append(_issue(split, manifest_path, row_number, "missing_source_type", "missing source_type"))
            source_type_counts["<missing>"] += 1
        elif source_type not in REAL_WORLD_SOURCE_TYPES:
            issues.append(
                _issue(
                    split,
                    manifest_path,
                    row_number,
                    "non_real_world_source_type",
                    f"non-real-world source_type: {source_type}",
                )
            )
            source_type_counts[source_type] += 1
        else:
            source_type_counts[source_type] += 1

        if require_provenance and not source_id:
            issues.append(
                _issue(
                    split,
                    manifest_path,
                    row_number,
                    "missing_source_id",
                    "missing source_id or clip_id for provenance tracking",
                )
            )

        if resolved_audio_path is None:
            continue
        audio_records.append(
            {
                "split": split,
                "manifest": str(manifest_path),
                "row": row_number,
                "audio_path": audio_path,
                "resolved_audio_path": str(resolved_audio_path),
            }
        )
        if not resolved_audio_path.exists():
            issues.append(_issue(split, manifest_path, row_number, "missing_audio_file", f"audio file not found: {audio_path}"))
            continue

        try:
            audio_info = _read_audio_info(resolved_audio_path)
        except RuntimeError as exc:
            issues.append(_issue(split, manifest_path, row_number, "unreadable_audio_file", str(exc)))
            continue

        duration_s = float(audio_info["duration_s"])
        sample_rate_counts[str(audio_info["sample_rate"])] += 1
        channel_counts[str(audio_info["channels"])] += 1
        if label in G1_AUDIO_LABELS:
            duration_by_label[label].append(duration_s)
        if duration_s < min_duration_s:
            issues.append(
                _issue(
                    split,
                    manifest_path,
                    row_number,
                    "audio_too_short",
                    f"audio duration {duration_s:.3f}s is shorter than {min_duration_s:.3f}s",
                )
            )
        if duration_s > max_duration_s:
            issues.append(
                _issue(
                    split,
                    manifest_path,
                    row_number,
                    "audio_too_long",
                    f"audio duration {duration_s:.3f}s is longer than {max_duration_s:.3f}s",
                )
            )

    return {
        "split": split,
        "manifest": str(manifest_path),
        "rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "sample_rate_counts": dict(sorted(sample_rate_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "duration_by_label": {label: _duration_summary(values) for label, values in sorted(duration_by_label.items())},
        "audio_records": audio_records,
        "issues": issues,
    }


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _resolve_audio_path(manifest_path: Path, audio_path: str) -> Path:
    path = Path(audio_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _read_audio_info(path: Path) -> dict[str, object]:
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        return _read_wav_info(path, exc)

    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"audio file is not readable: {path}") from exc
    if info.samplerate <= 0:
        raise RuntimeError(f"audio file has invalid sample rate: {path}")
    return {
        "duration_s": round(float(info.frames / info.samplerate), 6),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": info.format,
    }


def _read_wav_info(path: Path, import_error: ModuleNotFoundError) -> dict[str, object]:
    try:
        import wave
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile or wave is required to audit audio files") from exc

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            frames = int(handle.getnframes())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"audio file is not readable without soundfile installed: {path}") from import_error
    if sample_rate <= 0:
        raise RuntimeError(f"audio file has invalid sample rate: {path}")
    return {
        "duration_s": round(float(frames / sample_rate), 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "format": "WAV",
    }


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "total": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean(values), 6),
        "total": round(sum(values), 6),
    }


def _aggregate_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    durations: list[float] = []
    for report in reports:
        label_counts.update(report["label_counts"])
        source_type_counts.update(report["source_type_counts"])
        sample_rate_counts.update(report["sample_rate_counts"])
        channel_counts.update(report["channel_counts"])
        duration_by_label = report["duration_by_label"]
        if isinstance(duration_by_label, dict):
            for summary in duration_by_label.values():
                if isinstance(summary, dict) and summary.get("count"):
                    durations.extend([float(summary["mean"])] * int(summary["count"]))
    return {
        "rows": sum(int(report["rows"]) for report in reports),
        "label_counts": dict(sorted(label_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "sample_rate_counts": dict(sorted(sample_rate_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "duration_s": _duration_summary(durations),
    }


def _duplicate_audio_path_issues(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for report in reports:
        for record in report["audio_records"]:
            if isinstance(record, dict):
                by_path[str(record["resolved_audio_path"])].append(record)
    issues: list[dict[str, object]] = []
    for resolved_path, records in sorted(by_path.items()):
        if len(records) <= 1:
            continue
        issues.append(
            {
                "split": "aggregate",
                "manifest": "",
                "row": None,
                "kind": "duplicate_audio_path",
                "message": f"audio file appears in multiple manifest rows: {resolved_path}",
                "records": records,
            }
        )
    return issues


def _public_manifest_report(report: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in report.items()
        if key != "audio_records"
    }


def _label_coverage_issues(
    *,
    label_counts: dict[str, int],
    min_per_label: int,
    require_all_labels: bool,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    labels = G1_AUDIO_LABELS if require_all_labels else tuple(label_counts)
    for label in labels:
        count = int(label_counts.get(label, 0))
        if count < min_per_label:
            issues.append(
                {
                    "split": "aggregate",
                    "manifest": "",
                    "row": None,
                    "kind": "insufficient_label_count",
                    "message": f"label {label!r} has {count} rows; need at least {min_per_label}",
                }
            )
    return issues


def _issue(split: str, manifest_path: Path, row: int | None, kind: str, message: str) -> dict[str, object]:
    return {
        "split": split,
        "manifest": str(manifest_path),
        "row": row,
        "kind": kind,
        "message": message,
    }
