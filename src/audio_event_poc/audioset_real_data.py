from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from audio_event_poc.audioset_manifest import map_audioset_labels


AUDIOSET_METADATA_URLS = {
    "class_labels": "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "balanced_train": "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "eval": "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
    "unbalanced_train": "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
}


RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class AudioSetCandidate:
    clip_id: str
    ytid: str
    start_seconds: float
    end_seconds: float
    label: str
    audioset_labels: list[str]
    positive_mids: list[str]

    @property
    def source_id(self) -> str:
        return f"audioset:{self.ytid}:{self.start_seconds:.3f}:{self.end_seconds:.3f}"

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.ytid}"


def read_class_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        labels: dict[str, str] = {}
        for row in reader:
            mid = (row.get("mid") or "").strip()
            display_name = (row.get("display_name") or "").strip()
            if mid and display_name:
                labels[mid] = display_name
        return labels


def read_segment_candidates(segment_path: Path, mid_to_label: dict[str, str]) -> list[AudioSetCandidate]:
    candidates: list[AudioSetCandidate] = []
    with segment_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(_segment_csv_lines(handle))
        for row in reader:
            ytid = (row.get("YTID") or "").strip()
            if not ytid:
                continue
            positive_mids = _split_mids(row.get("positive_labels") or "")
            display_labels = [mid_to_label[mid] for mid in positive_mids if mid in mid_to_label]
            label = map_audioset_labels(display_labels)
            if label is None:
                continue
            start_seconds = float((row.get("start_seconds") or "0").strip())
            end_seconds = float((row.get("end_seconds") or "0").strip())
            candidates.append(
                AudioSetCandidate(
                    clip_id=build_clip_id(ytid, start_seconds, end_seconds),
                    ytid=ytid,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    label=label,
                    audioset_labels=display_labels,
                    positive_mids=positive_mids,
                )
            )
    return candidates


def build_candidates(
    segment_paths: Iterable[Path],
    class_labels_path: Path,
    *,
    per_label_limit: dict[str, int] | None = None,
) -> list[AudioSetCandidate]:
    mid_to_label = read_class_labels(class_labels_path)
    limits = dict(per_label_limit or {})
    counts: dict[str, int] = defaultdict(int)
    selected: list[AudioSetCandidate] = []
    for segment_path in segment_paths:
        for candidate in read_segment_candidates(segment_path, mid_to_label):
            limit = limits.get(candidate.label)
            if limit is not None and counts[candidate.label] >= limit:
                continue
            selected.append(candidate)
            counts[candidate.label] += 1
    return selected


def write_candidate_csv(candidates: Iterable[AudioSetCandidate], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "ytid",
        "start_seconds",
        "end_seconds",
        "label",
        "audioset_labels",
        "positive_mids",
        "source_type",
        "source_id",
        "youtube_url",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate_to_row(candidate))
    return output_path


def read_candidate_csv(path: Path) -> list[AudioSetCandidate]:
    candidates: list[AudioSetCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ytid = (row.get("ytid") or "").strip()
            start_seconds = float(row.get("start_seconds") or 0.0)
            end_seconds = float(row.get("end_seconds") or 0.0)
            candidates.append(
                AudioSetCandidate(
                    clip_id=(row.get("clip_id") or build_clip_id(ytid, start_seconds, end_seconds)).strip(),
                    ytid=ytid,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    label=(row.get("label") or "").strip(),
                    audioset_labels=_split_pipe(row.get("audioset_labels") or ""),
                    positive_mids=_split_pipe(row.get("positive_mids") or ""),
                )
            )
    return candidates


def candidate_to_row(candidate: AudioSetCandidate) -> dict[str, str]:
    return {
        "clip_id": candidate.clip_id,
        "ytid": candidate.ytid,
        "start_seconds": f"{candidate.start_seconds:.3f}",
        "end_seconds": f"{candidate.end_seconds:.3f}",
        "label": candidate.label,
        "audioset_labels": "|".join(candidate.audioset_labels),
        "positive_mids": "|".join(candidate.positive_mids),
        "source_type": "audioset",
        "source_id": candidate.source_id,
        "youtube_url": candidate.youtube_url,
    }


def download_candidates(
    candidates: Iterable[AudioSetCandidate],
    *,
    output_dir: Path,
    manifest_path: Path,
    failures_path: Path,
    sample_rate: int = 32_000,
    ffmpeg_path: str | None = None,
    runner: RunCommand | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    command_runner = runner or _run_command
    ffmpeg = ffmpeg_path or resolve_ffmpeg()
    if ffmpeg is None and not dry_run:
        raise RuntimeError("ffmpeg is required for AudioSet clip extraction")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    successes: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for candidate in candidates:
        output_path = output_dir / candidate.label / f"{candidate.clip_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            successes.append(_manifest_row(candidate, output_path, "dry_run"))
            continue
        assert ffmpeg is not None
        try:
            media_url = resolve_youtube_audio_url(candidate.youtube_url, runner=command_runner)
            extract_audio_segment(
                media_url=media_url,
                output_path=output_path,
                start_seconds=candidate.start_seconds,
                duration_seconds=max(0.01, candidate.end_seconds - candidate.start_seconds),
                sample_rate=sample_rate,
                ffmpeg_path=ffmpeg,
                runner=command_runner,
            )
            successes.append(_manifest_row(candidate, output_path, "downloaded"))
        except RuntimeError as exc:
            failures.append({**candidate_to_row(candidate), "error": str(exc)})

    _write_download_manifest(successes, manifest_path)
    with failures_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return {
        "requested": len(successes) + len(failures),
        "downloaded": len(successes),
        "failed": len(failures),
        "manifest": str(manifest_path),
        "failures": str(failures_path),
    }


def resolve_youtube_audio_url(youtube_url: str, *, runner: RunCommand | None = None) -> str:
    command_runner = runner or _run_command
    result = command_runner(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestaudio/best",
            "--no-playlist",
            "--get-url",
            youtube_url,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(_command_error("yt-dlp failed", result))
    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError("yt-dlp did not return an audio URL")
    return urls[-1]


def extract_audio_segment(
    *,
    media_url: str,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    ffmpeg_path: str,
    runner: RunCommand | None = None,
) -> None:
    command_runner = runner or _run_command
    result = command_runner(
        [
            ffmpeg_path,
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            media_url,
            "-t",
            f"{duration_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            str(output_path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(_command_error("ffmpeg segment extraction failed", result))


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_clip_id(ytid: str, start_seconds: float, end_seconds: float) -> str:
    safe_ytid = re.sub(r"[^A-Za-z0-9_-]+", "_", ytid).strip("_") or "unknown"
    return f"{safe_ytid}_{int(round(start_seconds * 1000)):09d}_{int(round(end_seconds * 1000)):09d}"


def _segment_csv_lines(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        if line.startswith("#"):
            stripped = line[1:].lstrip()
            if stripped.startswith("YTID,"):
                yield stripped
            continue
        if line.strip():
            yield line


def _split_mids(value: str) -> list[str]:
    return [item.strip().strip('"') for item in value.split(",") if item.strip().strip('"')]


def _split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _manifest_row(candidate: AudioSetCandidate, output_path: Path, status: str) -> dict[str, str]:
    row = candidate_to_row(candidate)
    row.update(
        {
            "audio_path": str(output_path),
            "download_status": status,
        }
    )
    return row


def _write_download_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    fieldnames = [
        "clip_id",
        "audio_path",
        "label",
        "audioset_labels",
        "source_type",
        "source_id",
        "ytid",
        "start_seconds",
        "end_seconds",
        "positive_mids",
        "youtube_url",
        "download_status",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def _command_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    return f"{prefix}: {detail}" if detail else prefix
