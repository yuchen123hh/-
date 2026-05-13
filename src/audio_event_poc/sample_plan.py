from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRequest:
    label: str
    count: int
    prompt: str


@dataclass(frozen=True)
class SampleItem:
    label: str
    index: int
    prompt: str
    path: Path


def build_sample_plan(sample_root: Path, requests: list[SampleRequest]) -> list[SampleItem]:
    plan: list[SampleItem] = []
    for request in requests:
        for index in range(1, request.count + 1):
            path = sample_root / request.label / f"{request.label}_{index:03d}.wav"
            plan.append(
                SampleItem(
                    label=request.label,
                    index=index,
                    prompt=request.prompt,
                    path=path,
                )
            )
    return plan


def manifest_rows(plan: list[SampleItem]) -> list[dict[str, str]]:
    return [
        {
            "audio_path": item.path.as_posix(),
            "label": item.label,
        }
        for item in plan
    ]
