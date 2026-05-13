from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EventDefinition:
    key: str
    label: str
    category: str
    threshold: float
    prompts: list[str]
    suppress: bool = False


def aggregate_prompt_scores(
    prompt_scores: Mapping[str, float],
    events: Sequence[EventDefinition],
) -> dict[str, float]:
    event_scores: dict[str, float] = {}
    for event in events:
        scores = [float(prompt_scores[prompt]) for prompt in event.prompts if prompt in prompt_scores]
        if scores:
            event_scores[event.key] = round(max(scores), 6)
    return event_scores


def select_detected_events(
    event_scores: Mapping[str, float],
    events: Sequence[EventDefinition],
    *,
    timestamp: float,
    max_events: int = 1,
    suppress_margin: float = 0.02,
) -> list[dict[str, object]]:
    event_by_key = {event.key: event for event in events}
    candidates: list[tuple[EventDefinition, float]] = []
    suppress_scores: list[float] = []

    for event_key, score in event_scores.items():
        event = event_by_key.get(event_key)
        if event is None:
            continue
        normalized_score = round(float(score), 6)
        if event.suppress:
            suppress_scores.append(normalized_score)
            continue
        if normalized_score >= event.threshold:
            candidates.append((event, normalized_score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    if candidates and suppress_scores:
        best_target_score = candidates[0][1]
        best_suppress_score = max(suppress_scores)
        if best_suppress_score >= best_target_score - suppress_margin:
            return []

    return [
        {
            "event": event.label,
            "event_key": event.key,
            "category": event.category,
            "score": score,
            "threshold": event.threshold,
            "timestamp": timestamp,
        }
        for event, score in candidates[:max_events]
    ]


def build_result(
    *,
    timestamp: float,
    audio_path: str,
    event_scores: Mapping[str, float],
    detected_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "audio_path": audio_path,
        "detected_events": list(detected_events),
        "scores": {key: round(float(value), 6) for key, value in event_scores.items()},
    }
