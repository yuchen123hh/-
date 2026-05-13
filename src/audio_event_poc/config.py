from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_event_poc.decision import EventDefinition


def load_event_config(path: str | Path) -> list[EventDefinition]:
    config_path = Path(path)
    data = _load_mapping(config_path)
    raw_events = data.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("event config must contain a non-empty 'events' list")

    return [_parse_event(raw_event) for raw_event in raw_events]


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError("event config root must be a mapping")
    return data


def _parse_event(raw_event: Any) -> EventDefinition:
    if not isinstance(raw_event, dict):
        raise ValueError("each event must be a mapping")

    key = _required_str(raw_event, "key")
    label = _required_str(raw_event, "label")
    category = _required_str(raw_event, "category")
    threshold = float(raw_event["threshold"])
    prompts = raw_event.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"event '{key}' must contain a non-empty string prompt list")

    return EventDefinition(
        key=key,
        label=label,
        category=category,
        threshold=threshold,
        prompts=prompts,
        suppress=bool(raw_event.get("suppress", False)),
    )


def _required_str(raw_event: dict[str, Any], key: str) -> str:
    value = raw_event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"event field '{key}' must be a non-empty string")
    return value
