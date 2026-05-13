from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
EVENT_TYPE = "audio_event"


@dataclass(frozen=True)
class G1EventDefinition:
    key: str
    label: str
    severity: str
    alertable: bool
    notify_guardian: bool
    trigger_alarm: bool


G1_EVENT_DEFINITIONS: dict[str, G1EventDefinition] = {
    "distress_call": G1EventDefinition("distress_call", "呼救声", "critical", True, True, True),
    "glass_break": G1EventDefinition("glass_break", "玻璃破碎声", "critical", True, True, True),
    "knock": G1EventDefinition("knock", "敲门声", "warning", True, True, False),
    "cough": G1EventDefinition("cough", "咳嗽声", "warning", True, True, False),
    "smoke_alarm": G1EventDefinition("smoke_alarm", "烟雾报警器声", "critical", True, True, True),
    "background": G1EventDefinition("background", "家庭背景声", "info", False, False, False),
}


def definition_for(event_key: str) -> G1EventDefinition:
    return G1_EVENT_DEFINITIONS[event_key]


def build_audio_event(
    *,
    event_key: str,
    confidence: float,
    threshold: float,
    start_time: float,
    end_time: float,
    detected_at: str,
    source: str,
    model: str,
    scores: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    definition = definition_for(event_key)
    _validate_probability("confidence", confidence)
    _validate_probability("threshold", threshold)
    normalized_scores = {key: _validated_score(key, value) for key, value in scores.items()}
    return {
        "type": EVENT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or uuid.uuid4().hex,
        "event_key": definition.key,
        "label": definition.label,
        "severity": definition.severity,
        "confidence": round(float(confidence), 6),
        "threshold": round(float(threshold), 6),
        "start_time": round(float(start_time), 3),
        "end_time": round(float(end_time), 3),
        "detected_at": detected_at,
        "source": source,
        "model": model,
        "scores": normalized_scores,
        "action": {
            "notify_guardian": definition.notify_guardian,
            "trigger_alarm": definition.trigger_alarm,
        },
        "metadata": dict(metadata or {}),
    }


def _validated_score(name: str, value: float) -> float:
    score = float(value)
    _validate_probability(name, score)
    return round(score, 6)


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
