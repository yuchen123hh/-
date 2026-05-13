from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from audio_event_poc.event_contract import build_audio_event, definition_for


DEFAULT_THRESHOLDS: dict[str, float] = {
    "distress_call": 0.62,
    "glass_break": 0.68,
    "knock": 0.58,
    "cough": 0.62,
    "smoke_alarm": 0.64,
}

DEFAULT_CONSECUTIVE_HITS: dict[str, int] = {
    "distress_call": 2,
    "glass_break": 1,
    "knock": 2,
    "cough": 2,
    "smoke_alarm": 2,
}


@dataclass(frozen=True)
class ScoreFrame:
    timestamp: float
    duration_s: float
    scores: Mapping[str, float]


@dataclass
class _CandidateState:
    count: int = 0
    first_timestamp: float = 0.0
    best_confidence: float = 0.0


class EventSmoother:
    def __init__(
        self,
        *,
        thresholds: Mapping[str, float] | None = None,
        consecutive_hits: Mapping[str, int] | None = None,
        cooldown_s: float = 8.0,
        source: str = "unitree_g1_mic",
        model: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.consecutive_hits = dict(consecutive_hits or DEFAULT_CONSECUTIVE_HITS)
        self.cooldown_s = float(cooldown_s)
        self.source = source
        self.model = model
        self.metadata = dict(metadata or {})
        self._states = {event_key: _CandidateState() for event_key in self.thresholds}
        self._last_emitted_at: dict[str, float] = {}

    def update(self, frame: ScoreFrame) -> list[dict[str, object]]:
        candidate = self._best_candidate(frame.scores)
        if candidate is None:
            self._reset_all()
            return []

        event_key, confidence = candidate
        if not definition_for(event_key).alertable:
            return []

        events: list[dict[str, object]] = []
        for key in list(self._states):
            if key != event_key:
                self._states[key] = _CandidateState()

        state = self._states.setdefault(event_key, _CandidateState())
        if state.count == 0:
            state.first_timestamp = frame.timestamp
            state.best_confidence = confidence
        state.count += 1
        state.best_confidence = max(state.best_confidence, confidence)

        required_hits = max(1, int(self.consecutive_hits.get(event_key, 1)))
        if state.count < required_hits:
            return []

        if self._in_cooldown(event_key, frame.timestamp):
            return []

        threshold = self.thresholds[event_key]
        metadata = {
            **self.metadata,
            "duration_s": frame.duration_s,
            "consecutive_hits": state.count,
        }
        events.append(
            build_audio_event(
                event_key=event_key,
                confidence=state.best_confidence,
                threshold=threshold,
                start_time=state.first_timestamp,
                end_time=frame.timestamp + frame.duration_s,
                detected_at=datetime.now(timezone.utc).isoformat(),
                source=self.source,
                model=self.model,
                scores=frame.scores,
                metadata=metadata,
            )
        )
        self._last_emitted_at[event_key] = frame.timestamp
        self._states[event_key] = _CandidateState()
        return events

    def _best_candidate(self, scores: Mapping[str, float]) -> tuple[str, float] | None:
        candidates: list[tuple[str, float]] = []
        for event_key, threshold in self.thresholds.items():
            if event_key == "background":
                continue
            score = float(scores.get(event_key, 0.0))
            if score >= float(threshold):
                candidates.append((event_key, score))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0]

    def _in_cooldown(self, event_key: str, timestamp: float) -> bool:
        last = self._last_emitted_at.get(event_key)
        return last is not None and timestamp - last < self.cooldown_s

    def _reset_all(self) -> None:
        for event_key in list(self._states):
            self._states[event_key] = _CandidateState()
