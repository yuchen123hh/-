from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.alarm import WebhookAlarmClient
from audio_event_poc.runtime import EventSmoother, ScoreFrame


def run_fake_realtime_smoke(webhook_url: str = "") -> dict[str, Any]:
    smoother = EventSmoother(
        thresholds={"smoke_alarm": 0.64},
        consecutive_hits={"smoke_alarm": 2},
        cooldown_s=8.0,
        model="efficientat_dymn10_as_v0_fake",
        metadata={"window_s": 2.0, "hop_s": 0.5, "runtime": "fake_smoke"},
    )
    alarm_client = WebhookAlarmClient(url=webhook_url, retry_delay_s=0.0)
    frames = [
        ScoreFrame(timestamp=0.0, duration_s=2.0, scores={"background": 0.95, "smoke_alarm": 0.1}),
        ScoreFrame(timestamp=0.5, duration_s=2.0, scores={"background": 0.2, "smoke_alarm": 0.82}),
        ScoreFrame(timestamp=1.0, duration_s=2.0, scores={"background": 0.1, "smoke_alarm": 0.89}),
    ]
    events: list[dict[str, object]] = []
    webhook_results: list[dict[str, object]] = []
    for frame in frames:
        for event in smoother.update(frame):
            events.append(event)
            webhook_results.append(alarm_client.send(event))
    return {"event_count": len(events), "events": events, "webhook_results": webhook_results}


def main() -> int:
    result = run_fake_realtime_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
