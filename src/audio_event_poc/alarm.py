from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


class AlarmTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout_s: float) -> int:
        ...


@dataclass
class HttpxAlarmTransport:
    def post_json(self, url: str, payload: dict[str, Any], timeout_s: float) -> int:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("webhook alarm requires httpx. Install project requirements first.") from exc
        response = httpx.post(url, json=payload, timeout=timeout_s)
        return int(response.status_code)


class WebhookAlarmClient:
    def __init__(
        self,
        *,
        url: str | None,
        transport: AlarmTransport | None = None,
        timeout_s: float = 3.0,
        max_retries: int = 2,
        retry_delay_s: float = 0.5,
    ) -> None:
        self.url = (url or "").strip()
        self.transport = transport or HttpxAlarmTransport()
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    def send(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.url:
            return {"sent": False, "reason": "webhook_disabled", "attempts": 0, "status_code": None}

        attempts = self.max_retries + 1
        last_status: int | None = None
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                last_status = self.transport.post_json(self.url, event, self.timeout_s)
                last_error = None
                if 200 <= last_status < 300:
                    return {"sent": True, "attempts": attempt, "status_code": last_status, "error": None}
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            if attempt < attempts and self.retry_delay_s > 0:
                time.sleep(self.retry_delay_s)

        return {
            "sent": False,
            "attempts": attempts,
            "status_code": last_status,
            "error": last_error,
        }
