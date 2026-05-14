from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_event_poc.alarm import WebhookAlarmClient
from audio_event_poc.runtime import EventSmoother, ScoreFrame


CLASSES = ["distress_call", "glass_break", "knock", "cough", "smoke_alarm", "background"]


class OnnxAudioClassifier:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("onnxruntime is required for G1 ONNX inference.") from exc
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_scores(self, audio) -> dict[str, float]:
        import numpy as np

        batch = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        output = self.session.run(None, {self.input_name: batch})[0][0]
        scores = np.asarray(output, dtype=np.float32)
        if scores.size != len(CLASSES):
            raise RuntimeError(f"model returned {scores.size} scores, expected {len(CLASSES)}")
        return {label: round(float(score), 6) for label, score in zip(CLASSES, scores, strict=True)}


def load_runtime_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load G1 runtime config.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"runtime config must be a mapping: {path}")
    return data


def iter_microphone_windows(*, sample_rate: int, window_s: float, hop_s: float, device: str | int | None):
    try:
        import numpy as np
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice and numpy are required for G1 microphone capture.") from exc

    window_frames = int(sample_rate * window_s)
    hop_frames = int(sample_rate * hop_s)
    buffer = np.zeros(0, dtype=np.float32)
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=device) as stream:
        while True:
            chunk, _ = stream.read(hop_frames)
            buffer = np.concatenate([buffer, chunk.reshape(-1)])
            if buffer.size >= window_frames:
                yield buffer[-window_frames:].copy()


def list_input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice is required to list G1 microphone devices.") from exc

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    rows: list[dict[str, object]] = []
    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi_name = ""
        if 0 <= hostapi_index < len(hostapis):
            hostapi_name = str(hostapis[hostapi_index].get("name", ""))
        rows.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "hostapi": hostapi_name,
                "max_input_channels": max_input_channels,
                "default_samplerate": float(device.get("default_samplerate", 0.0)),
            }
        )
    return rows


def print_input_devices() -> int:
    rows = list_input_devices()
    print(json.dumps({"input_devices": rows}, ensure_ascii=False, indent=2))
    return 0


def run_service(args: argparse.Namespace) -> int:
    if args.list_devices:
        return print_input_devices()
    config = load_runtime_config(args.config)
    model_config = config.get("model", {})
    runtime_config = config.get("runtime", {})
    model_path = Path(args.model)
    classifier = OnnxAudioClassifier(model_path)
    smoother = EventSmoother(
        thresholds=config.get("thresholds"),
        consecutive_hits=config.get("consecutive_hits"),
        cooldown_s=float(runtime_config.get("cooldown_s", 8.0)),
        source=str(model_config.get("source", "unitree_g1_mic")),
        model=f"efficientat_dymn10_as_{model_config.get('version', 'v0')}",
        metadata={"window_s": model_config.get("window_s", 2.0), "hop_s": model_config.get("hop_s", 0.5)},
    )
    alarm_client = WebhookAlarmClient(
        url=args.webhook_url if args.webhook_url is not None else runtime_config.get("webhook_url", ""),
        timeout_s=float(runtime_config.get("webhook_timeout_s", 3.0)),
        max_retries=int(runtime_config.get("webhook_max_retries", 2)),
        retry_delay_s=float(runtime_config.get("webhook_retry_delay_s", 0.5)),
    )

    log_path = Path(args.log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = int(model_config.get("sample_rate", 16000))
    window_s = float(model_config.get("window_s", 2.0))
    hop_s = float(model_config.get("hop_s", 0.5))
    start = time.monotonic()

    for window in iter_microphone_windows(
        sample_rate=sample_rate,
        window_s=window_s,
        hop_s=hop_s,
        device=args.device,
    ):
        timestamp = round(time.monotonic() - start, 3)
        scores = classifier.predict_scores(window)
        events = smoother.update(ScoreFrame(timestamp=timestamp, duration_s=window_s, scores=scores))
        for event in events:
            alarm_result = alarm_client.send(event)
            row = {"event": event, "alarm": alarm_result}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 real-time abnormal audio detection.")
    parser.add_argument("--model", required=True, help="Path to efficientat_g1_audio_v0.onnx or v1.onnx.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "g1_abnormal_events.yaml")
    parser.add_argument("--webhook-url", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-jsonl", default=str(ROOT / "logs" / "g1_audio_events.jsonl"))
    parser.add_argument("--list-devices", action="store_true", help="Print available microphone input devices and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_service(parse_args()))
