from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
RECORDINGS_DIR = STATIC_DIR / "audio_test_recordings"
LOG_DIR = ROOT / "logs"
LOG_JSONL = LOG_DIR / "audio_test_events.jsonl"
PANNS_MODEL_PATH = ROOT / "panns_data" / "Cnn14_mAP=0.431.pth"
PANNS_LABELS_PATH = ROOT / "panns_data" / "class_labels_indices.csv"
PANNS_HOME_MODEL_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
PANNS_HOME_LABELS_PATH = Path.home() / "panns_data" / "class_labels_indices.csv"
DEFAULT_LOCAL_ASR_MODEL = "models/faster-whisper-small-ct2"
DEFAULT_OPENAI_AUDIO_MODEL = "gpt-audio"
DEFAULT_REALTIME_MODEL = "gpt-realtime-1.5"
PANNS_MIN_BYTES = 300_000_000
STANDARD_LABELS = [
    "speech",
    "music",
    "clap",
    "knock",
    "cough",
    "alarm",
    "siren",
    "dog_bark",
    "running_water",
    "fan_noise",
    "keyboard_typing",
    "footsteps",
    "vehicle",
    "silence",
    "unknown",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def spectral_features(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if audio.size < 32 or float(np.max(np.abs(audio))) < 1e-8:
        return 0.0, 0.0
    windowed = audio[: min(audio.size, sample_rate * 5)]
    spectrum = np.abs(np.fft.rfft(windowed * np.hanning(windowed.size))) + 1e-12
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    flatness = geometric / arithmetic if arithmetic else 0.0
    return centroid, flatness


def analyze_environment_audio(wav_path: str | Path) -> dict[str, Any]:
    audio, sample_rate = read_wav_mono(wav_path)
    duration_s = round(float(audio.size / sample_rate), 3) if sample_rate else 0.0
    if audio.size == 0:
        rms = peak = 0.0
    else:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
    dbfs = 20.0 * math.log10(max(rms, 1e-9))
    frame = max(1, int(sample_rate * 0.02))
    frame_count = max(1, audio.size // frame)
    trimmed = audio[: frame_count * frame] if audio.size else audio
    frame_rms = (
        np.sqrt(np.mean(np.square(trimmed.reshape(frame_count, frame)), axis=1))
        if trimmed.size
        else np.array([0.0], dtype=np.float32)
    )
    active_ratio = float(np.mean(frame_rms > max(0.015, rms * 0.55)))
    centroid, flatness = spectral_features(audio, sample_rate)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))))) if audio.size > 1 else 0.0
    crest = peak / max(rms, 1e-9)

    if peak < 0.01 or rms < 0.002 or dbfs < -54:
        top_label = "silence"
    elif crest > 8.0 and active_ratio < 0.25:
        top_label = "impulse_clap_or_knock"
    elif centroid < 260 and rms > 0.01:
        top_label = "low_frequency_rumble"
    elif flatness > 0.32 and active_ratio > 0.45:
        top_label = "steady_noise"
    elif centroid > 900 and flatness < 0.22:
        top_label = "tonal_or_music"
    else:
        top_label = "speech_or_voice"

    heard = top_label != "silence"
    confidence = min(0.99, max(0.05, (peak * 0.65) + (active_ratio * 0.25) + (min(crest, 20) / 200)))
    return {
        "enabled": True,
        "model": "local_signal_features_v1",
        "heard": heard,
        "heard_confidence": round(float(confidence if heard else 1.0 - confidence), 3),
        "top_label": top_label,
        "summary_zh": f"轻量信号分析判断为：{top_label}",
        "duration_s": duration_s,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "dbfs": round(dbfs, 2),
        "active_ratio": round(active_ratio, 3),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_flatness": round(flatness, 4),
        "zero_crossing_rate": round(zcr, 4),
    }


def read_recent_events(log_path: str | Path = LOG_JSONL, limit: int = 20) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"error": "invalid_jsonl_line", "raw": line})
    return list(reversed(rows[-max(1, limit) :]))


def append_event(row: dict[str, Any]) -> None:
    ensure_dirs()
    with LOG_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_wav(raw_path: Path, wav_path: Path) -> None:
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_found")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(raw_path), "-ac", "1", "-ar", "16000", "-vn", str(wav_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg_failed").strip()[-2000:])


def empty_confidences() -> dict[str, float]:
    return {label: 0.0 for label in STANDARD_LABELS}


def map_signal_label(signal_label: str) -> str:
    return {
        "silence": "silence",
        "speech_or_voice": "speech",
        "impulse_clap_or_knock": "knock",
        "steady_noise": "fan_noise",
        "tonal_or_music": "music",
        "low_frequency_rumble": "vehicle",
    }.get(signal_label, "unknown")


def skipped_result(reason: str, backend: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "model": None,
        "heard": False,
        "heard_confidence": 0.0,
        "top_label": "unknown",
        "summary_zh": reason,
        "transcript": "",
        "confidences": empty_confidences(),
        "events": [],
        "elapsed_s": 0.0,
        "raw_text": "",
        "error": reason,
        "backend": backend,
    }


def resolve_panns_assets() -> tuple[Path, Path]:
    project_model_ok = PANNS_MODEL_PATH.exists() and PANNS_MODEL_PATH.stat().st_size >= PANNS_MIN_BYTES
    home_model_ok = PANNS_HOME_MODEL_PATH.exists() and PANNS_HOME_MODEL_PATH.stat().st_size >= PANNS_MIN_BYTES
    model_path = PANNS_MODEL_PATH if project_model_ok else PANNS_HOME_MODEL_PATH
    labels_path = PANNS_LABELS_PATH if PANNS_LABELS_PATH.exists() else PANNS_HOME_LABELS_PATH
    return model_path, labels_path


def labels_available() -> bool:
    model_path, labels_path = resolve_panns_assets()
    return model_path.exists() and model_path.stat().st_size >= PANNS_MIN_BYTES and labels_path.exists()


def local_model_status() -> dict[str, Any]:
    model_path, labels_path = resolve_panns_assets()
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    try:
        import faster_whisper  # noqa: F401
        whisper_dep = True
        whisper_error = None
    except Exception as exc:
        whisper_dep = False
        whisper_error = str(exc)
    try:
        import panns_inference  # noqa: F401

        panns_dep = True
        panns_error = None
    except Exception as exc:
        panns_dep = False
        panns_error = str(exc)
    asr_path = ROOT / DEFAULT_LOCAL_ASR_MODEL
    return {
        "available": labels_available(),
        "deps_available": whisper_dep and panns_dep,
        "whisper_dep_available": whisper_dep,
        "whisper_dep_error": whisper_error,
        "panns_dep_available": panns_dep,
        "panns_dep_error": panns_error,
        "panns_model_path": str(model_path),
        "panns_model_exists": model_path.exists(),
        "panns_model_bytes": model_bytes,
        "panns_model_complete": model_bytes >= PANNS_MIN_BYTES,
        "panns_labels_path": str(labels_path),
        "panns_labels_exists": labels_path.exists(),
        "asr_model_path": str(asr_path),
        "asr_model_exists": asr_path.exists(),
    }


def normalize_audioset_label(name: str) -> str | None:
    label = name.lower()
    if any(word in label for word in ["speech", "conversation", "narration"]):
        return "speech"
    if "cough" in label:
        return "cough"
    if "clapping" in label or "applause" in label:
        return "clap"
    if "knock" in label or "doorbell" in label:
        return "knock"
    if any(word in label for word in ["fire alarm", "smoke detector", "alarm", "alarm clock", "car alarm"]):
        return "alarm"
    if "siren" in label:
        return "siren"
    if any(word in label for word in ["water", "water tap", "faucet", "waterfall"]):
        return "running_water"
    if ("typing" in label or "keyboard" in label) and "musical" not in label:
        return "keyboard_typing"
    if "footstep" in label:
        return "footsteps"
    if any(word in label for word in ["vehicle", "car", "engine", "horn"]):
        return "vehicle"
    if any(word in label for word in ["music", "singing", "musical"]):
        return "music"
    if "dog" in label and "bark" in label:
        return "dog_bark"
    return None


def run_panns(wav_path: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from panns_inference import AudioTagging

    import librosa

    model_path, labels_path = resolve_panns_assets()
    audio, _ = librosa.load(str(wav_path), sr=32000, mono=True)
    model = AudioTagging(checkpoint_path=str(model_path), device="cpu")
    clipwise_output, _ = model.inference(audio[None, :])
    scores = np.asarray(clipwise_output[0], dtype=np.float32)
    rows: list[dict[str, str]] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    top_indices = np.argsort(scores)[::-1][:15]
    confidences = empty_confidences()
    top_labels: list[dict[str, Any]] = []
    for index in top_indices:
        row = rows[int(index)] if int(index) < len(rows) else {}
        name = row.get("display_name") or row.get("display_name\r") or row.get("name") or str(index)
        score = float(scores[int(index)])
        mapped = normalize_audioset_label(name)
        if mapped:
            confidences[mapped] = max(confidences[mapped], round(score, 4))
        top_labels.append({"audioset_label": name, "score": round(score, 4), "mapped_label": mapped})
    return confidences, top_labels


def run_whisper_if_needed(wav_path: Path, local_asr_model: str, top_label: str) -> tuple[str, float, str | None]:
    if top_label != "speech":
        return "", 0.0, None
    start = time.perf_counter()
    try:
        from faster_whisper import WhisperModel

        model_path = ROOT / local_asr_model
        model_name = str(model_path) if model_path.exists() else "small"
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav_path), language="zh")
        transcript = "".join(segment.text for segment in segments).strip()
        return transcript, round(time.perf_counter() - start, 3), None
    except Exception as exc:  # noqa: BLE001
        return "", round(time.perf_counter() - start, 3), str(exc)


def run_local_strong_model(wav_path: Path, signal: dict[str, Any], local_asr_model: str) -> dict[str, Any]:
    start = time.perf_counter()
    signal_label = map_signal_label(str(signal.get("top_label", "unknown")))
    confidences = empty_confidences()
    confidences[signal_label] = max(confidences[signal_label], float(signal.get("heard_confidence", 0.0)))
    panns_top_labels: list[dict[str, Any]] = []
    event_error = None
    event_elapsed_s = 0.0

    if labels_available():
        event_start = time.perf_counter()
        try:
            panns_confidences, panns_top_labels = run_panns(wav_path)
            event_elapsed_s = round(time.perf_counter() - event_start, 3)
            for label, score in panns_confidences.items():
                confidences[label] = max(confidences[label], score)
        except Exception as exc:  # noqa: BLE001
            event_elapsed_s = round(time.perf_counter() - event_start, 3)
            event_error = str(exc)
    else:
        model_path, labels_path = resolve_panns_assets()
        event_error = f"panns_assets_missing: {model_path} or {labels_path}"

    if str(signal.get("top_label")) == "impulse_clap_or_knock":
        confidences["knock"] = max(confidences["knock"], 0.72)
        confidences["clap"] = max(confidences["clap"], 0.62)
    top_label = max(confidences.items(), key=lambda item: item[1])[0]
    if signal.get("top_label") == "silence":
        top_label = "silence"
        confidences["silence"] = max(confidences["silence"], 0.9)

    transcript, asr_elapsed_s, asr_error = run_whisper_if_needed(wav_path, local_asr_model, top_label)
    heard = top_label != "silence" and max(confidences.values()) > 0.08
    events = [{"label": top_label, "confidence": round(confidences[top_label], 4)}] if heard else []
    return {
        "enabled": True,
        "model": "strong_local_panns_whisper",
        "heard": heard,
        "heard_confidence": round(float(confidences[top_label]), 4),
        "top_label": top_label,
        "summary_zh": f"B 本地强模型判断为：{top_label}",
        "transcript": transcript,
        "confidences": confidences,
        "events": events,
        "elapsed_s": round(time.perf_counter() - start, 3),
        "raw_text": json.dumps({"signal": signal, "panns_top_labels": panns_top_labels}, ensure_ascii=False),
        "error": event_error,
        "backend": "strong_local_panns_whisper",
        "asr_elapsed_s": asr_elapsed_s,
        "event_elapsed_s": event_elapsed_s,
        "asr_error": asr_error,
        "event_error": event_error,
        "signal_top_label": signal.get("top_label"),
        "event_model": "PANNs Cnn14 AudioSet",
        "panns_top_labels": panns_top_labels,
    }


OPENAI_AUDIO_PROMPT = """分析这段音频，识别人声和环境声。只返回紧凑 JSON，不要 Markdown。
字段: heard, heard_confidence, top_label, summary_zh, transcript, confidences, events。
top_label 和 confidences 的键必须归一化到:
speech,music,clap,knock,cough,alarm,siren,dog_bark,running_water,fan_noise,keyboard_typing,footsteps,vehicle,silence,unknown。"""


def run_openai_audio(wav_path: Path, model: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return skipped_result("openai_not_configured", "openai")
    start = time.perf_counter()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        with wav_path.open("rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("ascii")
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": OPENAI_AUDIO_PROMPT},
                            {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}},
                        ],
                    }
                ],
            )
        raw_text = getattr(response, "output_text", "") or str(response)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = {"heard": False, "heard_confidence": 0.0, "top_label": "unknown", "summary_zh": raw_text}
        confidences = empty_confidences()
        confidences.update({k: float(v) for k, v in parsed.get("confidences", {}).items() if k in confidences})
        return {
            "enabled": True,
            "model": model,
            "heard": bool(parsed.get("heard", False)),
            "heard_confidence": float(parsed.get("heard_confidence", 0.0)),
            "top_label": parsed.get("top_label", "unknown"),
            "summary_zh": parsed.get("summary_zh", ""),
            "transcript": parsed.get("transcript", ""),
            "confidences": confidences,
            "events": parsed.get("events", []),
            "elapsed_s": round(time.perf_counter() - start, 3),
            "raw_text": raw_text,
            "error": None,
            "backend": "openai",
        }
    except Exception as exc:  # noqa: BLE001
        result = skipped_result(str(exc), "openai")
        result["enabled"] = True
        result["model"] = model
        result["elapsed_s"] = round(time.perf_counter() - start, 3)
        return result


def create_app(args: argparse.Namespace | None = None) -> FastAPI:
    try:
        import httpx
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI app requires fastapi, uvicorn, python-multipart, python-dotenv, and httpx."
        ) from exc

    load_dotenv(ROOT / ".env")
    ensure_dirs()
    settings = args or argparse.Namespace(
        openai_model=DEFAULT_OPENAI_AUDIO_MODEL,
        realtime_model=DEFAULT_REALTIME_MODEL,
        local_model=DEFAULT_LOCAL_ASR_MODEL,
        default_backend="local",
    )
    app = FastAPI(title="G1 听觉测试页面")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML.replace("__DEFAULT_BACKEND__", settings.default_backend)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        ffmpeg = resolve_ffmpeg()
        proxy = {k: os.getenv(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"] if os.getenv(k)}
        local_status = local_model_status()
        return {
            "recordings_dir": str(RECORDINGS_DIR),
            "ffmpeg": {"available": bool(ffmpeg), "path": ffmpeg},
            "log_jsonl": str(LOG_JSONL),
            "openai_audio_configured": bool(os.getenv("OPENAI_API_KEY")),
            "openai_audio_model": settings.openai_model,
            "realtime_model": settings.realtime_model,
            "local_audio_model": str(resolve_panns_assets()[0]),
            "local_asr_model": settings.local_model,
            "local_audio_model_available": local_status["available"],
            "local_audio_model_status": "available" if local_status["available"] else "missing_panns_assets",
            "local_audio_model_details": local_status,
            "default_backend": settings.default_backend,
            "proxy": proxy,
            "server_time": utc_now(),
        }

    @app.post("/api/recordings")
    async def upload_recording(
        file: UploadFile = File(...),
        backend: str = Form("local"),
    ) -> JSONResponse:
        if backend not in {"local", "openai", "compare"}:
            raise HTTPException(status_code=400, detail="backend must be local, openai, or compare")
        event_id = uuid.uuid4().hex
        event_time = utc_now()
        suffix = Path(file.filename or "recording.webm").suffix or ".webm"
        raw_path = RECORDINGS_DIR / f"{event_id}{suffix}"
        wav_path = RECORDINGS_DIR / f"{event_id}.wav"
        content = await file.read()
        raw_path.write_bytes(content)
        row: dict[str, Any] = {
            "event_id": event_id,
            "event_time": event_time,
            "selected_backend": backend,
            "source_filename": file.filename,
            "content_type": file.content_type,
            "raw_file": str(raw_path),
            "wav_file": str(wav_path),
            "raw_url": f"/static/audio_test_recordings/{raw_path.name}",
            "wav_url": f"/static/audio_test_recordings/{wav_path.name}",
            "bytes": len(content),
            "analysis": None,
            "local_alternative": None,
            "openai_analysis": None,
        }
        try:
            convert_to_wav(raw_path, wav_path)
            signal = analyze_environment_audio(wav_path)
            row["analysis"] = signal
            row["local_alternative"] = (
                run_local_strong_model(wav_path, signal, settings.local_model)
                if backend in {"local", "compare"}
                else skipped_result("local_skipped", "strong_local_panns_whisper")
            )
            row["openai_analysis"] = (
                run_openai_audio(wav_path, settings.openai_model)
                if backend in {"openai", "compare"}
                else skipped_result("openai_skipped", "openai")
            )
        except Exception as exc:  # noqa: BLE001
            row["analysis"] = {"enabled": False, "error": str(exc), "top_label": "unknown", "heard": False}
            row["local_alternative"] = skipped_result("local_skipped_due_to_preprocess_error", "strong_local_panns_whisper")
            row["openai_analysis"] = skipped_result("openai_skipped_due_to_preprocess_error", "openai")
        append_event(row)
        return JSONResponse(row)

    @app.get("/api/recordings")
    async def recordings(limit: int = 20) -> dict[str, Any]:
        return {"items": read_recent_events(LOG_JSONL, limit=limit)}

    @app.post("/api/realtime/calls")
    async def realtime_calls(request: Request) -> PlainTextResponse:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="openai_not_configured")
        sdp = await request.body()
        url = f"https://api.openai.com/v1/realtime/calls?model={settings.realtime_model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/sdp",
            "OpenAI-Beta": "realtime=v1",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, content=sdp, headers=headers)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return PlainTextResponse(response.text, media_type="application/sdp")

    return app


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>G1 听觉测试页面</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --ink:#17202a; --muted:#667085; --line:#d9dee7; --ok:#16865a; --bad:#bc2f35; --blue:#2563eb; --panel:#fff; --amber:#b7791f; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:18px 24px 12px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:3; }
    h1 { margin:0 0 12px; font-size:24px; letter-spacing:0; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:7px; padding:9px 12px; cursor:pointer; font-size:14px; }
    button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .toolbar, .status, .modes, .metrics, .grid { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .pill { border:1px solid var(--line); border-radius:999px; padding:5px 9px; background:#fff; color:var(--muted); font-size:13px; }
    .pill.ok { color:var(--ok); border-color:#a9dec8; background:#effaf5; }
    .pill.bad { color:var(--bad); border-color:#efb0b3; background:#fff1f2; }
    .pill.warn { color:var(--amber); border-color:#f2d391; background:#fff8e8; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:16px 24px 24px; }
    section, aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    section { margin-bottom:14px; }
    h2 { margin:0 0 10px; font-size:16px; }
    .metric { flex:1 1 150px; border:1px solid var(--line); border-radius:8px; padding:12px; min-height:74px; }
    .metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
    .metric strong { font-size:21px; overflow-wrap:anywhere; }
    canvas { width:100%; height:120px; border:1px solid var(--line); border-radius:8px; background:#101828; display:block; }
    .volume { height:16px; border-radius:999px; border:1px solid var(--line); overflow:hidden; background:#eef2f7; }
    .volume > div { height:100%; width:0%; background:linear-gradient(90deg,#16a34a,#f59e0b,#dc2626); transition:width .08s linear; }
    .result-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px; max-height:280px; overflow:auto; background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:10px; }
    .bar { display:grid; grid-template-columns:130px 1fr 48px; gap:8px; align-items:center; margin:6px 0; font-size:12px; }
    .track { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
    .fill { height:100%; background:#2563eb; }
    .events li, .history li { border-bottom:1px solid var(--line); padding:9px 0; list-style:none; }
    ul { margin:0; padding:0; }
    audio { width:100%; margin-top:8px; }
    label { margin-right:12px; color:var(--muted); font-size:14px; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; padding:12px; } .result-grid { grid-template-columns:1fr; } header { padding:14px 12px; } }
  </style>
</head>
<body>
  <header>
    <h1>G1 听觉测试页面</h1>
    <div class="toolbar">
      <button id="startBtn" class="primary">开始录音</button>
      <button id="stopBtn" disabled>停止并分析</button>
      <button id="rtStartBtn">实时监听</button>
      <button id="rtStopBtn" disabled>停止实时</button>
      <button id="refreshBtn">刷新记录</button>
    </div>
    <div class="status" id="status" style="margin-top:10px"></div>
    <div class="modes" style="margin-top:10px">
      <label><input type="radio" name="backend" value="local" checked /> B 本地强模型</label>
      <label><input type="radio" name="backend" value="openai" /> A OpenAI</label>
      <label><input type="radio" name="backend" value="compare" /> A+B 对比</label>
    </div>
  </header>
  <main>
    <div>
      <section>
        <div class="metrics">
          <div class="metric"><span>是否听到</span><strong id="heard">-</strong></div>
          <div class="metric"><span>声音强度</span><strong id="strength">-</strong></div>
          <div class="metric"><span>最高类别</span><strong id="topLabel">-</strong></div>
          <div class="metric"><span>置信度</span><strong id="confidence">-</strong></div>
        </div>
      </section>
      <section>
        <h2>实时音量与波形</h2>
        <div class="volume"><div id="volumeFill"></div></div>
        <canvas id="wave" width="1100" height="180"></canvas>
        <audio id="playback" controls></audio>
      </section>
      <section>
        <div class="result-grid">
          <div><h2>本地信号分析</h2><pre id="signalResult">{}</pre></div>
          <div><h2>B 本地强模型</h2><pre id="localResult">{}</pre></div>
          <div><h2>A OpenAI 音频分析</h2><pre id="openaiResult">{}</pre></div>
        </div>
      </section>
      <section>
        <h2>置信度条形图</h2>
        <div id="bars"></div>
      </section>
      <section>
        <h2>事件列表</h2>
        <ul class="events" id="events"></ul>
      </section>
    </div>
    <aside>
      <h2>历史记录</h2>
      <ul class="history" id="history"></ul>
    </aside>
  </main>
<script>
const DEFAULT_BACKEND = "__DEFAULT_BACKEND__";
let mediaRecorder, chunks = [], stream, audioCtx, analyser, dataArray, rafId, realtimePc;
const $ = id => document.getElementById(id);
const labels = ["speech","music","clap","knock","cough","alarm","siren","dog_bark","running_water","fan_noise","keyboard_typing","footsteps","vehicle","silence","unknown"];

function setBackendDefault() {
  const input = document.querySelector(`input[name=backend][value="${DEFAULT_BACKEND}"]`);
  if (input) input.checked = true;
}
function selectedBackend() { return document.querySelector("input[name=backend]:checked").value; }
function pretty(obj) { return JSON.stringify(obj || {}, null, 2); }
function pill(text, state) { return `<span class="pill ${state}">${text}</span>`; }

async function refreshStatus() {
  const res = await fetch("/api/status");
  const s = await res.json();
  $("status").innerHTML = [
    pill(`OpenAI ${s.openai_audio_configured ? "已配置" : "未配置"}`, s.openai_audio_configured ? "ok" : "bad"),
    pill(`本地模型 ${s.local_audio_model_available ? "可用" : "缺失"}`, s.local_audio_model_available ? "ok" : "warn"),
    pill(`代理 ${Object.keys(s.proxy || {}).length ? "已检测" : "无"}`, "warn"),
    pill(`ffmpeg ${s.ffmpeg.available ? "可用" : "缺失"}`, s.ffmpeg.available ? "ok" : "bad")
  ].join("");
}

function draw() {
  if (!analyser) return;
  analyser.getByteTimeDomainData(dataArray);
  const canvas = $("wave"), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#7dd3fc"; ctx.lineWidth = 2; ctx.beginPath();
  let max = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const v = (dataArray[i] - 128) / 128; max = Math.max(max, Math.abs(v));
    const x = i / (dataArray.length - 1) * canvas.width;
    const y = canvas.height / 2 + v * canvas.height * 0.42;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  $("volumeFill").style.width = `${Math.min(100, max * 120)}%`;
  rafId = requestAnimationFrame(draw);
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.start();
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048;
  dataArray = new Uint8Array(analyser.frequencyBinCount);
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  draw();
  $("startBtn").disabled = true; $("stopBtn").disabled = false;
}

async function stopRecording() {
  return new Promise(resolve => {
    mediaRecorder.onstop = async () => {
      cancelAnimationFrame(rafId);
      stream.getTracks().forEach(t => t.stop());
      if (audioCtx) await audioCtx.close();
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      $("playback").src = URL.createObjectURL(blob);
      await upload(blob);
      $("startBtn").disabled = false; $("stopBtn").disabled = true;
      resolve();
    };
    mediaRecorder.stop();
  });
}

async function upload(blob) {
  const fd = new FormData();
  fd.append("file", blob, "recording.webm");
  fd.append("backend", selectedBackend());
  const res = await fetch("/api/recordings", { method: "POST", body: fd });
  const json = await res.json();
  renderResult(json);
  await loadHistory();
}

function renderResult(json) {
  const best = json.local_alternative?.enabled ? json.local_alternative : (json.openai_analysis?.enabled ? json.openai_analysis : json.analysis);
  $("heard").textContent = best?.heard ? "是" : "否";
  $("strength").textContent = json.analysis?.dbfs !== undefined ? `${json.analysis.dbfs} dBFS` : "-";
  $("topLabel").textContent = best?.top_label || "-";
  $("confidence").textContent = best?.heard_confidence !== undefined ? Number(best.heard_confidence).toFixed(3) : "-";
  $("signalResult").textContent = pretty(json.analysis);
  $("localResult").textContent = pretty(json.local_alternative);
  $("openaiResult").textContent = pretty(json.openai_analysis);
  renderBars(best?.confidences || {});
  renderEvents(best?.events || []);
}

function renderBars(conf) {
  $("bars").innerHTML = labels.map(label => {
    const v = Number(conf[label] || 0);
    return `<div class="bar"><span>${label}</span><div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, v*100))}%"></div></div><span>${v.toFixed(2)}</span></div>`;
  }).join("");
}

function renderEvents(events) {
  $("events").innerHTML = (events.length ? events : [{label:"无事件", confidence:0}]).map(e => `<li>${e.label || e.event || JSON.stringify(e)} <span class="pill">${Number(e.confidence || e.score || 0).toFixed(3)}</span></li>`).join("");
}

async function loadHistory() {
  const res = await fetch("/api/recordings?limit=20");
  const json = await res.json();
  $("history").innerHTML = (json.items || []).map(item => {
    const best = item.local_alternative?.enabled ? item.local_alternative : (item.openai_analysis?.enabled ? item.openai_analysis : item.analysis);
    return `<li><strong>${best?.top_label || "unknown"}</strong><br><small>${item.event_time}</small><audio controls src="${item.wav_url || item.raw_url}"></audio></li>`;
  }).join("");
}

async function startRealtime() {
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  realtimePc = new RTCPeerConnection();
  mic.getTracks().forEach(track => realtimePc.addTrack(track, mic));
  const dc = realtimePc.createDataChannel("oai-events");
  dc.onopen = () => dc.send(JSON.stringify({type:"session.update", session:{instructions:"只返回紧凑 JSON，字段包括 heard, top_label, summary_zh, confidences, events。"}}));
  dc.onmessage = e => {
    const li = document.createElement("li"); li.textContent = e.data; $("events").prepend(li);
  };
  const offer = await realtimePc.createOffer();
  await realtimePc.setLocalDescription(offer);
  const res = await fetch("/api/realtime/calls", { method:"POST", headers:{"Content-Type":"application/sdp"}, body:offer.sdp });
  if (!res.ok) throw new Error(await res.text());
  await realtimePc.setRemoteDescription({ type:"answer", sdp: await res.text() });
  $("rtStartBtn").disabled = true; $("rtStopBtn").disabled = false;
}
function stopRealtime() {
  if (realtimePc) realtimePc.close();
  $("rtStartBtn").disabled = false; $("rtStopBtn").disabled = true;
}

$("startBtn").onclick = () => startRecording().catch(alert);
$("stopBtn").onclick = () => stopRecording().catch(alert);
$("refreshBtn").onclick = () => { refreshStatus(); loadHistory(); };
$("rtStartBtn").onclick = () => startRealtime().catch(err => alert(err.message));
$("rtStopBtn").onclick = stopRealtime;
setBackendDefault(); refreshStatus(); loadHistory(); renderBars({});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_AUDIO_MODEL)
    parser.add_argument("--realtime-model", default=DEFAULT_REALTIME_MODEL)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_ASR_MODEL)
    parser.add_argument("--default-backend", choices=["local", "openai", "compare"], default="local")
    return parser.parse_args()


try:
    app = create_app()
except RuntimeError:
    app = None


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    uvicorn.run(create_app(cli_args), host=cli_args.host, port=cli_args.port)
