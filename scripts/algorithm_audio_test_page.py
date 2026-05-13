from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
RECORDINGS_DIR = STATIC_DIR / "algorithm_audio_recordings"
LOG_DIR = ROOT / "logs"
LOG_JSONL = LOG_DIR / "algorithm_audio_events.jsonl"
LABELS = [
    "silence",
    "impulse_clap_or_knock",
    "repeated_clicks_or_steps",
    "speech_or_voice",
    "steady_noise",
    "tonal_or_music_or_alarm",
    "low_frequency_rumble",
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


def frame_rms(audio: np.ndarray, sample_rate: int, frame_ms: float = 20.0) -> np.ndarray:
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    frame_count = max(1, audio.size // frame)
    trimmed = audio[: frame_count * frame] if audio.size else np.zeros(frame, dtype=np.float32)
    if trimmed.size < frame:
        trimmed = np.pad(trimmed, (0, frame - trimmed.size))
    return np.sqrt(np.mean(np.square(trimmed.reshape(frame_count, frame)), axis=1))


def spectrum_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size < 64 or float(np.max(np.abs(audio))) < 1e-8:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_flatness": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
            "dominant_freq_hz": 0.0,
            "tonal_ratio": 0.0,
        }
    segment = audio[: min(audio.size, sample_rate * 5)]
    windowed = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    power = np.square(spectrum)
    total_power = float(np.sum(power))
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    dominant_index = int(np.argmax(power))
    dominant_power = float(power[dominant_index])
    return {
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "low_band_ratio": float(np.sum(power[freqs < 250]) / total_power),
        "mid_band_ratio": float(np.sum(power[(freqs >= 250) & (freqs < 3000)]) / total_power),
        "high_band_ratio": float(np.sum(power[freqs >= 3000]) / total_power),
        "dominant_freq_hz": float(freqs[dominant_index]),
        "tonal_ratio": dominant_power / total_power,
    }


def onset_count(energy: np.ndarray) -> int:
    if energy.size < 3:
        return 0
    baseline = float(np.median(energy))
    spread = float(np.median(np.abs(energy - baseline))) + 1e-9
    threshold = max(0.025, baseline + spread * 7.0)
    peaks = 0
    armed = True
    for value in energy:
        if armed and value > threshold:
            peaks += 1
            armed = False
        elif value < threshold * 0.35:
            armed = True
    return peaks


def confidence_for(label: str, *, rms: float, peak: float, active_ratio: float, crest: float, onsets: int, tonal_ratio: float) -> float:
    if label == "silence":
        return min(0.99, max(0.5, 1.0 - peak * 12.0))
    if label == "repeated_clicks_or_steps":
        return min(0.96, 0.45 + min(onsets, 12) * 0.045 + min(crest, 16) * 0.015)
    if label == "impulse_clap_or_knock":
        return min(0.96, 0.45 + min(crest, 20) * 0.025 + peak * 0.2)
    if label == "tonal_or_music_or_alarm":
        return min(0.95, 0.45 + tonal_ratio * 1.2 + active_ratio * 0.25)
    if label == "low_frequency_rumble":
        return min(0.92, 0.48 + rms * 2.0 + active_ratio * 0.25)
    if label == "steady_noise":
        return min(0.9, 0.45 + active_ratio * 0.35)
    if label == "speech_or_voice":
        return min(0.86, 0.45 + active_ratio * 0.25 + min(peak, 0.8) * 0.15)
    return 0.35


def analyze_algorithmic_audio(wav_path: str | Path) -> dict[str, Any]:
    audio, sample_rate = read_wav_mono(wav_path)
    duration_s = round(float(audio.size / sample_rate), 3) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    dbfs = 20.0 * math.log10(max(rms, 1e-9))
    energy = frame_rms(audio, sample_rate, 20.0)
    active_threshold = max(0.012, rms * 0.55)
    active_ratio = float(np.mean(energy > active_threshold))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))))) if audio.size > 1 else 0.0
    crest = peak / max(rms, 1e-9)
    features = spectrum_features(audio, sample_rate)
    onsets = onset_count(energy)

    if peak < 0.01 or rms < 0.002 or dbfs < -54:
        top_label = "silence"
    elif onsets >= 4 and crest > 5.0 and active_ratio < 0.45:
        top_label = "repeated_clicks_or_steps"
    elif crest > 8.0 and active_ratio < 0.28:
        top_label = "impulse_clap_or_knock"
    elif features["low_band_ratio"] > 0.62 and features["dominant_freq_hz"] < 220 and rms > 0.015:
        top_label = "low_frequency_rumble"
    elif features["tonal_ratio"] > 0.18 and features["spectral_flatness"] < 0.18:
        top_label = "tonal_or_music_or_alarm"
    elif features["spectral_flatness"] > 0.32 and active_ratio > 0.35:
        top_label = "steady_noise"
    elif 0.015 <= rms and 0.08 <= active_ratio <= 0.95 and 0.01 <= zcr <= 0.22:
        top_label = "speech_or_voice"
    else:
        top_label = "unknown"

    confidence = confidence_for(
        top_label,
        rms=rms,
        peak=peak,
        active_ratio=active_ratio,
        crest=crest,
        onsets=onsets,
        tonal_ratio=features["tonal_ratio"],
    )
    confidences = {label: 0.0 for label in LABELS}
    confidences[top_label] = round(confidence, 3)
    return {
        "enabled": True,
        "algorithm": "dsp_rules_v1",
        "heard": top_label != "silence",
        "top_label": top_label,
        "confidence": round(confidence, 3),
        "summary_zh": f"纯算法规则判断为：{top_label}",
        "features": {
            "duration_s": duration_s,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "dbfs": round(dbfs, 2),
            "active_ratio": round(active_ratio, 3),
            "zero_crossing_rate": round(zcr, 4),
            "crest_factor": round(crest, 3),
            "onset_count": onsets,
            "onset_rate_per_s": round(onsets / max(duration_s, 1e-9), 3),
            **{key: round(value, 4) for key, value in features.items()},
        },
        "confidences": confidences,
        "events": [] if top_label == "silence" else [{"label": top_label, "confidence": round(confidence, 3)}],
        "limits_zh": "纯算法只能做粗分类，不等同于真实语义识别；复杂环境声建议用模型复核。",
    }


def append_event(row: dict[str, Any]) -> None:
    ensure_dirs()
    with LOG_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_recent_events(limit: int = 20) -> list[dict[str, Any]]:
    if not LOG_JSONL.exists():
        return []
    rows = []
    for line in LOG_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"error": "invalid_jsonl_line", "raw": line})
    return list(reversed(rows[-max(1, limit) :]))


def create_app() -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError("algorithm page requires fastapi, uvicorn, python-multipart, and imageio-ffmpeg") from exc

    ensure_dirs()
    app = FastAPI(title="G1 纯算法听觉测试页面")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        ffmpeg = resolve_ffmpeg()
        return {
            "mode": "algorithm_only",
            "model_required": False,
            "openai_required": False,
            "ffmpeg": {"available": bool(ffmpeg), "path": ffmpeg},
            "recordings_dir": str(RECORDINGS_DIR),
            "log_jsonl": str(LOG_JSONL),
            "labels": LABELS,
            "server_time": utc_now(),
        }

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> JSONResponse:
        event_id = uuid.uuid4().hex
        suffix = Path(file.filename or "recording.webm").suffix or ".webm"
        raw_path = RECORDINGS_DIR / f"{event_id}{suffix}"
        wav_path = RECORDINGS_DIR / f"{event_id}.wav"
        content = await file.read()
        raw_path.write_bytes(content)
        row: dict[str, Any] = {
            "event_id": event_id,
            "event_time": utc_now(),
            "source_filename": file.filename,
            "content_type": file.content_type,
            "raw_file": str(raw_path),
            "wav_file": str(wav_path),
            "raw_url": f"/static/algorithm_audio_recordings/{raw_path.name}",
            "wav_url": f"/static/algorithm_audio_recordings/{wav_path.name}",
            "bytes": len(content),
            "analysis": None,
        }
        try:
            convert_to_wav(raw_path, wav_path)
            row["analysis"] = analyze_algorithmic_audio(wav_path)
        except Exception as exc:  # noqa: BLE001
            row["analysis"] = {"enabled": False, "error": str(exc), "top_label": "unknown", "heard": False}
        append_event(row)
        return JSONResponse(row)

    @app.get("/api/events")
    async def events(limit: int = 20) -> dict[str, Any]:
        return {"items": read_recent_events(limit=limit)}

    return app


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>G1 纯算法听觉测试页面</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --ink:#17202a; --muted:#667085; --line:#d8dee8; --blue:#2563eb; --ok:#16865a; --bad:#bc2f35; --warn:#b7791f; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
    header { position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid var(--line); padding:16px 22px 12px; }
    h1 { margin:0 0 12px; font-size:24px; letter-spacing:0; }
    h2 { margin:0 0 10px; font-size:16px; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:7px; padding:9px 12px; cursor:pointer; font-size:14px; }
    button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .toolbar,.status,.metrics { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .pill { border:1px solid var(--line); border-radius:999px; padding:5px 9px; color:var(--muted); background:#fff; font-size:13px; }
    .pill.ok { color:var(--ok); border-color:#a9dec8; background:#effaf5; }
    .pill.bad { color:var(--bad); border-color:#efb0b3; background:#fff1f2; }
    .pill.warn { color:var(--warn); border-color:#f2d391; background:#fff8e8; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:16px 22px 24px; }
    section, aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:14px; }
    .metric { flex:1 1 150px; border:1px solid var(--line); border-radius:8px; padding:12px; min-height:74px; }
    .metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
    .metric strong { font-size:20px; overflow-wrap:anywhere; }
    canvas { width:100%; height:120px; border:1px solid var(--line); border-radius:8px; background:#101828; display:block; margin-top:10px; }
    .volume { height:16px; border-radius:999px; border:1px solid var(--line); overflow:hidden; background:#eef2f7; }
    .volume > div { height:100%; width:0%; background:linear-gradient(90deg,#16a34a,#f59e0b,#dc2626); transition:width .08s linear; }
    .two { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px; max-height:330px; overflow:auto; background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:10px; }
    .bar { display:grid; grid-template-columns:175px 1fr 48px; gap:8px; align-items:center; margin:6px 0; font-size:12px; }
    .track { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
    .fill { height:100%; background:#2563eb; }
    ul { margin:0; padding:0; }
    li { list-style:none; border-bottom:1px solid var(--line); padding:9px 0; }
    audio { width:100%; margin-top:8px; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; padding:12px; } .two { grid-template-columns:1fr; } header { padding:14px 12px; } }
  </style>
</head>
<body>
  <header>
    <h1>G1 纯算法听觉测试页面</h1>
    <div class="toolbar">
      <button id="startBtn" class="primary">开始录音</button>
      <button id="stopBtn" disabled>停止并分析</button>
      <button id="refreshBtn">刷新记录</button>
    </div>
    <div class="status" id="status" style="margin-top:10px"></div>
  </header>
  <main>
    <div>
      <section>
        <div class="metrics">
          <div class="metric"><span>是否听到</span><strong id="heard">-</strong></div>
          <div class="metric"><span>声音强度</span><strong id="strength">-</strong></div>
          <div class="metric"><span>算法类别</span><strong id="topLabel">-</strong></div>
          <div class="metric"><span>规则置信度</span><strong id="confidence">-</strong></div>
        </div>
      </section>
      <section>
        <h2>实时音量与波形</h2>
        <div class="volume"><div id="volumeFill"></div></div>
        <canvas id="wave" width="1100" height="180"></canvas>
        <audio id="playback" controls></audio>
      </section>
      <section>
        <div class="two">
          <div><h2>纯算法结果</h2><pre id="result">{}</pre></div>
          <div><h2>DSP 特征</h2><pre id="features">{}</pre></div>
        </div>
      </section>
      <section>
        <h2>类别置信度</h2>
        <div id="bars"></div>
      </section>
      <section>
        <h2>事件列表</h2>
        <ul id="events"></ul>
      </section>
    </div>
    <aside>
      <h2>历史记录</h2>
      <ul id="history"></ul>
    </aside>
  </main>
<script>
let mediaRecorder, chunks = [], stream, audioCtx, analyser, dataArray, rafId;
const $ = id => document.getElementById(id);
const labels = ["silence","impulse_clap_or_knock","repeated_clicks_or_steps","speech_or_voice","steady_noise","tonal_or_music_or_alarm","low_frequency_rumble","unknown"];
function pretty(obj) { return JSON.stringify(obj || {}, null, 2); }
function pill(text, state) { return `<span class="pill ${state}">${text}</span>`; }
async function refreshStatus() {
  const s = await (await fetch("/api/status")).json();
  $("status").innerHTML = [
    pill("纯算法 无需模型", "ok"),
    pill("OpenAI 无需配置", "ok"),
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
  const json = await (await fetch("/api/analyze", { method:"POST", body:fd })).json();
  renderResult(json);
  await loadHistory();
}
function renderResult(json) {
  const a = json.analysis || {};
  $("heard").textContent = a.heard ? "是" : "否";
  $("strength").textContent = a.features?.dbfs !== undefined ? `${a.features.dbfs} dBFS` : "-";
  $("topLabel").textContent = a.top_label || "-";
  $("confidence").textContent = a.confidence !== undefined ? Number(a.confidence).toFixed(3) : "-";
  $("result").textContent = pretty(a);
  $("features").textContent = pretty(a.features);
  renderBars(a.confidences || {});
  renderEvents(a.events || []);
}
function renderBars(conf) {
  $("bars").innerHTML = labels.map(label => {
    const v = Number(conf[label] || 0);
    return `<div class="bar"><span>${label}</span><div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, v*100))}%"></div></div><span>${v.toFixed(2)}</span></div>`;
  }).join("");
}
function renderEvents(events) {
  $("events").innerHTML = (events.length ? events : [{label:"无事件", confidence:0}]).map(e => `<li>${e.label || JSON.stringify(e)} <span class="pill">${Number(e.confidence || 0).toFixed(3)}</span></li>`).join("");
}
async function loadHistory() {
  const json = await (await fetch("/api/events?limit=20")).json();
  $("history").innerHTML = (json.items || []).map(item => {
    const a = item.analysis || {};
    return `<li><strong>${a.top_label || "unknown"}</strong><br><small>${item.event_time}</small><audio controls src="${item.wav_url || item.raw_url}"></audio></li>`;
  }).join("");
}
$("startBtn").onclick = () => startRecording().catch(err => alert(err.message));
$("stopBtn").onclick = () => stopRecording().catch(err => alert(err.message));
$("refreshBtn").onclick = () => { refreshStatus(); loadHistory(); };
refreshStatus(); loadHistory(); renderBars({});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    return parser.parse_args()


try:
    app = create_app()
except RuntimeError:
    app = None


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
