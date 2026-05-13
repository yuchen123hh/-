# G1 Abnormal Audio Detection Design

## Goal

Build a Unitree G1-ready abnormal audio event detection system for five event classes:

- `distress_call`: 呼救/尖叫/求助声
- `glass_break`: 玻璃破碎声
- `knock`: 敲门声
- `cough`: 咳嗽声
- `smoke_alarm`: 烟雾报警器声

The fixed model choice is EfficientAT DyMN10-AS, fine-tuned on a cloud RTX 4090 in two budgeted rounds: RMB 50 for v0, G1 field sampling, then RMB 50 for v1.

## Architecture

Training and robot runtime are separate.

- Training side runs on a rented RTX 4090 and produces a checkpoint, ONNX export, thresholds, and an evaluation report.
- G1 side runs local inference only. It reads microphone audio, performs sliding-window inference, smooths decisions, emits a unified `audio_event`, and sends critical events to an HTTP webhook.
- AudioSet provides the initial weakly labeled dataset. G1 field recordings provide the second-round domain adaptation data.

## Runtime Data Flow

```text
G1 microphone
 -> 16 kHz mono audio stream
 -> 2.0 s window / 0.5 s hop
 -> EfficientAT DyMN10-AS v0/v1
 -> per-class probabilities
 -> threshold + consecutive-hit smoothing + cooldown
 -> audio_event JSON
 -> JSONL log + HTTP webhook
```

## Event Contract

Every confirmed event is a JSON-serializable `audio_event` object:

```json
{
  "type": "audio_event",
  "schema_version": "1.0",
  "event_id": "uuid",
  "event_key": "smoke_alarm",
  "label": "烟雾报警器声",
  "severity": "critical",
  "confidence": 0.93,
  "threshold": 0.72,
  "start_time": 12.5,
  "end_time": 14.5,
  "detected_at": "2026-05-13T12:00:00+00:00",
  "source": "unitree_g1_mic",
  "model": "efficientat_dymn10_as_v0",
  "scores": {
    "distress_call": 0.12,
    "glass_break": 0.03,
    "knock": 0.05,
    "cough": 0.08,
    "smoke_alarm": 0.93,
    "background": 0.02
  },
  "action": {
    "notify_guardian": true,
    "trigger_alarm": true
  },
  "metadata": {
    "window_s": 2.0,
    "hop_s": 0.5
  }
}
```

## Event Severity

- `critical`: `distress_call`, `glass_break`, `smoke_alarm`
- `warning`: `knock`, `cough`
- `info`: `background` is never emitted as an alert event

## Accuracy Strategy

- Use EfficientAT DyMN10-AS because it is lighter and stronger than PANNs Cnn14 for this edge deployment.
- Use AudioSet labels for first-round data, including hard negatives from common home and robot sounds.
- Use G1 recordings for second-round domain adaptation, especially robot motor noise, fan noise, TV speech, kitchen sounds, footsteps, and false-trigger examples.
- Apply class-specific thresholds. `distress_call` and `smoke_alarm` prefer recall; `knock`, `cough`, and `glass_break` prefer precision with stronger smoothing.
- Treat Chinese distress calls as an audio-event plus optional ASR keyword problem. The first version detects shouting/screaming/help-like speech acoustics; field samples improve Chinese-specific behavior.

## Alarm Integration

The first integration target is HTTP webhook. The runtime sends one JSON POST per confirmed alert. The client supports timeout, retry count, retry delay, and cooldown is handled before webhook dispatch to avoid duplicate alarms.

## Deployment Constraints

- G1 does not train the model.
- Runtime must run CPU-first and should support ONNX Runtime when the exported model is available.
- Missing model assets produce explicit status errors and do not fake detections.
- Logs are JSONL so failures and field samples can be used for second-round training.

## Test Plan

- Unit tests for event schema, thresholds, smoothing, cooldown, webhook retry behavior, AudioSet manifest generation, and model asset checks.
- Offline evaluation script computes precision, recall, F1, confusion matrix, and threshold report from a manifest.
- Runtime smoke test uses deterministic fake model scores to validate real-time event emission without requiring GPU or model weights.
- G1 field test records false positives and missed detections for v1 training.
