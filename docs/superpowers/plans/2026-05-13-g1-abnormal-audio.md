# G1 Abnormal Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local engineering foundation for a Unitree G1 abnormal audio detector using EfficientAT DyMN10-AS, RTX 4090 cloud fine-tuning, unified `audio_event` output, webhook alarms, and real-time edge runtime.

**Architecture:** Keep model training, runtime decision logic, and alarm transport independent. Local tests use deterministic scores and do not require GPU. Cloud GPU work is prepared as scripts/configs and is started only after the user pays for compute.

**Tech Stack:** Python 3.10+, unittest, NumPy, sounddevice/soundfile, httpx, PyYAML, PyTorch/EfficientAT for cloud training, ONNX Runtime for optional G1 inference.

---

### Task 1: Event Contract

**Files:**
- Create: `src/audio_event_poc/event_contract.py`
- Test: `tests/test_event_contract.py`

- [ ] Write tests for canonical `audio_event` shape, severity/action policy, confidence validation, and JSON serialization.
- [ ] Run `python -m unittest tests.test_event_contract -v` and verify the tests fail before implementation.
- [ ] Implement dataclasses and helpers for event definitions and event creation.
- [ ] Re-run the test and verify pass.

### Task 2: Runtime Smoothing

**Files:**
- Create: `src/audio_event_poc/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] Write tests for per-class thresholds, consecutive hit confirmation, background suppression, cooldown, and event timing.
- [ ] Run `python -m unittest tests.test_runtime -v` and verify failure.
- [ ] Implement `EventSmoother` and `ScoreFrame`.
- [ ] Re-run the test and verify pass.

### Task 3: Alarm Webhook

**Files:**
- Create: `src/audio_event_poc/alarm.py`
- Test: `tests/test_alarm.py`

- [ ] Write tests using a fake transport for success, retry, and disabled webhook behavior.
- [ ] Run `python -m unittest tests.test_alarm -v` and verify failure.
- [ ] Implement `WebhookAlarmClient`.
- [ ] Re-run the test and verify pass.

### Task 4: EfficientAT Training Scaffolding

**Files:**
- Create: `config/g1_abnormal_events.yaml`
- Create: `src/audio_event_poc/audioset_manifest.py`
- Create: `training/efficientat/README.md`
- Create: `training/efficientat/train_g1_abnormal.py`
- Create: `training/efficientat/export_onnx.py`
- Test: `tests/test_audioset_manifest.py`

- [ ] Write manifest tests for AudioSet label mapping and balanced sampling limits.
- [ ] Run `python -m unittest tests.test_audioset_manifest -v` and verify failure.
- [ ] Implement label mapping and manifest planning.
- [ ] Add cloud training scripts that fail clearly if EfficientAT/PyTorch assets are missing.
- [ ] Re-run the manifest test and verify pass.

### Task 5: G1 Real-Time Runtime

**Files:**
- Create: `scripts/g1_realtime_audio_service.py`
- Create: `scripts/g1_fake_realtime_smoke.py`
- Modify: `README.md`
- Test: `tests/test_g1_realtime_smoke.py`

- [ ] Write a smoke test with deterministic fake score frames.
- [ ] Run `python -m unittest tests.test_g1_realtime_smoke -v` and verify failure.
- [ ] Implement a fake-model smoke entrypoint and a real service entrypoint that validates model assets before microphone startup.
- [ ] Update README with local prep, cloud payment gate, training, G1 deployment, and testing steps.
- [ ] Re-run the smoke test and full suite.

### Completion Gate

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Confirm no cloud GPU spending was started.
- [ ] Tell the user exactly what is ready locally and what command/payment gate starts RTX 4090 training.
