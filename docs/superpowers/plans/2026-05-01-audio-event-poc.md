# Audio Event PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-stage desktop PoC that detects knock, cough, and clap sounds in a quiet office and outputs structured JSON.

**Architecture:** Keep the model-facing code separate from the deterministic decision logic. CLAP produces prompt-level similarity scores, the decision layer aggregates scores by event, applies per-event thresholds, suppresses silence/unknown labels, and emits JSON.

**Tech Stack:** Python 3.10+, pytest, numpy, soundfile, librosa, PyYAML, laion-clap, optional sounddevice.

---

### Task 1: Decision Logic

**Files:**
- Create: `src/audio_event_poc/decision.py`
- Test: `tests/test_decision.py`

- [x] **Step 1: Write failing tests**

Tests cover prompt aggregation, threshold gating, silence/unknown suppression, and JSON result shape.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_decision -v`
Expected: FAIL because `audio_event_poc.decision` does not exist yet.

- [ ] **Step 3: Implement decision logic**

Create dataclasses for event definitions and detection outputs. Implement prompt score aggregation and event selection with per-class thresholds.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_decision -v`
Expected: PASS.

### Task 2: Configuration Loader

**Files:**
- Create: `src/audio_event_poc/config.py`
- Create: `config/events.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Tests load the YAML config and verify target/suppress event definitions are parsed into `EventDefinition`.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_config -v`
Expected: FAIL because `audio_event_poc.config` does not exist yet.

- [ ] **Step 3: Implement config loader**

Read YAML with PyYAML, validate required fields, and return event definitions.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_config -v`
Expected: PASS.

### Task 3: CLAP Classification Entry Point

**Files:**
- Create: `src/audio_event_poc/audio.py`
- Create: `src/audio_event_poc/clap_backend.py`
- Create: `scripts/classify_audio.py`

- [ ] **Step 1: Add CLI integration**

Load wav audio, resample to 48 kHz for CLAP, compute prompt embeddings, aggregate scores, apply thresholds, and print JSON.

- [ ] **Step 2: Smoke-test CLI help**

Run: `python scripts/classify_audio.py --help`
Expected: argparse help text with `--audio` and `--config`.

### Task 4: Recording and Evaluation Tools

**Files:**
- Create: `scripts/record_sample.py`
- Create: `scripts/run_once.py`
- Create: `scripts/evaluate_thresholds.py`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `pyproject.toml`

- [ ] **Step 1: Add recording script**

Support `sounddevice` when available and document Linux `arecord` fallback.

- [ ] **Step 2: Add threshold evaluation script**

Read a CSV manifest of labeled wav files, run classifier, and print per-label precision/recall counts.

- [ ] **Step 3: Document setup and acceptance criteria**

README includes environment setup, recording commands, expected directory layout, and first-stage acceptance targets.

### Self-Review

- Spec coverage: The plan covers the three target sounds, suppress labels, per-class thresholds, JSON output, and local evaluation.
- Placeholder scan: No placeholder requirements remain.
- Type consistency: All modules use `EventDefinition` and JSON-serializable dictionaries as the cross-module contract.
