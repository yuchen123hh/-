# EfficientAT DyMN10-AS Cloud Training

This folder is the cloud GPU handoff for the fixed training plan:

- GPU: RTX 4090 24GB
- Budget: RMB 50 for v0, then RMB 50 after G1 field sampling
- Model: EfficientAT DyMN10-AS AudioSet pretrained checkpoint
- Classes: `distress_call`, `glass_break`, `knock`, `cough`, `smoke_alarm`, `background`

Do not start the cloud GPU until the user approves paying for compute.

## Cloud Environment

Use a PyTorch CUDA image already supported by the rental platform, preferably:

```bash
python --version
python -m pip install -U pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Clone or install EfficientAT in the cloud workspace according to its upstream README. The local scripts intentionally fail with a clear error if the upstream package or checkpoint is missing.

## Expected Inputs

```text
data/g1_audio/train_manifest.csv
data/g1_audio/val_manifest.csv
checkpoints/dymn10_as_pretrained.pt
config/g1_abnormal_events.yaml
```

Manifest columns:

```text
audio_path,label,source_type
```

`source_type` is a hard training gate. Only real-world sources are accepted:

- `audioset`: real AudioSet clips selected from the AudioSet metadata/audio files.
- `g1_field`: real recordings captured from the Unitree G1 microphone in the target environment.

Synthetic, generated, mock, and smoke-test samples are rejected by `cloud_preflight.py` and `train_g1_abnormal.py`.

For paid training, also include `source_id` or `clip_id` in each row so every training item can be traced back to its AudioSet clip or G1 field recording.

## Round 1 Command

Audit the actual audio files before preflight:

```bash
python scripts/audit_g1_dataset.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --output reports/g1_audio_dataset_audit.json \
  --min-duration-s 0.5 \
  --max-duration-s 15 \
  --min-per-label 100 \
  --require-all-labels
```

Only continue when `ready_for_training` is `true`.

Run preflight before any paid training:

```bash
python training/efficientat/cloud_preflight.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --efficientat-root /workspace/EfficientAT \
  --min-per-label 100 \
  --require-all-labels
```

Only continue if `ready_for_paid_training` is `true` and the GPU name is an RTX 4090-class device. Stop and ask before starting the paid instance or training run.

```bash
python training/efficientat/train_g1_abnormal.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --efficientat-root /workspace/EfficientAT \
  --output-dir runs/g1_audio_v0 \
  --epochs 20 \
  --batch-size 48 \
  --freeze-backbone-epochs 3
```

## Export

```bash
python training/efficientat/export_onnx.py \
  --checkpoint runs/g1_audio_v0/best.pt \
  --output models/efficientat_g1_audio_v0.onnx
```

## Outputs To Bring Back To G1 Project

```text
models/efficientat_g1_audio_v0.onnx
runs/g1_audio_v0/best.pt
runs/g1_audio_v0/metrics.json
runs/g1_audio_v0/thresholds.yaml
```
