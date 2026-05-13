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
audio_path,label
```

## Round 1 Command

```bash
python training/efficientat/train_g1_abnormal.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --pretrained checkpoints/dymn10_as_pretrained.pt \
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
