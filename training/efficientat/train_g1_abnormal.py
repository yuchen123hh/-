from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from efficientat_adapter import CLASSES, build_dymn10_model, build_mel


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(path: Path) -> Counter[str]:
    rows = read_manifest(path)
    if not rows:
        raise RuntimeError(f"manifest is empty: {path}")
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=2):
        audio_path = row.get("audio_path", "").strip()
        label = row.get("label", "").strip()
        if not audio_path:
            raise RuntimeError(f"{path}:{index} missing audio_path")
        if label not in CLASSES:
            raise RuntimeError(f"{path}:{index} unsupported label '{label}'")
        counts[label] += 1
    return counts


class ManifestAudioDataset:
    def __init__(self, manifest_path: Path, *, sample_rate: int, clip_s: float) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required for training.") from exc
        self.torch = torch
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(sample_rate * clip_s)
        self.root = manifest_path.parent
        self.rows = read_manifest(manifest_path)
        self.label_to_index = {label: index for index, label in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (self.root / audio_path).resolve()
        waveform = self._load_audio(audio_path)
        target = self.torch.zeros(len(CLASSES), dtype=self.torch.float32)
        target[self.label_to_index[row["label"]]] = 1.0
        return waveform, target

    def _load_audio(self, path: Path):
        import librosa
        import numpy as np

        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < self.clip_samples:
            audio = np.pad(audio, (0, self.clip_samples - audio.size))
        elif audio.size > self.clip_samples:
            max_start = audio.size - self.clip_samples
            start = random.randint(0, max_start)
            audio = audio[start : start + self.clip_samples]
        return self.torch.from_numpy(audio.astype("float32"))


def freeze_backbone(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier")


def unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def train_one_epoch(*, model, mel, loader, optimizer, device, amp: bool) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    mel.train()
    losses: list[float] = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for waveform, target in loader:
        waveform = waveform.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            spec = mel(waveform)
            logits, _ = model(spec.unsqueeze(1))
            loss = F.binary_cross_entropy_with_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def evaluate(*, model, mel, loader, device) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn import metrics

    model.eval()
    mel.eval()
    targets: list[Any] = []
    outputs: list[Any] = []
    losses: list[float] = []
    with torch.no_grad():
        for waveform, target in loader:
            waveform = waveform.to(device)
            target = target.to(device)
            spec = mel(waveform)
            logits, _ = model(spec.unsqueeze(1))
            loss = F.binary_cross_entropy_with_logits(logits, target)
            targets.append(target.cpu().numpy())
            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
            losses.append(float(loss.cpu()))
    y_true = np.concatenate(targets, axis=0)
    y_score = np.concatenate(outputs, axis=0)
    try:
        macro_ap = float(metrics.average_precision_score(y_true, y_score, average="macro"))
    except ValueError:
        macro_ap = 0.0
    y_pred = np.zeros_like(y_score)
    y_pred[np.arange(y_score.shape[0]), np.argmax(y_score, axis=1)] = 1.0
    report = metrics.classification_report(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "val_loss": sum(losses) / max(1, len(losses)),
        "macro_ap": macro_ap,
        "classification_report": report,
    }


def write_thresholds(output_path: Path, *, default_threshold: float = 0.62) -> None:
    thresholds = {
        "distress_call": 0.62,
        "glass_break": 0.68,
        "knock": 0.58,
        "cough": 0.62,
        "smoke_alarm": 0.64,
    }
    text = "\n".join([f"{key}: {value:.2f}" for key, value in thresholds.items()])
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EfficientAT DyMN10-AS for G1 abnormal audio events.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--efficientat-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional local checkpoint; omitted uses dymn10_as auto-download.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-s", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print the cloud training plan.")
    args = parser.parse_args()

    train_counts = validate_manifest(args.train_manifest)
    val_counts = validate_manifest(args.val_manifest)
    if args.pretrained is not None and not args.pretrained.exists():
        raise RuntimeError(f"pretrained checkpoint not found: {args.pretrained}")
    if not (args.efficientat_root / "models" / "dymn" / "model.py").exists():
        raise RuntimeError(f"EfficientAT root is invalid: {args.efficientat_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": "efficientat_dymn10_as",
        "classes": CLASSES,
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "efficientat_root": str(args.efficientat_root),
        "pretrained": str(args.pretrained) if args.pretrained else "dymn10_as_auto_download",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "lr": args.lr,
        "sample_rate": args.sample_rate,
        "clip_s": args.clip_s,
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
    }
    (args.output_dir / "training_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RTX 4090 cloud training requires CUDA. Use --dry-run locally.")
    train_set = ManifestAudioDataset(args.train_manifest, sample_rate=args.sample_rate, clip_s=args.clip_s)
    val_set = ManifestAudioDataset(args.val_manifest, sample_rate=args.sample_rate, clip_s=args.clip_s)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_dymn10_model(
        efficientat_root=args.efficientat_root,
        num_classes=len(CLASSES),
        pretrained_name=None if args.pretrained else "dymn10_as",
        pretrained_path=args.pretrained,
    ).to(device)
    mel = build_mel(
        efficientat_root=args.efficientat_root,
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
    ).to(device)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_score = -1.0
    amp = True
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_backbone_epochs:
            freeze_backbone(model)
        elif epoch == args.freeze_backbone_epochs + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.25, weight_decay=1e-4)
        train_loss = train_one_epoch(model=model, mel=mel, loader=train_loader, optimizer=optimizer, device=device, amp=amp)
        val_metrics = evaluate(model=model, mel=mel, loader=val_loader, device=device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "classification_report"}, ensure_ascii=False), flush=True)
        score = float(val_metrics["macro_ap"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "plan": plan,
                    "best_macro_ap": best_score,
                },
                args.output_dir / "best.pt",
            )
            write_thresholds(args.output_dir / "thresholds.yaml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
