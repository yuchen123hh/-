from __future__ import annotations

import argparse
from pathlib import Path

from efficientat_adapter import CLASSES, WaveformToProbabilities, build_dymn10_model, build_mel


def main() -> int:
    parser = argparse.ArgumentParser(description="Export G1 EfficientAT checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--efficientat-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {args.checkpoint}")
    if not (args.efficientat_root / "models" / "dymn" / "model.py").exists():
        raise RuntimeError(f"EfficientAT root is invalid: {args.efficientat_root}")

    import torch

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_dymn10_model(
        efficientat_root=args.efficientat_root,
        num_classes=len(CLASSES),
        pretrained_name=None,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    mel = build_mel(
        efficientat_root=args.efficientat_root,
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
    )
    mel.eval()
    wrapper = WaveformToProbabilities(mel, model).module.eval()
    dummy = torch.zeros(1, int(args.sample_rate * args.window_s), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        input_names=["waveform"],
        output_names=["probabilities"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "probabilities": {0: "batch"}},
        opset_version=17,
    )
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
