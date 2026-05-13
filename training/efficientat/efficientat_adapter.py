from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


CLASSES = ["distress_call", "glass_break", "knock", "cough", "smoke_alarm", "background"]


def add_efficientat_root(efficientat_root: str | Path) -> Path:
    root = Path(efficientat_root).expanduser().resolve()
    if not (root / "models" / "dymn" / "model.py").exists():
        raise RuntimeError(f"EfficientAT root is invalid: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def build_dymn10_model(
    *,
    efficientat_root: str | Path,
    num_classes: int,
    pretrained_name: str | None = "dymn10_as",
    pretrained_path: str | Path | None = None,
):
    add_efficientat_root(efficientat_root)
    from models.dymn.model import get_model as get_dymn

    model = get_dymn(
        width_mult=1.0,
        pretrained_name=pretrained_name if pretrained_path is None else None,
        pretrain_final_temp=1.0,
        num_classes=num_classes,
    )
    if pretrained_path is not None:
        load_adapted_state_dict(model, Path(pretrained_path))
    return model


def build_mel(*, efficientat_root: str | Path, sample_rate: int, window_size: int, hop_size: int, n_fft: int, n_mels: int):
    add_efficientat_root(efficientat_root)
    from models.preprocess import AugmentMelSTFT

    return AugmentMelSTFT(
        n_mels=n_mels,
        sr=sample_rate,
        win_length=window_size,
        hopsize=hop_size,
        n_fft=n_fft,
        freqm=0,
        timem=0,
        fmin=0,
        fmax=None,
        fmin_aug_range=1,
        fmax_aug_range=1,
    )


def load_adapted_state_dict(model, checkpoint_path: Path) -> None:
    import torch

    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint does not contain a state dict: {checkpoint_path}")

    current = model.state_dict()
    adapted: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            adapted[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(adapted, strict=False)
    if skipped:
        print(f"Skipped incompatible checkpoint keys: {len(skipped)}")


class WaveformToProbabilities:
    def __init__(self, mel, model) -> None:
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, mel_module, model_module) -> None:
                super().__init__()
                self.mel = mel_module
                self.model = model_module

            def forward(self, waveform):
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                spec = self.mel(waveform)
                logits, _ = self.model(spec.unsqueeze(1))
                return torch.sigmoid(logits.float())

        self.module = _Wrapper(mel, model)

