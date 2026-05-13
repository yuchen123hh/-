from __future__ import annotations

import os
import urllib.request
from pathlib import Path


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
CLAP_REPO_PATH = "lukewys/laion_clap/resolve/main"
DEFAULT_CHECKPOINT_NAME = "630k-audioset-best.pt"


def configured_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def checkpoint_url(filename: str = DEFAULT_CHECKPOINT_NAME) -> str:
    return f"{configured_hf_endpoint()}/{CLAP_REPO_PATH}/{filename}"


def resolve_checkpoint_path(
    *,
    checkpoint_dir: Path | None = None,
    filename: str = DEFAULT_CHECKPOINT_NAME,
    download: bool = True,
) -> Path:
    target_dir = checkpoint_dir or Path(__file__).resolve().parents[2] / ".models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists():
        return target
    if not download:
        raise FileNotFoundError(f"CLAP checkpoint not found: {target}")

    url = checkpoint_url(filename)
    print(f"Downloading CLAP checkpoint from {url}")
    urllib.request.urlretrieve(url, target)
    return target
