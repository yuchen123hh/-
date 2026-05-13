from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from audio_event_poc.audio import load_audio_mono
from audio_event_poc.decision import EventDefinition
from audio_event_poc.model_assets import configured_hf_endpoint, resolve_checkpoint_path


class ClapAudioClassifier:
    def __init__(self, *, enable_fusion: bool = False) -> None:
        os.environ.setdefault("HF_ENDPOINT", configured_hf_endpoint())
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import laion_clap
        except ModuleNotFoundError as exc:
            raise RuntimeError("CLAP backend requires laion-clap. Install requirements first.") from exc

        with contextlib.redirect_stdout(sys.stderr):
            self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
            checkpoint = resolve_checkpoint_path()
            self.model.load_ckpt(ckpt=str(checkpoint), verbose=False)

    def score_file(self, audio_path: str | Path, events: Sequence[EventDefinition]) -> dict[str, float]:
        prompts = [prompt for event in events for prompt in event.prompts]
        text_embeddings = self._text_embeddings(prompts)
        audio_embedding = self._audio_embedding(Path(audio_path))
        similarities = _cosine_similarity(audio_embedding, text_embeddings)
        return {
            prompt: round(float(score), 6)
            for prompt, score in zip(prompts, similarities, strict=True)
        }

    def _text_embeddings(self, prompts: list[str]) -> np.ndarray:
        try:
            embeddings = self.model.get_text_embedding(prompts, use_tensor=False)
        except TypeError:
            embeddings = self.model.get_text_embedding(prompts)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _audio_embedding(self, audio_path: Path) -> np.ndarray:
        audio = load_audio_mono(audio_path)
        audio_batch = audio.reshape(1, -1)

        if hasattr(self.model, "get_audio_embedding_from_data"):
            try:
                embedding = self.model.get_audio_embedding_from_data(x=audio_batch, use_tensor=False)
            except TypeError:
                embedding = self.model.get_audio_embedding_from_data(audio_batch)
        elif hasattr(self.model, "get_audio_embedding_from_floatx"):
            embedding = self.model.get_audio_embedding_from_floatx(audio)
        else:
            try:
                embedding = self.model.get_audio_embedding_from_filelist(x=[str(audio_path)], use_tensor=False)
            except TypeError:
                embedding = self.model.get_audio_embedding_from_filelist([str(audio_path)])

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim > 1:
            embedding = embedding[0]
        return embedding


def _cosine_similarity(audio_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    audio_norm = np.linalg.norm(audio_embedding)
    text_norms = np.linalg.norm(text_embeddings, axis=1)
    if audio_norm == 0 or np.any(text_norms == 0):
        raise RuntimeError("CLAP returned a zero embedding; cannot compute cosine similarity")
    return np.dot(text_embeddings, audio_embedding) / (text_norms * audio_norm)
