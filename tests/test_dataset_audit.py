import csv
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from audio_event_poc.dataset_audit import audit_manifests


class DatasetAuditTests(unittest.TestCase):
    def test_audit_accepts_real_world_audio_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_wav(base / "cough_train.wav", duration_s=1.0)
            self._write_wav(base / "cough_val.wav", duration_s=1.0)
            train_manifest = base / "train_manifest.csv"
            val_manifest = base / "val_manifest.csv"
            self._write_manifest(train_manifest, [("cough_train.wav", "cough", "audioset", "ytid-cough")])
            self._write_manifest(val_manifest, [("cough_val.wav", "cough", "audioset", "ytid-cough-val")])

            report = audit_manifests(
                [("train", train_manifest), ("val", val_manifest)],
                min_duration_s=0.5,
                min_per_label=2,
            )

        self.assertTrue(report["ready_for_training"], report["failures"])
        self.assertEqual(report["aggregate"]["label_counts"]["cough"], 2)
        self.assertEqual(report["aggregate"]["source_type_counts"]["audioset"], 2)

    def test_audit_rejects_non_real_world_sources_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            audio = base / "cough.wav"
            self._write_wav(audio, duration_s=1.0)
            manifest = base / "manifest.csv"
            self._write_manifest(manifest, [("cough.wav", "cough", "synthetic", "")])

            report = audit_manifests([("train", manifest)], require_provenance=True)

        self.assertFalse(report["ready_for_training"])
        messages = "\n".join(str(item["message"]) for item in report["failures"])
        self.assertIn("non-real-world source_type", messages)
        self.assertIn("missing source_id", messages)

    def test_audit_rejects_missing_and_too_short_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            short_audio = base / "short.wav"
            self._write_wav(short_audio, duration_s=0.1)
            manifest = base / "manifest.csv"
            self._write_manifest(
                manifest,
                [
                    ("short.wav", "cough", "audioset", "short-id"),
                    ("missing.wav", "knock", "g1_field", "g1-001"),
                ],
            )

            report = audit_manifests([("train", manifest)], min_duration_s=0.5)

        self.assertFalse(report["ready_for_training"])
        kinds = {item["kind"] for item in report["failures"]}
        self.assertIn("audio_too_short", kinds)
        self.assertIn("missing_audio_file", kinds)

    def _write_manifest(self, path: Path, rows: list[tuple[str, str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "source_type", "source_id"])
            writer.writeheader()
            for audio_path, label, source_type, source_id in rows:
                writer.writerow(
                    {
                        "audio_path": audio_path,
                        "label": label,
                        "source_type": source_type,
                        "source_id": source_id,
                    }
                )

    def _write_wav(self, path: Path, *, duration_s: float, sample_rate: int = 16_000) -> None:
        sample_count = int(duration_s * sample_rate)
        audio = np.zeros(sample_count, dtype=np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(audio.tobytes())


if __name__ == "__main__":
    unittest.main()
