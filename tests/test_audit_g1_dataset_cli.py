import csv
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class AuditG1DatasetCliTests(unittest.TestCase):
    def test_cli_writes_report_and_returns_success_for_real_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_wav(base / "cough_train.wav")
            self._write_wav(base / "cough_val.wav")
            train_manifest = base / "train_manifest.csv"
            val_manifest = base / "val_manifest.csv"
            self._write_manifest(train_manifest, "cough_train.wav")
            self._write_manifest(val_manifest, "cough_val.wav")
            report_path = base / "audit.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_g1_dataset.py"),
                    "--train-manifest",
                    str(train_manifest),
                    "--val-manifest",
                    str(val_manifest),
                    "--output",
                    str(report_path),
                    "--min-per-label",
                    "2",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(report_path.exists())
            self.assertIn('"ready_for_training": true', result.stdout)

    def test_cli_returns_failure_for_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            train_manifest = base / "train_manifest.csv"
            val_manifest = base / "val_manifest.csv"
            self._write_manifest(train_manifest, "missing.wav")
            self._write_manifest(val_manifest, "missing.wav")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_g1_dataset.py"),
                    "--train-manifest",
                    str(train_manifest),
                    "--val-manifest",
                    str(val_manifest),
                    "--allow-missing-provenance",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("audio file not found", result.stdout)

    def _write_manifest(self, path: Path, audio_path: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "source_type", "source_id"])
            writer.writeheader()
            writer.writerow(
                {
                    "audio_path": audio_path,
                    "label": "cough",
                    "source_type": "audioset",
                    "source_id": f"id-{path.stem}",
                }
            )

    def _write_wav(self, path: Path, *, duration_s: float = 1.0, sample_rate: int = 16_000) -> None:
        audio = np.zeros(int(duration_s * sample_rate), dtype=np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(audio.tobytes())


if __name__ == "__main__":
    unittest.main()
