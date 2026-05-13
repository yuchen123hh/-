import csv
import tempfile
import unittest
from pathlib import Path

from training.efficientat.train_g1_abnormal import validate_manifest


class TrainingManifestValidationTests(unittest.TestCase):
    def test_validate_manifest_accepts_real_world_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "train_manifest.csv"
            self._write_manifest(manifest, [("a.wav", "cough", "audioset"), ("b.wav", "knock", "g1_field")])

            counts = validate_manifest(manifest)

        self.assertEqual(counts["cough"], 1)
        self.assertEqual(counts["knock"], 1)

    def test_validate_manifest_rejects_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "train_manifest.csv"
            self._write_manifest(manifest, [("a.wav", "cough", "synthetic")])

            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_validate_manifest_rejects_missing_source_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "train_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["audio_path", "label"])
                writer.writeheader()
                writer.writerow({"audio_path": "a.wav", "label": "cough"})

            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def _write_manifest(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "source_type"])
            writer.writeheader()
            for audio_path, label, source_type in rows:
                writer.writerow({"audio_path": audio_path, "label": label, "source_type": source_type})


if __name__ == "__main__":
    unittest.main()
