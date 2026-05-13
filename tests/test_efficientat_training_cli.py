import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EfficientATTrainingCliTests(unittest.TestCase):
    def test_dry_run_writes_training_plan_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            efficientat_root = base / "EfficientAT"
            (efficientat_root / "models" / "dymn").mkdir(parents=True)
            (efficientat_root / "models" / "dymn" / "model.py").write_text("", encoding="utf-8")
            train_manifest = base / "train_manifest.csv"
            val_manifest = base / "val_manifest.csv"
            self._write_manifest(train_manifest, [("a.wav", "cough", "audioset")])
            self._write_manifest(val_manifest, [("b.wav", "smoke_alarm", "audioset")])
            output_dir = base / "run"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "training" / "efficientat" / "train_g1_abnormal.py"),
                    "--train-manifest",
                    str(train_manifest),
                    "--val-manifest",
                    str(val_manifest),
                    "--efficientat-root",
                    str(efficientat_root),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            plan = (output_dir / "training_plan.json").read_text(encoding="utf-8")
            self.assertIn("efficientat_dymn10_as", plan)
            self.assertIn("smoke_alarm", plan)

    def _write_manifest(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "source_type"])
            writer.writeheader()
            for audio_path, label, source_type in rows:
                writer.writerow({"audio_path": audio_path, "label": label, "source_type": source_type})


if __name__ == "__main__":
    unittest.main()
