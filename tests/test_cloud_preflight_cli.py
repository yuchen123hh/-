import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CloudPreflightCliTests(unittest.TestCase):
    def test_preflight_check_reports_cpu_environment_without_failing_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            efficientat_root = base / "EfficientAT"
            (efficientat_root / "models" / "dymn").mkdir(parents=True)
            (efficientat_root / "models" / "dymn" / "model.py").write_text("", encoding="utf-8")
            train_manifest = base / "train_manifest.csv"
            val_manifest = base / "val_manifest.csv"
            self._write_manifest(train_manifest)
            self._write_manifest(val_manifest)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "training" / "efficientat" / "cloud_preflight.py"),
                    "--train-manifest",
                    str(train_manifest),
                    "--val-manifest",
                    str(val_manifest),
                    "--efficientat-root",
                    str(efficientat_root),
                    "--allow-cpu",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cuda_available", result.stdout)
            self.assertIn("train_counts", result.stdout)
            self.assertIn("ready_for_paid_training", result.stdout)

    def _write_manifest(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "label"])
            writer.writeheader()
            writer.writerow({"audio_path": "sample.wav", "label": "cough"})


if __name__ == "__main__":
    unittest.main()
