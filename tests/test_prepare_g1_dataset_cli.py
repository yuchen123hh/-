import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareG1DatasetCliTests(unittest.TestCase):
    def test_build_manifest_from_metadata_splits_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata = base / "metadata.csv"
            audio_root = base / "audio"
            audio_root.mkdir()
            self._write_metadata(
                metadata,
                [
                    ("cough-a", "cough-a.wav", "Cough"),
                    ("cough-b", "cough-b.wav", "Cough"),
                    ("alarm-a", "alarm-a.wav", "Smoke detector, smoke alarm"),
                    ("bg-a", "bg-a.wav", "Domestic sounds, home sounds"),
                ],
            )
            output_dir = base / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_g1_dataset.py"),
                    "build-manifest",
                    "--metadata-csv",
                    str(metadata),
                    "--audio-root",
                    str(audio_root),
                    "--output-dir",
                    str(output_dir),
                    "--val-ratio",
                    "0.25",
                    "--seed",
                    "7",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            train_rows = self._read_rows(output_dir / "train_manifest.csv")
            val_rows = self._read_rows(output_dir / "val_manifest.csv")
            all_labels = {row["label"] for row in train_rows + val_rows}
            self.assertEqual(all_labels, {"cough", "smoke_alarm", "background"})
            self.assertTrue(all(row["audio_path"].endswith(".wav") for row in train_rows + val_rows))
            self.assertEqual({row["source_type"] for row in train_rows + val_rows}, {"audioset"})
            self.assertTrue(all(row["source_id"] for row in train_rows + val_rows))

    def test_write_g1_collection_plan_contains_required_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "g1_collection_plan.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_g1_dataset.py"),
                    "g1-plan",
                    "--output",
                    str(output),
                    "--per-event",
                    "2",
                    "--background",
                    "3",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self._read_rows(output)
            counts = {}
            for row in rows:
                counts[row["label"]] = counts.get(row["label"], 0) + 1
            self.assertEqual(counts["distress_call"], 2)
            self.assertEqual(counts["glass_break"], 2)
            self.assertEqual(counts["knock"], 2)
            self.assertEqual(counts["cough"], 2)
            self.assertEqual(counts["smoke_alarm"], 2)
            self.assertEqual(counts["background"], 3)
            self.assertIn("unitree_g1_mic", rows[0]["source"])

    def _write_metadata(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip_id", "audio_path", "audioset_labels", "source_type"])
            writer.writeheader()
            for clip_id, audio_path, labels in rows:
                writer.writerow(
                    {
                        "clip_id": clip_id,
                        "audio_path": audio_path,
                        "audioset_labels": labels,
                        "source_type": "audioset",
                    }
                )

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
