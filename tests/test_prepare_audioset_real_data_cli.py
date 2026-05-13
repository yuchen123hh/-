import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareAudioSetRealDataCliTests(unittest.TestCase):
    def test_build_candidates_cli_writes_real_source_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            class_labels = base / "class_labels_indices.csv"
            segments = base / "segments.csv"
            output = base / "candidates.csv"
            class_labels.write_text(
                "index,mid,display_name\n"
                "0,/m/cough,Cough\n"
                "1,/m/alarm,\"Smoke detector, smoke alarm\"\n",
                encoding="utf-8",
            )
            segments.write_text(
                "# YTID,start_seconds,end_seconds,positive_labels\n"
                "abc,1.000,11.000,\"/m/cough\"\n"
                "alarm,2.000,12.000,\"/m/alarm\"\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_audioset_real_data.py"),
                    "build-candidates",
                    "--class-labels-csv",
                    str(class_labels),
                    "--segments-csv",
                    str(segments),
                    "--output",
                    str(output),
                    "--limit",
                    "cough=1",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            rows = self._read_rows(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({row["source_type"] for row in rows}, {"audioset"})
        self.assertEqual({row["label"] for row in rows}, {"cough", "smoke_alarm"})

    def test_download_audio_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            candidates = base / "candidates.csv"
            manifest = base / "downloaded_manifest.csv"
            failures = base / "failures.jsonl"
            self._write_candidates(candidates)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_audioset_real_data.py"),
                    "download-audio",
                    "--candidates-csv",
                    str(candidates),
                    "--output-dir",
                    str(base / "audio"),
                    "--manifest",
                    str(manifest),
                    "--failures",
                    str(failures),
                    "--dry-run",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=20,
            )

            rows = self._read_rows(manifest)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]["source_type"], "audioset")
        self.assertEqual(rows[0]["download_status"], "dry_run")

    def _write_candidates(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_id",
                    "ytid",
                    "start_seconds",
                    "end_seconds",
                    "label",
                    "audioset_labels",
                    "positive_mids",
                    "source_type",
                    "source_id",
                    "youtube_url",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "clip_id": "abc_000001000_000011000",
                    "ytid": "abc",
                    "start_seconds": "1.000",
                    "end_seconds": "11.000",
                    "label": "cough",
                    "audioset_labels": "Cough",
                    "positive_mids": "/m/cough",
                    "source_type": "audioset",
                    "source_id": "audioset:abc:1.000:11.000",
                    "youtube_url": "https://www.youtube.com/watch?v=abc",
                }
            )

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
