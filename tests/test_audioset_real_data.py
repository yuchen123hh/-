import csv
import tempfile
import unittest
from pathlib import Path

from audio_event_poc.audioset_real_data import (
    AudioSetCandidate,
    build_candidates,
    download_candidates,
    extract_audio_segment,
    read_candidate_csv,
    resolve_youtube_audio_url,
    write_candidate_csv,
)


class AudioSetRealDataTests(unittest.TestCase):
    def test_build_candidates_from_official_csv_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            class_labels = base / "class_labels_indices.csv"
            segments = base / "balanced_train_segments.csv"
            class_labels.write_text(
                "index,mid,display_name\n"
                '0,/m/cough,Cough\n'
                '1,/m/home,"Domestic sounds, home sounds"\n'
                "2,/m/glass,Glass\n",
                encoding="utf-8",
            )
            segments.write_text(
                "# Segments csv created by AudioSet\n"
                "# YTID,start_seconds,end_seconds,positive_labels\n"
                'abc123,30.000,40.000,"/m/cough,/m/home"\n'
                'glass1,10.000,20.000,"/m/glass"\n'
                'skip1,0.000,10.000,"/m/unknown"\n',
                encoding="utf-8",
            )

            candidates = build_candidates([segments], class_labels, per_label_limit={"cough": 1})

        self.assertEqual([candidate.label for candidate in candidates], ["cough", "glass_break"])
        self.assertEqual(candidates[0].source_id, "audioset:abc123:30.000:40.000")
        self.assertIn("Cough", candidates[0].audioset_labels)

    def test_candidate_csv_round_trip_preserves_real_source_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.csv"
            candidates = [
                AudioSetCandidate(
                    clip_id="abc_000030000_000040000",
                    ytid="abc",
                    start_seconds=30.0,
                    end_seconds=40.0,
                    label="cough",
                    audioset_labels=["Cough"],
                    positive_mids=["/m/cough"],
                )
            ]

            write_candidate_csv(candidates, path)
            loaded = read_candidate_csv(path)

        self.assertEqual(loaded[0].ytid, "abc")
        self.assertEqual(loaded[0].label, "cough")
        self.assertEqual(loaded[0].source_id, "audioset:abc:30.000:40.000")

    def test_download_candidates_dry_run_writes_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            candidate = AudioSetCandidate(
                clip_id="abc_000030000_000040000",
                ytid="abc",
                start_seconds=30.0,
                end_seconds=40.0,
                label="cough",
                audioset_labels=["Cough"],
                positive_mids=["/m/cough"],
            )

            report = download_candidates(
                [candidate],
                output_dir=base / "audio",
                manifest_path=base / "downloaded_manifest.csv",
                failures_path=base / "failures.jsonl",
                dry_run=True,
            )

            rows = self._read_rows(base / "downloaded_manifest.csv")

        self.assertEqual(report["requested"], 1)
        self.assertEqual(report["downloaded"], 1)
        self.assertEqual(rows[0]["source_type"], "audioset")
        self.assertEqual(rows[0]["source_id"], "audioset:abc:30.000:40.000")
        self.assertTrue(rows[0]["audio_path"].endswith("abc_000030000_000040000.wav"))

    def test_resolve_youtube_audio_url_uses_yt_dlp(self):
        commands = []

        def runner(command):
            commands.append(command)
            return _completed(stdout="https://media.example/audio.webm\n")

        media_url = resolve_youtube_audio_url("https://www.youtube.com/watch?v=abc", runner=runner)

        self.assertEqual(media_url, "https://media.example/audio.webm")
        self.assertIn("yt_dlp", commands[0])
        self.assertIn("--get-url", commands[0])

    def test_extract_audio_segment_uses_ffmpeg_crop_command(self):
        commands = []

        def runner(command):
            commands.append(command)
            return _completed()

        extract_audio_segment(
            media_url="https://media.example/audio.webm",
            output_path=Path("out.wav"),
            start_seconds=30.0,
            duration_seconds=10.0,
            sample_rate=32_000,
            ffmpeg_path="ffmpeg",
            runner=runner,
        )

        command = commands[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-ss", command)
        self.assertIn("30.000", command)
        self.assertIn("-t", command)
        self.assertIn("10.000", command)
        self.assertIn("32000", command)

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    class Completed:
        def __init__(self) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    return Completed()


if __name__ == "__main__":
    unittest.main()
