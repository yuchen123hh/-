import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.audio_test_page import analyze_environment_audio, read_recent_events


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


class AudioTestPageTests(unittest.TestCase):
    def test_analyze_environment_audio_detects_silence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "silence.wav"
            write_wav(wav_path, np.zeros(16000, dtype=np.float32))

            result = analyze_environment_audio(wav_path)

        self.assertEqual(result["top_label"], "silence")
        self.assertFalse(result["heard"])
        self.assertEqual(result["duration_s"], 1.0)

    def test_analyze_environment_audio_detects_impulse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "impulse.wav"
            audio = np.zeros(16000, dtype=np.float32)
            audio[4000:4040] = 0.9
            write_wav(wav_path, audio)

            result = analyze_environment_audio(wav_path)

        self.assertEqual(result["top_label"], "impulse_clap_or_knock")
        self.assertTrue(result["heard"])

    def test_read_recent_events_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"
            rows = [
                {"event_id": "1", "event_time": "old"},
                {"event_id": "2", "event_time": "new"},
            ]
            log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            events = read_recent_events(log_path, limit=1)

        self.assertEqual(events, [rows[1]])


if __name__ == "__main__":
    unittest.main()
