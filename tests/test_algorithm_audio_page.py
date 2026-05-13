import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.algorithm_audio_test_page import analyze_algorithmic_audio


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


class AlgorithmAudioPageTests(unittest.TestCase):
    def analyze(self, audio: np.ndarray) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.wav"
            write_wav(path, audio)
            return analyze_algorithmic_audio(path)

    def test_detects_silence(self):
        result = self.analyze(np.zeros(16000, dtype=np.float32))

        self.assertEqual(result["top_label"], "silence")
        self.assertFalse(result["heard"])

    def test_detects_single_impulse(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[5000:5040] = np.hanning(40) * 0.95

        result = self.analyze(audio)

        self.assertEqual(result["top_label"], "impulse_clap_or_knock")

    def test_detects_repeated_clicks(self):
        audio = np.zeros(32000, dtype=np.float32)
        pulse = np.hanning(30).astype(np.float32) * 0.65
        for offset in range(3000, 23000, 1800):
            audio[offset : offset + pulse.size] += pulse

        result = self.analyze(audio)

        self.assertEqual(result["top_label"], "repeated_clicks_or_steps")

    def test_detects_tonal_sound(self):
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = 0.35 * np.sin(2 * np.pi * 880 * t)

        result = self.analyze(audio.astype(np.float32))

        self.assertEqual(result["top_label"], "tonal_or_music_or_alarm")

    def test_detects_low_frequency_rumble(self):
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = 0.45 * np.sin(2 * np.pi * 90 * t)

        result = self.analyze(audio.astype(np.float32))

        self.assertEqual(result["top_label"], "low_frequency_rumble")


if __name__ == "__main__":
    unittest.main()
