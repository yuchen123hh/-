import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.g1_realtime_audio_service import list_input_devices, print_input_devices


class G1RealtimeServiceTests(unittest.TestCase):
    def test_list_input_devices_filters_output_capable_devices(self):
        devices = [
            {"name": "speaker", "max_input_channels": 0, "hostapi": 0, "default_samplerate": 48000},
            {"name": "mic0", "max_input_channels": 1, "hostapi": 0, "default_samplerate": 16000},
            {"name": "mic1", "max_input_channels": 2, "hostapi": 1, "default_samplerate": 48000},
        ]
        hostapis = [{"name": "ALSA"}, {"name": "PulseAudio"}]
        fake_sd = SimpleNamespace(
            query_devices=lambda: devices,
            query_hostapis=lambda: hostapis,
        )

        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            rows = list_input_devices()

        self.assertEqual([row["name"] for row in rows], ["mic0", "mic1"])
        self.assertEqual(rows[0]["hostapi"], "ALSA")
        self.assertEqual(rows[1]["hostapi"], "PulseAudio")

    def test_print_input_devices_emits_json(self):
        with patch(
            "scripts.g1_realtime_audio_service.list_input_devices",
            return_value=[{"index": 1, "name": "mic0", "hostapi": "ALSA", "max_input_channels": 1, "default_samplerate": 16000.0}],
        ):
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = print_input_devices()

        payload = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["input_devices"][0]["name"], "mic0")


if __name__ == "__main__":
    unittest.main()
