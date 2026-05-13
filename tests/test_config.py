import tempfile
import textwrap
import unittest
from pathlib import Path

from audio_event_poc.config import load_event_config


class ConfigTests(unittest.TestCase):
    def test_load_event_config_returns_event_definitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "events.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    {
                      "events": [
                        {
                          "key": "knock",
                          "label": "敲门",
                          "category": "interaction",
                          "threshold": 0.46,
                          "prompts": ["a person knocking on a door"]
                        },
                        {
                          "key": "silence",
                          "label": "安静办公室",
                          "category": "suppress",
                          "threshold": 0.0,
                          "prompts": ["a quiet office"],
                          "suppress": true
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            events = load_event_config(config_path)

        self.assertEqual(events[0].key, "knock")
        self.assertEqual(events[0].label, "敲门")
        self.assertEqual(events[0].threshold, 0.46)
        self.assertEqual(events[0].prompts, ["a person knocking on a door"])
        self.assertFalse(events[0].suppress)
        self.assertEqual(events[1].key, "silence")
        self.assertTrue(events[1].suppress)

    def test_load_event_config_rejects_missing_events_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "events.yaml"
            config_path.write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_event_config(config_path)


if __name__ == "__main__":
    unittest.main()
