import unittest

from audio_event_poc.audioset_manifest import (
    AUDIOSET_TO_G1_LABELS,
    AudioSetClip,
    build_balanced_manifest,
    map_audioset_labels,
)


class AudioSetManifestTests(unittest.TestCase):
    def test_label_mapping_covers_required_classes(self):
        self.assertIn("Screaming", AUDIOSET_TO_G1_LABELS["distress_call"])
        self.assertIn("Glass", AUDIOSET_TO_G1_LABELS["glass_break"])
        self.assertIn("Knock", AUDIOSET_TO_G1_LABELS["knock"])
        self.assertIn("Cough", AUDIOSET_TO_G1_LABELS["cough"])
        self.assertIn("Smoke detector, smoke alarm", AUDIOSET_TO_G1_LABELS["smoke_alarm"])

    def test_map_audioset_labels_returns_matching_g1_label(self):
        self.assertEqual(map_audioset_labels(["Speech", "Cough"]), "cough")
        self.assertEqual(map_audioset_labels(["Smoke detector, smoke alarm"]), "smoke_alarm")
        self.assertEqual(map_audioset_labels(["Domestic sounds, home sounds"]), "background")

    def test_build_balanced_manifest_limits_each_label(self):
        clips = [
            AudioSetClip("a", "a.wav", ["Cough"]),
            AudioSetClip("b", "b.wav", ["Cough"]),
            AudioSetClip("c", "c.wav", ["Knock"]),
            AudioSetClip("d", "d.wav", ["Glass"]),
            AudioSetClip("e", "e.wav", ["Domestic sounds, home sounds"]),
            AudioSetClip("f", "f.wav", ["Inside, small room"]),
        ]

        rows = build_balanced_manifest(clips, per_label_limit={"cough": 1, "background": 1})

        self.assertEqual(
            rows,
            [
                {
                    "clip_id": "a",
                    "audio_path": "a.wav",
                    "label": "cough",
                    "audioset_labels": "Cough",
                    "source_type": "audioset",
                },
                {
                    "clip_id": "c",
                    "audio_path": "c.wav",
                    "label": "knock",
                    "audioset_labels": "Knock",
                    "source_type": "audioset",
                },
                {
                    "clip_id": "d",
                    "audio_path": "d.wav",
                    "label": "glass_break",
                    "audioset_labels": "Glass",
                    "source_type": "audioset",
                },
                {
                    "clip_id": "e",
                    "audio_path": "e.wav",
                    "label": "background",
                    "audioset_labels": "Domestic sounds, home sounds",
                    "source_type": "audioset",
                },
            ],
        )

    def test_target_labels_win_over_background(self):
        label = map_audioset_labels(["Domestic sounds, home sounds", "Glass"])

        self.assertEqual(label, "glass_break")


if __name__ == "__main__":
    unittest.main()
