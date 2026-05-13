import unittest
from pathlib import Path

from audio_event_poc.sample_plan import SampleRequest, build_sample_plan, manifest_rows


class SamplePlanTests(unittest.TestCase):
    def test_build_sample_plan_creates_zero_padded_paths_per_label(self):
        requests = [
            SampleRequest(label="knock", count=2, prompt="敲门"),
            SampleRequest(label="cough", count=1, prompt="咳嗽"),
        ]

        plan = build_sample_plan(Path("samples"), requests)

        self.assertEqual([item.path for item in plan], [
            Path("samples/knock/knock_001.wav"),
            Path("samples/knock/knock_002.wav"),
            Path("samples/cough/cough_001.wav"),
        ])
        self.assertEqual([item.label for item in plan], ["knock", "knock", "cough"])
        self.assertEqual(plan[0].prompt, "敲门")

    def test_manifest_rows_use_forward_slashes_for_portable_csv(self):
        requests = [SampleRequest(label="clap", count=1, prompt="拍手")]
        plan = build_sample_plan(Path("samples"), requests)

        rows = manifest_rows(plan)

        self.assertEqual(rows, [{"audio_path": "samples/clap/clap_001.wav", "label": "clap"}])


if __name__ == "__main__":
    unittest.main()
