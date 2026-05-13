import unittest

from scripts.g1_fake_realtime_smoke import run_fake_realtime_smoke


class G1RealtimeSmokeTests(unittest.TestCase):
    def test_fake_realtime_smoke_emits_one_smoke_alarm(self):
        result = run_fake_realtime_smoke()

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["event_key"], "smoke_alarm")
        self.assertEqual(result["events"][0]["type"], "audio_event")
        self.assertEqual(result["webhook_results"][0]["sent"], False)
        self.assertEqual(result["webhook_results"][0]["reason"], "webhook_disabled")


if __name__ == "__main__":
    unittest.main()
