import unittest

from audio_event_poc.runtime import EventSmoother, ScoreFrame


class RuntimeTests(unittest.TestCase):
    def test_requires_consecutive_hits_before_emitting_event(self):
        smoother = EventSmoother(
            thresholds={"smoke_alarm": 0.7},
            consecutive_hits={"smoke_alarm": 2},
            cooldown_s=10.0,
            model="efficientat_dymn10_as_v0",
        )

        first = smoother.update(
            ScoreFrame(timestamp=10.0, duration_s=2.0, scores={"smoke_alarm": 0.91, "background": 0.02})
        )
        second = smoother.update(
            ScoreFrame(timestamp=10.5, duration_s=2.0, scores={"smoke_alarm": 0.94, "background": 0.01})
        )

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["event_key"], "smoke_alarm")
        self.assertEqual(second[0]["start_time"], 10.0)
        self.assertEqual(second[0]["end_time"], 12.5)
        self.assertEqual(second[0]["confidence"], 0.94)

    def test_background_never_emits(self):
        smoother = EventSmoother(
            thresholds={"background": 0.1},
            consecutive_hits={"background": 1},
            cooldown_s=10.0,
            model="efficientat_dymn10_as_v0",
        )

        events = smoother.update(
            ScoreFrame(timestamp=1.0, duration_s=2.0, scores={"background": 0.99})
        )

        self.assertEqual(events, [])

    def test_cooldown_suppresses_duplicate_events(self):
        smoother = EventSmoother(
            thresholds={"glass_break": 0.65},
            consecutive_hits={"glass_break": 1},
            cooldown_s=5.0,
            model="efficientat_dymn10_as_v0",
        )

        first = smoother.update(
            ScoreFrame(timestamp=1.0, duration_s=2.0, scores={"glass_break": 0.9})
        )
        duplicate = smoother.update(
            ScoreFrame(timestamp=3.0, duration_s=2.0, scores={"glass_break": 0.92})
        )
        after_cooldown = smoother.update(
            ScoreFrame(timestamp=7.1, duration_s=2.0, scores={"glass_break": 0.93})
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(duplicate, [])
        self.assertEqual(len(after_cooldown), 1)

    def test_below_threshold_resets_consecutive_hits(self):
        smoother = EventSmoother(
            thresholds={"knock": 0.55},
            consecutive_hits={"knock": 2},
            cooldown_s=10.0,
            model="efficientat_dymn10_as_v0",
        )

        smoother.update(ScoreFrame(timestamp=1.0, duration_s=2.0, scores={"knock": 0.7}))
        smoother.update(ScoreFrame(timestamp=1.5, duration_s=2.0, scores={"knock": 0.2}))
        events = smoother.update(ScoreFrame(timestamp=2.0, duration_s=2.0, scores={"knock": 0.75}))

        self.assertEqual(events, [])

    def test_best_non_background_score_is_selected(self):
        smoother = EventSmoother(
            thresholds={"knock": 0.55, "cough": 0.6},
            consecutive_hits={"knock": 1, "cough": 1},
            cooldown_s=10.0,
            model="efficientat_dymn10_as_v0",
        )

        events = smoother.update(
            ScoreFrame(timestamp=4.0, duration_s=2.0, scores={"knock": 0.7, "cough": 0.82, "background": 0.9})
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_key"], "cough")


if __name__ == "__main__":
    unittest.main()
