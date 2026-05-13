import json
import unittest

from audio_event_poc.event_contract import (
    G1_EVENT_DEFINITIONS,
    build_audio_event,
    definition_for,
)


class EventContractTests(unittest.TestCase):
    def test_build_audio_event_uses_canonical_shape(self):
        event = build_audio_event(
            event_key="smoke_alarm",
            confidence=0.93,
            threshold=0.72,
            start_time=12.5,
            end_time=14.5,
            detected_at="2026-05-13T12:00:00+00:00",
            source="unitree_g1_mic",
            model="efficientat_dymn10_as_v0",
            scores={
                "distress_call": 0.12,
                "glass_break": 0.03,
                "knock": 0.05,
                "cough": 0.08,
                "smoke_alarm": 0.93,
                "background": 0.02,
            },
            metadata={"window_s": 2.0, "hop_s": 0.5},
            event_id="fixed-id",
        )

        self.assertEqual(event["type"], "audio_event")
        self.assertEqual(event["schema_version"], "1.0")
        self.assertEqual(event["event_id"], "fixed-id")
        self.assertEqual(event["event_key"], "smoke_alarm")
        self.assertEqual(event["label"], "烟雾报警器声")
        self.assertEqual(event["severity"], "critical")
        self.assertEqual(event["confidence"], 0.93)
        self.assertEqual(event["threshold"], 0.72)
        self.assertEqual(event["start_time"], 12.5)
        self.assertEqual(event["end_time"], 14.5)
        self.assertEqual(event["detected_at"], "2026-05-13T12:00:00+00:00")
        self.assertEqual(event["source"], "unitree_g1_mic")
        self.assertEqual(event["model"], "efficientat_dymn10_as_v0")
        self.assertTrue(event["action"]["notify_guardian"])
        self.assertTrue(event["action"]["trigger_alarm"])
        self.assertEqual(event["metadata"], {"window_s": 2.0, "hop_s": 0.5})
        json.dumps(event, ensure_ascii=False)

    def test_warning_events_notify_without_triggering_alarm(self):
        event = build_audio_event(
            event_key="cough",
            confidence=0.81,
            threshold=0.65,
            start_time=1.0,
            end_time=3.0,
            detected_at="2026-05-13T12:00:00+00:00",
            source="unitree_g1_mic",
            model="efficientat_dymn10_as_v0",
            scores={"cough": 0.81, "background": 0.1},
        )

        self.assertEqual(event["severity"], "warning")
        self.assertTrue(event["action"]["notify_guardian"])
        self.assertFalse(event["action"]["trigger_alarm"])

    def test_background_is_not_alertable(self):
        definition = definition_for("background")

        self.assertFalse(definition.alertable)
        self.assertFalse(definition.notify_guardian)
        self.assertFalse(definition.trigger_alarm)

    def test_rejects_unknown_event_key(self):
        with self.assertRaises(KeyError):
            definition_for("unknown_event")

    def test_rejects_confidence_outside_probability_range(self):
        with self.assertRaises(ValueError):
            build_audio_event(
                event_key="knock",
                confidence=1.4,
                threshold=0.6,
                start_time=1.0,
                end_time=2.0,
                detected_at="2026-05-13T12:00:00+00:00",
                source="unitree_g1_mic",
                model="efficientat_dymn10_as_v0",
                scores={"knock": 1.4},
            )

    def test_all_expected_g1_events_are_defined(self):
        self.assertEqual(
            set(G1_EVENT_DEFINITIONS),
            {"distress_call", "glass_break", "knock", "cough", "smoke_alarm", "background"},
        )


if __name__ == "__main__":
    unittest.main()
