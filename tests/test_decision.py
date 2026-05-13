import unittest

from audio_event_poc.decision import (
    EventDefinition,
    aggregate_prompt_scores,
    build_result,
    select_detected_events,
)


class DecisionTests(unittest.TestCase):
    def test_aggregate_prompt_scores_uses_best_prompt_per_event(self):
        events = [
            EventDefinition(
                key="knock",
                label="敲门",
                category="interaction",
                threshold=0.46,
                prompts=["knock_direct", "knock_variant"],
            ),
            EventDefinition(
                key="cough",
                label="咳嗽",
                category="human",
                threshold=0.44,
                prompts=["cough_direct"],
            ),
        ]
        prompt_scores = {
            "knock_direct": 0.41,
            "knock_variant": 0.52,
            "cough_direct": 0.37,
        }

        scores = aggregate_prompt_scores(prompt_scores, events)

        self.assertEqual(scores, {"knock": 0.52, "cough": 0.37})

    def test_select_detected_events_returns_target_above_threshold(self):
        events = [
            EventDefinition("knock", "敲门", "interaction", 0.46, ["knock"]),
            EventDefinition("other", "其他声音", "suppress", 0.0, ["other"], suppress=True),
        ]

        detections = select_detected_events(
            {"knock": 0.51, "other": 0.32},
            events,
            timestamp=1714368000.123,
        )

        self.assertEqual(
            detections,
            [
                {
                    "event": "敲门",
                    "event_key": "knock",
                    "category": "interaction",
                    "score": 0.51,
                    "threshold": 0.46,
                    "timestamp": 1714368000.123,
                }
            ],
        )

    def test_select_detected_events_returns_empty_when_score_below_threshold(self):
        events = [EventDefinition("cough", "咳嗽", "human", 0.44, ["cough"])]

        detections = select_detected_events(
            {"cough": 0.41},
            events,
            timestamp=1714368000.0,
        )

        self.assertEqual(detections, [])

    def test_select_detected_events_suppresses_silence_and_other_labels(self):
        events = [
            EventDefinition("clap", "拍手", "interaction", 0.48, ["clap"]),
            EventDefinition("silence", "安静办公室", "suppress", 0.0, ["silence"], suppress=True),
            EventDefinition("other", "其他声音", "suppress", 0.0, ["other"], suppress=True),
        ]

        detections = select_detected_events(
            {"clap": 0.47, "silence": 0.76, "other": 0.63},
            events,
            timestamp=1714368000.0,
        )

        self.assertEqual(detections, [])

    def test_select_detected_events_suppresses_when_suppress_label_scores_higher(self):
        events = [
            EventDefinition("clap", "拍手", "interaction", 0.48, ["clap"]),
            EventDefinition("silence", "安静办公室", "suppress", 0.0, ["silence"], suppress=True),
        ]

        detections = select_detected_events(
            {"clap": 0.52, "silence": 0.76},
            events,
            timestamp=1714368000.0,
        )

        self.assertEqual(detections, [])

    def test_build_result_contains_scores_and_detected_events(self):
        events = [
            EventDefinition("clap", "拍手", "interaction", 0.48, ["clap"]),
            EventDefinition("silence", "安静办公室", "suppress", 0.0, ["silence"], suppress=True),
        ]
        event_scores = {"clap": 0.66, "silence": 0.21}
        detections = select_detected_events(event_scores, events, timestamp=1714368000.0)

        result = build_result(
            timestamp=1714368000.0,
            audio_path="samples/clap_001.wav",
            event_scores=event_scores,
            detected_events=detections,
        )

        self.assertEqual(result["timestamp"], 1714368000.0)
        self.assertEqual(result["audio_path"], "samples/clap_001.wav")
        self.assertEqual(result["detected_events"][0]["event_key"], "clap")
        self.assertEqual(result["scores"]["clap"], 0.66)
        self.assertEqual(result["scores"]["silence"], 0.21)


if __name__ == "__main__":
    unittest.main()
