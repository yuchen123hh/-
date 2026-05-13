import unittest

from audio_event_poc.alarm import WebhookAlarmClient


class FakeTransport:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []

    def post_json(self, url, payload, timeout_s):
        self.calls.append({"url": url, "payload": payload, "timeout_s": timeout_s})
        return self.statuses.pop(0)


class AlarmTests(unittest.TestCase):
    def test_disabled_client_does_not_send(self):
        transport = FakeTransport([200])
        client = WebhookAlarmClient(url="", transport=transport)

        result = client.send({"event_key": "smoke_alarm"})

        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "webhook_disabled")
        self.assertEqual(transport.calls, [])

    def test_successful_webhook_posts_event(self):
        transport = FakeTransport([204])
        client = WebhookAlarmClient(url="http://127.0.0.1/alarm", transport=transport)

        result = client.send({"event_key": "smoke_alarm"})

        self.assertTrue(result["sent"])
        self.assertEqual(result["status_code"], 204)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]["url"], "http://127.0.0.1/alarm")
        self.assertEqual(transport.calls[0]["payload"]["event_key"], "smoke_alarm")

    def test_retries_until_success(self):
        transport = FakeTransport([500, 502, 200])
        client = WebhookAlarmClient(
            url="http://127.0.0.1/alarm",
            transport=transport,
            max_retries=2,
            retry_delay_s=0.0,
        )

        result = client.send({"event_key": "glass_break"})

        self.assertTrue(result["sent"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["status_code"], 200)

    def test_reports_failure_after_retries(self):
        transport = FakeTransport([500, 503])
        client = WebhookAlarmClient(
            url="http://127.0.0.1/alarm",
            transport=transport,
            max_retries=1,
            retry_delay_s=0.0,
        )

        result = client.send({"event_key": "distress_call"})

        self.assertFalse(result["sent"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["status_code"], 503)


if __name__ == "__main__":
    unittest.main()
