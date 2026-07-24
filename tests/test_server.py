import hashlib
import unittest

import server


class ServerUnitTests(unittest.TestCase):
    def setUp(self):
        server.RATE_BUCKETS.clear()
        server.DAILY_BUCKET.clear()

    def test_sanitize_payload_keeps_only_bounded_facts(self):
        result = server.sanitize_payload(
            {
                "intent": "专注",
                "scene": "工位",
                "light": "明亮",
                "tidy": "一般",
                "brightness": 999,
                "detections": [{"name": "电脑", "confidence": 4}],
                "cards": [{"role": "你的答案", "name": "愚者", "direction": "正位", "keyword": "开始"}],
                "localSuggestions": ["先清出桌面。"],
                "variationId": "test-id",
                "image": "must-not-pass-through",
            }
        )
        self.assertEqual(result["brightness"], 255)
        self.assertEqual(result["detections"], [{"name": "电脑", "confidence": 1.0}])
        self.assertNotIn("image", result)

    def test_model_request_disables_storage_and_uses_strict_schema(self):
        request = server.build_model_request({"scene": "工位"})
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(request["text"]["format"]["schema"]["properties"]["actions"]["minItems"], 3)

    def test_origin_allowlist_is_exact(self):
        self.assertTrue(server.is_origin_allowed(None))
        self.assertTrue(server.is_origin_allowed("https://liab-bb.github.io/"))
        self.assertFalse(server.is_origin_allowed("https://liab-bb.github.io.attacker.example"))

    def test_direct_client_key_ignores_untrusted_proxy_header(self):
        original = server.TRUST_PROXY
        server.TRUST_PROXY = False
        try:
            key = server.client_rate_key({"CF-Connecting-IP": "203.0.113.9"}, "127.0.0.1")
        finally:
            server.TRUST_PROXY = original
        expected = hashlib.sha256(b"127.0.0.1").hexdigest()[:16]
        self.assertEqual(key, expected)

    def test_rate_and_daily_limits_are_independent(self):
        original_minute = server.RATE_LIMIT_PER_MINUTE
        original_daily = server.DAILY_REQUEST_LIMIT
        server.RATE_LIMIT_PER_MINUTE = 2
        server.DAILY_REQUEST_LIMIT = 3
        try:
            self.assertIsNone(server.consume_request_capacity("a", 1.0))
            self.assertIsNone(server.consume_request_capacity("a", 2.0))
            self.assertEqual(server.consume_request_capacity("a", 3.0), "rate_limited")
            self.assertIsNone(server.consume_request_capacity("b", 4.0))
            self.assertEqual(server.consume_request_capacity("c", 5.0), "daily_budget_reached")
        finally:
            server.RATE_LIMIT_PER_MINUTE = original_minute
            server.DAILY_REQUEST_LIMIT = original_daily


if __name__ == "__main__":
    unittest.main()
