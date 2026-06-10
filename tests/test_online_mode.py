import json
import os
import socket
import unittest
from unittest.mock import patch

import qwen_client
from medical_prompt import MEDICAL_SYSTEM_PROMPT, build_qwen_messages
from qwen_config import QWEN_API_KEY, QWEN_MODEL
from qwen_client import QwenClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "请及时补液，严重时线下就医。"}}]},
            ensure_ascii=False,
        ).encode("utf-8")


class OnlineModeTestCase(unittest.TestCase):
    def test_medical_prompt_sets_strict_safety_boundaries(self):
        self.assertIn("不能替代医生诊断", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("不要编造", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("急诊", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("遵医嘱", MEDICAL_SYSTEM_PROMPT)

    def test_medical_prompt_asks_for_non_markdown_answer_style(self):
        self.assertIn("不要使用 Markdown", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("*、-、#", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("自然中文短段落", MEDICAL_SYSTEM_PROMPT)

    def test_build_qwen_messages_includes_retrieved_context(self):
        messages = build_qwen_messages("腹痛怎么办？", retrieved_context="Top 1: 腹痛问答")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Top 1: 腹痛问答", messages[1]["content"])
        self.assertIn("腹痛怎么办？", messages[1]["content"])

    def test_qwen_client_builds_openai_compatible_request(self):
        captured = {}

        def fake_transport(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        client = QwenClient(api_key="test-key", transport=fake_transport, timeout=9)

        answer = client.chat("发热怎么办？")

        self.assertIn("补液", answer)
        self.assertEqual(
            captured["url"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], QWEN_MODEL)
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["timeout"], 9)

    def test_qwen_client_defaults_to_code_config_not_environment(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            client = QwenClient()

        self.assertEqual(client.api_key, QWEN_API_KEY)
        self.assertEqual(client.model, QWEN_MODEL)
        self.assertNotEqual(client.api_key, "env-key")

    def test_placeholder_code_api_key_is_not_treated_as_configured(self):
        client = QwenClient(api_key="请在这里填写你的千问API Key")

        self.assertFalse(client.is_configured)

    def test_qwen_client_retries_timeout_before_success(self):
        attempts = []

        def flaky_transport(request, timeout):
            attempts.append(timeout)
            if len(attempts) == 1:
                raise socket.timeout("read operation timed out")
            return FakeResponse()

        client = QwenClient(
            api_key="test-key",
            transport=flaky_transport,
            timeout=60,
            max_retries=2,
            retry_delay=0,
        )

        answer = client.chat("发热怎么办？")

        self.assertIn("补液", answer)
        self.assertEqual(attempts, [60, 60])

    def test_qwen_client_reports_timeout_after_retries(self):
        def timeout_transport(request, timeout):
            raise TimeoutError("read operation timed out")

        client = QwenClient(
            api_key="test-key",
            transport=timeout_transport,
            timeout=60,
            max_retries=1,
            retry_delay=0,
        )

        with self.assertRaises(RuntimeError) as context:
            client.chat("发热怎么办？")

        self.assertIn("超时", str(context.exception))

    def test_qwen_client_cleans_common_markdown_marks(self):
        raw = "### 建议\n- 多喝水\n* 注意休息\n- 如症状加重，请及时就医"

        cleaned = qwen_client.clean_model_answer(raw)

        self.assertNotIn("###", cleaned)
        self.assertNotIn("- 多喝水", cleaned)
        self.assertNotIn("* 注意休息", cleaned)
        self.assertIn("多喝水", cleaned)
        self.assertIn("注意休息", cleaned)

    def test_qwen_client_requires_api_key(self):
        client = QwenClient(api_key="")

        with self.assertRaises(ValueError):
            client.chat("咳嗽怎么办？")


if __name__ == "__main__":
    unittest.main()
