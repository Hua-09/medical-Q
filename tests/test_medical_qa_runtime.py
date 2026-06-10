import threading
import unittest

from medical_qa_cli import QARecord, SearchResult
from medical_qa_runtime import (
    MedicalQAApplication,
    build_retrieved_context,
    format_local_answer,
)


class FakeLocalSystem:
    def search(self, query, *, top_k, threshold):
        record = QARecord(
            department="消化科",
            title="腹泻腹痛怎么办？",
            ask="最近腹泻并伴随腹痛。",
            answer="建议清淡饮食，注意补液，症状严重时及时就医。",
            source="IM_内科",
        )
        return [SearchResult(rank=1, score=0.72, record=record)]


class FakeQwenClient:
    def __init__(self, configured=True, fail=False):
        self.calls = []
        self.is_configured = configured
        self.model = "qwen-test"
        self.fail = fail

    def chat(self, question, *, retrieved_context=None):
        self.calls.append((question, retrieved_context))
        if self.fail:
            raise RuntimeError("千问超时")
        return "在线回答：请注意补液，严重时及时就医。"


class RuntimeTestCase(unittest.TestCase):
    def test_format_local_answer_returns_answer_and_evidence(self):
        record = QARecord(
            department="消化科",
            title="腹泻怎么办？",
            ask="连续腹泻。",
            answer="注意补液并清淡饮食。",
            source="IM_内科",
        )
        result = SearchResult(rank=1, score=0.8, record=record)

        payload = format_local_answer("腹泻怎么办？", [result])

        self.assertEqual(payload["mode"], "local")
        self.assertIn("注意补液", payload["answer"])
        self.assertEqual(payload["evidence"][0]["title"], "腹泻怎么办？")
        self.assertEqual(payload["evidence"][0]["score"], 0.8)
        self.assertIn("不能替代医生诊断", payload["warning"])

    def test_build_retrieved_context_summarizes_top_results(self):
        record = QARecord(
            department="消化科",
            title="腹泻怎么办？",
            ask="连续腹泻。",
            answer="注意补液并清淡饮食。",
            source="IM_内科",
        )

        context = build_retrieved_context(
            [SearchResult(rank=1, score=0.8, record=record)]
        )

        self.assertIn("Top 1", context)
        self.assertIn("腹泻怎么办？", context)
        self.assertIn("注意补液", context)

    def test_application_dispatches_local_mode(self):
        app = MedicalQAApplication(local_system_factory=lambda: FakeLocalSystem())

        payload = app.answer("腹泻怎么办？", mode="local", top_k=1, threshold=0.1)

        self.assertEqual(payload["mode"], "local")
        self.assertEqual(payload["evidence"][0]["department"], "消化科")

    def test_application_dispatches_online_mode(self):
        qwen = FakeQwenClient()
        app = MedicalQAApplication(
            local_system_factory=lambda: FakeLocalSystem(),
            qwen_client=qwen,
        )

        payload = app.answer("腹泻怎么办？", mode="online")

        self.assertEqual(payload["mode"], "online")
        self.assertIn("在线回答", payload["answer"])
        self.assertEqual(qwen.calls[0][1], None)

    def test_application_dispatches_hybrid_mode_with_context(self):
        qwen = FakeQwenClient()
        app = MedicalQAApplication(
            local_system_factory=lambda: FakeLocalSystem(),
            qwen_client=qwen,
        )

        payload = app.answer("腹泻怎么办？", mode="hybrid", top_k=1, threshold=0.1)

        self.assertEqual(payload["mode"], "hybrid")
        self.assertIn("在线回答", payload["answer"])
        self.assertIn("腹泻腹痛怎么办？", qwen.calls[0][1])
        self.assertEqual(payload["evidence"][0]["source"], "IM_内科")

    def test_health_reports_online_model_from_code_config(self):
        qwen = FakeQwenClient(configured=True)
        app = MedicalQAApplication(
            local_system_factory=lambda: FakeLocalSystem(),
            qwen_client=qwen,
        )

        payload = app.health()

        self.assertEqual(payload["online_model"], "qwen-test")
        self.assertEqual(payload["online_ready"], True)

    def test_application_can_preload_local_system_at_startup(self):
        calls = []

        def factory():
            calls.append("loaded")
            return FakeLocalSystem()

        app = MedicalQAApplication(
            local_system_factory=factory,
            preload_local=True,
        )

        self.assertEqual(calls, ["loaded"])
        self.assertEqual(app.health()["local_ready"], True)

    def test_application_can_load_local_system_in_background(self):
        started = threading.Event()
        release = threading.Event()

        def factory():
            started.set()
            release.wait(timeout=2)
            return FakeLocalSystem()

        app = MedicalQAApplication(
            local_system_factory=factory,
            background_load=True,
        )

        self.assertTrue(started.wait(timeout=1))
        self.assertEqual(app.health()["local_ready"], False)
        self.assertEqual(app.health()["local_loading"], True)

        payload = app.answer("腹泻怎么办？", mode="local", top_k=1, threshold=0.1)
        self.assertIn("正在加载", payload["answer"])

        release.set()
        self.assertTrue(app.wait_for_local_load(timeout=2))
        self.assertEqual(app.health()["local_ready"], True)
        self.assertEqual(app.health()["local_loading"], False)
        self.assertEqual(app.health()["local_progress"]["percent"], 100)

    def test_health_reports_local_index_progress(self):
        app = MedicalQAApplication(local_system_factory=lambda: FakeLocalSystem())

        payload = app.health()

        self.assertIn("local_progress", payload)
        self.assertIn("percent", payload["local_progress"])
        self.assertIn("stage", payload["local_progress"])

    def test_configure_departments_restarts_background_load(self):
        calls = []
        release = threading.Event()

        def factory():
            calls.append("loaded")
            if len(calls) > 1:
                release.wait(timeout=2)
            return FakeLocalSystem()

        app = MedicalQAApplication(
            local_system_factory=factory,
            preload_local=True,
        )

        app.configure_departments(("Pediatric_儿科",), background_load=True)

        self.assertEqual(app.health()["departments"], ["Pediatric_儿科"])
        self.assertEqual(app.health()["local_ready"], False)
        self.assertEqual(app.health()["local_loading"], True)
        release.set()
        self.assertTrue(app.wait_for_local_load(timeout=2))

    def test_hybrid_mode_falls_back_to_local_answer_when_qwen_fails(self):
        app = MedicalQAApplication(
            local_system_factory=lambda: FakeLocalSystem(),
            qwen_client=FakeQwenClient(fail=True),
        )

        payload = app.answer("腹泻怎么办？", mode="hybrid", top_k=1, threshold=0.1)

        self.assertEqual(payload["mode"], "hybrid")
        self.assertIn("本地检索参考", payload["answer"])
        self.assertIn("建议清淡饮食", payload["answer"])
        self.assertEqual(payload["evidence"][0]["title"], "腹泻腹痛怎么办？")
        self.assertIn("在线模式调用失败", payload["warning"])


if __name__ == "__main__":
    unittest.main()
