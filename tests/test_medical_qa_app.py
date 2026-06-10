import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from medical_qa_app import create_request_handler


class FakeApplication:
    def __init__(self):
        self.departments = ["IM_内科", "Surgical_外科"]
        self.configured_background_load = None

    def health(self):
        return {
            "status": "ok",
            "online_ready": True,
            "local_ready": False,
            "local_loading": True,
            "local_progress": {"percent": 30, "stage": "测试加载"},
            "online_model": "qwen-test",
            "departments": self.departments,
        }

    def answer(self, question, *, mode, top_k, threshold):
        return {
            "mode": mode,
            "answer": f"回答：{question}",
            "evidence": [{"rank": 1, "title": "测试依据", "score": 0.5}],
            "warning": "不能替代医生诊断。",
            "top_k": top_k,
            "threshold": threshold,
        }

    def configure_departments(self, departments, *, background_load=True):
        self.departments = list(departments)
        self.configured_background_load = background_load


class MedicalQAAppTestCase(unittest.TestCase):
    def run_server(self, ui_text="<html>医疗问答助手</html>"):
        tmp_dir = tempfile.TemporaryDirectory()
        ui_path = Path(tmp_dir.name) / "ui_preview.html"
        ui_path.write_text(ui_text, encoding="utf-8")
        self.fake_application = FakeApplication()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_request_handler(self.fake_application, ui_path),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(tmp_dir.cleanup)
        return server

    def test_root_serves_ui_html(self):
        server = self.run_server()

        body = urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/", timeout=5
        ).read().decode("utf-8")

        self.assertIn("医疗问答助手", body)

    def test_health_endpoint_returns_json(self):
        server = self.run_server()

        body = urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/api/health", timeout=5
        ).read().decode("utf-8")

        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["online_ready"], True)
        self.assertEqual(payload["online_model"], "qwen-test")

    def test_ask_endpoint_returns_answer_json(self):
        server = self.run_server()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/ask",
            data=json.dumps(
                {"question": "腹泻怎么办？", "mode": "local", "top_k": 2, "threshold": 0.1}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        body = urllib.request.urlopen(request, timeout=5).read().decode("utf-8")

        payload = json.loads(body)
        self.assertEqual(payload["mode"], "local")
        self.assertIn("腹泻怎么办？", payload["answer"])
        self.assertEqual(payload["top_k"], 2)
        self.assertEqual(payload["threshold"], 0.1)

    def test_config_endpoint_updates_department_scope(self):
        server = self.run_server()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/config",
            data=json.dumps({"scope": "pediatric"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        body = urllib.request.urlopen(request, timeout=5).read().decode("utf-8")

        payload = json.loads(body)
        self.assertEqual(payload["departments"], ["Pediatric_儿科"])
        self.assertEqual(self.fake_application.configured_background_load, True)

    def test_main_loads_local_system_in_background_before_serving(self):
        source = Path("medical_qa_app.py").read_text(encoding="utf-8")

        self.assertIn("background_load=True", source)
        self.assertNotIn("preload_local=True", source)


if __name__ == "__main__":
    unittest.main()
