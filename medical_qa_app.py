from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from medical_qa_runtime import MedicalQAApplication, RuntimeConfig


DEPARTMENT_SCOPES: dict[str, tuple[str, ...]] = {
    "default": ("IM_内科", "Surgical_外科"),
    "internal": ("IM_内科",),
    "surgical": ("Surgical_外科",),
    "pediatric": ("Pediatric_儿科",),
    "gynecology": ("OAGD_妇产科",),
    "andriatria": ("Andriatria_男科",),
    "oncology": ("Oncology_肿瘤科",),
    "all": (
        "IM_内科",
        "Surgical_外科",
        "OAGD_妇产科",
        "Pediatric_儿科",
        "Andriatria_男科",
        "Oncology_肿瘤科",
    ),
}


def resource_path(relative_path: str | Path) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / relative_path


def read_json_request(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length) if content_length else b"{}"
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


def write_response(
    handler: BaseHTTPRequestHandler,
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_json(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, object],
    *,
    status: int = 200,
) -> None:
    write_response(
        handler,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        status=status,
        content_type="application/json; charset=utf-8",
    )


def create_request_handler(
    application: MedicalQAApplication,
    ui_path: str | Path,
):
    ui_file = Path(ui_path)

    class MedicalQARequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/ui_preview.html"}:
                write_response(
                    self,
                    ui_file.read_bytes(),
                    content_type="text/html; charset=utf-8",
                )
                return
            if self.path == "/api/health":
                write_json(self, application.health())
                return
            write_json(self, {"error": "接口不存在。"}, status=404)

        def do_POST(self) -> None:
            if self.path == "/api/config":
                try:
                    payload = read_json_request(self)
                    scope = str(payload.get("scope", "default"))
                    departments = DEPARTMENT_SCOPES.get(scope)
                    if departments is None:
                        write_json(self, {"error": "未知的数据范围。"}, status=400)
                        return
                    application.configure_departments(
                        departments,
                        background_load=True,
                    )
                    response = application.health()
                    response["scope"] = scope
                    write_json(self, response)
                except Exception as exc:
                    write_json(self, {"error": str(exc)}, status=500)
                return

            if self.path != "/api/ask":
                write_json(self, {"error": "接口不存在。"}, status=404)
                return

            try:
                payload = read_json_request(self)
                question = str(payload.get("question", ""))
                mode = str(payload.get("mode", "local"))
                top_k = int(payload.get("top_k", 3))
                threshold = float(payload.get("threshold", 0.05))
                result = application.answer(
                    question,
                    mode=mode,
                    top_k=top_k,
                    threshold=threshold,
                )
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=500)
                return

            write_json(self, result)

    return MedicalQARequestHandler


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="医疗问答助手 Web/EXE 入口")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument(
        "--data-dir",
        default=str(resource_path("Data_数据")),
        help="数据目录",
    )
    parser.add_argument(
        "--departments",
        nargs="+",
        default=["IM_内科", "Surgical_外科"],
        help="加载的数据子目录",
    )
    parser.add_argument("--max-records", type=int, default=None, help="最多加载记录数")
    parser.add_argument("--cache-dir", default=".cache", help="本地索引缓存目录")
    return parser


def run_server(
    application: MedicalQAApplication,
    *,
    host: str,
    port: int,
    ui_path: Path,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (host, port),
        create_request_handler(application, ui_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = RuntimeConfig(
        data_dir=Path(args.data_dir),
        departments=tuple(args.departments),
        max_records=args.max_records,
        cache_dir=Path(args.cache_dir),
    )

    print("正在启动医疗问答助手，本地检索索引将在后台加载...")
    application = MedicalQAApplication(config=config, background_load=True)

    server = run_server(
        application,
        host=args.host,
        port=args.port,
        ui_path=resource_path("ui_preview.html"),
    )
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"医疗问答助手已启动：{url}")
    print("本地数据加载完成前，可先使用在线问答模式。")
    print("关闭此窗口即可停止服务。")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
