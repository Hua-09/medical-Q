from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Callable

from medical_prompt import build_qwen_messages
from qwen_config import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MAX_RETRIES,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    QWEN_TIMEOUT_SECONDS,
)


QWEN_COMPATIBLE_BASE_URL = QWEN_BASE_URL


def default_transport(
    request: urllib.request.Request,
    timeout: float,
):
    return urllib.request.urlopen(request, timeout=timeout)


def clean_model_answer(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[*-]\s+", "", line)
        line = re.sub(r"^(\d+[.)]、?)\s*", r"\1 ", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")

        if re.fullmatch(r"\|?[-:\s|]{3,}\|?", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            line = "，".join(cells)

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


class QwenClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = QWEN_COMPATIBLE_BASE_URL,
        timeout: float = QWEN_TIMEOUT_SECONDS,
        max_retries: int = QWEN_MAX_RETRIES,
        retry_delay: float = 1.0,
        max_tokens: int | None = QWEN_MAX_TOKENS,
        temperature: float | None = None,
        transport: Callable[[urllib.request.Request, float], object] = default_transport,
    ) -> None:
        self.api_key = api_key if api_key is not None else QWEN_API_KEY
        self.model = model or QWEN_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.max_tokens = max_tokens
        self.temperature = QWEN_TEMPERATURE if temperature is None else temperature
        self.transport = transport

    @property
    def is_configured(self) -> bool:
        api_key = self.api_key.strip()
        return bool(api_key) and not api_key.startswith("请在这里填写")

    def chat(self, question: str, *, retrieved_context: str | None = None) -> str:
        if not self.is_configured:
            raise ValueError("千问 API Key 未配置，请在 qwen_config.py 中填写 QWEN_API_KEY。")

        payload: dict[str, object] = {
            "model": self.model,
            "messages": build_qwen_messages(question, retrieved_context=retrieved_context),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = int(self.max_tokens)

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        data = json.loads(self._read_with_retries(request))
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("千问 API 返回格式异常。") from exc
        return clean_model_answer(content)

    def _read_with_retries(self, request: urllib.request.Request) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self.transport(request, self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore").strip()
                message = detail or str(exc.reason)
                raise RuntimeError(
                    f"千问 API HTTP {exc.code}：{message}。当前模型：{self.model}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        if isinstance(last_error, (TimeoutError, socket.timeout)):
            raise RuntimeError(
                f"千问 API 请求超时，请检查网络、模型名或稍后重试。当前模型：{self.model}"
            ) from last_error

        raise RuntimeError(
            f"千问 API 请求失败，请检查网络、API Key 或模型名。当前模型：{self.model}，错误：{last_error}"
        ) from last_error
