from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import joblib

from medical_qa_cli import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEPARTMENTS,
    MedicalQASystem,
    SearchResult,
    load_records_from_departments,
)
from qwen_client import QwenClient


ProgressCallback = Callable[[int, str], None]

LOCAL_WARNING = (
    "本回答基于历史医患问答检索，仅供课程演示和健康咨询参考，不能替代医生诊断。"
    "如症状严重、持续加重或出现急症表现，请及时线下就医。"
)

ONLINE_WARNING = (
    "在线回答由千问 API 生成，仅供健康咨询参考，不能替代医生诊断。"
)


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    departments: tuple[str, ...] = DEFAULT_DEPARTMENTS
    encoding: str = "gb18030"
    max_records: int | None = None
    max_features: int | None = 80000
    cache_dir: Path = Path(".cache")


def _report_progress(
    progress_callback: ProgressCallback | None,
    percent: int,
    stage: str,
) -> None:
    if progress_callback:
        progress_callback(percent, stage)


def evidence_to_dict(result: SearchResult) -> dict[str, object]:
    record = result.record
    return {
        "rank": result.rank,
        "score": round(float(result.score), 4),
        "department": record.department,
        "title": record.title,
        "ask": record.ask,
        "answer": record.answer,
        "source": record.source,
    }


def format_local_answer(
    question: str,
    results: Sequence[SearchResult],
) -> dict[str, object]:
    if not results:
        return {
            "mode": "local",
            "answer": "抱歉，本地问答库中未检索到足够相似的问题。可以换一种描述，或切换到在线问答模式。",
            "evidence": [],
            "warning": LOCAL_WARNING,
        }

    best = results[0]
    answer = (
        f"根据本地问答库中相似问题“{best.record.title}”的医生回复，"
        f"参考建议如下：{best.record.answer}"
    )
    return {
        "mode": "local",
        "answer": answer,
        "evidence": [evidence_to_dict(result) for result in results],
        "warning": LOCAL_WARNING,
    }


def format_loading_answer(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "answer": "本地索引正在加载中，请稍候再使用本地检索或混合模式。在线问答模式可先使用。",
        "evidence": [],
        "warning": "本地数据加载完成后，页面顶部状态会显示“本地索引已就绪”。",
    }


def build_retrieved_context(results: Sequence[SearchResult]) -> str:
    lines: list[str] = []
    for result in results:
        record = result.record
        lines.append(
            f"Top {result.rank} | 相似度 {result.score:.4f} | "
            f"科室 {record.department} | 问题：{record.title} | "
            f"患者描述：{record.ask} | 医生回复：{record.answer}"
        )
    return "\n".join(lines)


def cache_path_for_config(config: RuntimeConfig) -> Path:
    key = "|".join(
        [
            str(Path(config.data_dir)),
            ",".join(config.departments),
            str(config.max_records),
            str(config.max_features),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return Path(config.cache_dir) / f"medical_qa_system_{digest}.joblib"


def build_local_system(
    config: RuntimeConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> MedicalQASystem:
    _report_progress(progress_callback, 20, "读取本地 CSV 数据")
    records = load_records_from_departments(
        config.data_dir,
        config.departments,
        encoding=config.encoding,
        max_records=config.max_records,
    )
    _report_progress(progress_callback, 70, "构建 TF-IDF 检索索引")
    return MedicalQASystem(records, max_features=config.max_features)


def load_or_build_local_system(
    config: RuntimeConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> MedicalQASystem:
    cache_path = cache_path_for_config(config)
    if cache_path.exists():
        _report_progress(progress_callback, 35, "读取本地索引缓存")
        system = joblib.load(cache_path)
        _report_progress(progress_callback, 90, "索引缓存读取完成")
        return system

    system = build_local_system(config, progress_callback=progress_callback)
    _report_progress(progress_callback, 88, "保存索引缓存")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(system, cache_path)
    return system


class MedicalQAApplication:
    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        local_system_factory: Callable[[], MedicalQASystem] | None = None,
        qwen_client: QwenClient | None = None,
        preload_local: bool = False,
        background_load: bool = False,
    ) -> None:
        self.config = config or RuntimeConfig()
        self._local_system: MedicalQASystem | None = None
        self._local_loading = False
        self._local_load_error: str | None = None
        self._local_load_thread: threading.Thread | None = None
        self._local_progress: dict[str, object] = {
            "percent": 0,
            "stage": "等待加载",
        }
        self._local_lock = threading.RLock()
        self._load_generation = 0
        self.qwen_client = qwen_client or QwenClient()

        self._local_system_factory = local_system_factory or (
            lambda: load_or_build_local_system(
                self.config,
                progress_callback=self._set_local_progress,
            )
        )

        if preload_local:
            self.load_local_system()
        elif background_load:
            self.start_background_load()

    def _set_local_progress(self, percent: int, stage: str) -> None:
        clamped = max(0, min(int(percent), 100))
        with self._local_lock:
            self._local_progress = {
                "percent": clamped,
                "stage": stage,
            }

    @property
    def local_system(self) -> MedicalQASystem:
        with self._local_lock:
            if self._local_system is not None:
                return self._local_system
            self._local_loading = True
            self._local_load_error = None
        self._set_local_progress(5, "准备加载本地索引")

        try:
            system = self._local_system_factory()
        except Exception as exc:
            with self._local_lock:
                self._local_load_error = str(exc)
                self._local_loading = False
            self._set_local_progress(0, "本地索引加载失败")
            raise

        with self._local_lock:
            self._local_system = system
            self._local_loading = False
            self._local_load_error = None
        self._set_local_progress(100, "本地索引已就绪")
        return system

    def load_local_system(self) -> MedicalQASystem:
        return self.local_system

    def start_background_load(self, *, force: bool = False) -> None:
        with self._local_lock:
            if not force and (self._local_system is not None or self._local_loading):
                return
            self._load_generation += 1
            generation = self._load_generation
            self._local_loading = True
            self._local_load_error = None
            self._local_progress = {
                "percent": 5,
                "stage": "准备加载本地索引",
            }

        def load_worker() -> None:
            try:
                system = self._local_system_factory()
            except Exception as exc:
                with self._local_lock:
                    if generation != self._load_generation:
                        return
                    self._local_load_error = str(exc)
                    self._local_loading = False
                    self._local_progress = {
                        "percent": 0,
                        "stage": "本地索引加载失败",
                    }
                return

            with self._local_lock:
                if generation != self._load_generation:
                    return
                self._local_system = system
                self._local_load_error = None
                self._local_loading = False
                self._local_progress = {
                    "percent": 100,
                    "stage": "本地索引已就绪",
                }

        thread = threading.Thread(
            target=load_worker,
            name="medical-qa-local-index-loader",
            daemon=True,
        )
        with self._local_lock:
            self._local_load_thread = thread
        thread.start()

    def configure_departments(
        self,
        departments: Sequence[str],
        *,
        background_load: bool = True,
    ) -> None:
        self.config = replace(self.config, departments=tuple(departments))
        with self._local_lock:
            self._load_generation += 1
            self._local_system = None
            self._local_loading = False
            self._local_load_error = None
            self._local_progress = {
                "percent": 0,
                "stage": "等待加载",
            }

        if background_load:
            self.start_background_load(force=True)

    def wait_for_local_load(self, timeout: float | None = None) -> bool:
        thread = self._local_load_thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._local_lock:
            return self._local_system is not None and not self._local_loading

    def local_is_ready(self) -> bool:
        with self._local_lock:
            return self._local_system is not None

    def local_is_loading(self) -> bool:
        with self._local_lock:
            return self._local_loading

    def search_local(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
    ) -> list[SearchResult]:
        return self.local_system.search(question, top_k=top_k, threshold=threshold)

    def answer(
        self,
        question: str,
        *,
        mode: str = "local",
        top_k: int = 3,
        threshold: float = 0.05,
    ) -> dict[str, object]:
        question = question.strip()
        if not question:
            return {
                "mode": mode,
                "answer": "请输入需要咨询的健康问题。",
                "evidence": [],
                "warning": LOCAL_WARNING,
            }

        normalized_mode = mode if mode in {"local", "online", "hybrid"} else "local"
        top_k = max(1, min(int(top_k), 10))
        threshold = max(0.0, float(threshold))

        if normalized_mode == "online":
            return self._answer_online(question, mode="online")

        if not self.local_is_ready():
            if self.local_is_loading():
                return format_loading_answer(normalized_mode)
            if self._local_load_error:
                return {
                    "mode": normalized_mode,
                    "answer": f"本地索引加载失败：{self._local_load_error}",
                    "evidence": [],
                    "warning": "请检查数据目录、CSV 文件和程序运行权限。",
                }

        if normalized_mode == "local":
            return format_local_answer(
                question,
                self.search_local(question, top_k=top_k, threshold=threshold),
            )

        return self._answer_hybrid(question, top_k=top_k, threshold=threshold)

    def _answer_hybrid(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
    ) -> dict[str, object]:
        local_results = self.search_local(question, top_k=top_k, threshold=threshold)
        evidence = [evidence_to_dict(result) for result in local_results]
        retrieved_context = build_retrieved_context(local_results)

        try:
            answer = self.qwen_client.chat(
                question,
                retrieved_context=retrieved_context,
            )
        except Exception as exc:
            local_payload = format_local_answer(question, local_results)
            return {
                "mode": "hybrid",
                "answer": f"本地检索参考（千问在线回答失败）：\n{local_payload['answer']}",
                "evidence": evidence,
                "warning": f"在线模式调用失败：{exc}。已优先返回本地检索结果。",
            }

        return {
            "mode": "hybrid",
            "answer": answer,
            "evidence": evidence,
            "warning": ONLINE_WARNING,
        }

    def _answer_online(
        self,
        question: str,
        *,
        mode: str,
        retrieved_context: str | None = None,
        evidence: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        try:
            answer = self.qwen_client.chat(
                question,
                retrieved_context=retrieved_context,
            )
        except Exception as exc:
            return {
                "mode": mode,
                "answer": f"在线模式暂时不可用：{exc}",
                "evidence": evidence or [],
                "warning": "请检查千问 API Key、模型名或网络连接，也可以切换到本地检索模式。",
            }

        return {
            "mode": mode,
            "answer": answer,
            "evidence": evidence or [],
            "warning": ONLINE_WARNING,
        }

    def health(self) -> dict[str, object]:
        with self._local_lock:
            local_ready = self._local_system is not None
            local_loading = self._local_loading
            local_error = self._local_load_error
            local_progress = dict(self._local_progress)
        return {
            "status": "ok",
            "local_ready": local_ready,
            "local_loading": local_loading,
            "local_error": local_error,
            "local_progress": local_progress,
            "online_ready": self.qwen_client.is_configured,
            "online_model": getattr(self.qwen_client, "model", ""),
            "departments": list(self.config.departments),
            "max_records": self.config.max_records,
        }
