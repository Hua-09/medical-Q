from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = ("department", "title", "ask", "answer")
DEFAULT_DATA_DIR = Path("Data_数据")
DEFAULT_DEPARTMENTS = ("IM_内科", "Surgical_外科")
DEFAULT_MEDICAL_TERMS = (
    "高血压",
    "糖尿病",
    "冠心病",
    "心律失常",
    "心肌梗死",
    "脑梗塞",
    "脑出血",
    "急性胃肠炎",
    "胃肠炎",
    "胃溃疡",
    "十二指肠溃疡",
    "胃食管反流",
    "反流性食管炎",
    "胆囊炎",
    "胆囊结石",
    "胆结石",
    "脂肪肝",
    "肝炎",
    "肝硬化",
    "肾结石",
    "肾炎",
    "尿毒症",
    "支气管炎",
    "慢性支气管炎",
    "支气管哮喘",
    "肺炎",
    "肺结核",
    "肺气肿",
    "甲状腺结节",
    "甲状腺功能亢进",
    "甲亢",
    "贫血",
    "痛风",
    "类风湿关节炎",
    "腰椎间盘突出",
    "颈椎间盘突出",
    "腰肌劳损",
    "阑尾炎",
    "痔疮",
    "肛裂",
    "肛周脓肿",
    "前列腺增生",
    "乳腺增生",
    "腹股沟疝",
    "静脉曲张",
)
DEFAULT_SYNONYM_GROUPS = (
    ("腹泻", "拉肚子", "拉稀", "泻肚", "闹肚子"),
    ("腹痛", "肚子疼", "肚子痛", "肚痛"),
    ("发热", "发烧", "低烧", "高烧"),
    ("咳嗽", "咳", "干咳"),
    ("高血压", "血压高", "血压偏高"),
    ("糖尿病", "血糖高", "血糖偏高"),
    ("胃痛", "胃疼", "胃部疼痛"),
    ("反酸", "烧心", "胃酸"),
    ("便秘", "大便干", "排便困难"),
    ("痔疮", "痔", "肛门肿物"),
    ("头晕", "眩晕", "头昏"),
    ("胸闷", "胸口闷", "气短"),
)
DEFAULT_STOP_WORDS = {
    "的",
    "了",
    "和",
    "是",
    "我",
    "也",
    "就",
    "都",
    "而",
    "及",
    "与",
    "着",
    "或",
    "一个",
    "没有",
    "您好",
    "你好",
    "请问",
    "医生",
    "谢谢",
    "一下",
    "进行",
    "患者",
    "可能",
    "需要",
    "可以",
    "已经",
}
PUNCTUATION_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class QARecord:
    department: str
    title: str
    ask: str
    answer: str
    source: str = ""

    @property
    def search_text(self) -> str:
        return build_search_text(self)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    record: QARecord


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load_word_file(path: str | Path, *, encoding: str = "utf-8") -> set[str]:
    words: set[str] = set()
    with Path(path).open("r", encoding=encoding) as file:
        for line in file:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            words.add(word)
    return words


def load_stop_words(path: str | Path, *, encoding: str = "utf-8") -> set[str]:
    return load_word_file(path, encoding=encoding)


def register_medical_terms(
    terms: Iterable[str] = DEFAULT_MEDICAL_TERMS, *, frequency: int = 200000
) -> None:
    for term in terms:
        term = normalize_text(term)
        if term:
            jieba.add_word(term, freq=frequency)


def tokenize_chinese(
    text: str, stop_words: Iterable[str] | None = None
) -> list[str]:
    stop_word_set = set(stop_words or DEFAULT_STOP_WORDS)
    tokens: list[str] = []
    for token in jieba.cut(normalize_text(text)):
        token = token.strip()
        if not token:
            continue
        if token in stop_word_set:
            continue
        if PUNCTUATION_PATTERN.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def build_search_text(record: QARecord, *, title_weight: int = 2) -> str:
    title_weight = max(1, title_weight)
    title_parts = [record.title] * title_weight if record.title else []
    parts = title_parts + [record.ask]
    return normalize_text(" ".join(part for part in parts if part))


def build_synonym_map(
    synonym_groups: Sequence[Sequence[str]] = DEFAULT_SYNONYM_GROUPS,
) -> dict[str, tuple[str, ...]]:
    synonym_map: dict[str, tuple[str, ...]] = {}
    for group in synonym_groups:
        normalized_group = tuple(
            word for word in (normalize_text(item) for item in group) if word
        )
        for word in normalized_group:
            synonym_map[word] = tuple(item for item in normalized_group if item != word)
    return synonym_map


def expand_query_text(
    query: str,
    synonym_groups: Sequence[Sequence[str]] = DEFAULT_SYNONYM_GROUPS,
) -> str:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return ""

    additions: list[str] = []
    synonym_map = build_synonym_map(synonym_groups)
    for word, synonyms in synonym_map.items():
        if word not in normalized_query:
            continue
        for synonym in synonyms:
            if synonym not in normalized_query and synonym not in additions:
                additions.append(synonym)

    if not additions:
        return normalized_query
    return normalize_text(f"{normalized_query} {' '.join(additions)}")


def load_records_from_csv(
    csv_path: str | Path,
    *,
    encoding: str = "gb18030",
    source: str | None = None,
    limit: int | None = None,
) -> list[QARecord]:
    path = Path(csv_path)
    records: list[QARecord] = []
    source_name = source or path.parent.name

    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{path} 缺少必要字段: {', '.join(missing_columns)}")

        for row in reader:
            title = normalize_text(row.get("title"))
            ask = normalize_text(row.get("ask"))
            answer = normalize_text(row.get("answer"))
            if not answer or not (title or ask):
                continue

            records.append(
                QARecord(
                    department=normalize_text(row.get("department")),
                    title=title,
                    ask=ask,
                    answer=answer,
                    source=source_name,
                )
            )
            if limit is not None and len(records) >= limit:
                break

    return records


def load_records_from_departments(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    departments: Sequence[str] = DEFAULT_DEPARTMENTS,
    *,
    encoding: str = "gb18030",
    max_records: int | None = None,
) -> list[QARecord]:
    base_dir = Path(data_dir)
    records: list[QARecord] = []

    for department_dir in departments:
        folder = base_dir / department_dir
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"未在 {folder} 中找到 CSV 数据文件")

        for csv_file in csv_files:
            remaining = None if max_records is None else max_records - len(records)
            if remaining is not None and remaining <= 0:
                return records
            records.extend(
                load_records_from_csv(
                    csv_file,
                    encoding=encoding,
                    source=department_dir,
                    limit=remaining,
                )
            )

    return records


class MedicalQASystem:
    def __init__(
        self,
        records: Sequence[QARecord],
        *,
        stop_words: Iterable[str] | None = None,
        medical_terms: Iterable[str] | None = None,
        synonym_groups: Sequence[Sequence[str]] = DEFAULT_SYNONYM_GROUPS,
        title_weight: int = 2,
        enable_query_expansion: bool = True,
        max_features: int | None = 80000,
    ) -> None:
        if not records:
            raise ValueError("问答数据为空，无法构建检索系统")

        register_medical_terms(DEFAULT_MEDICAL_TERMS)
        if medical_terms:
            register_medical_terms(medical_terms)

        self.records = list(records)
        self.stop_words = set(stop_words or DEFAULT_STOP_WORDS)
        self.synonym_groups = tuple(tuple(group) for group in synonym_groups)
        self.title_weight = max(1, title_weight)
        self.enable_query_expansion = enable_query_expansion
        self.document_texts = [
            build_search_text(record, title_weight=self.title_weight)
            for record in self.records
        ]
        self.vectorizer = TfidfVectorizer(
            analyzer=self._analyze,
            max_features=max_features,
        )
        self.question_vectors = self.vectorizer.fit_transform(self.document_texts)

    def _analyze(self, text: str) -> list[str]:
        return tokenize_chinese(text, self.stop_words)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        threshold: float = 0.1,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        top_k = max(1, min(top_k, len(self.records)))
        query_text = (
            expand_query_text(query, self.synonym_groups)
            if self.enable_query_expansion
            else query
        )
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.question_vectors).ravel()

        if top_k == len(similarities):
            candidate_indices = np.arange(len(similarities))
        else:
            candidate_indices = np.argpartition(similarities, -top_k)[-top_k:]

        sorted_indices = candidate_indices[
            np.argsort(similarities[candidate_indices])[::-1]
        ]
        results: list[SearchResult] = []
        for index in sorted_indices:
            score = float(similarities[index])
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=score,
                    record=self.records[int(index)],
                )
            )
        return results


def format_result(result: SearchResult) -> str:
    record = result.record
    return (
        f"Top {result.rank} | 相似度: {result.score:.4f} | "
        f"来源: {record.source} | 科室: {record.department}\n"
        f"匹配问题: {record.title}\n"
        f"患者描述: {record.ask}\n"
        f"医生回复: {record.answer}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 jieba + TF-IDF + 余弦相似度的医疗问答命令行系统"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="数据根目录")
    parser.add_argument(
        "--departments",
        nargs="+",
        default=list(DEFAULT_DEPARTMENTS),
        help="要加载的数据子目录，默认加载内科和外科",
    )
    parser.add_argument("--encoding", default="gb18030", help="CSV 文件编码")
    parser.add_argument("--top-k", type=int, default=3, help="返回的候选答案数量")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="最低余弦相似度阈值"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="最多加载多少条问答，调试时可设置较小数值",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=80000,
        help="TF-IDF 最大词表规模，设置 0 表示不限制",
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=2,
        help="标题在检索文本中的重复权重，数值越大标题越重要",
    )
    parser.add_argument(
        "--stop-words",
        default=None,
        help="自定义停用词文件，UTF-8 编码，每行一个词",
    )
    parser.add_argument(
        "--medical-terms",
        default=None,
        help="自定义医学词典文件，UTF-8 编码，每行一个词",
    )
    parser.add_argument(
        "--no-query-expansion",
        action="store_true",
        help="关闭常见医学同义词查询扩展",
    )
    parser.add_argument("--once", default=None, help="只查询一次后退出")
    return parser


def print_results(results: Sequence[SearchResult]) -> None:
    if not results:
        print("抱歉，未检索到足够相似的医疗问答，请尝试换一种描述。")
        return

    for result in results:
        print("\n" + format_result(result))


def run_once(system: MedicalQASystem, query: str, top_k: int, threshold: float) -> None:
    print(f"\n用户提问: {query}")
    print_results(system.search(query, top_k=top_k, threshold=threshold))


def interactive_loop(system: MedicalQASystem, top_k: int, threshold: float) -> None:
    print("\n医疗问答系统已启动，输入“退出”结束。")
    print("提示：本系统基于历史医患问答检索，结果仅供课程演示和学习参考。")
    while True:
        query = input("\n请输入您的医疗问题：\n> ").strip()
        if query.lower() in {"退出", "exit", "quit"}:
            print("感谢使用，再见！")
            break
        if not query:
            print("请输入有效问题。")
            continue
        print_results(system.search(query, top_k=top_k, threshold=threshold))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    max_features = None if args.max_features == 0 else args.max_features
    stop_words = set(DEFAULT_STOP_WORDS)
    if args.stop_words:
        stop_words.update(load_stop_words(args.stop_words))

    medical_terms = set(DEFAULT_MEDICAL_TERMS)
    if args.medical_terms:
        medical_terms.update(load_word_file(args.medical_terms))

    print(f"正在加载数据目录: {args.data_dir}")
    print(f"加载科室数据: {', '.join(args.departments)}")
    records = load_records_from_departments(
        args.data_dir,
        args.departments,
        encoding=args.encoding,
        max_records=args.max_records,
    )
    print(f"已加载问答记录: {len(records)} 条")
    print("正在构建 TF-IDF 向量索引，请稍候...")
    system = MedicalQASystem(
        records,
        stop_words=stop_words,
        medical_terms=medical_terms,
        title_weight=args.title_weight,
        enable_query_expansion=not args.no_query_expansion,
        max_features=max_features,
    )
    print(
        f"索引构建完成，问题矩阵维度: "
        f"{system.question_vectors.shape[0]} x {system.question_vectors.shape[1]}"
    )

    if args.once:
        run_once(system, args.once, args.top_k, args.threshold)
    else:
        interactive_loop(system, args.top_k, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
