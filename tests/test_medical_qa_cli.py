import csv
import tempfile
import unittest
from pathlib import Path

from medical_qa_cli import (
    MedicalQASystem,
    QARecord,
    build_search_text,
    load_stop_words,
    load_records_from_csv,
    register_medical_terms,
    tokenize_chinese,
)


class MedicalQATestCase(unittest.TestCase):
    def test_load_records_from_gb18030_csv_combines_title_and_ask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            with csv_path.open("w", encoding="gb18030", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["department", "title", "ask", "answer"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "department": "心血管科",
                        "title": "高血压怎么治疗？",
                        "ask": "最近体检发现血压偏高，应该怎么办？",
                        "answer": "建议低盐饮食，并在医生指导下治疗。",
                    }
                )

            records = load_records_from_csv(csv_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].department, "心血管科")
        self.assertEqual(records[0].title, "高血压怎么治疗？")
        self.assertIn("高血压怎么治疗", records[0].search_text)
        self.assertIn("血压偏高", records[0].search_text)

    def test_tokenize_chinese_filters_stop_words_and_punctuation(self):
        tokens = tokenize_chinese("高血压应该怎么治疗？谢谢医生！", {"应该", "怎么", "谢谢"})

        self.assertIn("高血压", tokens)
        self.assertIn("治疗", tokens)
        self.assertNotIn("应该", tokens)
        self.assertNotIn("怎么", tokens)
        self.assertNotIn("？", tokens)

    def test_register_medical_terms_keeps_domain_terms_as_whole_tokens(self):
        register_medical_terms(["急性胃肠炎", "颈椎间盘突出"])

        tokens = tokenize_chinese("急性胃肠炎和颈椎间盘突出需要及时就诊")

        self.assertIn("急性胃肠炎", tokens)
        self.assertIn("颈椎间盘突出", tokens)

    def test_load_stop_words_reads_custom_stop_word_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stop_word_path = Path(tmp_dir) / "stopwords.txt"
            stop_word_path.write_text(
                "# comment\n应该\n\n请问\n", encoding="utf-8"
            )

            stop_words = load_stop_words(stop_word_path)

        self.assertIn("应该", stop_words)
        self.assertIn("请问", stop_words)
        self.assertNotIn("# comment", stop_words)

    def test_build_search_text_weights_title_terms(self):
        record = QARecord(
            department="消化科",
            title="胃痛反酸怎么办？",
            ask="饭后不舒服。",
            answer="建议规律饮食。",
            source="内科",
        )

        search_text = build_search_text(record, title_weight=3)

        self.assertEqual(search_text.count("胃痛反酸怎么办？"), 3)
        self.assertIn("饭后不舒服。", search_text)

    def test_search_returns_top_k_results_sorted_by_similarity(self):
        records = [
            QARecord(
                department="心血管科",
                title="高血压如何治疗？",
                ask="血压高，头晕，想知道怎么控制高血压。",
                answer="建议低盐饮食，规律监测血压，必要时遵医嘱服药。",
                source="内科",
            ),
            QARecord(
                department="肛肠",
                title="拉肚子腹痛怎么办？",
                ask="连续腹泻并伴随腹痛。",
                answer="建议清淡饮食，必要时检查大便常规。",
                source="外科",
            ),
            QARecord(
                department="消化科",
                title="胃痛反酸如何处理？",
                ask="饭后胃痛反酸。",
                answer="建议规律饮食，必要时去消化内科就诊。",
                source="内科",
            ),
        ]
        system = MedicalQASystem(records)

        results = system.search("高血压头晕应该如何治疗？", top_k=2, threshold=0.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].record.department, "心血管科")
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_search_expands_common_medical_synonyms(self):
        records = [
            QARecord(
                department="消化科",
                title="腹泻怎么办？",
                ask="连续腹泻并伴随腹痛。",
                answer="建议清淡饮食，必要时检查大便常规。",
                source="内科",
            ),
            QARecord(
                department="心血管科",
                title="高血压如何治疗？",
                ask="血压高，头晕。",
                answer="建议低盐饮食。",
                source="内科",
            ),
        ]
        system = MedicalQASystem(records)

        results = system.search("拉肚子", top_k=1, threshold=0.01)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.department, "消化科")

    def test_search_returns_empty_when_best_score_is_below_threshold(self):
        records = [
            QARecord(
                department="心血管科",
                title="高血压如何治疗？",
                ask="血压高，头晕，想知道怎么控制高血压。",
                answer="建议低盐饮食，规律监测血压，必要时遵医嘱服药。",
                source="内科",
            )
        ]
        system = MedicalQASystem(records)

        results = system.search("图书馆几点关门？", top_k=3, threshold=0.1)

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
