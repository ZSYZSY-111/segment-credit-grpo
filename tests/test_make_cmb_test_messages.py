import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_utils import parse_messages_record
from scripts.make_cmb_test_messages import convert, main


def question(qid, stem, options, qtype="单项选择题"):
    return {"id": qid, "exam_type": "医师考试", "exam_class": "临床", "exam_subject": "内科", "question_type": qtype,
            "question": stem, "option": options}


class ConvertTests(unittest.TestCase):
    def test_rows_round_trip_through_the_record_parser(self):
        questions = [question(1, "题干一", {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": ""}),
                     question(2, "题干二", {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": "戊"}, "多项选择题"),
                     question(3, "题干三", {"A": "甲", "B": "甲", "C": "丙", "D": "丁"})]
        answers = {"1": "B", "2": "AC"}
        rows, meta, dropped, counts = convert(questions, answers)
        self.assertEqual(counts["converted"], 2)
        self.assertEqual([d["reason"] for d in dropped], ["no_answer"])
        parsed = parse_messages_record(rows[0])
        self.assertEqual(parsed["answer"], "B")
        self.assertEqual(parsed["sample_id"], meta[0]["sample_id"])
        self.assertIn("E. ", rows[0]["messages"][1]["content"])      # trailing empty option rendered as the evaluator does
        self.assertEqual(parse_messages_record(rows[1])["answer"], "AC")
        self.assertEqual(meta[1]["question_type"], "多项选择题")

    def test_malformed_rows_are_counted_by_reason(self):
        questions = [question(1, "题干", {"A": "甲", "B": "甲", "C": "丙", "D": "丁"})]
        rows, _, dropped, counts = convert(questions, {"1": "A"})
        self.assertEqual(rows, [])
        self.assertEqual(counts["duplicate_option_text"], 1)
        self.assertEqual(dropped[0]["reason"], "duplicate_option_text")

    def test_cli_writes_messages_meta_and_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "q.json").write_text(json.dumps([question(7, "题干", {"A": "甲", "B": "乙", "C": "丙", "D": "丁"})]), encoding="utf-8")
            (root / "a.json").write_text(json.dumps([{"id": 7, "answer": "D"}]), encoding="utf-8")
            out = root / "cmb_test_messages.jsonl"
            self.assertEqual(main(["--questions", str(root / "q.json"), "--answers", str(root / "a.json"), "--output", str(out)]), 0)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 1)
            meta = json.loads((root / "cmb_test_messages.meta.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((meta["id"], meta["answer"]), ("7", "D"))
            self.assertEqual((root / "cmb_test_messages.dropped.jsonl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
