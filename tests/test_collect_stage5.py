import json
import tempfile
import unittest
from pathlib import Path

from tools.collect_stage5 import answer_letter_early, arm_statistics, collect, curve_from_log_history


class CollectTests(unittest.TestCase):
    def test_curve_thinning_and_key_mapping(self):
        history = []
        for step in range(1, 13):
            history.append({"step": step, "reward": 0.5 + step / 100, "completions/mean_length": 100 - step,
                            "credit/phi_last_mean": 0.7, "frac_reward_zero_std": 0.1, "loss": 0.0})
            history.append({"step": step, "eval_reward": 0.6})       # evaluation entries carry no reward key
        curve = curve_from_log_history(history)
        self.assertEqual([p["step"] for p in curve], [1.0, 5.0, 10.0, 12.0])
        self.assertEqual(curve[0]["mean_length"], 99.0)
        self.assertEqual(curve[0]["phi_last_mean"], 0.7)
        self.assertNotIn("entropy", curve[0])
        self.assertEqual(curve_from_log_history([]), [])

    def test_answer_letter_early(self):
        self.assertTrue(answer_letter_early("解析：答案是B，因为……", "B"))
        self.assertFalse(answer_letter_early("解析：本题考查肝硬化的并发症，" + "很长的内容" * 10 + "答案：B", "B"))
        self.assertFalse(answer_letter_early("解析：Vitamin BCD 缺乏", "B"))   # part of a word, not a standalone letter
        self.assertTrue(answer_letter_early("解析：Vitamin B 缺乏", "B"))       # a standalone letter counts, whatever it means
        self.assertTrue(answer_letter_early("解析：选 C。", None))              # no reference letters: any standalone letter counts

    def test_collect_reads_a_synthetic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = out / "stage5_segment_credit_seed7"
            (run / "eval_cmexam_val").mkdir(parents=True)
            (run / "run_config.json").write_text(json.dumps({"arguments": {"segment_credit_lambda": 2.0, "credit_mode": "segment",
                                                                            "seed": 7, "max_steps": 150, "other": 1},
                                                              "batch_geometry": {"completions_per_step": 128},
                                                              "resolved": {"gpu": "RTX"}, "versions": {"trl": "1.8.0"}}), encoding="utf-8")
            (run / "segment_credit.json").write_text(json.dumps({"credit_lambda": 2.0}), encoding="utf-8")
            (run / "trainer_state.json").write_text(json.dumps({"log_history": [
                {"step": 1, "reward": 0.7, "completions/mean_length": 140.0}, {"step": 5, "reward": 0.75, "completions/mean_length": 90.0}]}), encoding="utf-8")
            (run / "train_results.json").write_text(json.dumps({"train_runtime": 7200.0}), encoding="utf-8")
            (run / "eval_cmexam_val" / "headroom_report.json").write_text(json.dumps({"overall": {"pass1_greedy": 0.75}, "slices": {}}), encoding="utf-8")
            rows = [{"sample_id": f"s{i}", "greedy_correct": i < 3, "greedy_answer": "A" if i < 3 else None,
                     "greedy_text": "解析：A 是对的。\n答案：A" if i == 0 else "解析：" + "内容" * 30 + "\n答案：A"} for i in range(4)]
            (run / "eval_cmexam_val" / "headroom_records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            (out / "stage5_standard_vs_segment_credit_seed7.json").write_text(json.dumps({"overall": {"delta_b_minus_a": 1.0}}), encoding="utf-8")
            (out / "stage5_seeds.json").write_text(json.dumps({"arms": {}}), encoding="utf-8")
            report = collect(out)
            run_report = report["runs"]["stage5_segment_credit_seed7"]
            self.assertEqual((run_report["name"], run_report["seed"]), ("segment_credit", 7))
            self.assertEqual(run_report["config"], {"segment_credit_lambda": 2.0, "credit_mode": "segment", "max_steps": 150, "seed": 7})
            self.assertEqual(run_report["train_runtime_hours"], 2.0)
            self.assertEqual([p["step"] for p in run_report["curve"]], [1.0, 5.0])
            arm = run_report["eval_cmexam_val"]["arm"]
            self.assertEqual((arm["n"], arm["correct"], arm["accuracy"]), (4, 3, 75.0))
            self.assertEqual(arm["format_failure_pct"], 25.0)
            self.assertEqual(arm["answer_letter_in_first_20_chars_pct"], 25.0)
            self.assertIn("stage5_standard_vs_segment_credit_seed7", report["comparisons"])
            self.assertEqual(report["seeds_summary"], {"arms": {}})
            self.assertIsNone(report["stage4_start"])

    def test_arm_statistics_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text("", encoding="utf-8")
            self.assertIsNone(arm_statistics(path))


if __name__ == "__main__":
    unittest.main()
