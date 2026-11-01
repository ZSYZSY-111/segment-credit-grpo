import unittest

from tools.make_tables import render, runs_table


def run(name, seed, acc, chars=50.0):
    return {"name": name, "seed": seed, "train_runtime_hours": 2.5,
            "curve": [{"step": 1.0, "reward": 0.7}, {"step": 150.0, "reward": 0.86, "token_frac_nonzero_adv": 0.91, "entropy": 0.4}],
            "eval_cmexam_val": {"arm": {"accuracy": acc, "ci95": [acc - 1, acc + 1], "format_failure_pct": 0.1, "output_chars_mean": chars,
                                        "output_chars_median": chars - 10, "answer_letter_in_first_20_chars_pct": 12.9}}}


class TableTests(unittest.TestCase):
    def test_runs_table_orders_arms_and_fills_blanks(self):
        runs = {"b": run("segment_credit", 42, 83.7), "a": run("standard", 7, 82.7), "c": {"name": "uniform", "seed": 42, "curve": []}}
        text = runs_table(runs, ["standard", "segment_credit", "uniform"])
        rows = text.splitlines()[2:]
        self.assertTrue(rows[0].startswith("| standard | 7 | 82.70 | 81.70 – 83.70"))
        self.assertTrue(rows[1].startswith("| segment_credit | 42 | 83.70"))
        self.assertIn("0.91 | 0.40 |", rows[1])
        self.assertTrue(rows[2].startswith("| uniform | 42 |  |"))

    def test_render_with_summary(self):
        summary = {"arms": {"standard": {"accuracy_across_seeds": {"n": 2, "mean": 82.7, "std": 0.3, "min": 82.5, "max": 82.9}}},
                   "comparisons": {"segment_credit − standard": {
                       "same_seed": {"42": {"acc_a": 82.7, "acc_b": 83.7, "delta_b_minus_a": 1.0, "delta_ci": [0.6, 1.5], "mcnemar_p": 1e-5, "discordance": 0.034, "mde_points": 0.64}},
                       "same_seed_delta": {"n": 1, "mean": 1.0, "std": None, "min": 1.0, "max": 1.0, "positive": 1, "significant_positive": 1},
                       "cross_seed": {"delta": {"n": 1, "mean": 1.0, "std": None, "min": 1.0, "max": 1.0}}}}}
        report = {"runs": {"x": run("standard", 42, 82.7)}, "seeds_summary": summary,
                  "stage4_start": {"overall": {"pass1_greedy": 0.775, "questions": 6657}}}
        text = render(report, ["standard"])
        self.assertIn("| standard | 2 | 82.70 | 0.30 | 82.50 | 82.90 |", text)
        self.assertIn("| 42 | 82.70 | 83.70 | +1.00 | +0.60 to +1.50 | 1e-05 | 3.4% | 0.64 |", text)
        self.assertIn("positive in 1/1, significant in 1/1", text)
        self.assertIn("pass@1 greedy 77.50 on 6657 questions", text)


if __name__ == "__main__":
    unittest.main()
