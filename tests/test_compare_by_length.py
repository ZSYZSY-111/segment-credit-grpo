import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_by_length import bucket_labels, bucket_of, compare_by_length, main


def record(sid, correct, text):
    return {"sample_id": sid, "greedy_correct": correct, "greedy_answer": "A", "greedy_text": text, "slices": {}}


class LengthComparisonTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual([bucket_of(n, [40, 80]) for n in (0, 39, 40, 79, 80, 500)], [0, 0, 1, 1, 2, 2])
        self.assertEqual(bucket_labels([40, 80]), ["[0, 40)", "[40, 80)", "[80, ∞)"])

    def test_stratified_difference_removes_a_pure_length_effect(self):
        # A: 50 short (90% right) + 50 long (50% right); B: everything short, same per-length accuracy as A.
        # The raw gap is a length effect only; within A's length buckets the two arms differ by a known amount.
        a, b = {}, {}
        for i in range(100):
            short = i < 50
            text_a = "短" * 10 if short else "长" * 100
            correct_a = (i % 10 != 0) if short else (i % 2 == 0)
            a[f"s{i}"] = record(f"s{i}", correct_a, text_a)
            b[f"s{i}"] = record(f"s{i}", correct_a, "短" * 10)            # identical correctness, all short
        report = compare_by_length(a, b, [40], resamples=200)
        self.assertEqual(report["common"], 100)
        self.assertEqual(report["overall"]["delta_b_minus_a"], 0.0)
        self.assertEqual(report["stratified_delta_b_minus_a"], 0.0)
        self.assertEqual(report["same_bucket"]["n"], 50)                   # only A's short outputs share B's bucket
        self.assertEqual(report["length_b"]["median"], 10)
        self.assertEqual(sorted(report["by_bucket_of_a"]), ["[0, 40)", "[40, ∞)"])
        # now B fixes 10 of the long questions: the gap must show up in the long bucket only
        for i in range(50, 60):
            b[f"s{i}"]["greedy_correct"] = True
        report = compare_by_length(a, b, [40], resamples=200)
        self.assertGreater(report["by_bucket_of_a"]["[40, ∞)"]["delta_b_minus_a"], 0)
        self.assertEqual(report["by_bucket_of_a"]["[0, 40)"]["delta_b_minus_a"], 0.0)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb, out = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl", Path(tmp) / "out.json"
            pa.write_text("\n".join(json.dumps(record(f"s{i}", i % 2 == 0, "x" * (10 + 10 * i))) for i in range(12)) + "\n", encoding="utf-8")
            pb.write_text("\n".join(json.dumps(record(f"s{i}", i % 3 == 0, "x" * 20)) for i in range(12)) + "\n", encoding="utf-8")
            self.assertEqual(main(["--a", str(pa), "--b", str(pb), "--edges", "50,100", "--resamples", "100", "--json", str(out)]), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["edges"], [50, 100])
            self.assertEqual(main(["--a", str(pa), "--b", str(pb), "--edges", ""]), 1)


if __name__ == "__main__":
    unittest.main()
