import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_seeds import (
    build_report,
    collect_runs,
    compare_arms,
    main,
    parse_compare,
    parse_list,
    spread,
)


def write_records(path, correct_flags):
    rows = [{"sample_id": f"s{i}", "greedy_correct": flag, "greedy_answer": "A", "slices": {}}
            for i, flag in enumerate(correct_flags)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def flags(n, predicate):
    return [bool(predicate(i)) for i in range(n)]


class SpreadAndParsingTests(unittest.TestCase):
    def test_spread_uses_sample_standard_deviation(self):
        s = spread([1.0, 2.0, 3.0])
        self.assertEqual((s["n"], s["mean"], s["std"], s["min"], s["max"]), (3, 2.0, 1.0, 1.0, 3.0))
        self.assertIsNone(spread([5.0])["std"])
        self.assertEqual(spread([])["n"], 0)

    def test_parse_list_accepts_commas_and_spaces(self):
        self.assertEqual(parse_list("42,7 2024"), ["42", "7", "2024"])
        self.assertEqual(parse_list(""), [])

    def test_parse_compare_reports_b_minus_a(self):
        self.assertEqual(parse_compare("segment_credit:standard"), ("segment_credit", "standard"))
        with self.assertRaises(ValueError):
            parse_compare("standard")
        with self.assertRaises(ValueError):
            parse_compare(":standard")


class CollectRunsTests(unittest.TestCase):
    def test_pattern_expansion_skips_missing_files_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, seed in (("a", "1"), ("a", "2"), ("b", "1")):
                (root / f"{arm}_seed{seed}").mkdir()
                write_records(root / f"{arm}_seed{seed}" / "records.jsonl", flags(10, lambda i: True))
            runs, missing = collect_runs([], str(root / "{arm}_seed{seed}" / "records.jsonl"), ["a", "b"], ["1", "2"])
            self.assertEqual(sorted(runs["a"]), ["1", "2"])
            self.assertEqual(sorted(runs["b"]), ["1"])
            self.assertEqual([(m["arm"], m["seed"]) for m in missing], [("b", "2")])
            with self.assertRaises(ValueError):
                collect_runs([("a", 1, str(root / "a_seed1" / "records.jsonl"))],
                             str(root / "{arm}_seed{seed}" / "records.jsonl"), ["a"], ["1"])


class CompareArmsTests(unittest.TestCase):
    def test_same_seed_and_cross_seed_differences(self):
        n = 100
        a = {"1": dict((f"s{i}", i % 2 == 0) for i in range(n)), "2": dict((f"s{i}", i % 2 == 1) for i in range(n))}
        b = {"1": dict((f"s{i}", i % 2 == 0 or i < 20) for i in range(n)), "2": dict((f"s{i}", i % 2 == 1 or i < 20) for i in range(n))}
        report = compare_arms(a, b, resamples=200)
        self.assertEqual(sorted(report["same_seed"]), ["1", "2"])
        for seed, pair in report["same_seed"].items():
            self.assertEqual(pair["delta_b_minus_a"], 10.0)
            self.assertEqual(pair["cells"]["only_a"], 0)
            self.assertEqual(pair["cells"]["only_b"], 10)
            self.assertLess(pair["mcnemar_p"], 0.05)
        d = report["same_seed_delta"]
        self.assertEqual((d["n"], d["mean"], d["std"], d["positive"], d["significant_positive"]), (2, 10.0, 0.0, 2, 2))
        self.assertEqual(len(report["cross_seed"]["pairs"]), 4)
        self.assertEqual({(p["seed_a"], p["seed_b"]) for p in report["cross_seed"]["pairs"]},
                         {("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")})
        self.assertEqual(report["cross_seed"]["delta"]["mean"], 10.0)

    def test_single_seed_has_no_standard_deviation(self):
        a = {"42": dict((f"s{i}", i < 50) for i in range(100))}
        b = {"42": dict((f"s{i}", i < 55) for i in range(100))}
        report = compare_arms(a, b, resamples=100)
        self.assertEqual(report["same_seed"]["42"]["delta_b_minus_a"], 5.0)
        self.assertIsNone(report["same_seed_delta"]["std"])
        self.assertEqual(report["cross_seed"]["delta"]["n"], 1)


class CliTests(unittest.TestCase):
    def test_cli_writes_json_and_lists_missing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for arm, seed, predicate in (("standard", "42", lambda i: i % 2 == 0), ("standard", "7", lambda i: i % 2 == 1),
                                         ("segment_credit", "42", lambda i: i % 2 == 0 or i < 10),
                                         ("segment_credit", "7", lambda i: i % 2 == 1 or i < 10)):
                paths[(arm, seed)] = root / f"{arm}_{seed}.jsonl"
                write_records(paths[(arm, seed)], flags(60, predicate))
            out = root / "seeds.json"
            argv = ["--compare", "segment_credit:standard", "--json", str(out), "--resamples", "100",
                    "--run", "standard", "2024", str(root / "absent.jsonl")]
            for (arm, seed), path in paths.items():
                argv += ["--run", arm, seed, str(path)]
            self.assertEqual(main(argv), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["arms"]["standard"]["accuracy_across_seeds"]["n"], 2)
            self.assertEqual(report["arms"]["standard"]["seeds"]["42"]["accuracy"], 50.0)
            comparison = report["comparisons"]["segment_credit − standard"]
            self.assertEqual(sorted(comparison["same_seed"]), ["42", "7"])
            self.assertEqual(comparison["same_seed_delta"]["positive"], 2)
            self.assertEqual([m["seed"] for m in report["missing"]], ["2024"])

    def test_cli_fails_without_records_or_with_unknown_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            write_records(path, flags(5, lambda i: True))
            self.assertEqual(main(["--run", "a", "1", str(Path(tmp) / "none.jsonl")]), 1)
            self.assertEqual(main(["--run", "a", "1", str(path), "--compare", "b:a"]), 1)
            self.assertEqual(main(["--run", "a", "1", str(path), "--pattern", "no_placeholders.jsonl"]), 1)

    def test_build_report_without_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            write_records(path, flags(8, lambda i: i < 6))
            report = build_report({"a": {"1": str(path)}}, [], [], resamples=50)
            self.assertEqual(report["arms"]["a"]["seeds"]["1"]["accuracy"], 75.0)
            self.assertEqual(report["comparisons"], {})


if __name__ == "__main__":
    unittest.main()
