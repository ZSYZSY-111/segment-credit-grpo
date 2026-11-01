#!/usr/bin/env python3
"""Aggregate per-question records across training seeds: per-seed paired comparisons, cross-seed means and the
spread of the paired difference.

Input: one headroom_records.jsonl (eval_validation.py) per arm and seed, given explicitly or through a pattern with
{arm} and {seed} placeholders:

  python scripts/aggregate_seeds.py --pattern "training_outputs/stage5_{arm}_seed{seed}/eval_cmexam_val/headroom_records.jsonl" \\
      --arms standard,segment_credit --seeds 42,7,2024 --compare segment_credit:standard --json out.json
  python scripts/aggregate_seeds.py --run standard 42 a42.jsonl --run standard 7 a7.jsonl \\
      --run segment_credit 42 b42.jsonl --run segment_credit 7 b7.jsonl --compare segment_credit:standard

Per arm: every seed's accuracy with its Wilson interval, and the mean, sample standard deviation, minimum and maximum
across seeds. Per comparison B:A (reported as B − A):
  * same-seed pairs, B_s against A_s for every seed both arms have: the 2×2 table, the difference with a paired
    bootstrap interval, the exact McNemar p-value, the discordance and the minimum detectable effect;
  * the spread of those differences across seeds, with how many are positive and how many are significant;
  * every cross-seed pair, B_s against A_t, and the minimum, mean and maximum of their differences, so the
    conclusion does not rest on one pairing of seeds.
Record files that do not exist yet are listed and skipped, so the summary can be run while seeds are still training.
"""
import argparse
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

try:
    from .compare_headroom_records import load_records, summarize
    from .score_predictions import wilson
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_headroom_records import load_records, summarize
    from score_predictions import wilson


def parse_list(text):
    """'42,7 2024' → ['42', '7', '2024']."""
    return [item for item in re.split(r"[,\s]+", (text or "").strip()) if item]


def expand_pattern(pattern, arms, seeds):
    if "{arm}" not in pattern or "{seed}" not in pattern:
        raise ValueError("--pattern needs both {arm} and {seed} placeholders")
    return [(arm, seed, pattern.format(arm=arm, seed=seed)) for arm in arms for seed in seeds]


def collect_runs(run_triples, pattern=None, arms=(), seeds=()):
    """Return ({arm: {seed: path}}, missing) with only the files that exist; naming one run twice is an error."""
    runs, missing, seen = OrderedDict(), [], set()
    triples = [tuple(item) for item in (run_triples or [])]
    if pattern:
        triples.extend(expand_pattern(pattern, arms, seeds))
    for arm, seed, path in triples:
        seed = str(seed)
        if (arm, seed) in seen:
            raise ValueError(f"run {arm} seed {seed} is given twice")
        seen.add((arm, seed))
        if not Path(path).is_file():
            missing.append({"arm": arm, "seed": seed, "path": str(path)})
            continue
        runs.setdefault(arm, OrderedDict())[seed] = str(path)
    return runs, missing


def load_correct(path):
    return OrderedDict((sid, bool(record["greedy_correct"])) for sid, record in load_records(path).items())


def spread(values):
    """Mean, sample standard deviation (None with fewer than two values), minimum and maximum."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
    return {"n": len(values), "mean": round(mean, 2), "std": None if std is None else round(std, 2),
            "min": round(min(values), 2), "max": round(max(values), 2)}


def arm_summary(correct_by_seed):
    per_seed = OrderedDict()
    for seed, correct in correct_by_seed.items():
        n, k = len(correct), sum(correct.values())
        acc, lo, hi = wilson(k, n)
        per_seed[seed] = {"n": n, "correct": k, "accuracy": round(100 * acc, 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)]}
    return {"seeds": per_seed, "accuracy_across_seeds": spread([v["accuracy"] for v in per_seed.values()])}


def paired(correct_a, correct_b, resamples, seed):
    common = [sid for sid in correct_a if sid in correct_b]
    out = summarize([(correct_a[sid], correct_b[sid]) for sid in common], resamples, seed)
    out["only_in_a"] = len(correct_a) - len(common)
    out["only_in_b"] = len(correct_b) - len(common)
    return out


def compare_arms(correct_a_by_seed, correct_b_by_seed, resamples=10000, seed=42, alpha=0.05):
    """B − A. same_seed: one paired comparison per seed both arms have; cross_seed: every B seed against every A seed."""
    same = OrderedDict()
    for s, correct_a in correct_a_by_seed.items():
        if s in correct_b_by_seed:
            same[s] = paired(correct_a, correct_b_by_seed[s], resamples, seed)
    cross = [{"seed_a": sa, "seed_b": sb, **paired(ca, cb, resamples, seed)}
             for sa, ca in correct_a_by_seed.items() for sb, cb in correct_b_by_seed.items()]
    same_deltas = [v["delta_b_minus_a"] for v in same.values()]
    return {
        "same_seed": same,
        "same_seed_delta": {**spread(same_deltas),
                            "positive": sum(1 for d in same_deltas if d > 0),
                            "significant_positive": sum(1 for v in same.values() if v["delta_b_minus_a"] > 0 and v["mcnemar_p"] < alpha)},
        "cross_seed": {"pairs": cross, "delta": spread([p["delta_b_minus_a"] for p in cross])},
    }


def parse_compare(spec):
    if ":" not in spec:
        raise ValueError(f"--compare takes B:A and reports B − A, got {spec!r}")
    b, a = (part.strip() for part in spec.split(":", 1))
    if not a or not b:
        raise ValueError(f"--compare needs two arm names, got {spec!r}")
    return b, a


def build_report(runs, missing, compares, resamples=10000, seed=42):
    correct = OrderedDict((arm, OrderedDict((s, load_correct(p)) for s, p in by_seed.items())) for arm, by_seed in runs.items())
    report = {"runs": {arm: dict(by_seed) for arm, by_seed in runs.items()}, "missing": missing,
              "arms": OrderedDict((arm, arm_summary(by_seed)) for arm, by_seed in correct.items()),
              "comparisons": OrderedDict()}
    for b, a in compares:
        if a not in correct or b not in correct:
            raise ValueError(f"comparison {b}:{a} names an arm without any record file")
        report["comparisons"][f"{b} − {a}"] = {"a": a, "b": b, **compare_arms(correct[a], correct[b], resamples, seed)}
    return report


def _spread_text(s, signed=False):
    fmt = "{:+.2f}" if signed else "{:.2f}"
    text = f"mean {fmt.format(s['mean'])}"
    if s["std"] is not None:
        text += f", std {s['std']:.2f}, min {fmt.format(s['min'])}, max {fmt.format(s['max'])}"
    return text


def render(report):
    lines = []
    for arm, summary in report["arms"].items():
        lines += [f"### {arm}", "", "| seed | n | accuracy | 95% CI |", "|---|---:|---:|---|"]
        for s, v in summary["seeds"].items():
            lines.append(f"| {s} | {v['n']} | {v['accuracy']:.2f} | {v['ci95'][0]:.2f} – {v['ci95'][1]:.2f} |")
        a = summary["accuracy_across_seeds"]
        lines += ["", f"{a['n']} seed(s): {_spread_text(a)}", ""]
    for name, comp in report["comparisons"].items():
        lines += [f"### {name}", "", f"| seed | {comp['a']} | {comp['b']} | B − A | bootstrap 95% | McNemar p | discordance | MDE |",
                  "|---|---:|---:|---:|---|---:|---:|---:|"]
        for s, v in comp["same_seed"].items():
            lines.append(f"| {s} | {v['acc_a']:.2f} | {v['acc_b']:.2f} | {v['delta_b_minus_a']:+.2f} | "
                         f"{v['delta_ci'][0]:+.2f} to {v['delta_ci'][1]:+.2f} | {v['mcnemar_p']:.2g} | "
                         f"{100 * v['discordance']:.1f}% | {v['mde_points']} |")
        d = comp["same_seed_delta"]
        lines.append("")
        if d["n"]:
            lines.append(f"same-seed difference over {d['n']} seed(s): {_spread_text(d, signed=True)}; "
                         f"positive in {d['positive']}/{d['n']}, significant (p < 0.05) in {d['significant_positive']}/{d['n']}")
        c = comp["cross_seed"]["delta"]
        lines.append(f"cross-seed pairs ({c['n']}): {_spread_text(c, signed=True)}")
        lines.append("")
    if report["missing"]:
        lines.append("missing record files (skipped): " + ", ".join(f"{m['arm']} seed {m['seed']}" for m in report["missing"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=3, action="append", default=[], metavar=("ARM", "SEED", "PATH"),
                        help="one record file; repeatable")
    parser.add_argument("--pattern", default=None, help="path with {arm} and {seed} placeholders, expanded over --arms × --seeds")
    parser.add_argument("--arms", default="", help="comma-separated arm names for --pattern")
    parser.add_argument("--seeds", default="", help="comma- or space-separated seeds for --pattern")
    parser.add_argument("--compare", action="append", default=[], metavar="B:A", help="report B − A; repeatable")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    try:
        runs, missing = collect_runs(args.run, args.pattern, parse_list(args.arms), parse_list(args.seeds))
        if not runs:
            raise ValueError("no record file found")
        report = build_report(runs, missing, [parse_compare(spec) for spec in args.compare], args.resamples, args.seed)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
