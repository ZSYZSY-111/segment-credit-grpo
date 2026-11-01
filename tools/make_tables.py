#!/usr/bin/env python3
"""Render the Stage 5 results collected by tools/collect_stage5.py as Markdown tables for the README and docs.

  python tools/make_tables.py results/stage5_runs.json [--arms standard,segment_credit,uniform,shuffled,sequence]

Three tables: one row per run (accuracy with Wilson interval, format failures, output length, early answer letters,
runtime, and the last-step training metrics), the same-seed paired differences against standard GRPO, and the
cross-seed summary from aggregate_seeds.py. Missing pieces are left blank rather than failing, so the tables can be
rendered while runs are still finishing.
"""
import argparse
import json
import sys
from collections import OrderedDict


def fmt(value, digits=2, signed=False):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def ci(values):
    return f"{values[0]:.2f} – {values[1]:.2f}" if values else ""


def runs_table(runs, arm_order):
    lines = ["| arm | seed | accuracy | 95% CI | format fail % | chars mean / median | letter in first 20 chars % | runtime h | last-step reward | tokens with non-zero advantage | entropy |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = sorted(runs.values(), key=lambda r: (arm_order.index(r["name"]) if r["name"] in arm_order else len(arm_order), r["name"] or "", r["seed"] or 0))
    for run in ordered:
        arm = (run.get("eval_cmexam_val") or {}).get("arm") or {}
        last = run["curve"][-1] if run.get("curve") else {}
        lines.append("| " + " | ".join([
            run["name"] or "", str(run["seed"] or ""), fmt(arm.get("accuracy")), ci(arm.get("ci95")), fmt(arm.get("format_failure_pct")),
            f"{fmt(arm.get('output_chars_mean'), 0)} / {fmt(arm.get('output_chars_median'), 0)}" if arm else "",
            fmt(arm.get("answer_letter_in_first_20_chars_pct"), 1), fmt(run.get("train_runtime_hours")),
            fmt(last.get("reward"), 3), fmt(last.get("token_frac_nonzero_adv"), 2), fmt(last.get("entropy"), 2)]) + " |")
    return "\n".join(lines)


def same_seed_table(summary):
    lines = []
    for label, comp in (summary or {}).get("comparisons", {}).items():
        lines += [f"**{label}**", "", "| seed | standard | arm | difference | bootstrap 95% | McNemar p | discordance | MDE |", "|---|---:|---:|---:|---|---:|---:|---:|"]
        for seed, v in comp["same_seed"].items():
            lines.append(f"| {seed} | {v['acc_a']:.2f} | {v['acc_b']:.2f} | {v['delta_b_minus_a']:+.2f} | "
                         f"{v['delta_ci'][0]:+.2f} to {v['delta_ci'][1]:+.2f} | {v['mcnemar_p']:.2g} | {100 * v['discordance']:.1f}% | {v['mde_points']} |")
        d = comp["same_seed_delta"]
        if d["n"]:
            spread = f"mean {d['mean']:+.2f}" + (f", std {d['std']:.2f}, min {d['min']:+.2f}, max {d['max']:+.2f}" if d["std"] is not None else "")
            lines.append(f"\nacross {d['n']} seed(s): {spread}; positive in {d['positive']}/{d['n']}, significant in {d['significant_positive']}/{d['n']}")
        c = comp["cross_seed"]["delta"]
        if c["n"]:
            lines.append(f"all {c['n']} cross-seed pairings: mean {c['mean']:+.2f}, min {c['min']:+.2f}, max {c['max']:+.2f}")
        lines.append("")
    return "\n".join(lines)


def arms_summary_table(summary):
    lines = ["| arm | seeds | mean accuracy | std | min | max |", "|---|---:|---:|---:|---:|---:|"]
    for arm, s in (summary or {}).get("arms", {}).items():
        a = s["accuracy_across_seeds"]
        lines.append(f"| {arm} | {a['n']} | {fmt(a['mean'])} | {fmt(a['std'])} | {fmt(a['min'])} | {fmt(a['max'])} |")
    return "\n".join(lines)


def render(report, arm_order):
    parts = ["### Runs", "", runs_table(report["runs"], arm_order), ""]
    if report.get("seeds_summary"):
        parts += ["### Accuracy across seeds", "", arms_summary_table(report["seeds_summary"]), "",
                  "### Paired differences against standard GRPO", "", same_seed_table(report["seeds_summary"])]
    start = report.get("stage4_start")
    if start:
        parts += [f"RL start (segment-weighted SFT, seed 42): pass@1 greedy {100 * start['overall']['pass1_greedy']:.2f} on {start['overall']['questions']} questions", ""]
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", default="results/stage5_runs.json")
    parser.add_argument("--arms", default="standard,segment_credit,uniform,shuffled,sequence,lambda1,lambda4,entropy,g16_standard,g16_segment,long_standard,long_segment,adaptive")
    args = parser.parse_args(argv)
    report = json.load(open(args.results, encoding="utf-8"), object_pairs_hook=OrderedDict)
    print(render(report, [a for a in args.arms.split(",") if a]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
