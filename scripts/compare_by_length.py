#!/usr/bin/env python3
"""Length-controlled paired comparison of two record files (eval_validation headroom_records.jsonl).

Both GRPO arms learn to write shorter rationales, and shorter rationales mislead less often, so an accuracy gap could
be a length effect. This script joins the two files on sample_id, buckets questions by output length and reports:

  * the length distribution of each arm (mean, median, quartiles of greedy_text characters);
  * the paired difference inside each length bucket, where the bucket is defined by arm A's output length (so the
    stratification does not depend on B's behaviour), with bootstrap interval and exact McNemar p;
  * the stratified difference: the per-bucket differences weighted by bucket size, which is the gap at equal length;
  * the same-bucket difference: only pairs whose two outputs fall in the same length bucket.

  python scripts/compare_by_length.py --a standard.jsonl --b segment_credit.jsonl --edges 40,80,160 [--json out.json]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

try:
    from .compare_headroom_records import load_records, summarize
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_headroom_records import load_records, summarize


def output_length(record):
    return len((record.get("greedy_text") or "").strip())


def bucket_of(length, edges):
    """Index of the first edge greater than length; len(edges) for the open-ended last bucket."""
    for index, edge in enumerate(edges):
        if length < edge:
            return index
    return len(edges)


def bucket_labels(edges):
    labels, start = [], 0
    for edge in edges:
        labels.append(f"[{start}, {edge})")
        start = edge
    labels.append(f"[{start}, ∞)")
    return labels


def quantiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"mean": round(sum(ordered) / len(ordered), 1), "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75)}


def compare_by_length(records_a, records_b, edges, resamples=10000, seed=42):
    common = [sid for sid in records_a if sid in records_b]
    labels = bucket_labels(edges)
    by_bucket = OrderedDict((label, []) for label in labels)
    same_bucket = []
    for sid in common:
        a, b = records_a[sid], records_b[sid]
        pair = (bool(a["greedy_correct"]), bool(b["greedy_correct"]))
        ba, bb = bucket_of(output_length(a), edges), bucket_of(output_length(b), edges)
        by_bucket[labels[ba]].append(pair)
        if ba == bb:
            same_bucket.append(pair)
    buckets = OrderedDict()
    weighted, total = 0.0, 0
    for label, pairs in by_bucket.items():
        if not pairs:
            continue
        summary = summarize(pairs, resamples, seed)
        buckets[label] = summary
        weighted += summary["delta_b_minus_a"] * len(pairs)
        total += len(pairs)
    all_pairs = [(bool(records_a[s]["greedy_correct"]), bool(records_b[s]["greedy_correct"])) for s in common]
    return {
        "common": len(common), "edges": list(edges),
        "length_a": quantiles([output_length(records_a[s]) for s in common]),
        "length_b": quantiles([output_length(records_b[s]) for s in common]),
        "overall": summarize(all_pairs, resamples, seed) if all_pairs else None,
        "by_bucket_of_a": buckets,
        "stratified_delta_b_minus_a": round(weighted / total, 2) if total else None,
        "same_bucket": {"n": len(same_bucket), **summarize(same_bucket, resamples, seed)} if same_bucket else None,
    }


def render(report, label_a, label_b):
    lines = [f"shared questions {report['common']}; output length (chars) {label_a}: {report['length_a']}  {label_b}: {report['length_b']}", ""]
    if report["overall"]:
        o = report["overall"]
        lines.append(f"overall {label_b} − {label_a} = {o['delta_b_minus_a']:+.2f} (bootstrap 95% {o['delta_ci'][0]:+.2f} to {o['delta_ci'][1]:+.2f}, p = {o['mcnemar_p']:.2g})")
    lines += ["", f"| {label_a} length bucket | n | {label_a} | {label_b} | B − A | 95% CI | p |", "|---|---:|---:|---:|---:|---|---:|"]
    for label, s in report["by_bucket_of_a"].items():
        lines.append(f"| {label} | {s['n']} | {s['acc_a']:.2f} | {s['acc_b']:.2f} | {s['delta_b_minus_a']:+.2f} | "
                     f"{s['delta_ci'][0]:+.2f} to {s['delta_ci'][1]:+.2f} | {s['mcnemar_p']:.2g} |")
    lines.append("")
    if report["stratified_delta_b_minus_a"] is not None:
        lines.append(f"stratified (equal-length) difference: {report['stratified_delta_b_minus_a']:+.2f}")
    if report["same_bucket"]:
        s = report["same_bucket"]
        lines.append(f"pairs in the same length bucket ({s['n']}): {s['delta_b_minus_a']:+.2f} "
                     f"({s['delta_ci'][0]:+.2f} to {s['delta_ci'][1]:+.2f}, p = {s['mcnemar_p']:.2g})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--edges", default="40,80,160", help="bucket edges in characters of the greedy output")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    edges = sorted({int(e) for e in args.edges.split(",") if e.strip()})
    if not edges:
        print("error: --edges needs at least one integer", file=sys.stderr)
        return 1
    report = compare_by_length(load_records(args.a), load_records(args.b), edges, args.resamples, args.seed)
    if report["common"] == 0:
        print("the two record files share no sample_id", file=sys.stderr)
        return 1
    print(render(report, args.label_a, args.label_b))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
