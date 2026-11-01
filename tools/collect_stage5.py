#!/usr/bin/env python3
"""Collect every Stage 5 run under an output directory into one results JSON.

For each training_outputs/stage5_<name>_seed<seed>/ the script gathers the recorded configuration (run_config.json,
segment_credit.json), the training curve (trainer_state.json log_history, every fifth step plus the first and the
last), the training runtime (train_results.json), the CMExam validation evaluation (headroom_report.json plus the
arm-level statistics computed from headroom_records.jsonl: Wilson interval, format failures, output length, share of
outputs that name an answer letter within their first 20 characters) and, when present, the CMB-test evaluation.
Per-seed paired comparisons (stage5_standard_vs_<name>_seed<seed>.json) and the cross-seed summary
(stage5_seeds.json) are copied in as they are.

  python tools/collect_stage5.py --out-dir training_outputs --output results/stage5_runs.json
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_predictions import wilson  # noqa: E402

RUN_DIR = re.compile(r"^stage5_(?P<name>.+)_seed(?P<seed>\d+)$")
CURVE_KEYS = OrderedDict([
    ("reward", "reward"),
    ("mean_length", "completions/mean_length"),
    ("frac_reward_zero_std", "frac_reward_zero_std"),
    ("entropy", "entropy"),
    ("token_frac_nonzero_adv", "credit/token_frac_nonzero_adv"),
    ("token_frac_nonzero_seq_adv", "credit/token_frac_nonzero_seq_adv"),
    ("phi_first_mean", "credit/phi_first_mean"),
    ("phi_last_mean", "credit/phi_last_mean"),
    ("abs_credit_mean", "credit/abs_credit_mean"),
    ("letter_mass_mean", "credit/letter_mass_mean"),
    ("probe_seconds", "credit/probe_seconds"),
    ("mem/peak_alloc_gb", "mem/peak_alloc_gb"),
])
CONFIG_KEYS = ("segment_credit_lambda", "credit_mode", "reward_mode", "max_steps", "num_generations", "prompts_per_step",
               "learning_rate", "entropy_coef", "seed", "generation_chunk", "segment_probe_batch", "segment_max_probes")
LETTER = re.compile(r"(?<![A-Za-z])[A-G](?![A-Za-z])")


def load_json(path):
    path = Path(path)
    return json.load(path.open(encoding="utf-8")) if path.is_file() else None


def curve_from_log_history(log_history, every=5):
    """Training entries (those carrying a reward), thinned to steps 1, every fifth step, and the last one."""
    entries = [e for e in log_history if "reward" in e and "step" in e]
    if not entries:
        return []
    last_step = entries[-1]["step"]
    curve = []
    for entry in entries:
        step = entry["step"]
        if not (step == 1 or step % every == 0 or step == last_step):
            continue
        point = {"step": float(step)}
        for out_key, in_key in CURVE_KEYS.items():
            if in_key in entry and entry[in_key] is not None:
                point[out_key] = round(float(entry[in_key]), 4)
        curve.append(point)
    return curve


def answer_letter_early(text, answer, chars=20):
    """Does an answer letter appear within the first `chars` characters of the rationale (after the prefix)?"""
    body = (text or "").strip()
    if body.startswith("解析："):
        body = body[len("解析："):]
    head = body[:chars]
    letters = set(answer or "")
    return any(m.group(0) in letters for m in LETTER.finditer(head)) if letters else bool(LETTER.search(head))


def arm_statistics(records_path):
    """The per-arm statistics the recorded results report, from eval_validation's headroom_records.jsonl."""
    rows = [json.loads(l) for l in Path(records_path).open(encoding="utf-8") if l.strip()]
    if not rows:
        return None
    n = len(rows)
    correct = sum(1 for r in rows if r.get("greedy_correct"))
    acc, lo, hi = wilson(correct, n)
    lengths = sorted(len((r.get("greedy_text") or "").strip()) for r in rows)
    early = sum(1 for r in rows if answer_letter_early(r.get("greedy_text"), r.get("greedy_answer") or r.get("gold")))
    return {
        "n": n, "correct": correct, "accuracy": round(100 * acc, 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "format_failure_pct": round(100 * sum(1 for r in rows if r.get("greedy_answer") is None) / n, 2),
        "output_chars_mean": round(sum(lengths) / n, 1), "output_chars_median": lengths[n // 2],
        "share_under_40_chars_pct": round(100 * sum(1 for x in lengths if x < 40) / n, 1),
        "answer_letter_in_first_20_chars_pct": round(100 * early / n, 1),
    }


def collect_run(run_dir):
    run_dir = Path(run_dir)
    out = {"name": None, "seed": None}
    match = RUN_DIR.match(run_dir.name)
    if match:
        out["name"], out["seed"] = match.group("name"), int(match.group("seed"))
    config = load_json(run_dir / "run_config.json") or {}
    arguments = config.get("arguments", {})
    out["config"] = {k: arguments.get(k) for k in CONFIG_KEYS if k in arguments}
    out["batch_geometry"] = config.get("batch_geometry")
    out["versions"] = config.get("versions")
    out["gpu"] = (config.get("resolved") or {}).get("gpu")
    out["segment_credit"] = load_json(run_dir / "segment_credit.json")
    state = load_json(run_dir / "trainer_state.json") or load_json(run_dir / "final_adapter" / "trainer_state.json")
    out["curve"] = curve_from_log_history(state.get("log_history", [])) if state else []
    train = load_json(run_dir / "train_results.json")
    out["train_runtime_hours"] = round(train["train_runtime"] / 3600, 2) if train and "train_runtime" in train else None
    for key, sub in (("eval_cmexam_val", "eval_cmexam_val"), ("eval_cmb_test", "eval_cmb_test")):
        report = load_json(run_dir / sub / "headroom_report.json")
        records = run_dir / sub / "headroom_records.jsonl"
        if report or records.is_file():
            out[key] = {"overall": (report or {}).get("overall"), "slices": (report or {}).get("slices"),
                        "arm": arm_statistics(records) if records.is_file() else None}
    return out


def collect(out_dir):
    out_dir = Path(out_dir)
    runs = OrderedDict()
    for run_dir in sorted(p for p in out_dir.glob("stage5_*_seed*") if p.is_dir()):
        runs[run_dir.name] = collect_run(run_dir)
    comparisons = OrderedDict()
    for path in sorted(out_dir.glob("stage5_standard_vs_*_seed*.json")):
        comparisons[path.stem] = load_json(path)
    return {"runs": runs, "comparisons": comparisons, "seeds_summary": load_json(out_dir / "stage5_seeds.json"),
            "stage4_start": load_json(out_dir / "stage4_weighted_seed42" / "eval_cmexam_val" / "headroom_report.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="training_outputs")
    parser.add_argument("--output", default="results/stage5_runs.json")
    args = parser.parse_args(argv)
    report = collect(args.out_dir)
    if not report["runs"]:
        print(f"no stage5_*_seed* directories under {args.out_dir}", file=sys.stderr)
        return 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for key, run in report["runs"].items():
        arm = (run.get("eval_cmexam_val") or {}).get("arm") or {}
        print(f"{key}: steps={len(run['curve'])} points, runtime_h={run['train_runtime_hours']}, "
              f"val_acc={arm.get('accuracy')} (n={arm.get('n')}), chars={arm.get('output_chars_mean')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
