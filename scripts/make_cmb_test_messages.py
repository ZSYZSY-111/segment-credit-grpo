#!/usr/bin/env python3
"""Convert the official CMB-Exam test split (questions + answers) to the messages JSONL that eval_validation.py reads,
so rationale-mode adapters can be scored on CMB-test with the same per-question records as on the CMExam split.

  python scripts/make_cmb_test_messages.py --questions data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json \\
      --answers data/CMB/CMB-test-choice-answer.json --output data/cmb_test_messages.jsonl

Rows are rendered with the direct-answer template (eval_validation re-renders the prompt for the requested mode), the
assistant turn is the sorted answer letters, and trailing empty options are kept exactly as the official evaluator
renders them. A sidecar <output>.meta.jsonl carries sample_id, the official id, exam_type, exam_class, exam_subject and
question_type for slicing. Rows the row parser rejects are counted per reason and listed in <output>.dropped.jsonl.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from .build_cmb_train_sft import parse_cmb_row
    from .cmexam_prompts import MODE_DIRECT, build_prompt_messages
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_cmb_train_sft import parse_cmb_row
    from cmexam_prompts import MODE_DIRECT, build_prompt_messages


def load_answers(path):
    rows = json.load(open(path, encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("answers") or list(rows.values())
    return {str(r["id"]): r["answer"] for r in rows}


def convert(questions, answers):
    """Returns (rows, meta, dropped, counts); rows are {"messages": [...]} dicts."""
    rows, meta, dropped, counts = [], [], [], Counter()
    for index, question in enumerate(questions):
        qid = str(question.get("id", index + 1))
        answer = answers.get(qid)
        if answer is None:
            counts["no_answer"] += 1
            dropped.append({"id": qid, "reason": "no_answer"})
            continue
        try:
            parsed = parse_cmb_row({**question, "answer": answer})
        except ValueError as error:
            counts[str(error)] += 1
            dropped.append({"id": qid, "reason": str(error)})
            continue
        messages = build_prompt_messages(parsed["question"], parsed["options"], MODE_DIRECT)
        messages.append({"role": "assistant", "content": parsed["answer"]})
        user_content = "\n".join(m["content"] for m in messages[:-1] if m["role"] == "user")
        sample_id = hashlib.sha256(user_content.encode("utf-8")).hexdigest()
        rows.append({"messages": messages})
        meta.append({"sample_id": sample_id, "id": qid, "exam_type": question.get("exam_type", ""),
                     "exam_class": question.get("exam_class", ""), "exam_subject": question.get("exam_subject", ""),
                     "question_type": question.get("question_type", ""), "answer": parsed["answer"]})
        counts["converted"] += 1
    return rows, meta, dropped, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    questions = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("data") or questions.get("questions") or list(questions.values())
    rows, meta, dropped, counts = convert(questions, load_answers(args.answers))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output.with_suffix(".meta.jsonl").open("w", encoding="utf-8") as handle:
        for row in meta:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output.with_suffix(".dropped.jsonl").open("w", encoding="utf-8") as handle:
        for row in dropped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"questions": len(questions), **counts}, ensure_ascii=False))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
