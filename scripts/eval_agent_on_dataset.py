from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.answer_normalizer import AnswerContext, DEFAULT_NORMALIZER  # noqa: E402
from agent.agent_utils import extract_final_answer  # noqa: E402
from agent.error_analysis import (  # noqa: E402
    build_error_report,
    diagnose_record,
    render_error_report_markdown,
)
from user_agent import ReasoningAgent  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("idx", line_number)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def compact_diagnostic_trace(trace: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the control-flow signals needed for attribution without full model text."""
    useful_steps = {
        "route",
        "candidate",
        "candidate_rejected",
        "attack_verifier",
        "refine",
        "consensus_lock",
        "final_judge",
        "final_judge_rejected",
        "solver_error",
        "attack_error",
        "refine_error",
        "final_judge_error",
        "select_final_response",
    }
    compact: List[Dict[str, Any]] = []
    for entry in trace:
        if entry.get("step") not in useful_steps:
            continue
        content = entry.get("content")
        if isinstance(content, dict):
            content = {
                key: value
                for key, value in content.items()
                if key not in {"response_preview", "response_tail", "report_preview"}
            }
        compact.append({"step": entry.get("step"), "content": content})
    return compact


def answer_key(value: str, answer_type: str = "", contest: str = "") -> str:
    extracted = extract_final_answer(value)
    context = AnswerContext(answer_type=answer_type, contest=contest)
    return DEFAULT_NORMALIZER.canonicalize(extracted or value, context)


def is_match(prediction: str, expected: str, answer_type: str = "", contest: str = "") -> bool:
    pred_key = answer_key(prediction, answer_type, contest)
    expected_key = answer_key(expected, answer_type, contest)
    if not pred_key or not expected_key:
        return False
    return pred_key == expected_key


def classify_error(prediction: str, expected: str) -> str:
    if is_match(prediction, expected):
        return "correct"
    raw = str(prediction or "").strip()
    lowered = raw.lower().replace(" ", "")
    if not raw:
        return "empty_prediction"
    if "<最终答案>" in raw or "<final" in lowered or "<answer" in lowered:
        return "placeholder_leak"
    if "i need to output" in lowered or "outputformat" in lowered or "selected:" in lowered:
        return "format_meta_leak"
    if len(lowered) > 120:
        return "long_non_answer"
    if any(token in lowered for token in ["final_answer", "reason:", "verdict:", "confidence:"]):
        return "unstripped_protocol_text"
    return "math_or_normalization_mismatch"


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row.get("is_correct"))
    by_subject: Dict[str, Counter] = defaultdict(Counter)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    by_contest: Dict[str, Counter] = defaultdict(Counter)
    error_types = Counter()

    for row in records:
        value = "correct" if row.get("is_correct") else "wrong"
        error_types[str(row.get("error_type", value))] += 1
        subjects = row.get("subject") or ["unknown"]
        if isinstance(subjects, str):
            subjects = [subjects]
        for subject in subjects:
            by_subject[str(subject)][value] += 1
        by_type[str(row.get("answer_type", "unknown"))][value] += 1
        by_contest[str(row.get("contest", "unknown"))][value] += 1

    def collapse(counter_map: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
        output = {}
        for key, counter in sorted(counter_map.items()):
            subtotal = counter["correct"] + counter["wrong"]
            accuracy = counter["correct"] / subtotal if subtotal else 0.0
            output[key] = {
                "total": subtotal,
                "correct": counter["correct"],
                "accuracy": round(accuracy, 4),
            }
        return output

    error_report = build_error_report(records)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_subject": collapse(by_subject),
        "by_answer_type": collapse(by_type),
        "by_contest": collapse(by_contest),
        "by_error_type": dict(error_types.most_common()),
        "error_analysis": {
            "diagnostic_coverage": error_report["diagnostic_coverage"],
            "by_primary_cause": error_report["by_primary_cause"],
            "by_root_cause": error_report["by_root_cause"],
        },
    }


def run_agent(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from agent.llm_client import InternChatClient

    dataset = load_jsonl(args.dataset)
    if args.ids:
        requested_ids = set(args.ids)
        dataset = [item for item in dataset if str(item.get("id")) in requested_ids]
    if args.offset:
        dataset = dataset[args.offset :]
    if args.limit:
        dataset = dataset[: args.limit]

    client = InternChatClient()
    agent = ReasoningAgent(client=client)
    results = []
    start_time = time.time()

    for position, item in enumerate(dataset, start=1):
        item_start = time.time()
        try:
            agent_result = agent.solve(
                problem=item["problem"],
                metadata={
                    "idx": item.get("idx", position - 1),
                    "source_id": item.get("id"),
                    "subject": item.get("subject"),
                    "contest": item.get("contest"),
                    "answer_type": item.get("answer_type"),
                    "difficulty": item.get("difficulty"),
                },
            )
            final_response = str(agent_result.get("final_response", ""))
            status = "success"
            error = None
            trace = agent_result.get("trace", [])
        except Exception as exc:  # noqa: BLE001 - keep evaluating remaining rows.
            final_response = ""
            status = "error"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            trace = []

        expected = str(item.get("canonical_answer") or item.get("answer") or "")
        answer_type = str(item.get("answer_type") or "")
        contest = str(item.get("contest") or "")
        record = {
            "idx": item.get("idx", position - 1),
            "id": item.get("id"),
            "contest": item.get("contest"),
            "year": item.get("year"),
            "problem_no": item.get("problem_no"),
            "subject": item.get("subject"),
            "answer_type": item.get("answer_type"),
            "status": status,
            "expected": expected,
            "prediction": final_response,
            "expected_key": answer_key(expected, answer_type, contest),
            "prediction_key": answer_key(final_response, answer_type, contest),
            "is_correct": is_match(final_response, expected, answer_type, contest),
            "error_type": classify_error(final_response, expected),
            "elapsed_seconds": round(time.time() - item_start, 3),
            "error": error,
            "trace": trace if args.keep_trace else compact_diagnostic_trace(trace),
        }
        record["diagnosis"] = diagnose_record(record)
        record["error_type"] = record["diagnosis"]["primary_cause"]
        results.append(record)
        print(
            f"[{position}/{len(dataset)}] {record['id']} "
            f"{'OK' if record['is_correct'] else 'WRONG'} "
            f"pred={record['prediction_key']} expected={record['expected_key']} "
            f"elapsed={record['elapsed_seconds']}s",
            flush=True,
        )

    print(f"Total elapsed: {round(time.time() - start_time, 3)}s")
    return results


def evaluate_existing(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.predictions)
    for row in rows:
        answer_type = str(row.get("answer_type") or "")
        contest = str(row.get("contest") or "")
        row["is_correct"] = is_match(
            str(row.get("prediction", "")), str(row.get("expected", "")), answer_type, contest
        )
        row["prediction_key"] = answer_key(str(row.get("prediction", "")), answer_type, contest)
        row["expected_key"] = answer_key(str(row.get("expected", "")), answer_type, contest)
        row["diagnosis"] = diagnose_record(row)
        row["error_type"] = row["diagnosis"]["primary_cause"]
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ReasoningAgent on a JSONL math dataset.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_math_aime_500.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval_outputs" / "agent_eval.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "eval_outputs" / "agent_eval_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Evaluate only the listed dataset record IDs.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many records before applying --limit.",
    )
    parser.add_argument("--keep-trace", action="store_true")
    parser.add_argument("--error-report", type=Path, help="JSON error-analysis report path.")
    parser.add_argument("--error-report-md", type=Path, help="Markdown error-analysis report path.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Evaluate an existing prediction JSONL instead of running the agent.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.predictions:
        records = evaluate_existing(args)
    else:
        records = run_agent(args)
    summary = evaluate_records(records)
    error_report = build_error_report(records)
    write_jsonl(args.output, records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.error_report or args.output.with_name(f"{args.output.stem}_error_report.json")
    markdown_path = args.error_report_md or args.output.with_name(f"{args.output.stem}_error_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(error_report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_error_report_markdown(error_report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {args.output}")
    print(f"Wrote summary to {args.summary}")
    print(f"Wrote error report to {report_path}")
    print(f"Wrote Markdown error report to {markdown_path}")


if __name__ == "__main__":
    main()
