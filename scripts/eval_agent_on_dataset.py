from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.answer_normalizer import (  # noqa: E402
    is_placeholder_answer,
)
from agent.answer_contract import AnswerContract, has_terminal_answer  # noqa: E402
from agent.agent_utils import safe_exception_message, sanitize_trace_text  # noqa: E402
from agent.answer_matching import answer_key as _answer_key, answers_match  # noqa: E402
from agent.error_analysis import (  # noqa: E402
    build_error_report,
    diagnose_record,
    render_error_report_markdown,
)
from user_agent import ReasoningAgent  # noqa: E402


NON_SCORABLE_BENCHMARK_FLAGS = {"non_unique_open_response", "omitted_choice_dependency"}


def detect_benchmark_flags(problem: str) -> List[str]:
    """Flag stems that cannot be scored reliably after choices were removed."""
    text = " ".join(str(problem or "").lower().split())
    flags: List[str] = []
    if re_search(r"\banother point\b.*\bon (?:this|the) line\b", text):
        flags.append("non_unique_open_response")
    if re_search(r"\bwhich of the following\b", text) and not re_search(
        r"(?:^|\s)[(\[]?[a-e][)\].:]\s", text
    ):
        flags.append("omitted_choice_dependency")
    return flags


def re_search(pattern: str, text: str) -> bool:
    # Local helper keeps the module's public surface focused on evaluation.
    import re

    return bool(re.search(pattern, text, re.IGNORECASE))


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import sympy

        sympy_version = str(sympy.__version__)
    except ImportError:
        sympy_version = "unavailable"
    source_files = [
        ROOT / "agent" / "reasoning_agent.py",
        ROOT / "agent" / "agent_utils.py",
        ROOT / "agent" / "answer_contract.py",
        ROOT / "agent" / "answer_matching.py",
        ROOT / "agent" / "answer_normalizer.py",
        ROOT / "agent" / "answer_selection_agent.py",
        ROOT / "agent" / "error_analysis.py",
        ROOT / "agent" / "tool_verify.py",
        ROOT / "agent" / "rlot_policy.json",
        ROOT / "scripts" / "eval_agent_on_dataset.py",
    ]
    dataset = getattr(args, "dataset", None)
    predictions = getattr(args, "predictions", None)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": {"sympy": sympy_version},
        "model": os.getenv("INTERN_MODEL", "intern-s2-preview"),
        "inputs": {
            "dataset": str(dataset or ""),
            "dataset_sha256": sha256_file(dataset),
            "predictions": str(predictions or ""),
            "predictions_sha256": sha256_file(predictions),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_files
        },
        "records": len(records),
        "unique_ids": len({str(row.get("id")) for row in records}),
    }


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
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
    temporary.replace(path)


def compact_diagnostic_trace(trace: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the control-flow signals needed for attribution without full model text."""
    useful_steps = {
        "route",
        "candidate",
        "candidate_rejected",
        "answer_recovery",
        "answer_recovery_error",
        "attack_verifier",
        "refine",
        "consensus_lock",
        "final_judge",
        "final_judge_rejected",
        "answer_selector",
        "solver_error",
        "attack_error",
        "refine_error",
        "final_judge_error",
        "tool_verify",
        "tool_equivalence_cluster",
        "select_final_response",
        "navigator_state",
        "navigator_policy",
        "navigator_action",
        "navigator_action_error",
        "navigator_forced_terminate",
        "navigator_summary",
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
    return _answer_key(value, answer_type, contest)


def is_match(prediction: str, expected: str, answer_type: str = "", contest: str = "") -> bool:
    return answers_match(prediction, expected, answer_type, contest)


def classify_error(prediction: str, expected: str) -> str:
    if is_match(prediction, expected):
        return "correct"
    raw = str(prediction or "").strip()
    lowered = raw.lower().replace(" ", "")
    if not raw:
        return "empty_prediction"
    if is_placeholder_answer(raw):
        return "placeholder_leak"
    if "<最终答案>" in raw or "<final" in lowered or "<answer" in lowered:
        return "placeholder_leak"
    if (
        "i need to output" in lowered
        or "ineedtooutput" in lowered
        or "iwillputtheanswer" in lowered
        or "afterthecolon" in lowered
        or "outputformat" in lowered
        or "selected:" in lowered
    ):
        return "format_meta_leak"
    if len(lowered) > 120:
        return "long_non_answer"
    if any(token in lowered for token in ["final_answer", "reason:", "verdict:", "confidence:"]):
        return "unstripped_protocol_text"
    return "math_or_normalization_mismatch"


def _trace_contents(row: Dict[str, Any], step: str) -> List[Dict[str, Any]]:
    return [
        entry["content"]
        for entry in row.get("trace") or []
        if entry.get("step") == step and isinstance(entry.get("content"), dict)
    ]


def build_pipeline_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_oracle_ids: List[str] = []
    correct_candidate_lost: List[str] = []
    candidates_total = 0
    protocol_candidates = 0
    recovered_candidates = 0
    contract_rejections = 0
    tool_verdicts: Counter[str] = Counter()
    verifier_verdicts: Counter[str] = Counter()
    selector_sources: Counter[str] = Counter()
    navigator_actions: Counter[str] = Counter()
    navigator_calls: List[int] = []

    for row in records:
        expected = str(row.get("expected", ""))
        answer_type = str(row.get("answer_type") or "")
        contest = str(row.get("contest") or "")
        candidate_answers: List[str] = []
        for step, answer_field in (("candidate", "extracted_answer"), ("refine", "refined_answer")):
            for candidate in _trace_contents(row, step):
                candidates_total += 1
                answer = str(candidate.get(answer_field) or candidate.get("normalized_answer") or "")
                candidate_answers.append(answer)
                protocol_value = candidate.get("protocol_compliant")
                if protocol_value is True or (
                    protocol_value is None and has_terminal_answer(candidate.get("response_tail", ""))
                ):
                    protocol_candidates += 1
                recovered_candidates += int(bool(candidate.get("recovered")))
        oracle_correct = any(
            is_match(answer, expected, answer_type, contest) for answer in candidate_answers if answer
        )
        if oracle_correct:
            candidate_oracle_ids.append(str(row.get("id")))
            if not row.get("is_correct"):
                correct_candidate_lost.append(str(row.get("id")))

        contract_rejections += sum(
            item.get("reason") == "answer_contract_mismatch"
            for step in ("candidate_rejected", "refine_rejected")
            for item in _trace_contents(row, step)
        )
        tool_verdicts.update(
            str(item.get("verdict") or "UNKNOWN") for item in _trace_contents(row, "tool_verify")
        )
        verifier_verdicts.update(
            str(item.get("verdict") or "UNKNOWN") for item in _trace_contents(row, "attack_verifier")
        )
        selector_sources.update(
            str(item.get("source") or "unknown") for item in _trace_contents(row, "answer_selector")
        )
        for summary in _trace_contents(row, "navigator_summary"):
            navigator_actions.update(str(action) for action in summary.get("actions") or [])
            if isinstance(summary.get("calls_used"), int):
                navigator_calls.append(summary["calls_used"])

    total = len(records)
    return {
        "final_accuracy": round(sum(bool(row.get("is_correct")) for row in records) / total, 4)
        if total
        else 0.0,
        "candidate_oracle_accuracy": round(len(candidate_oracle_ids) / total, 4) if total else 0.0,
        "candidate_oracle_correct": len(candidate_oracle_ids),
        "correct_candidate_lost": {
            "count": len(correct_candidate_lost),
            "ids": correct_candidate_lost,
        },
        "candidate_protocol": {
            "total": candidates_total,
            "compliant": protocol_candidates,
            "rate": round(protocol_candidates / candidates_total, 4) if candidates_total else 0.0,
            "recovered": recovered_candidates,
        },
        "answer_contract_rejections": contract_rejections,
        "tool_verdicts": dict(tool_verdicts.most_common()),
        "verifier_verdicts": dict(verifier_verdicts.most_common()),
        "selector_sources": dict(selector_sources.most_common()),
        "navigator": {
            "actions": dict(navigator_actions.most_common()),
            "average_calls": round(sum(navigator_calls) / len(navigator_calls), 4)
            if navigator_calls
            else 0.0,
        },
    }


def build_benchmark_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    id_counts = Counter(str(row.get("id")) for row in records)
    flagged = [
        row
        for row in records
        if set(row.get("benchmark_flags") or []) & NON_SCORABLE_BENCHMARK_FLAGS
    ]
    scorable = [row for row in records if row not in flagged]
    scorable_correct = sum(bool(row.get("is_correct")) for row in scorable)
    return {
        "duplicate_ids": sorted(identifier for identifier, count in id_counts.items() if count > 1),
        "adjudication_needed": {
            "count": len(flagged),
            "ids": [str(row.get("id")) for row in flagged],
        },
        "scorable_total": len(scorable),
        "scorable_correct": scorable_correct,
        "scorable_accuracy": round(scorable_correct / len(scorable), 4) if scorable else 0.0,
    }


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row.get("is_correct"))
    by_subject: Dict[str, Counter] = defaultdict(Counter)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    by_contest: Dict[str, Counter] = defaultdict(Counter)
    by_contract: Dict[str, Counter] = defaultdict(Counter)
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
        contract = row.get("answer_contract") or {}
        by_contract[str(contract.get("kind", "unknown"))][value] += 1

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
        "schema_version": 2,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_subject": collapse(by_subject),
        "by_answer_type": collapse(by_type),
        "by_contest": collapse(by_contest),
        "by_answer_contract": collapse(by_contract),
        "by_error_type": dict(error_types.most_common()),
        "error_analysis": {
            "diagnostic_coverage": error_report["diagnostic_coverage"],
            "heuristic_attribution_rate": error_report["heuristic_attribution_rate"],
            "by_primary_cause": error_report["by_primary_cause"],
            "by_root_cause": error_report["by_root_cause"],
        },
        "pipeline_metrics": build_pipeline_metrics(records),
        "benchmark_quality": build_benchmark_quality(records),
    }


def write_evaluation_artifacts(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    *,
    announce: bool = False,
) -> None:
    summary = evaluate_records(records)
    manifest = build_run_manifest(args, records)
    summary["run_manifest"] = manifest
    error_report = build_error_report(records)
    write_jsonl(args.output, records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = args.summary.with_name(f".{args.summary.name}.tmp")
    summary_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_temporary.replace(args.summary)
    report_path = args.error_report or args.output.with_name(
        f"{args.output.stem}_error_report.json"
    )
    markdown_path = args.error_report_md or args.output.with_name(
        f"{args.output.stem}_error_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report_path.with_name(f".{report_path.name}.tmp")
    markdown_temporary = markdown_path.with_name(f".{markdown_path.name}.tmp")
    report_temporary.write_text(
        json.dumps(error_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_temporary.write_text(
        render_error_report_markdown(error_report),
        encoding="utf-8",
    )
    report_temporary.replace(report_path)
    markdown_temporary.replace(markdown_path)
    manifest_path = getattr(args, "manifest", None) or args.output.with_name(
        f"{args.output.stem}_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    if announce:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Wrote predictions to {args.output}")
        print(f"Wrote summary to {args.summary}")
        print(f"Wrote error report to {report_path}")
        print(f"Wrote Markdown error report to {markdown_path}")
        print(f"Wrote run manifest to {manifest_path}")


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
                    "requires_proof": item.get("requires_proof"),
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
                "message": safe_exception_message(exc),
                "traceback": sanitize_trace_text(traceback.format_exc(), limit=1200),
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
            "source": item.get("source"),
            "answer_contract": AnswerContract.infer(
                str(item.get("problem") or ""),
                integer_only=answer_type.lower() in {"integer", "integer_mod_1000"}
                or contest.lower() == "aime",
                proof_mode=answer_type.lower() == "proof" or bool(item.get("requires_proof")),
            ).as_dict(),
            "benchmark_flags": detect_benchmark_flags(str(item.get("problem") or "")),
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
        write_evaluation_artifacts(args, results)
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
    dataset_by_id: Dict[str, Dict[str, Any]] = {}
    if args.dataset.is_file():
        dataset_by_id = {str(item.get("id")): item for item in load_jsonl(args.dataset)}
    for row in rows:
        answer_type = str(row.get("answer_type") or "")
        contest = str(row.get("contest") or "")
        dataset_item = dataset_by_id.get(str(row.get("id")), {})
        problem = str(dataset_item.get("problem") or row.get("problem") or "")
        row["answer_contract"] = AnswerContract.infer(
            problem,
            integer_only=answer_type.lower() in {"integer", "integer_mod_1000"}
            or contest.lower() == "aime",
            proof_mode=answer_type.lower() == "proof" or bool(dataset_item.get("requires_proof")),
        ).as_dict()
        row["benchmark_flags"] = detect_benchmark_flags(problem)
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
    parser.add_argument("--manifest", type=Path, help="Reproducibility manifest path.")
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
    write_evaluation_artifacts(args, records, announce=True)


if __name__ == "__main__":
    main()
