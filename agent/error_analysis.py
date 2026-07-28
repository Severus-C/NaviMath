from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from .answer_matching import answers_match
from .answer_normalizer import AnswerContext, DEFAULT_NORMALIZER, is_placeholder_answer


ROOT_CAUSE_LABELS = {
    "format_error": "格式错误",
    "routing_error": "路由错误",
    "wrong_consensus": "多候选一致但都错",
    "verifier_misjudgment": "verifier 误判",
    "final_judge_wrong_selection": "final judge 选错",
    "latex_normalization_failure": "LaTeX 归一化失败",
    "solver_math_error": "求解/数学错误",
    "runtime_error": "运行时错误",
    "answer_contract_mismatch": "答案契约不匹配",
    "benchmark_ambiguity": "基准题面歧义",
}


def _context(row: Dict[str, Any]) -> AnswerContext:
    return AnswerContext(
        answer_type=str(row.get("answer_type") or ""),
        contest=str(row.get("contest") or ""),
    )


def _key(value: Any, row: Dict[str, Any]) -> str:
    return DEFAULT_NORMALIZER.canonicalize(value, _context(row))


def _matches(value: Any, expected: Any, row: Dict[str, Any]) -> bool:
    return answers_match(
        value,
        expected,
        str(row.get("answer_type") or ""),
        str(row.get("contest") or ""),
    )


def _trace_entries(row: Dict[str, Any], step: str) -> List[Dict[str, Any]]:
    entries = []
    for entry in row.get("trace") or []:
        if entry.get("step") == step and isinstance(entry.get("content"), dict):
            entries.append(entry["content"])
    return entries


def diagnose_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Produce evidence-backed, multi-label diagnostics for one evaluation row."""
    if row.get("is_correct"):
        return {"primary_cause": "correct", "root_causes": [], "evidence": []}

    expected_key = _key(row.get("expected", ""), row)
    prediction = str(row.get("prediction") or "").strip()
    prediction_key = _key(prediction, row)
    causes: List[str] = []
    evidence: List[Dict[str, Any]] = []

    benchmark_flags = list(row.get("benchmark_flags") or [])
    if benchmark_flags:
        causes.append("benchmark_ambiguity")
        evidence.append({"signal": "benchmark_ambiguity", "flags": benchmark_flags})

    if row.get("status") == "error" or row.get("error"):
        causes.append("runtime_error")
        evidence.append({"signal": "runtime_error", "detail": row.get("error")})

    protocol_tokens = (
        "final_answer",
        "verdict:",
        "confidence:",
        "selected:",
        "<answer",
        "<final",
        "iwillputtheanswer",
        "afterthecolon",
    )
    if (
        not prediction
        or not prediction_key
        or is_placeholder_answer(prediction)
        or any(token in prediction.lower() for token in protocol_tokens)
    ):
        causes.append("format_error")
        evidence.append({"signal": "unusable_final_output", "prediction": prediction[:240]})

    latex_residue = re.findall(
        r"\\(?:frac|dfrac|tfrac|begin|end|boxed|left|right|text|operatorname)|[{}]",
        prediction_key,
    )
    if prediction_key.startswith("{") and prediction_key.endswith("}"):
        latex_residue = [token for token in latex_residue if token not in {"{", "}"}]
    if latex_residue:
        causes.append("latex_normalization_failure")
        evidence.append({"signal": "latex_tokens_remain", "tokens": sorted(set(latex_residue))})

    candidate_answers: List[str] = []
    for candidate in _trace_entries(row, "candidate"):
        answer = str(candidate.get("extracted_answer") or candidate.get("normalized_answer") or "")
        if answer:
            candidate_answers.append(answer)
    for refined in _trace_entries(row, "refine"):
        answer = str(refined.get("refined_answer") or refined.get("refined_normalized_answer") or "")
        if answer:
            candidate_answers.append(answer)

    for lock in _trace_entries(row, "consensus_lock"):
        lock_key = _key(lock.get("answer", ""), row)
        support = sum(_matches(answer, lock.get("answer", ""), row) for answer in candidate_answers)
        if lock_key and not _matches(lock.get("answer", ""), row.get("expected", ""), row) and support >= 2:
            causes.append("wrong_consensus")
            evidence.append(
                {
                    "signal": "wrong_consensus_lock",
                    "answer": lock_key,
                    "support": support,
                }
            )

    contract_rejections = [
        item
        for step in ("candidate_rejected", "refine_rejected")
        for item in _trace_entries(row, step)
        if item.get("reason") == "answer_contract_mismatch"
    ]
    if contract_rejections:
        causes.append("answer_contract_mismatch")
        evidence.append(
            {
                "signal": "answer_contract_mismatch",
                "items": [
                    {
                        "role": item.get("role"),
                        "refined_answer": item.get("refined_answer", ""),
                    }
                    for item in contract_rejections
                ],
            }
        )

    verifier_errors = []
    for verifier in _trace_entries(row, "attack_verifier"):
        candidate_answer = verifier.get("candidate_answer", "")
        answer_key = _key(candidate_answer, row)
        verdict = str(verifier.get("verdict") or "").upper()
        accepted = verdict in {"ACCEPT", "CORRECT", "A"}
        rejected = verdict in {"REJECT", "INCORRECT", "WRONG", "B"}
        matches_expected = _matches(candidate_answer, row.get("expected", ""), row)
        if accepted and answer_key and not matches_expected:
            verifier_errors.append({"kind": "false_accept", "answer": answer_key, "role": verifier.get("role")})
        elif rejected and matches_expected:
            verifier_errors.append({"kind": "false_reject", "answer": answer_key, "role": verifier.get("role")})
    if verifier_errors:
        causes.append("verifier_misjudgment")
        evidence.append({"signal": "verifier_misjudgment", "items": verifier_errors})

    judges = _trace_entries(row, "final_judge")
    if judges:
        judge_key = _key(judges[-1].get("normalized") or judges[-1].get("answer", ""), row)
        judge_answer = judges[-1].get("answer") or judges[-1].get("normalized") or ""
        correct_candidate_available = any(
            _matches(answer, row.get("expected", ""), row) for answer in candidate_answers
        )
        if not _matches(judge_answer, row.get("expected", ""), row) and correct_candidate_available:
            causes.append("final_judge_wrong_selection")
            evidence.append(
                {
                    "signal": "correct_candidate_available",
                    "judge_answer": judge_key,
                    "expected": expected_key,
                }
            )

    routes = _trace_entries(row, "route")
    subjects = row.get("subject") or []
    if isinstance(subjects, str):
        subjects = [subjects]
    aliases = {
        "counting_and_probability": {"probability", "combinatorics"},
        "prealgebra": {"algebra"},
        "precalculus": {"algebra", "calculus"},
    }
    expected_subjects = set()
    for subject in subjects:
        normalized_subject = str(subject).lower()
        expected_subjects.update(aliases.get(normalized_subject, {normalized_subject}))
    if routes and expected_subjects:
        routed = str(routes[-1].get("subject") or "unknown").lower()
        if routed not in expected_subjects:
            causes.append("routing_error")
            evidence.append(
                {"signal": "subject_mismatch", "routed": routed, "dataset_subjects": sorted(expected_subjects)}
            )

    if not causes:
        causes.append("solver_math_error")
        evidence.append({"signal": "wrong_answer_without_downstream_failure", "prediction": prediction_key})

    # The first item is the actionable primary cause. Downstream selection errors
    # outrank broad solver disagreement, while format/runtime failures outrank both.
    priority = [
        "runtime_error",
        "benchmark_ambiguity",
        "format_error",
        "latex_normalization_failure",
        "answer_contract_mismatch",
        "final_judge_wrong_selection",
        "verifier_misjudgment",
        "wrong_consensus",
        "routing_error",
        "solver_math_error",
    ]
    unique_causes = list(dict.fromkeys(causes))
    primary = next(cause for cause in priority if cause in unique_causes)
    return {"primary_cause": primary, "root_causes": unique_causes, "evidence": evidence}


def build_error_report(records: Iterable[Dict[str, Any]], example_limit: int = 5) -> Dict[str, Any]:
    rows = list(records)
    diagnostics = []
    primary_counts: Counter[str] = Counter()
    root_counts: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        diagnostic = row.get("diagnosis") or diagnose_record(row)
        if diagnostic["primary_cause"] == "correct":
            continue
        primary = diagnostic["primary_cause"]
        primary_counts[primary] += 1
        root_counts.update(diagnostic["root_causes"])
        item = {
            "id": row.get("id"),
            "expected": row.get("expected_key") or _key(row.get("expected", ""), row),
            "prediction": row.get("prediction_key") or _key(row.get("prediction", ""), row),
            "evidence": diagnostic["evidence"],
        }
        if len(examples.setdefault(primary, [])) < example_limit:
            examples[primary].append(item)
        diagnostics.append({"id": row.get("id"), **diagnostic})

    wrong = len(diagnostics)
    attributed = sum(count for cause, count in primary_counts.items() if cause != "solver_math_error")
    attribution_rate = round(attributed / wrong, 4) if wrong else 1.0
    return {
        "schema_version": 1,
        "total": len(rows),
        "correct": len(rows) - wrong,
        "wrong": wrong,
        # Compatibility alias. This is heuristic attribution, not validated
        # root-cause accuracy.
        "diagnostic_coverage": attribution_rate,
        "heuristic_attribution_rate": attribution_rate,
        "by_primary_cause": dict(primary_counts.most_common()),
        "by_root_cause": dict(root_counts.most_common()),
        "labels_zh": ROOT_CAUSE_LABELS,
        "examples": examples,
        "records": diagnostics,
    }


def render_error_report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# 错因分析报告",
        "",
        f"- 样本：{report['total']}",
        f"- 正确：{report['correct']}",
        f"- 错误：{report['wrong']}",
        f"- 启发式归因率：{report['heuristic_attribution_rate']:.1%}",
        "",
        "## 主错因分布",
        "",
        "| 错因 | 数量 | 占错误比例 |",
        "|---|---:|---:|",
    ]
    wrong = report["wrong"]
    for cause, count in report["by_primary_cause"].items():
        label = ROOT_CAUSE_LABELS.get(cause, cause)
        ratio = count / wrong if wrong else 0.0
        lines.append(f"| {label} (`{cause}`) | {count} | {ratio:.1%} |")
    lines.extend(["", "## 典型样本", ""])
    for cause, items in report["examples"].items():
        lines.append(f"### {ROOT_CAUSE_LABELS.get(cause, cause)}")
        lines.append("")
        for item in items:
            lines.append(
                f"- `{item['id']}`：预测 `{item['prediction']}`，期望 `{item['expected']}`；"
                f"证据 `{item['evidence']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
