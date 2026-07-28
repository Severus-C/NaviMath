from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent_utils import SkillRegistry  # noqa: E402
from scripts.distill_skills_from_harp import train_router  # noqa: E402


EXPECTED_DOMAINS = {
    "prealgebra": {"algebra"},
    "precalculus": {"algebra", "calculus"},
    "counting_and_probability": {"combinatorics", "probability"},
}


def expected_domains(subjects: Iterable[str]) -> Set[str]:
    output: Set[str] = set()
    for subject in subjects:
        key = str(subject).lower()
        output.update(EXPECTED_DOMAINS.get(key, {key}))
    return output


def load_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    return rows


def evaluate(records: List[Dict[str, Any]], router: Dict[str, Any] | None = None) -> Dict[str, Any]:
    registry = SkillRegistry()
    if router is not None:
        registry.catalog.router = router
    correct = 0
    template_hits = 0
    proof_template_hits = 0
    proof_total = 0
    by_subject: Dict[str, Counter] = defaultdict(Counter)
    template_counts: Counter[str] = Counter()
    misses = []

    for row in records:
        route = registry.route(str(row.get("problem") or ""))
        expected = expected_domains(row.get("subject") or ["unknown"])
        hit = route.subject in expected
        correct += int(hit)
        for subject in expected:
            by_subject[subject]["total"] += 1
            by_subject[subject]["correct"] += int(hit)
        if route.templates:
            template_hits += 1
            template_counts.update(template.id for template in route.templates)
        is_proof = row.get("answer_type") == "proof" or bool(row.get("requires_proof"))
        if is_proof:
            proof_total += 1
            proof_template_hits += int(bool(route.templates))
        if not hit and len(misses) < 30:
            misses.append(
                {
                    "id": row.get("id"),
                    "expected": sorted(expected),
                    "predicted": route.subject,
                    "templates": [template.id for template in route.templates],
                }
            )

    return {
        "total": len(records),
        "domain_correct": correct,
        "domain_accuracy": round(correct / len(records), 4) if records else 0.0,
        "template_coverage": round(template_hits / len(records), 4) if records else 0.0,
        "proof_template_coverage": round(proof_template_hits / proof_total, 4) if proof_total else 0.0,
        "by_expected_domain": {
            key: {
                "total": value["total"],
                "correct": value["correct"],
                "accuracy": round(value["correct"] / value["total"], 4),
            }
            for key, value in sorted(by_subject.items())
        },
        "top_templates": dict(template_counts.most_common(20)),
        "sample_misses": misses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SkillRegistry routing and template coverage.")
    parser.add_argument(
        "--inputs", nargs="+", type=Path,
        default=[ROOT / "data" / "public_math_harp_all_short.jsonl", ROOT / "data" / "public_math_proof_harp.jsonl"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "eval_outputs" / "skill_routing_summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_jsonl(args.inputs)
    train = []
    holdout = []
    for row in records:
        digest = hashlib.sha256(str(row.get("id") or "").encode("utf-8")).digest()
        (holdout if digest[0] % 5 == 0 else train).append(row)
    report = {
        "full_artifact": evaluate(records),
        "deterministic_20pct_holdout": {
            "train_records": len(train),
            "test_records": len(holdout),
            **evaluate(holdout, router=train_router(train)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
