from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .agent_utils import (
    Candidate,
    Route,
    parse_verdict,
    sanitize_trace_text,
    score_candidates,
    truncate_text,
)
from .answer_contract import AnswerContract
from .tool_verify import REJECTED, VERIFIED


@dataclass(frozen=True)
class SelectionDecision:
    selected_index: int | None
    source: str
    reason: str
    response: str = ""
    eligible_indices: tuple[int, ...] = ()
    rule_scores: Dict[int, float] = field(default_factory=dict)
    checks: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "source": self.source,
            "reason": self.reason,
            "eligible_indices": list(self.eligible_indices),
            "rule_scores": {str(key): round(value, 4) for key, value in self.rule_scores.items()},
            "checks": self.checks,
            "response_preview": sanitize_trace_text(self.response, 900),
        }


class AnswerSelectionAgent:
    """Constrained sub-agent that may select, but never create, an answer.

    Deterministic rules establish the eligible set and evidence scores.  The
    language-model call is only allowed to return an index from that set.  A
    malformed or out-of-range decision fails closed so the caller can use its
    deterministic fallback.
    """

    _CHECK_NAMES = (
        "TARGET_MATCH",
        "CONSTRAINTS",
        "DERIVATION",
        "UNITS_FORMAT",
        "EVIDENCE",
    )

    def select(
        self,
        *,
        problem: str,
        route: Route,
        candidates: Sequence[Candidate],
        chat: Callable[..., str],
        max_tokens: int,
    ) -> SelectionDecision:
        contract = AnswerContract.infer(problem, proof_mode=route.proof_mode)
        eligible = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if candidate.display_answer()
            and candidate.contract_valid
            and contract.accepts(candidate.display_answer(), candidate.content)
            and candidate.tool_verdict != REJECTED
        ]
        if not eligible:
            return SelectionDecision(None, "no_eligible_candidate", "all candidates failed hard rules")
        if len(eligible) == 1:
            index, _ = eligible[0]
            return SelectionDecision(
                index,
                "single_eligible_candidate",
                "only one candidate survived hard rules",
                eligible_indices=(index,),
                rule_scores={index: self._rule_score(candidates[index], candidates)},
            )

        score_candidates(list(candidates))
        rule_scores = {
            index: self._rule_score(candidate, candidates) for index, candidate in eligible
        }
        shortlist = sorted(
            eligible,
            key=lambda item: rule_scores[item[0]],
            reverse=True,
        )[:4]
        prompt = self._prompt(problem, route, shortlist, rule_scores, contract)
        response = str(chat(prompt, temperature=0.0, max_tokens=max_tokens))
        selected_match = re.search(r"^SELECTED_INDEX\s*:\s*(\d+)\s*$", response, re.I | re.M)
        checks = self._parse_checks(response)
        shortlist_indices = tuple(index for index, _ in shortlist)
        if not selected_match:
            return SelectionDecision(
                None,
                "selector_rejected",
                "selection sub-agent omitted SELECTED_INDEX",
                response,
                shortlist_indices,
                rule_scores,
                checks,
            )

        selected_index = int(selected_match.group(1))
        if selected_index not in shortlist_indices:
            return SelectionDecision(
                None,
                "selector_rejected",
                "selection sub-agent returned an ineligible index",
                response,
                shortlist_indices,
                rule_scores,
                checks,
            )
        return SelectionDecision(
            selected_index,
            "selection_subagent",
            self._parse_reason(response) or "selected after rule and condition audit",
            response,
            shortlist_indices,
            rule_scores,
            checks,
        )

    @staticmethod
    def _rule_score(candidate: Candidate, candidates: Sequence[Candidate]) -> float:
        score = float(candidate.score)
        verdict = parse_verdict(candidate.attack_report)
        if candidate.tool_verdict == VERIFIED:
            score += 5.0 * max(0.5, candidate.tool_confidence)
        if verdict == "ACCEPT":
            score += 0.4
        elif verdict == "REJECT":
            score -= 0.5
        independent_support = {
            item.consensus_origin()
            for item in candidates
            if item.key() == candidate.key() and item.consensus_origin()
        }
        score += 1.5 * max(0, len(independent_support) - 1)
        if candidate.role.startswith("debate_synthesizer") and candidate.tool_verdict != VERIFIED:
            score -= 0.5
        return score

    def _prompt(
        self,
        problem: str,
        route: Route,
        shortlist: Sequence[tuple[int, Candidate]],
        rule_scores: Dict[int, float],
        contract: AnswerContract,
    ) -> str:
        blocks: List[str] = []
        for index, candidate in shortlist:
            blocks.append(
                f"""CANDIDATE_INDEX: {index}
answer: {candidate.display_answer()}
normalized: {candidate.key()}
role: {candidate.role}
origin: {candidate.origin()}
rule_score: {rule_scores[index]:.4f}
tool_verdict: {candidate.tool_verdict}
tool_report: {truncate_text(str(candidate.tool_report), 500)}
verifier_verdict: {parse_verdict(candidate.attack_report)}
verifier_report: {truncate_text(candidate.attack_report, 700)}
solution_head: {truncate_text(candidate.content, 1000)}
solution_tail: {candidate.content[-1200:]}"""
            )

        proof_rule = (
            "For a proof task, prefer the complete proof that closes every required case."
            if route.proof_mode
            else "For an answer task, verify that the exact requested quantity is returned."
        )
        return f"""You are AnswerSelectionAgent, an independent final-selection sub-agent.
You may audit and choose an existing candidate, but you are forbidden to create,
rewrite, repair, or calculate a new final answer. {proof_rule}

Audit rules, in priority order:
1. TARGET_MATCH: the candidate answers exactly what the question asks, not an intermediate value.
2. CONSTRAINTS: it satisfies all domains, cases, signs, integrality, and geometric conditions.
3. DERIVATION: its displayed answer is actually supported by its derivation.
4. UNITS_FORMAT: units, percentage scale, tuple order, and exact-vs-approximate form are appropriate.
5. EVIDENCE: deterministic tool evidence outranks model confidence. A verifier report is fallible.
6. Agreement counts only when derivations are genuinely independent; Debate may merely echo a proposal.
7. Select only one CANDIDATE_INDEX shown below. Never output an answer expression.
8. The answer contract is binding: {contract.instruction()}
9. Never introduce an unstated approximation such as pi=22/7.

Output exactly these six lines:
SELECTED_INDEX: one listed integer
TARGET_MATCH: PASS or FAIL
CONSTRAINTS: PASS or FAIL
DERIVATION: PASS or FAIL
UNITS_FORMAT: PASS or FAIL
EVIDENCE: one short reason without writing a replacement answer

Subject: {route.subject}
Problem:
{problem}

Candidates:
{chr(10).join(blocks)}
"""

    def _parse_checks(self, response: str) -> Dict[str, str]:
        checks: Dict[str, str] = {}
        for name in self._CHECK_NAMES:
            match = re.search(rf"^{name}\s*:\s*(.+?)\s*$", response, re.I | re.M)
            if match:
                checks[name.lower()] = match.group(1).strip()
        return checks

    @staticmethod
    def _parse_reason(response: str) -> str:
        match = re.search(r"^EVIDENCE\s*:\s*(.+?)\s*$", response, re.I | re.M)
        return match.group(1).strip() if match else ""


__all__ = ["AnswerSelectionAgent", "SelectionDecision"]
