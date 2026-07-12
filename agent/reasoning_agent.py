from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .agent_utils import (
    Candidate,
    Route,
    SkillRegistry,
    canonical_key,
    cluster_candidates,
    extract_final_answer,
    is_plausible_final_answer,
    normalize_latex_answer,
    parse_confidence,
    parse_verdict,
    score_candidates,
    select_best_candidate,
    trace_record,
    truncate_text,
)


CORE_SYSTEM_PROMPT = """你是一个竞赛数学智能体，目标是在隐藏评测中最大化最终答案正确率。
你必须严谨、怀疑自己的计算，并优先输出可判分的精确答案。
除非题目要求证明，否则最终答案必须短而明确。"""


@dataclass
class AgentConfig:
    mode: str = "aggressive"
    max_tokens_solver: int = 4096
    max_tokens_verifier: int = 1024
    max_tokens_judge: int = 2048
    easy_roles: int = 3
    medium_roles: int = 4
    hard_roles: int = 5
    attack_top_k: int = 2
    refine_top_k: int = 1
    use_final_judge: bool = True
    trace_response_chars: int = 900


class ReasoningAgent:
    """Aggressive multi-stage math reasoning agent for objective scoring."""

    def __init__(self, client: Any, config: AgentConfig | None = None, *args: Any, **kwargs: Any) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self.registry = SkillRegistry()

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)
        trace: List[Dict[str, Any]] = []
        route = self.registry.route(problem)
        subject_hint = metadata.get("subject")
        if isinstance(subject_hint, list):
            subject_hint = subject_hint[0] if subject_hint else ""
        hint = str(subject_hint or "").lower()
        hint_map = {
            "counting_and_probability": "probability",
            "precalculus": "algebra",
        }
        hint = hint_map.get(hint, hint)
        hinted_skill = next((skill for skill in self.registry.skills if skill.name == hint), None)
        if hinted_skill is not None:
            route.subject = hinted_skill.name
            route.skill = hinted_skill
        try:
            benchmark_difficulty = int(metadata.get("difficulty") or 0)
        except (TypeError, ValueError):
            benchmark_difficulty = 0
        if benchmark_difficulty:
            route.difficulty_score = max(route.difficulty_score, benchmark_difficulty)
        source_id = str(metadata.get("source_id", "")).lower()
        contest = str(metadata.get("contest", "")).lower()
        answer_type = str(metadata.get("answer_type", "")).lower()
        integer_only = answer_type == "integer" or contest == "aime" or "aime" in source_id
        trace.append(
            trace_record(
                "route",
                {
                    "idx": idx,
                    "subject": route.subject,
                    "skill": route.skill.name,
                    "difficulty_score": route.difficulty_score,
                    "difficulty_label": route.difficulty_label,
                    "proof_mode": route.proof_mode,
                    "action_plan": route.action_plan,
                },
            )
        )

        candidates = self._generate_candidates(problem, route, idx, trace, integer_only)
        if not any(self._answer_is_valid(candidate.display_answer(), route, integer_only) for candidate in candidates):
            fallback = self._fallback_solve(problem, route, idx, trace, integer_only)
            candidates.append(fallback)

        self._attack_and_refine(problem, route, idx, candidates, trace, integer_only)
        score_candidates(candidates)
        consensus_answer = self._unique_consensus_answer(candidates, route, integer_only)
        if consensus_answer:
            judged_answer = consensus_answer
            trace.append(
                trace_record(
                    "consensus_lock",
                    {
                        "answer": consensus_answer,
                        "reason": "unique_answer_cluster_with_at_least_two_candidates",
                    },
                )
            )
        else:
            judged_answer = self._final_judge(problem, route, idx, candidates, trace, integer_only)

        if self._answer_is_valid(judged_answer, route, integer_only):
            final_response = judged_answer if route.proof_mode else normalize_latex_answer(judged_answer)
            final_key = canonical_key(final_response)
            selected = max(candidates, key=lambda item: int(item.key() == final_key) * 100 + item.score)
        else:
            selected = select_best_candidate(candidates)
            final_response = selected.display_answer()
            if not route.proof_mode:
                final_response = normalize_latex_answer(final_response)

        if route.proof_mode and len(final_response) < 120:
            final_response = selected.refined_content or selected.content

        final_response = str(final_response).strip() or "Unable to determine"
        trace.append(
            trace_record(
                "select_final_response",
                {
                    "final_response": truncate_text(final_response, 500),
                    "normalized": canonical_key(final_response),
                    "selected_role": selected.role,
                    "selected_score": round(selected.score, 3),
                    "clusters": {
                        key: [candidate.role for candidate in group]
                        for key, group in cluster_candidates(candidates).items()
                    },
                },
            )
        )
        return {"final_response": final_response, "trace": trace}

    def _generate_candidates(
        self,
        problem: str,
        route: Route,
        idx: Any,
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> List[Candidate]:
        roles = self._solver_roles(route)
        candidates: List[Candidate] = []
        for role_index, role in enumerate(roles):
            prompt = self._solver_prompt(problem, route, role, integer_only)
            try:
                response = self._chat(
                    prompt,
                    temperature=self._solver_temperature(role_index, route),
                    max_tokens=self.config.max_tokens_solver,
                )
            except Exception as exc:  # noqa: BLE001 - keep the rest of the ensemble alive.
                trace.append(trace_record("solver_error", {"role": role["name"], "error": str(exc)}))
                continue

            extracted = extract_final_answer(
                response,
                proof_mode=route.proof_mode,
                integer_only=integer_only,
                problem=problem,
            )
            normalized = canonical_key(extracted) if not route.proof_mode else "<proof>"
            candidate = Candidate(
                role=role["name"],
                content=response,
                extracted_answer=extracted,
                normalized_answer=normalized,
                confidence=parse_confidence(response),
            )
            if not self._answer_is_valid(extracted, route, integer_only):
                candidates.append(candidate)
                trace.append(
                    trace_record(
                        "candidate_rejected",
                        {
                            "role": role["name"],
                            "reason": "no_plausible_final_answer",
                            "response_preview": truncate_text(response, self.config.trace_response_chars),
                            "response_tail": response[-self.config.trace_response_chars :],
                        },
                    )
                )
                continue
            candidates.append(candidate)
            trace.append(
                trace_record(
                    "candidate",
                    {
                        "role": role["name"],
                        "extracted_answer": truncate_text(extracted, 240),
                        "normalized_answer": normalized,
                        "confidence": round(candidate.confidence, 3),
                        "response_preview": truncate_text(response, self.config.trace_response_chars),
                        "response_tail": response[-self.config.trace_response_chars :],
                    },
                )
            )
        return candidates

    def _attack_and_refine(
        self,
        problem: str,
        route: Route,
        idx: Any,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> None:
        score_candidates(candidates)
        top_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[: self.config.attack_top_k]
        for rank, candidate in enumerate(top_candidates):
            attack_prompt = self._attack_prompt(problem, route, candidate)
            try:
                attack = self._chat(
                    attack_prompt,
                    temperature=0.0,
                    max_tokens=self.config.max_tokens_verifier,
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(trace_record("attack_error", {"role": candidate.role, "error": str(exc)}))
                continue
            candidate.attack_report = attack
            verdict = parse_verdict(attack)
            corrected = extract_final_answer(
                attack,
                proof_mode=False,
                integer_only=integer_only,
                problem=problem,
            )
            if verdict in {"REJECT", "INCORRECT", "B", "错误", "不通过"} and rank < self.config.refine_top_k:
                self._refine_candidate(problem, route, candidate, attack, trace, integer_only)
            elif (
                corrected
                and "FINAL_ANSWER" in attack
                and not route.proof_mode
                and self._answer_is_valid(corrected, route, integer_only)
            ):
                candidate.refined_answer = corrected
                candidate.refined_normalized_answer = canonical_key(corrected)

            trace.append(
                trace_record(
                    "attack_verifier",
                    {
                        "role": candidate.role,
                        "verdict": verdict,
                        "candidate_answer": truncate_text(candidate.display_answer(), 240),
                        "report_preview": truncate_text(attack, self.config.trace_response_chars),
                    },
                )
            )

    def _refine_candidate(
        self,
        problem: str,
        route: Route,
        candidate: Candidate,
        attack: str,
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> None:
        prompt = self._refine_prompt(problem, route, candidate, attack)
        try:
            response = self._chat(
                prompt,
                temperature=0.2,
                max_tokens=self.config.max_tokens_solver,
            )
        except Exception as exc:  # noqa: BLE001
            trace.append(trace_record("refine_error", {"role": candidate.role, "error": str(exc)}))
            return

        refined_answer = extract_final_answer(
            response,
            proof_mode=route.proof_mode,
            integer_only=integer_only,
            problem=problem,
        )
        if not self._answer_is_valid(refined_answer, route, integer_only):
            trace.append(
                trace_record(
                    "refine_rejected",
                    {
                        "role": candidate.role,
                        "reason": "no_plausible_final_answer",
                        "response_preview": truncate_text(response, self.config.trace_response_chars),
                    },
                )
            )
            return

        candidate.refined_content = response
        candidate.refined_answer = refined_answer
        candidate.refined_normalized_answer = (
            "<proof>" if route.proof_mode else canonical_key(candidate.refined_answer)
        )
        trace.append(
            trace_record(
                "refine",
                {
                    "role": candidate.role,
                    "refined_answer": truncate_text(candidate.refined_answer, 240),
                    "refined_normalized_answer": candidate.refined_normalized_answer,
                    "response_preview": truncate_text(response, self.config.trace_response_chars),
                },
            )
        )

    def _final_judge(
        self,
        problem: str,
        route: Route,
        idx: Any,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> str:
        if not self.config.use_final_judge or len(candidates) < 2:
            return ""
        score_candidates(candidates)
        shortlist = sorted(candidates, key=lambda item: item.score, reverse=True)[:4]
        prompt = self._judge_prompt(problem, route, shortlist)
        try:
            response = self._chat(prompt, temperature=0.0, max_tokens=self.config.max_tokens_judge)
        except Exception as exc:  # noqa: BLE001
            trace.append(trace_record("final_judge_error", {"error": str(exc)}))
            return ""
        selected_match = re.search(r"SELECTED_INDEX\s*:\s*(\d+)", response, flags=re.IGNORECASE)
        if not selected_match:
            answer = extract_final_answer(
                response,
                proof_mode=route.proof_mode,
                integer_only=integer_only,
                problem=problem,
            )
            trace.append(
                trace_record(
                    "final_judge_rejected",
                    {
                        "answer": truncate_text(answer, 300),
                        "normalized": canonical_key(answer) if answer else "",
                        "reason": "missing_selected_index_used_constrained_extraction",
                    },
                )
            )
        else:
            selected_index = int(selected_match.group(1))
            answer = shortlist[selected_index].display_answer() if selected_index < len(shortlist) else ""
            if not self._answer_is_valid(answer, route, integer_only):
                answer = extract_final_answer(
                    response,
                    proof_mode=route.proof_mode,
                    integer_only=integer_only,
                    problem=problem,
                )
        trace.append(
            trace_record(
                "final_judge",
                {
                    "answer": truncate_text(answer, 300),
                    "normalized": canonical_key(answer) if not route.proof_mode else "<proof>",
                    "response_preview": truncate_text(response, self.config.trace_response_chars),
                },
            )
        )
        return answer

    def _fallback_solve(
        self,
        problem: str,
        route: Route,
        idx: Any,
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> Candidate:
        prompt = self._solver_prompt(
            problem,
            route,
            {
                "name": "fallback_direct",
                "mission": "Solve directly and return the best possible final answer.",
                "style": "minimal",
            },
            integer_only,
        )
        try:
            response = self._chat(prompt, temperature=0.1, max_tokens=self.config.max_tokens_solver)
        except Exception as exc:  # noqa: BLE001
            response = f"FINAL_ANSWER: Unable to determine\nCONFIDENCE: 0\nERROR: {exc}"
        extracted = extract_final_answer(
            response,
            proof_mode=route.proof_mode,
            integer_only=integer_only,
            problem=problem,
        )
        candidate = Candidate(
            role="fallback_direct",
            content=response,
            extracted_answer=extracted,
            normalized_answer="<proof>" if route.proof_mode else canonical_key(extracted),
            confidence=parse_confidence(response),
        )
        trace.append(
            trace_record(
                "fallback_candidate",
                {
                    "extracted_answer": truncate_text(extracted, 240),
                    "response_preview": truncate_text(response, self.config.trace_response_chars),
                },
            )
        )
        return candidate

    @staticmethod
    def _answer_is_valid(answer: str, route: Route, integer_only: bool) -> bool:
        if not answer:
            return False
        if route.proof_mode:
            return is_plausible_final_answer(answer, proof_mode=True)
        if integer_only:
            key = canonical_key(answer)
            return bool(re.fullmatch(r"\d{1,3}", key)) and 0 <= int(key) <= 999
        return is_plausible_final_answer(answer)

    def _unique_consensus_answer(
        self,
        candidates: List[Candidate],
        route: Route,
        integer_only: bool,
    ) -> str:
        valid = [
            candidate
            for candidate in candidates
            if self._answer_is_valid(candidate.display_answer(), route, integer_only)
            and parse_verdict(candidate.attack_report) != "REJECT"
        ]
        clusters = cluster_candidates(valid)
        ranked = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
        if not ranked or len(ranked[0][1]) < 2:
            return ""
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            return ""
        return ranked[0][1][0].display_answer()

    def _solver_roles(self, route: Route) -> List[Dict[str, str]]:
        direct = {
            "name": "direct_solver",
            "mission": "Solve the problem directly with the shortest reliable route.",
            "style": "Be decisive; avoid unnecessary branching.",
        }
        domain_roles: Dict[str, List[Dict[str, str]]] = {
            "combinatorics": [
                {
                    "name": "complement_counter",
                    "mission": "Solve by complement counting or inclusion-exclusion and track overcounting exactly.",
                    "style": "Define the sample space and count every case explicitly.",
                },
                {
                    "name": "casework_counter",
                    "mission": "Build a disjoint case split and independently sum the cases.",
                    "style": "Check that cases are exhaustive and non-overlapping.",
                },
                {
                    "name": "recurrence_counter",
                    "mission": "Look for a recurrence, contribution count, or double-counting argument.",
                    "style": "Verify the result on small instances before generalizing.",
                },
            ],
            "probability": [
                {
                    "name": "complement_probability_solver",
                    "mission": "Compute the probability through exact complement counting in a clearly defined sample space.",
                    "style": "Keep fractions exact and finish every transformation requested by the question.",
                },
                {
                    "name": "casework_probability_solver",
                    "mission": "Independently enumerate favorable outcomes using disjoint cases.",
                    "style": "Audit adjacency, cyclic symmetry, and overcounting carefully.",
                },
                {
                    "name": "probability_sanity_checker",
                    "mission": "Solve independently, then check the probability lies in [0,1] and reduce the final fraction.",
                    "style": "Return the requested numerator/denominator expression, not the probability itself.",
                },
            ],
            "geometry": [
                {
                    "name": "coordinate_geometry_solver",
                    "mission": "Choose coordinates and derive the requested quantity from exact equations.",
                    "style": "Resolve all geometric branches using the diagram conditions.",
                },
                {
                    "name": "synthetic_geometry_solver",
                    "mission": "Use circle, chord, similarity, power, area, or volume theorems synthetically.",
                    "style": "State why the selected configuration is the intended one.",
                },
                {
                    "name": "geometry_cross_checker",
                    "mission": "Solve with a second analytic representation and check dimensions and feasibility.",
                    "style": "Reject diagram-based assumptions that are not implied by the statement.",
                },
            ],
            "number_theory": [
                {
                    "name": "modular_solver",
                    "mission": "Use congruences, divisibility, valuations, and digit constraints constructively.",
                    "style": "Verify the final integer against every modular condition.",
                },
                {
                    "name": "constructive_search_solver",
                    "mission": "Derive a finite systematic search or recurrence and prove minimality or completeness.",
                    "style": "Do not stop at an intermediate multiple or residue.",
                },
                {
                    "name": "number_theory_checker",
                    "mission": "Solve independently and substitute the proposed integer back into the original conditions.",
                    "style": "Check minimality and requested post-processing such as division or remainder.",
                },
            ],
            "algebra": [
                {
                    "name": "constraint_interval_solver",
                    "mission": "Convert the conditions into exact integer intervals or inequalities and locate boundary transitions without brute-force enumeration.",
                    "style": "Use floor and ceiling formulas and test only transition points.",
                },
                {
                    "name": "diophantine_solver",
                    "mission": "Model integer variables and solve the resulting Diophantine constraints symbolically.",
                    "style": "Exploit congruences and bounds before checking a small number of cases.",
                },
                {
                    "name": "algebra_cross_checker",
                    "mission": "Derive an independent symbolic solution and verify the requested extremal value.",
                    "style": "Avoid listing long sequences of routine cases.",
                },
            ],
        }
        generic_roles = [
            {
                "name": "theorem_solver",
                "mission": "Identify the relevant theorem, invariant, formula, or standard contest pattern.",
                "style": "Name the core method before applying it.",
            },
            {
                "name": "decomposition_solver",
                "mission": "Decompose the problem into lemmas or subcomputations, then synthesize.",
                "style": "Use explicit subgoals and verify each one.",
            },
            {
                "name": "contest_trick_solver",
                "mission": "Look for a fast olympiad-style transformation, substitution, or counting shortcut.",
                "style": "Prefer elegant reductions but still verify the final answer.",
            },
            {
                "name": "formal_symbolic_solver",
                "mission": "Translate the problem into equations, algebraic identities, cases, or exact symbolic checks.",
                "style": "Emphasize exact calculation and answer format.",
            },
            {
                "name": "skeptic_solver",
                "mission": "Solve while actively watching for traps, hidden conditions, and arithmetic mistakes.",
                "style": "After solving, perform a self-audit before the final answer.",
            },
        ]
        roles = [direct] + domain_roles.get(route.subject, []) + generic_roles
        if route.proof_mode:
            return roles[:6]
        if route.difficulty_label == "easy":
            return roles[: self.config.easy_roles]
        if route.difficulty_label == "medium":
            return roles[: self.config.medium_roles]
        return roles[: self.config.hard_roles]

    @staticmethod
    def _solver_temperature(role_index: int, route: Route) -> float:
        if route.proof_mode:
            return min(0.7, 0.25 + role_index * 0.08)
        return min(0.75, 0.2 + role_index * 0.09)

    def _solver_prompt(
        self,
        problem: str,
        route: Route,
        role: Dict[str, str],
        integer_only: bool,
    ) -> str:
        proof_instruction = (
            "本题要求证明。final_response 应该是一份完整且可检查的证明，而不是短答案。"
            if route.proof_mode
            else "本题不要求证明。最终只输出最简洁、精确、可判分的答案。"
        )
        traps = "\n".join(f"- {trap}" for trap in route.skill.traps) or "- Check hidden assumptions."
        answer_constraint = (
            "This is an AIME-style integer task. FINAL_ANSWER must contain only one integer from 0 to 999."
            if integer_only
            else "FINAL_ANSWER must contain only the exact answer requested by the problem."
        )
        return f"""{CORE_SYSTEM_PROMPT}

你现在扮演 `{role['name']}`。
任务: {role['mission']}
风格: {role['style']}

题型路由:
- subject: {route.subject}
- difficulty: {route.difficulty_label} ({route.difficulty_score}/10)
- RLoT action plan: {' -> '.join(route.action_plan)}

命中的 Skill:
- name: {route.skill.name}
- description: {route.skill.description}
- strategy: {route.skill.strategy}
- common traps:
{traps}
- answer format: {route.skill.answer_format}

{proof_instruction}
{answer_constraint}

Hard brevity rule: do not restate the problem and do not present alternative
solutions. As soon as the requested answer is computed, finish the current
check, emit FINAL_ANSWER, and stop. Keep the solution under 900 words.

输出格式必须严格包含以下字段。不要复制字段说明，不要输出尖括号占位符。
SOLUTION:
写必要推理，尽量简洁但不能跳关键步骤。

SELF_CHECK:
检查计算、条件、边界、题目是否真正被回答。

FINAL_ANSWER: 在这里写实际最终答案；若是证明题，写完整证明摘要或证明主体。
CONFIDENCE: 写 0 到 100 的整数。

题目:
{problem}
"""

    def _attack_prompt(self, problem: str, route: Route, candidate: Candidate) -> str:
        target = candidate.refined_content or candidate.content
        return f"""{CORE_SYSTEM_PROMPT}

你是攻击型数学 verifier。你的职责是找错，不要被候选解答的语气说服。
先独立计算题目，再检查候选推理是否严格支持下面这个被抽取的最终答案。
如果长推理正确但抽取答案不等于真正结论，也必须 REJECT 并给出修正答案。
请检查候选解答是否真正解决题目，重点查:
- 建模是否正确
- 是否漏掉条件/边界/特殊情况
- 计算、符号、单位、计数是否有错
- 最终答案格式是否回答了题目

输出格式:
VERDICT: ACCEPT 或 REJECT
ISSUES: 若无问题写 none；若有问题列出关键漏洞。
CORRECTED_FINAL_ANSWER: 如果能修正，给出修正后的最终答案；否则重复原最终答案或写 UNKNOWN。

题型: {route.subject}
题目:
{problem}

被抽取的候选最终答案:
{candidate.display_answer() or "EMPTY"}

候选解答:
{target}
"""

    def _refine_prompt(self, problem: str, route: Route, candidate: Candidate, attack: str) -> str:
        return f"""{CORE_SYSTEM_PROMPT}

请根据 verifier 的攻击报告修正候选解答。若 verifier 错了，也要说明并保留正确答案。
输出格式:
SOLUTION:
写修正后的推理。

FINAL_ANSWER: 写修正后的实际最终答案。
CONFIDENCE: 写 0 到 100 的整数。

题目:
{problem}

原候选:
{candidate.content}

攻击报告:
{attack}
"""

    def _judge_prompt(self, problem: str, route: Route, candidates: List[Candidate]) -> str:
        blocks = []
        for index, candidate in enumerate(candidates):
            blocks.append(
                f"""Candidate {index}
role: {candidate.role}
score_hint: {candidate.score:.3f}
answer: {candidate.display_answer()}
normalized: {candidate.key()}
attack_verdict: {parse_verdict(candidate.attack_report)}
attack_report: {truncate_text(candidate.attack_report, 700)}
solution_head: {truncate_text(candidate.refined_content or candidate.content, 600)}
solution_tail: {(candidate.refined_content or candidate.content)[-1200:]}
"""
            )
        proof_instruction = "Choose the most complete proof." if route.proof_mode else "Choose the most likely exact answer."
        has_usable_answer = any(candidate.display_answer() for candidate in candidates)
        if not has_usable_answer and not route.proof_mode:
            return f"""{CORE_SYSTEM_PROMPT}

All candidate solutions lost their machine-readable final answer, usually due
to truncation. Recover the requested answer from their derivations and solve
any unfinished last step yourself. Do not choose a candidate index.

Output exactly:
FINAL_ANSWER: one integer from 0 to 999

Problem:
{problem}

Candidate derivations:
{chr(10).join(blocks)}
"""
        return f"""{CORE_SYSTEM_PROMPT}

You are the final judge. Compare the candidates using mathematical correctness,
agreement, and verifier reports. You must select one existing candidate. Do not
create a new answer and do not continue solving the problem.
{proof_instruction}

Output exactly one line:
SELECTED_INDEX: integer from 0 to {len(candidates) - 1}

Subject: {route.subject}
Problem:
{problem}

Candidates:
{chr(10).join(blocks)}
"""

    def _chat(self, prompt: str, temperature: float, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": CORE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response)
