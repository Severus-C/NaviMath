from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .answer_selection_agent import AnswerSelectionAgent
from .answer_contract import AnswerContract, has_terminal_answer
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
    safe_exception_message,
    sanitize_trace_text,
    score_candidates,
    select_best_candidate,
    trace_record,
    truncate_text,
)
from .tool_verify import REJECTED, ToolVerify
from .rlot_navigator import ACTIONS, NavigationDecision, RLoTNavigator, RLoTState


CORE_SYSTEM_PROMPT = """你是一个竞赛数学智能体，目标是在隐藏评测中最大化最终答案正确率。
你必须严谨、怀疑自己的计算，并优先输出可判分的精确答案。
除非题目要求证明，否则最终答案必须短而明确。"""


@dataclass
class AgentConfig:
    mode: str = "aggressive"
    max_tokens_solver: int = 4096
    max_tokens_verifier: int = 1024
    max_tokens_judge: int = 2048
    max_tokens_recovery: int = 160
    answer_recovery_min_chars: int = 3000
    use_answer_recovery: bool = True
    easy_roles: int = 3
    medium_roles: int = 4
    hard_roles: int = 5
    attack_top_k: int = 2
    refine_top_k: int = 1
    use_final_judge: bool = True
    use_tool_verify: bool = True
    trace_response_chars: int = 900
    use_rlot_navigator: bool = True
    rlot_model_path: str = ""
    rlot_confidence_margin: float | None = None
    rlot_max_steps: int = 5
    rlot_easy_call_budget: int = 6
    rlot_medium_call_budget: int = 8
    rlot_hard_call_budget: int = 10
    rlot_proof_call_budget: int = 11
    rlot_min_candidates: int = 2


class ReasoningAgent:
    """Aggressive multi-stage math reasoning agent for objective scoring."""

    def __init__(self, client: Any, config: AgentConfig | None = None, *args: Any, **kwargs: Any) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self.registry = SkillRegistry()
        self.tool_verify = ToolVerify()
        self.answer_selector = AnswerSelectionAgent()
        context_name = f"navimath_{id(self)}"
        self._active_contract_context: ContextVar[AnswerContract] = ContextVar(
            f"{context_name}_active_contract",
            default=AnswerContract(),
        )
        self._rlot_call_budget_context: ContextVar[int | None] = ContextVar(
            f"{context_name}_rlot_call_budget",
            default=None,
        )
        self._rlot_calls_used_context: ContextVar[int] = ContextVar(
            f"{context_name}_rlot_calls_used",
            default=0,
        )
        self.navigator = RLoTNavigator(
            model_path=self.config.rlot_model_path or None,
            confidence_margin=self.config.rlot_confidence_margin,
        )

    @property
    def _active_contract(self) -> AnswerContract:
        return self._active_contract_context.get()

    @_active_contract.setter
    def _active_contract(self, value: AnswerContract) -> None:
        self._active_contract_context.set(value)

    @property
    def _rlot_call_budget(self) -> int | None:
        return self._rlot_call_budget_context.get()

    @_rlot_call_budget.setter
    def _rlot_call_budget(self, value: int | None) -> None:
        self._rlot_call_budget_context.set(value)

    @property
    def _rlot_calls_used(self) -> int:
        return self._rlot_calls_used_context.get()

    @_rlot_calls_used.setter
    def _rlot_calls_used(self, value: int) -> None:
        self._rlot_calls_used_context.set(value)

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)
        trace: List[Dict[str, Any]] = []
        route = self.registry.route(problem)
        subject_hint = metadata.get("subject")
        if isinstance(subject_hint, list):
            subject_hint = subject_hint[0] if subject_hint else ""
        hint = str(subject_hint or "").lower()
        hint_map = {
            "prealgebra": "algebra",
        }
        hint = hint_map.get(hint, hint)
        if hint == "counting_and_probability":
            hint = route.subject if route.subject in {"combinatorics", "probability"} else "combinatorics"
        if hint == "precalculus":
            hint = route.subject if route.subject in {"algebra", "calculus"} else "algebra"
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
        if answer_type == "proof" or bool(metadata.get("requires_proof")):
            route.proof_mode = True
            route.difficulty_score = max(route.difficulty_score, 7)
            route.action_plan = self.registry._action_plan(route.difficulty_score, proof_mode=True)
        route.templates = self.registry.catalog.match(
            problem,
            route.subject,
            proof_mode=route.proof_mode,
        )
        integer_only = (
            answer_type in {"integer", "integer_mod_1000"}
            or contest == "aime"
            or "aime" in source_id
        )
        self._active_contract = AnswerContract.infer(
            problem,
            integer_only=integer_only,
            proof_mode=route.proof_mode,
        )
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
                    "answer_contract": self._active_contract.as_dict(),
                    "action_plan": route.action_plan,
                    "navigator": {
                        "enabled": self.config.use_rlot_navigator,
                        "model_loaded": self.navigator.loaded,
                        "parameter_count": self.navigator.parameter_count,
                        "model_artifact": self.navigator.model_path.name,
                        "load_error": self.navigator.load_error or None,
                    },
                    "distilled_templates": [
                        {
                            "id": template.id,
                            "name": template.name,
                            "support": template.support,
                            "proof_support": template.proof_support,
                        }
                        for template in route.templates
                    ],
                },
            )
        )

        if self.config.use_rlot_navigator:
            self._rlot_call_budget = self._navigator_call_budget(route)
            self._rlot_calls_used = 0
            try:
                return self._solve_with_navigator(
                    problem,
                    route,
                    idx,
                    trace,
                    integer_only,
                    answer_type,
                )
            finally:
                self._rlot_call_budget = None

        candidates = self._generate_candidates(problem, route, idx, trace, integer_only)
        if not any(
            candidate.contract_valid
            and self._answer_is_valid(candidate.display_answer(), route, integer_only)
            for candidate in candidates
        ):
            fallback = self._fallback_solve(problem, route, idx, trace, integer_only)
            candidates.append(fallback)

        self._tool_verify_candidates(problem, answer_type, candidates, trace, phase="initial")
        self._attack_and_refine(problem, route, idx, candidates, trace, integer_only)
        self._tool_verify_candidates(problem, answer_type, candidates, trace, phase="post_refine")
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
                    "final_response": sanitize_trace_text(final_response, 500),
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

    def _navigator_call_budget(self, route: Route) -> int:
        if route.proof_mode:
            return max(2, self.config.rlot_proof_call_budget)
        if route.difficulty_label == "hard":
            return max(2, self.config.rlot_hard_call_budget)
        if route.difficulty_label == "medium":
            return max(2, self.config.rlot_medium_call_budget)
        return max(2, self.config.rlot_easy_call_budget)

    def _solve_with_navigator(
        self,
        problem: str,
        route: Route,
        idx: Any,
        trace: List[Dict[str, Any]],
        integer_only: bool,
        answer_type: str,
    ) -> Dict[str, Any]:
        candidates: List[Candidate] = []
        history: List[str] = []
        decisions: List[NavigationDecision] = []
        max_steps = max(2, self.config.rlot_max_steps)

        for step_index in range(max_steps):
            state = self._navigator_state(
                candidates,
                route,
                history,
                step_index,
                max_steps,
                integer_only,
            )
            valid_actions = self._navigator_valid_actions(
                state,
                history,
                step_index,
                max_steps,
            )
            heuristic = self.navigator.heuristic_action(state, history, valid_actions)
            decision = self.navigator.decide(state, valid_actions, heuristic)
            decisions.append(decision)
            trace.append(
                trace_record(
                    "navigator_state",
                    {
                        "step_index": step_index,
                        "calls_used": self._rlot_calls_used,
                        "call_budget": self._rlot_call_budget,
                        **state.as_dict(),
                    },
                )
            )
            trace.append(trace_record("navigator_policy", decision.as_dict()))

            action = decision.action
            route.action_plan = [*history, action]
            calls_before = self._rlot_calls_used
            candidate_count_before = len(candidates)
            error = ""
            try:
                self._execute_navigator_action(
                    action,
                    problem,
                    route,
                    idx,
                    candidates,
                    trace,
                    integer_only,
                    answer_type,
                    step_index,
                )
            except Exception as exc:  # noqa: BLE001 - force termination with surviving work.
                error = safe_exception_message(exc)
                trace.append(
                    trace_record(
                        "navigator_action_error",
                        {"action": action, "step_index": step_index, "error": error},
                    )
                )
            history.append(action)
            trace.append(
                trace_record(
                    "navigator_action",
                    {
                        "step_index": step_index,
                        "action": action,
                        "source": decision.source,
                        "calls_consumed": self._rlot_calls_used - calls_before,
                        "calls_used": self._rlot_calls_used,
                        "candidates_added": len(candidates) - candidate_count_before,
                        "error": error or None,
                    },
                )
            )
            if action == "Terminate":
                break
            if self._navigator_calls_remaining() <= 0:
                history.append("Terminate")
                trace.append(
                    trace_record(
                        "navigator_forced_terminate",
                        {"reason": "call_budget_exhausted", "step_index": step_index},
                    )
                )
                break
        else:
            history.append("Terminate")
            trace.append(
                trace_record(
                    "navigator_forced_terminate",
                    {"reason": "maximum_action_count_reached", "step_index": max_steps},
                )
            )

        if (
            not any(
                candidate.contract_valid
                and self._answer_is_valid(candidate.display_answer(), route, integer_only)
                for candidate in candidates
            )
            and self._navigator_calls_remaining() > 0
        ):
            candidates.append(self._fallback_solve(problem, route, idx, trace, integer_only))
        final_response = self._finalize_navigator_candidates(
            problem,
            route,
            idx,
            candidates,
            trace,
            integer_only,
        )
        trace.append(
            trace_record(
                "navigator_summary",
                {
                    "actions": history,
                    "calls_used": self._rlot_calls_used,
                    "call_budget": self._rlot_call_budget,
                    "learned_decisions": sum(
                        decision.source == "learned_policy" for decision in decisions
                    ),
                    "fallback_decisions": sum(
                        decision.source != "learned_policy" for decision in decisions
                    ),
                    "model_loaded": self.navigator.loaded,
                    "parameter_count": self.navigator.parameter_count,
                },
            )
        )
        return {"final_response": final_response, "trace": trace}

    def _navigator_state(
        self,
        candidates: Sequence[Candidate],
        route: Route,
        history: Sequence[str],
        step_index: int,
        max_steps: int,
        integer_only: bool,
    ) -> RLoTState:
        valid = [
            candidate
            for candidate in candidates
            if candidate.contract_valid
            and self._answer_is_valid(candidate.display_answer(), route, integer_only)
        ]
        independent_candidate_count = len({candidate.origin() for candidate in candidates})
        independent_valid_count = len({candidate.origin() for candidate in valid})
        if route.proof_mode:
            distinct_count = len(valid)
            accepted = sum(parse_verdict(candidate.attack_report) == "ACCEPT" for candidate in valid)
            agreement = accepted / max(1, len(valid)) if accepted else (0.5 if valid else 0.0)
        else:
            clusters = cluster_candidates(valid)
            distinct_count = len(clusters)
            agreement = max(
                (
                    len({candidate.consensus_origin() for candidate in group} - {""})
                    for group in clusters.values()
                ),
                default=0,
            ) / max(1, independent_valid_count)

        verifier_values = []
        tool_values = []
        for candidate in candidates:
            verdict = parse_verdict(candidate.attack_report)
            if verdict == "ACCEPT":
                verifier_values.append(1.0)
            elif verdict == "REJECT":
                verifier_values.append(-1.0)
            if candidate.tool_verdict == "VERIFIED":
                tool_values.append(max(0.25, candidate.tool_confidence))
            elif candidate.tool_verdict == REJECTED:
                tool_values.append(-max(0.25, candidate.tool_confidence))

        verifier_signal = sum(verifier_values) / len(verifier_values) if verifier_values else 0.0
        tool_signal = sum(tool_values) / len(tool_values) if tool_values else 0.0
        mean_confidence = (
            sum(candidate.confidence for candidate in candidates) / len(candidates)
            if candidates
            else 0.0
        )
        call_budget = max(1, int(self._rlot_call_budget or 1))
        return RLoTState.from_signals(
            difficulty=route.difficulty_score,
            proof_mode=route.proof_mode,
            domain=route.subject,
            candidate_count=independent_candidate_count,
            valid_answer_count=independent_valid_count,
            distinct_answer_count=distinct_count,
            agreement=agreement,
            mean_confidence=mean_confidence,
            verifier_signal=verifier_signal,
            tool_signal=tool_signal,
            budget_ratio=self._navigator_calls_remaining() / call_budget,
            step_ratio=step_index / max(1, max_steps - 1),
            previous_action=history[-1] if history else "",
        )

    def _navigator_valid_actions(
        self,
        state: RLoTState,
        history: Sequence[str],
        step_index: int,
        max_steps: int,
    ) -> List[str]:
        remaining = self._navigator_calls_remaining()
        if state.answer_available and (
            step_index >= max_steps - 1
            or remaining <= 0
            or (
                state.candidate_count >= self.config.rlot_min_candidates
                and state.completeness_within_step >= 3
                and state.agreement >= 0.8
                and min(state.verifier_signal, state.tool_signal) >= 0.0
            )
        ):
            return ["Terminate"]
        if step_index >= max_steps - 1:
            return ["Terminate"] if state.answer_available else ["ReasonOneStep"]

        costs = {
            "ReasonOneStep": 1,
            "Decompose": 1,
            "Debate": 1,
            "Refine": 1,
            "Terminate": 0,
        }
        valid = [action for action in ACTIONS if costs[action] <= remaining]
        if not history and "Refine" in valid:
            valid.remove("Refine")
        if not state.answer_available:
            valid = [action for action in valid if action not in {"Refine", "Terminate"}]
        elif (
            state.candidate_count < self.config.rlot_min_candidates
            and state.tool_signal <= 0.5
            and remaining > 1
        ):
            valid = [action for action in valid if action != "Terminate"]
        if history.count("ReasonOneStep") >= 2:
            valid = [action for action in valid if action != "ReasonOneStep"]
        if "Decompose" in history:
            valid = [action for action in valid if action != "Decompose"]
        if history.count("Debate") >= 2:
            valid = [action for action in valid if action != "Debate"]
        if history.count("Refine") >= 2:
            valid = [action for action in valid if action != "Refine"]
        if not valid:
            return ["Terminate"] if state.answer_available else ["ReasonOneStep"]
        return valid

    def _execute_navigator_action(
        self,
        action: str,
        problem: str,
        route: Route,
        idx: Any,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
        answer_type: str,
        step_index: int,
    ) -> None:
        if action == "Terminate":
            return
        if action == "Refine":
            self._navigator_refine(problem, route, candidates, trace, integer_only)
            self._tool_verify_candidates(
                problem,
                answer_type,
                candidates,
                trace,
                phase=f"navigator_{step_index}_refine",
            )
            return

        roles = self._solver_roles(route)
        used_roles = {candidate.role for candidate in candidates}
        prompt_override = ""
        if action == "Decompose":
            role = next(
                (role for role in roles if role["name"] == "decomposition_solver"),
                {
                    "name": "decomposition_solver",
                    "mission": "Break the task into minimal independent subproblems, solve them, then synthesize.",
                    "style": "State explicit subgoals and verify every dependency.",
                },
            )
        elif action == "Debate":
            debate_index = 1 + sum(
                candidate.role.startswith("debate_synthesizer") for candidate in candidates
            )
            role = {
                "name": f"debate_synthesizer_{debate_index}",
                "mission": "Compare independent approaches and continue from the most reliable one.",
                "style": "Resolve disagreements mathematically instead of voting by rhetoric.",
            }
            prompt_override = self._debate_prompt(problem, route, candidates, integer_only)
        else:
            role = next(
                (role for role in roles if role["name"] not in used_roles),
                {
                    "name": f"reason_step_{len(candidates) + 1}",
                    "mission": "Advance the solution by one decisive mathematical step and complete it if ready.",
                    "style": "Use the shortest reliable derivation and audit the result.",
                },
            )

        candidate = self._generate_candidate(
            problem,
            route,
            role,
            len(candidates),
            trace,
            integer_only,
            prompt_override=prompt_override,
        )
        if candidate is not None:
            candidates.append(candidate)
        self._tool_verify_candidates(
            problem,
            answer_type,
            candidates,
            trace,
            phase=f"navigator_{step_index}_{action.lower()}",
        )

    def _navigator_refine(
        self,
        problem: str,
        route: Route,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> None:
        refined_origins = {candidate.origin() for candidate in candidates if candidate.origin_role}
        usable = [
            candidate
            for candidate in candidates
            if candidate.display_answer()
            and candidate.contract_valid
            and not candidate.origin_role
            and candidate.origin() not in refined_origins
        ]
        if not usable:
            return
        score_candidates(usable)
        candidate = max(
            usable,
            key=lambda item: item.score + (0.5 if not item.attack_report else 0.0),
        )
        if candidate.attack_report:
            attack = candidate.attack_report
            verdict = parse_verdict(attack)
        else:
            attack = self._chat(
                self._attack_prompt(problem, route, candidate),
                temperature=0.0,
                max_tokens=self.config.max_tokens_verifier,
            )
            candidate.attack_report = attack
            verdict = parse_verdict(attack)
            trace.append(
                trace_record(
                    "attack_verifier",
                    {
                        "role": candidate.role,
                        "verdict": verdict,
                        "candidate_answer": sanitize_trace_text(candidate.display_answer(), 240),
                        "report_preview": sanitize_trace_text(attack, self.config.trace_response_chars),
                        "navigator_action": "Refine",
                    },
                )
            )
        if verdict == "REJECT" and self._navigator_calls_remaining() > 0:
            refined = self._refine_candidate(problem, route, candidate, attack, trace, integer_only)
            if refined is not None:
                candidates.append(refined)

    def _finalize_navigator_candidates(
        self,
        problem: str,
        route: Route,
        idx: Any,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> str:
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.contract_valid
            and self._answer_is_valid(candidate.display_answer(), route, integer_only)
        ]
        if not valid_candidates:
            trace.append(
                trace_record(
                    "select_final_response",
                    {
                        "final_response": "Unable to determine",
                        "normalized": "",
                        "selected_role": None,
                        "selected_score": None,
                        "clusters": {},
                        "reason": "no_valid_candidate",
                    },
                )
            )
            return "Unable to determine"
        score_candidates(valid_candidates)
        consensus_answer = self._unique_consensus_answer(valid_candidates, route, integer_only)
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
        elif self._navigator_calls_remaining() > 0:
            judged_answer = self._final_judge(
                problem,
                route,
                idx,
                valid_candidates,
                trace,
                integer_only,
            )
        else:
            judged_answer = ""

        if self._answer_is_valid(judged_answer, route, integer_only):
            final_response = judged_answer if route.proof_mode else normalize_latex_answer(judged_answer)
            final_key = canonical_key(final_response)
            selected = max(
                valid_candidates,
                key=lambda item: int(item.key() == final_key) * 100 + item.score,
            )
        else:
            selected = select_best_candidate(valid_candidates)
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
                    "final_response": sanitize_trace_text(final_response, 500),
                    "normalized": canonical_key(final_response),
                    "selected_role": selected.role,
                    "selected_score": round(selected.score, 3),
                    "clusters": {
                        key: [candidate.role for candidate in group]
                        for key, group in cluster_candidates(valid_candidates).items()
                    },
                },
            )
        )
        return final_response

    def _navigator_calls_remaining(self) -> int:
        if self._rlot_call_budget is None:
            return 10**9
        return max(0, self._rlot_call_budget - self._rlot_calls_used)

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
            candidate = self._generate_candidate(
                problem,
                route,
                role,
                role_index,
                trace,
                integer_only,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _generate_candidate(
        self,
        problem: str,
        route: Route,
        role: Dict[str, str],
        role_index: int,
        trace: List[Dict[str, Any]],
        integer_only: bool,
        prompt_override: str = "",
    ) -> Candidate | None:
        prompt = prompt_override or self._solver_prompt(problem, route, role, integer_only)
        try:
            response = self._chat(
                prompt,
                temperature=self._solver_temperature(role_index, route),
                max_tokens=self.config.max_tokens_solver,
            )
        except Exception as exc:  # noqa: BLE001 - keep the rest of the ensemble alive.
            trace.append(
                trace_record("solver_error", {"role": role["name"], "error": safe_exception_message(exc)})
            )
            return None

        extracted, protocol_compliant, recovered, contract_valid = self._extract_or_recover_answer(
            response=response,
            problem=problem,
            route=route,
            integer_only=integer_only,
            role=role["name"],
            trace=trace,
        )
        normalized = canonical_key(extracted) if not route.proof_mode else "<proof>"
        candidate = Candidate(
            role=role["name"],
            content=response,
            extracted_answer=extracted,
            normalized_answer=normalized,
            confidence=parse_confidence(response),
            protocol_compliant=protocol_compliant,
            contract_valid=contract_valid,
            recovered=recovered,
        )
        if not self._answer_is_valid(extracted, route, integer_only) or not contract_valid:
            trace.append(
                trace_record(
                    "candidate_rejected",
                    {
                        "role": role["name"],
                        "reason": (
                            "answer_contract_mismatch"
                            if extracted and not contract_valid
                            else "no_plausible_final_answer"
                        ),
                        "answer_contract": self._active_contract.as_dict(),
                        "response_preview": sanitize_trace_text(response, self.config.trace_response_chars),
                        "response_tail": sanitize_trace_text(
                            response[-self.config.trace_response_chars :],
                            self.config.trace_response_chars,
                        ),
                    },
                )
            )
            return candidate
        trace.append(
            trace_record(
                "candidate",
                {
                    "role": role["name"],
                    "extracted_answer": sanitize_trace_text(extracted, 240),
                    "normalized_answer": normalized,
                    "confidence": round(candidate.confidence, 3),
                    "protocol_compliant": protocol_compliant,
                    "contract_valid": contract_valid,
                    "recovered": recovered,
                    "response_preview": sanitize_trace_text(response, self.config.trace_response_chars),
                    "response_tail": sanitize_trace_text(
                        response[-self.config.trace_response_chars :],
                        self.config.trace_response_chars,
                    ),
                },
            )
        )
        return candidate

    def _extract_or_recover_answer(
        self,
        *,
        response: str,
        problem: str,
        route: Route,
        integer_only: bool,
        role: str,
        trace: List[Dict[str, Any]],
    ) -> tuple[str, bool, bool, bool]:
        extracted = extract_final_answer(
            response,
            proof_mode=route.proof_mode,
            integer_only=integer_only,
            problem=problem,
        )
        protocol_compliant = route.proof_mode or has_terminal_answer(response)
        contract_valid = route.proof_mode or self._active_contract.accepts(extracted, response)
        should_recover = (
            self.config.use_answer_recovery
            and not route.proof_mode
            and not protocol_compliant
            and self._navigator_calls_remaining() > 0
            and (not contract_valid or len(response) >= self.config.answer_recovery_min_chars)
        )
        if not should_recover:
            return extracted, protocol_compliant, False, contract_valid

        prompt = f"""You are a final-answer serialization worker, not a new solver.
Read the existing reasoning below and return only the conclusion it already supports.
Do not introduce assumptions, approximations, examples, or a different solution.
{self._active_contract.instruction()}

Output exactly one line:
FINAL_ANSWER: <answer>

Problem:
{problem}

Existing reasoning:
{response}
"""
        try:
            recovered_response = self._chat(
                prompt,
                temperature=0.0,
                max_tokens=self.config.max_tokens_recovery,
            )
        except Exception as exc:  # noqa: BLE001 - keep the original extraction.
            trace.append(
                trace_record(
                    "answer_recovery_error",
                    {"role": role, "error": safe_exception_message(exc)},
                )
            )
            return extracted, protocol_compliant, False, contract_valid

        recovered_answer = extract_final_answer(
            recovered_response,
            proof_mode=False,
            integer_only=integer_only,
            problem=problem,
        )
        recovered_valid = self._active_contract.accepts(recovered_answer, response)
        trace.append(
            trace_record(
                "answer_recovery",
                {
                    "role": role,
                    "previous_answer": sanitize_trace_text(extracted, 240),
                    "recovered_answer": sanitize_trace_text(recovered_answer, 240),
                    "accepted": bool(
                        recovered_valid
                        and self._answer_is_valid(recovered_answer, route, integer_only)
                    ),
                    "answer_contract": self._active_contract.as_dict(),
                    "response_preview": sanitize_trace_text(recovered_response, 400),
                },
            )
        )
        if recovered_valid and self._answer_is_valid(recovered_answer, route, integer_only):
            return recovered_answer, has_terminal_answer(recovered_response), True, True
        return extracted, protocol_compliant, False, contract_valid

    def _tool_verify_candidates(
        self,
        problem: str,
        answer_type: str,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        phase: str,
    ) -> None:
        if not self.config.use_tool_verify:
            return

        active: List[Candidate] = []
        for candidate in candidates:
            if phase == "post_refine" and not candidate.origin_role:
                continue
            answer = candidate.display_answer()
            if not answer or not candidate.contract_valid:
                continue
            result = self.tool_verify.verify_candidate(problem, answer, answer_type)
            candidate.tool_verdict = result.verdict
            candidate.tool_confidence = result.confidence
            candidate.tool_report = result.as_dict()
            active.append(candidate)
            trace.append(
                trace_record(
                    "tool_verify",
                    {
                        "phase": phase,
                        "role": candidate.role,
                        "answer": sanitize_trace_text(answer, 240),
                        **result.as_dict(),
                    },
                )
            )

        # SymPy-equivalent expressions should vote in the same consensus cluster
        # even when their surface forms differ (for example, x*(x+1) vs x^2+x).
        if len(active) < 2:
            return
        answers = [candidate.display_answer() for candidate in active]
        groups = self.tool_verify.equivalent_groups(answers)
        merged_groups = []
        for group in groups:
            if len(group) < 2:
                continue
            representative = active[group[0]].key()
            roles = []
            for index in group:
                candidate = active[index]
                roles.append(candidate.role)
                if candidate.refined_answer:
                    candidate.refined_normalized_answer = representative
                else:
                    candidate.normalized_answer = representative
            merged_groups.append({"normalized": representative, "roles": roles})
        if merged_groups:
            trace.append(
                trace_record(
                    "tool_equivalence_cluster",
                    {"phase": phase, "groups": merged_groups},
                )
            )

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
        refined_candidates: List[Candidate] = []
        for rank, candidate in enumerate(top_candidates):
            attack_prompt = self._attack_prompt(problem, route, candidate)
            try:
                attack = self._chat(
                    attack_prompt,
                    temperature=0.0,
                    max_tokens=self.config.max_tokens_verifier,
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    trace_record(
                        "attack_error",
                        {"role": candidate.role, "error": safe_exception_message(exc)},
                    )
                )
                continue
            candidate.attack_report = attack
            verdict = parse_verdict(attack)
            if verdict in {"REJECT", "INCORRECT", "B", "错误", "不通过"} and rank < self.config.refine_top_k:
                refined = self._refine_candidate(problem, route, candidate, attack, trace, integer_only)
                if refined is not None:
                    refined_candidates.append(refined)

            trace.append(
                trace_record(
                    "attack_verifier",
                    {
                        "role": candidate.role,
                        "verdict": verdict,
                        "candidate_answer": sanitize_trace_text(candidate.display_answer(), 240),
                        "report_preview": sanitize_trace_text(attack, self.config.trace_response_chars),
                    },
                )
            )
        candidates.extend(refined_candidates)

    def _refine_candidate(
        self,
        problem: str,
        route: Route,
        candidate: Candidate,
        attack: str,
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> Candidate | None:
        prompt = self._refine_prompt(problem, route, candidate, attack)
        try:
            response = self._chat(
                prompt,
                temperature=0.2,
                max_tokens=self.config.max_tokens_solver,
            )
        except Exception as exc:  # noqa: BLE001
            trace.append(
                trace_record(
                    "refine_error",
                    {"role": candidate.role, "error": safe_exception_message(exc)},
                )
            )
            return None

        refined_answer, protocol_compliant, recovered, contract_valid = self._extract_or_recover_answer(
            response=response,
            problem=problem,
            route=route,
            integer_only=integer_only,
            role=f"{candidate.role}_refined",
            trace=trace,
        )
        if not self._answer_is_valid(refined_answer, route, integer_only) or not contract_valid:
            trace.append(
                trace_record(
                    "refine_rejected",
                    {
                        "role": candidate.role,
                        "reason": (
                            "answer_contract_mismatch"
                            if refined_answer and not contract_valid
                            else "no_plausible_final_answer"
                        ),
                        "original_answer": sanitize_trace_text(candidate.display_answer(), 240),
                        "refined_answer": sanitize_trace_text(refined_answer, 240),
                        "answer_contract": self._active_contract.as_dict(),
                        "response_preview": sanitize_trace_text(response, self.config.trace_response_chars),
                    },
                )
            )
            return None

        refined_candidate = Candidate(
            role=f"{candidate.role}_refined",
            content=response,
            extracted_answer=refined_answer,
            normalized_answer="<proof>" if route.proof_mode else canonical_key(refined_answer),
            confidence=parse_confidence(response),
            origin_role=candidate.origin(),
            protocol_compliant=protocol_compliant,
            contract_valid=contract_valid,
            recovered=recovered,
        )
        trace.append(
            trace_record(
                "refine",
                {
                    "role": refined_candidate.role,
                    "source_role": candidate.role,
                    "refined_answer": sanitize_trace_text(refined_candidate.extracted_answer, 240),
                    "refined_normalized_answer": refined_candidate.normalized_answer,
                    "protocol_compliant": protocol_compliant,
                    "contract_valid": contract_valid,
                    "recovered": recovered,
                    "response_preview": sanitize_trace_text(response, self.config.trace_response_chars),
                },
            )
        )
        return refined_candidate

    def _final_judge(
        self,
        problem: str,
        route: Route,
        idx: Any,
        candidates: List[Candidate],
        trace: List[Dict[str, Any]],
        integer_only: bool,
    ) -> str:
        if not self.config.use_final_judge:
            return ""
        selectable = [
            candidate
            for candidate in candidates
            if candidate.contract_valid
            and self._answer_is_valid(candidate.display_answer(), route, integer_only)
        ]
        if not selectable:
            return ""
        try:
            decision = self.answer_selector.select(
                problem=problem,
                route=route,
                candidates=selectable,
                chat=self._chat,
                max_tokens=self.config.max_tokens_judge,
            )
        except Exception as exc:  # noqa: BLE001
            trace.append(trace_record("final_judge_error", {"error": safe_exception_message(exc)}))
            return ""
        trace.append(trace_record("answer_selector", decision.as_dict()))
        if decision.selected_index is None:
            trace.append(
                trace_record(
                    "final_judge_rejected",
                    {
                        "answer": "",
                        "normalized": "",
                        "reason": decision.reason,
                    },
                )
            )
            trace.append(
                trace_record(
                    "final_judge",
                    {
                        "answer": "",
                        "normalized": "",
                        "response_preview": sanitize_trace_text(
                            decision.response,
                            self.config.trace_response_chars,
                        ),
                        "accepted": False,
                        "source": decision.source,
                    },
                )
            )
            return ""
        answer = selectable[decision.selected_index].display_answer()
        if not self._answer_is_valid(answer, route, integer_only):
            return ""
        trace.append(
            trace_record(
                "final_judge",
                {
                    "answer": sanitize_trace_text(answer, 300),
                    "normalized": canonical_key(answer) if not route.proof_mode else "<proof>",
                    "response_preview": sanitize_trace_text(
                        decision.response,
                        self.config.trace_response_chars,
                    ),
                    "source": decision.source,
                    "checks": decision.checks,
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
            response = (
                "FINAL_ANSWER: Unable to determine\n"
                f"CONFIDENCE: 0\nERROR: {safe_exception_message(exc)}"
            )
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
                    "extracted_answer": sanitize_trace_text(extracted, 240),
                    "response_preview": sanitize_trace_text(response, self.config.trace_response_chars),
                },
            )
        )
        return candidate

    def _answer_is_valid(self, answer: str, route: Route, integer_only: bool) -> bool:
        if not answer:
            return False
        if route.proof_mode:
            return is_plausible_final_answer(answer, proof_mode=True)
        if integer_only:
            key = canonical_key(answer)
            return bool(re.fullmatch(r"\d{1,3}", key)) and 0 <= int(key) <= 999
        return is_plausible_final_answer(answer) and self._active_contract.accepts(answer)

    def _unique_consensus_answer(
        self,
        candidates: List[Candidate],
        route: Route,
        integer_only: bool,
    ) -> str:
        # Proof candidates intentionally share the placeholder key `<proof>`;
        # treating that as semantic consensus would always select the first proof.
        if route.proof_mode:
            return ""
        valid = [
            candidate
            for candidate in candidates
            if candidate.contract_valid
            and self._answer_is_valid(candidate.display_answer(), route, integer_only)
            and candidate.tool_verdict != REJECTED
        ]
        clusters = cluster_candidates(valid)
        ranked = sorted(
            clusters.items(),
            key=lambda item: len({candidate.consensus_origin() for candidate in item[1]} - {""}),
            reverse=True,
        )
        if not ranked or len({candidate.consensus_origin() for candidate in ranked[0][1]} - {""}) < 2:
            return ""
        if len(ranked) > 1 and (
            len({candidate.consensus_origin() for candidate in ranked[0][1]} - {""})
            == len({candidate.consensus_origin() for candidate in ranked[1][1]} - {""})
        ):
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
        template_roles = [
            {
                "name": template.solver_role,
                "mission": f"Apply the distilled pattern '{template.name}': {template.strategy}",
                "style": "Follow the template checklist, but abandon it immediately if its hypotheses do not fit.",
            }
            for template in route.templates[:2]
        ]
        proof_roles = (
            [
                {
                    "name": "proof_architect",
                    "mission": "Design a complete lemma-level proof before writing details; track every quantifier and equality case.",
                    "style": "Prefer the shortest rigorous proof whose dependencies can all be checked.",
                },
                {
                    "name": "counterexample_skeptic",
                    "mission": "Try small, boundary, degenerate, and equality cases before proving the surviving statement.",
                    "style": "Actively search for hidden assumptions and circular steps.",
                },
            ]
            if route.proof_mode
            else []
        )
        roles = [direct] + proof_roles + template_roles + domain_roles.get(route.subject, []) + generic_roles
        unique_roles: List[Dict[str, str]] = []
        seen_roles = set()
        for role in roles:
            if role["name"] in seen_roles:
                continue
            seen_roles.add(role["name"])
            unique_roles.append(role)
        roles = unique_roles
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

    @staticmethod
    def _distilled_guidance(route: Route, verifier: bool = False) -> str:
        if not route.templates:
            return "- No fine-grained template matched; solve from first principles."
        blocks = []
        for template in route.templates:
            checklist = template.verifier_checklist if verifier else template.traps
            checklist_label = "verification checklist" if verifier else "failure modes"
            blocks.append(
                "\n".join(
                    [
                        f"- template: {template.name} ({template.id})",
                        f"  HARP support: {template.support} problems; proof support: {template.proof_support}",
                        f"  strategy: {template.strategy}",
                        f"  answer schema: {template.answer_schema}",
                        f"  proof methods: {', '.join(template.proof_methods) or 'derive from structure'}",
                        f"  {checklist_label}: {'; '.join(checklist) or 'check all hypotheses and edge cases'}",
                    ]
                )
            )
        return "\n".join(blocks)

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
        contract_instruction = self._active_contract.instruction()
        distilled_guidance = self._distilled_guidance(route)
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

HARP-distilled fine-grained templates (use only after checking fit):
{distilled_guidance}

{proof_instruction}
{answer_constraint}
Answer contract: {contract_instruction}

Hard brevity rule: do not restate the problem, discuss these instructions, or
present alternative solutions. As soon as the answer is computed, finish one
check and stop. Keep the complete response under 900 words.

Use SOLUTION and SELF_CHECK as short section headings. End with exactly two
machine-readable lines beginning with `FINAL_ANSWER:` and `CONFIDENCE:`.
Place the computed mathematical answer immediately after the first colon and
one integer from 0 to 100 immediately after the second colon.
Never put a schema label, placeholder, or explanatory sentence after
FINAL_ANSWER.

题目:
{problem}
"""

    def _debate_prompt(
        self,
        problem: str,
        route: Route,
        candidates: Sequence[Candidate],
        integer_only: bool,
    ) -> str:
        proposals = []
        for index, candidate in enumerate(candidates[-4:]):
            proposals.append(
                f"""Proposal {index}
role: {candidate.role}
answer: {candidate.display_answer() or 'EMPTY'}
tool_verdict: {candidate.tool_verdict}
attack_verdict: {parse_verdict(candidate.attack_report)}
derivation: {truncate_text(candidate.content, 1800)}"""
            )
        answer_constraint = (
            "FINAL_ANSWER must contain only one integer from 0 to 999."
            if integer_only
            else "FINAL_ANSWER must contain the exact requested answer."
        )
        contract_instruction = self._active_contract.instruction()
        return f"""{CORE_SYSTEM_PROMPT}

You are the Debate logic block in an RLoT reasoning episode. Independently
recompute the problem, compare the proposals below, identify any shared
unstated assumption, and continue from the mathematically strongest approach.
Do not select by majority vote when the majority is wrong. Finish with a
machine-readable provisional answer.

Output exactly:
SOLUTION:
brief comparison and corrected derivation

FINAL_ANSWER: the best exact provisional answer
CONFIDENCE: an integer from 0 to 100

{answer_constraint}
Answer contract: {contract_instruction}
Subject: {route.subject}
Problem:
{problem}

Existing proposals:
{chr(10).join(proposals) if proposals else 'No proposal exists; debate plausible approaches internally.'}
"""

    def _attack_prompt(self, problem: str, route: Route, candidate: Candidate) -> str:
        target = candidate.refined_content or candidate.content
        distilled_verifier_guidance = self._distilled_guidance(route, verifier=True)
        return f"""{CORE_SYSTEM_PROMPT}

HARP-distilled verifier checklist:
{distilled_verifier_guidance}

你是独立数学 verifier。ACCEPT 与 REJECT 同样正常；不要预设候选一定有错。
先独立计算题目，再对称地检查候选推理是否严格支持下面这个被抽取的最终答案。
模型 verifier 只提供软证据；没有明确反例、矛盾或遗漏时必须 ACCEPT。
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
答案契约: {self._active_contract.instruction()}
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
verifier 的 REJECT 只是软证据。除非能指出明确数学矛盾，否则不得改变原答案。
修正结果必须继续满足同一个答案契约：{self._active_contract.instruction()}
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

    def _chat(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if self._rlot_call_budget is not None:
            if self._rlot_calls_used >= self._rlot_call_budget:
                raise RuntimeError("RLoT call budget exhausted")
            self._rlot_calls_used += 1
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
