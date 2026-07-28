from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ACTIONS: Tuple[str, ...] = (
    "ReasonOneStep",
    "Decompose",
    "Debate",
    "Refine",
    "Terminate",
)

DOMAINS: Tuple[str, ...] = (
    "algebra",
    "number_theory",
    "geometry",
    "combinatorics",
    "probability",
    "calculus",
    "linear_algebra",
    "complex_analysis",
    "abstract_algebra",
    "optimization",
    "real_analysis",
    "differential_equations",
    "topology",
    "statistics",
    "numerical_analysis",
    "discrete_mathematics",
    "differential_geometry",
    "logic_set_theory",
)

SELF_EVALUATION_FIELDS: Tuple[str, ...] = (
    "correctness_of_modeling",
    "clarity_for_further_reasoning",
    "correctness_of_calculation",
    "complexity_to_final_answer",
    "alternative_methods",
    "closeness_to_final_solution",
    "completeness_within_step",
)

MODEL_FILENAME = "rlot_policy.json"
MODEL_SCHEMA_VERSION = 1


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class RLoTState:
    """Low-dimensional reasoning state used by the RLoT policy.

    The first seven values are the paper's 1--3 self-evaluation aspects. The
    remaining fields are observable runtime context; they let one compact
    policy transfer across domains while still respecting call budgets.
    """

    correctness_of_modeling: int = 1
    clarity_for_further_reasoning: int = 1
    correctness_of_calculation: int = 1
    complexity_to_final_answer: int = 1
    alternative_methods: int = 1
    closeness_to_final_solution: int = 1
    completeness_within_step: int = 1
    difficulty: int = 1
    proof_mode: bool = False
    agreement: float = 0.0
    verifier_signal: float = 0.0
    tool_signal: float = 0.0
    answer_available: bool = False
    budget_ratio: float = 1.0
    step_ratio: float = 0.0
    previous_action: str = ""
    domain: str = "general_contest_math"
    candidate_count: int = 0
    valid_answer_count: int = 0
    distinct_answer_count: int = 0

    @classmethod
    def from_signals(
        cls,
        *,
        difficulty: int,
        proof_mode: bool,
        domain: str,
        candidate_count: int,
        valid_answer_count: int,
        distinct_answer_count: int,
        agreement: float,
        mean_confidence: float,
        verifier_signal: float,
        tool_signal: float,
        budget_ratio: float,
        step_ratio: float,
        previous_action: str = "",
    ) -> "RLoTState":
        answer_available = valid_answer_count > 0
        agreement = _clip(agreement, 0.0, 1.0)
        verifier_signal = _clip(verifier_signal, -1.0, 1.0)
        tool_signal = _clip(tool_signal, -1.0, 1.0)
        parse_ratio = valid_answer_count / max(1, candidate_count)

        if tool_signal < -0.25 or verifier_signal < -0.5:
            modeling = 1
        elif tool_signal > 0.5 or (
            candidate_count >= 2 and agreement >= 0.75 and verifier_signal >= 0.0
        ):
            modeling = 3
        else:
            modeling = 2 if answer_available else 1

        if parse_ratio >= 0.8 and mean_confidence >= 0.55:
            clarity = 3
        elif answer_available or candidate_count:
            clarity = 2
        else:
            clarity = 1

        if tool_signal > 0.5:
            calculation = 3
        elif tool_signal < -0.25:
            calculation = 1
        elif verifier_signal > 0.25 and agreement >= 0.5:
            calculation = 3
        else:
            calculation = 2 if answer_available else 1

        if answer_available and agreement >= 0.75:
            complexity = 3
        elif answer_available:
            complexity = 2
        else:
            complexity = 1

        if candidate_count >= 3 and distinct_answer_count >= 2:
            alternatives = 3
        elif candidate_count >= 2 or previous_action in {"Decompose", "Debate"}:
            alternatives = 2
        else:
            alternatives = 1

        if answer_available and agreement >= 0.75 and min(verifier_signal, tool_signal) >= 0.0:
            closeness = 3
        elif answer_available:
            closeness = 2
        else:
            closeness = 1

        if answer_available and (
            tool_signal > 0.5
            or verifier_signal > 0.25
            or (candidate_count >= 2 and agreement >= 0.75)
        ):
            completeness = 3
        elif answer_available:
            completeness = 2
        else:
            completeness = 1

        return cls(
            correctness_of_modeling=modeling,
            clarity_for_further_reasoning=clarity,
            correctness_of_calculation=calculation,
            complexity_to_final_answer=complexity,
            alternative_methods=alternatives,
            closeness_to_final_solution=closeness,
            completeness_within_step=completeness,
            difficulty=max(1, min(10, int(difficulty))),
            proof_mode=bool(proof_mode),
            agreement=agreement,
            verifier_signal=verifier_signal,
            tool_signal=tool_signal,
            answer_available=answer_available,
            budget_ratio=_clip(budget_ratio, 0.0, 1.0),
            step_ratio=_clip(step_ratio, 0.0, 1.0),
            previous_action=previous_action if previous_action in ACTIONS else "",
            domain=domain,
            candidate_count=max(0, int(candidate_count)),
            valid_answer_count=max(0, int(valid_answer_count)),
            distinct_answer_count=max(0, int(distinct_answer_count)),
        )

    def vector(self) -> List[float]:
        values = [
            (float(getattr(self, name)) - 1.0) / 2.0
            for name in SELF_EVALUATION_FIELDS
        ]
        values.extend(
            [
                self.difficulty / 10.0,
                float(self.proof_mode),
                self.agreement,
                (self.verifier_signal + 1.0) / 2.0,
                (self.tool_signal + 1.0) / 2.0,
                float(self.answer_available),
                self.budget_ratio,
                self.step_ratio,
            ]
        )
        values.extend(float(self.previous_action == action) for action in ACTIONS)
        values.extend(float(self.domain == domain) for domain in DOMAINS)
        if len(values) != 38:
            raise RuntimeError(f"Unexpected RLoT feature count: {len(values)}")
        return values

    def as_dict(self) -> Dict[str, Any]:
        return {
            "self_evaluation": {
                name: int(getattr(self, name)) for name in SELF_EVALUATION_FIELDS
            },
            "difficulty": self.difficulty,
            "proof_mode": self.proof_mode,
            "domain": self.domain,
            "candidate_count": self.candidate_count,
            "valid_answer_count": self.valid_answer_count,
            "distinct_answer_count": self.distinct_answer_count,
            "agreement": round(self.agreement, 4),
            "verifier_signal": round(self.verifier_signal, 4),
            "tool_signal": round(self.tool_signal, 4),
            "answer_available": self.answer_available,
            "budget_ratio": round(self.budget_ratio, 4),
            "step_ratio": round(self.step_ratio, 4),
            "previous_action": self.previous_action or None,
        }


@dataclass(frozen=True)
class NavigationDecision:
    action: str
    source: str
    q_values: Dict[str, float]
    margin: float
    valid_actions: Tuple[str, ...]
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source,
            "q_values": {key: round(value, 5) for key, value in self.q_values.items()},
            "margin": round(self.margin, 5),
            "valid_actions": list(self.valid_actions),
            "reason": self.reason,
        }


class RLoTNavigator:
    """Inference-only Double-Dueling-DQN policy with a conservative fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_margin: float | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else Path(__file__).with_name(MODEL_FILENAME)
        self.payload: Dict[str, Any] = {}
        self.layers: Dict[str, Any] = {}
        self.loaded = False
        self.load_error = ""
        self.confidence_margin = 0.06 if confidence_margin is None else float(confidence_margin)
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
                raise ValueError("unsupported schema version")
            if tuple(payload.get("actions", [])) != ACTIONS:
                raise ValueError("action order mismatch")
            if int(payload.get("input_size", 0)) != 38:
                raise ValueError("feature size mismatch")
            layers = payload["network"]
            self._validate_layers(layers)
            self.payload = payload
            self.layers = layers
            self.confidence_margin = float(
                payload.get("inference", {}).get("confidence_margin", self.confidence_margin)
            )
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 - absent/corrupt models must fail open.
            self.load_error = f"{type(exc).__name__}: model artifact could not be loaded"
            self.loaded = False

    @staticmethod
    def _validate_layers(layers: Mapping[str, Any]) -> None:
        expected = {
            "w1": (38, 32),
            "b1": (32,),
            "w2": (32, 32),
            "b2": (32,),
            "value_w": (32, 1),
            "value_b": (1,),
            "advantage_w": (32, 5),
            "advantage_b": (5,),
        }
        for name, shape in expected.items():
            value = layers.get(name)
            if value is None:
                raise ValueError(f"missing layer {name}")
            actual = RLoTNavigator._shape(value)
            if actual != shape:
                raise ValueError(f"layer {name} has shape {actual}, expected {shape}")

    @staticmethod
    def _shape(value: Any) -> Tuple[int, ...]:
        if not isinstance(value, list):
            return ()
        if not value:
            return (0,)
        if isinstance(value[0], list):
            widths = {len(row) for row in value if isinstance(row, list)}
            if len(widths) != 1 or len(value) == 0:
                return (len(value), -1)
            return (len(value), widths.pop())
        return (len(value),)

    @property
    def parameter_count(self) -> int:
        if self.payload:
            return int(self.payload.get("parameter_count", 0))
        return 0

    def q_values(self, state: RLoTState) -> Dict[str, float]:
        if not self.loaded:
            return {}
        hidden1 = self._relu(self._dense(state.vector(), self.layers["w1"], self.layers["b1"]))
        hidden2 = self._relu(self._dense(hidden1, self.layers["w2"], self.layers["b2"]))
        value = self._dense(hidden2, self.layers["value_w"], self.layers["value_b"])[0]
        advantages = self._dense(
            hidden2,
            self.layers["advantage_w"],
            self.layers["advantage_b"],
        )
        advantage_mean = sum(advantages) / len(advantages)
        return {
            action: value + advantages[index] - advantage_mean
            for index, action in enumerate(ACTIONS)
        }

    @staticmethod
    def _dense(values: Sequence[float], weights: Sequence[Sequence[float]], bias: Sequence[float]) -> List[float]:
        return [
            float(bias[column])
            + sum(float(values[row]) * float(weights[row][column]) for row in range(len(values)))
            for column in range(len(bias))
        ]

    @staticmethod
    def _relu(values: Iterable[float]) -> List[float]:
        return [max(0.0, float(value)) for value in values]

    def decide(
        self,
        state: RLoTState,
        valid_actions: Sequence[str],
        heuristic_action: str,
    ) -> NavigationDecision:
        valid = tuple(action for action in ACTIONS if action in set(valid_actions))
        if not valid:
            valid = ("Terminate",)
        if heuristic_action not in valid:
            heuristic_action = valid[0]

        q_values = self.q_values(state)
        if not q_values:
            return NavigationDecision(
                action=heuristic_action,
                source="rule_fallback",
                q_values={},
                margin=0.0,
                valid_actions=valid,
                reason=self.load_error or "model unavailable",
            )

        ranked = sorted(valid, key=lambda action: q_values[action], reverse=True)
        margin = math.inf if len(ranked) == 1 else q_values[ranked[0]] - q_values[ranked[1]]
        if margin < self.confidence_margin:
            return NavigationDecision(
                action=heuristic_action,
                source="rule_fallback",
                q_values=q_values,
                margin=margin,
                valid_actions=valid,
                reason=f"Q margin below {self.confidence_margin:.4f}",
            )
        return NavigationDecision(
            action=ranked[0],
            source="learned_policy",
            q_values=q_values,
            margin=margin,
            valid_actions=valid,
            reason="highest valid Double-Dueling-DQN action value",
        )

    @staticmethod
    def heuristic_action(
        state: RLoTState,
        history: Sequence[str],
        valid_actions: Sequence[str],
    ) -> str:
        valid = set(valid_actions)

        def choose(*actions: str) -> str:
            return next((action for action in actions if action in valid), next(iter(valid), "Terminate"))

        if not history:
            if state.proof_mode or state.difficulty >= 7:
                return choose("Decompose", "ReasonOneStep", "Debate")
            return choose("ReasonOneStep", "Decompose", "Debate")
        if not state.answer_available:
            return choose("Decompose", "Debate", "ReasonOneStep", "Terminate")
        if state.tool_signal < -0.25 or state.verifier_signal < -0.4:
            return choose("Refine", "Debate", "ReasonOneStep", "Terminate")
        if state.agreement < 0.67 and state.candidate_count < 3:
            return choose("Debate", "Decompose", "Refine", "ReasonOneStep", "Terminate")
        if state.proof_mode and "Refine" not in history:
            return choose("Refine", "Debate", "Terminate")
        if state.candidate_count == 1 and "Refine" not in history:
            return choose("Refine", "Debate", "ReasonOneStep", "Terminate")
        if state.completeness_within_step >= 3 and state.agreement >= 0.67:
            return choose("Terminate", "Refine", "Debate")
        return choose("Refine", "Debate", "Terminate", "ReasonOneStep")


def count_parameters(input_size: int = 38, hidden_size: int = 32, action_size: int = 5) -> int:
    return (
        input_size * hidden_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + hidden_size
        + 1
        + hidden_size * action_size
        + action_size
    )
