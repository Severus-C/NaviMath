from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rlot_navigator import (  # noqa: E402
    ACTIONS,
    DOMAINS,
    MODEL_SCHEMA_VERSION,
    RLoTNavigator,
    RLoTState,
    count_parameters,
)


ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
ACTION_COST = {
    "ReasonOneStep": 0.035,
    "Decompose": 0.05,
    "Debate": 0.075,
    "Refine": 0.06,
    "Terminate": 0.01,
}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


@dataclass
class EpisodeSignals:
    episode_id: str
    domain: str
    difficulty: int
    proof_mode: bool
    candidate_count: int
    valid_answer_count: int
    distinct_answer_count: int
    agreement: float
    mean_confidence: float
    verifier_signal: float
    tool_signal: float
    correct: bool
    observed_plan: List[str]
    source: str


@dataclass
class Transition:
    episode_id: str
    state: RLoTState
    action: str
    reward: float
    next_state: RLoTState
    done: bool
    valid_actions: Tuple[str, ...]
    next_valid_actions: Tuple[str, ...]
    heuristic_action: str
    source: str
    successful: bool


def _trace_entries(record: Dict[str, Any], step: str) -> List[Dict[str, Any]]:
    return [
        entry.get("content", {})
        for entry in record.get("trace", [])
        if entry.get("step") == step and isinstance(entry.get("content"), dict)
    ]


def _map_plan(plan: Iterable[Any]) -> List[str]:
    mapped = []
    for raw in plan:
        action = str(raw)
        if action in {"Attack", "Verify"}:
            action = "Refine"
        if action not in ACTIONS:
            continue
        if mapped and mapped[-1] == action == "Refine":
            continue
        mapped.append(action)
    if not mapped:
        mapped = ["ReasonOneStep", "Refine", "Terminate"]
    if mapped[-1] != "Terminate":
        mapped.append("Terminate")
    if mapped[0] == "Refine":
        mapped[0] = "ReasonOneStep"
    return mapped[:5]


def record_to_signals(record: Dict[str, Any], source: str) -> EpisodeSignals | None:
    route_items = _trace_entries(record, "route")
    if not route_items:
        return None
    route = route_items[0]
    candidates = _trace_entries(record, "candidate")
    rejected = _trace_entries(record, "candidate_rejected")
    fallbacks = _trace_entries(record, "fallback_candidate")
    all_candidates = [*candidates, *rejected, *fallbacks]
    normalized = [
        str(item.get("normalized_answer") or item.get("extracted_answer") or "").strip()
        for item in [*candidates, *fallbacks]
    ]
    normalized = [value for value in normalized if value and value != "<proof>"]
    counts: Dict[str, int] = {}
    for value in normalized:
        counts[value] = counts.get(value, 0) + 1
    proof_mode = bool(route.get("proof_mode"))
    valid_count = len(candidates) + len(fallbacks)
    if proof_mode:
        distinct_count = valid_count
        agreement = 0.5 if valid_count else 0.0
    else:
        distinct_count = len(counts)
        agreement = max(counts.values(), default=0) / max(1, len(normalized))

    verifier_values = []
    for item in _trace_entries(record, "attack_verifier"):
        verdict = str(item.get("verdict", "")).upper()
        if verdict in {"ACCEPT", "CORRECT"}:
            verifier_values.append(1.0)
        elif verdict in {"REJECT", "INCORRECT"}:
            verifier_values.append(-1.0)
    verifier_signal = (
        sum(verifier_values) / len(verifier_values) if verifier_values else 0.0
    )

    tool_values = []
    for item in _trace_entries(record, "tool_verify"):
        verdict = str(item.get("verdict", "")).upper()
        confidence = max(0.25, float(item.get("confidence") or 0.0))
        if verdict == "VERIFIED":
            tool_values.append(confidence)
        elif verdict == "REJECTED":
            tool_values.append(-confidence)
    tool_signal = sum(tool_values) / len(tool_values) if tool_values else 0.0
    mean_confidence = (
        sum(float(item.get("confidence") or 0.0) for item in candidates) / len(candidates)
        if candidates
        else 0.0
    )
    episode_id = f"{source}:{record.get('id', record.get('idx', 'unknown'))}"
    return EpisodeSignals(
        episode_id=episode_id,
        domain=str(route.get("subject") or "general_contest_math"),
        difficulty=max(1, min(10, int(route.get("difficulty_score") or 1))),
        proof_mode=proof_mode,
        candidate_count=len(all_candidates),
        valid_answer_count=valid_count,
        distinct_answer_count=distinct_count,
        agreement=agreement,
        mean_confidence=mean_confidence,
        verifier_signal=verifier_signal,
        tool_signal=tool_signal,
        correct=bool(record.get("is_correct")),
        observed_plan=_map_plan(route.get("action_plan") or []),
        source="evaluation_trace",
    )


def load_trace_signals(paths: Sequence[Path]) -> Tuple[List[EpisodeSignals], Dict[str, Any]]:
    signals = []
    files = []
    for path in sorted(paths):
        if path.name.endswith("_analyzed.jsonl"):
            continue
        rows = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                item = record_to_signals(record, path.name)
                if item is not None:
                    signals.append(item)
                    rows += 1
        files.append({"path": _display_path(path), "usable_episodes": rows})
    metadata = {
        "files": files,
        "episodes": len(signals),
        "correct": sum(item.correct for item in signals),
        "wrong": sum(not item.correct for item in signals),
    }
    return signals, metadata


def _valid_actions(state: RLoTState, history: Sequence[str], final_step: bool = False) -> Tuple[str, ...]:
    if final_step:
        return ("Terminate",)
    valid = list(ACTIONS)
    if not history:
        valid.remove("Refine")
        valid.remove("Terminate")
    elif not state.answer_available:
        valid = [action for action in valid if action not in {"Refine", "Terminate"}]
    if "Decompose" in history:
        valid = [action for action in valid if action != "Decompose"]
    if history.count("ReasonOneStep") >= 2:
        valid = [action for action in valid if action != "ReasonOneStep"]
    return tuple(valid or ["Terminate"])


def _state_from_progress(
    signals: EpisodeSignals,
    history: Sequence[str],
    step_index: int,
    total_steps: int,
) -> RLoTState:
    reasoning_actions = [action for action in history if action != "Terminate"]
    generated = sum(action in {"ReasonOneStep", "Decompose"} for action in reasoning_actions)
    debated = "Debate" in reasoning_actions
    refined = "Refine" in reasoning_actions
    candidate_count = min(signals.candidate_count, max(generated, 1 if debated else 0))
    if debated:
        candidate_count = max(candidate_count, signals.candidate_count)
    valid_count = min(signals.valid_answer_count, candidate_count)
    if signals.candidate_count and candidate_count:
        valid_count = max(
            valid_count,
            round(signals.valid_answer_count * candidate_count / signals.candidate_count),
        )
    distinct_count = min(signals.distinct_answer_count, valid_count)
    if valid_count and distinct_count == 0 and not signals.proof_mode:
        distinct_count = 1
    agreement = signals.agreement if debated or candidate_count >= 2 else (1.0 if valid_count else 0.0)
    verifier_signal = signals.verifier_signal if refined else 0.0
    tool_signal = signals.tool_signal if reasoning_actions else 0.0
    return RLoTState.from_signals(
        difficulty=signals.difficulty,
        proof_mode=signals.proof_mode,
        domain=signals.domain,
        candidate_count=candidate_count,
        valid_answer_count=valid_count,
        distinct_answer_count=distinct_count,
        agreement=agreement,
        mean_confidence=signals.mean_confidence,
        verifier_signal=verifier_signal,
        tool_signal=tool_signal,
        budget_ratio=max(0.0, (total_steps - step_index) / max(1, total_steps)),
        step_ratio=step_index / max(1, total_steps),
        previous_action=history[-1] if history else "",
    )


def _potential(state: RLoTState) -> float:
    selected = (
        state.correctness_of_modeling,
        state.correctness_of_calculation,
        state.closeness_to_final_solution,
        state.completeness_within_step,
    )
    return sum((value - 1.0) / 2.0 for value in selected) / len(selected)


def trace_episode_transitions(signals: EpisodeSignals) -> List[Transition]:
    sequence = signals.observed_plan
    transitions = []
    history: List[str] = []
    state = _state_from_progress(signals, history, 0, len(sequence))
    for index, action in enumerate(sequence):
        done = action == "Terminate" or index == len(sequence) - 1
        valid = _valid_actions(state, history, final_step=done and action == "Terminate")
        if action not in valid:
            action = RLoTNavigator.heuristic_action(state, history, valid)
        next_history = [*history, action]
        next_state = _state_from_progress(signals, next_history, index + 1, len(sequence))
        next_valid = _valid_actions(next_state, next_history, final_step=done)
        reward = 0.22 * (_potential(next_state) - _potential(state)) - ACTION_COST[action]
        if done:
            reward += 1.0 if signals.correct else -1.0
        elif signals.correct:
            reward += 0.025
        heuristic = RLoTNavigator.heuristic_action(state, history, valid)
        transitions.append(
            Transition(
                episode_id=signals.episode_id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_actions=valid,
                next_valid_actions=next_valid,
                heuristic_action=heuristic,
                source=signals.source,
                successful=signals.correct,
            )
        )
        state = next_state
        history = next_history
        if done:
            break
    return transitions


def paper_prior_signals() -> List[EpisodeSignals]:
    priors = []
    patterns = {
        2: [
            ["ReasonOneStep", "Refine", "Terminate"],
            ["ReasonOneStep", "Decompose", "Terminate"],
        ],
        5: [
            ["ReasonOneStep", "Refine", "Debate", "Terminate"],
            ["ReasonOneStep", "Decompose", "Refine", "Terminate"],
        ],
        8: [
            ["Decompose", "Refine", "ReasonOneStep", "Terminate"],
            ["ReasonOneStep", "Decompose", "Debate", "Terminate"],
        ],
    }
    for domain in DOMAINS:
        for difficulty, domain_patterns in patterns.items():
            for pattern_index, pattern in enumerate(domain_patterns):
                priors.append(
                    EpisodeSignals(
                        episode_id=f"paper:{domain}:{difficulty}:{pattern_index}",
                        domain=domain,
                        difficulty=difficulty,
                        proof_mode=difficulty >= 8 and pattern_index == 0,
                        candidate_count=3 if "Debate" in pattern else 2,
                        valid_answer_count=3 if "Debate" in pattern else 2,
                        distinct_answer_count=2 if "Debate" in pattern else 1,
                        agreement=2 / 3 if "Debate" in pattern else 1.0,
                        mean_confidence=0.78,
                        verifier_signal=0.8 if "Refine" in pattern else 0.2,
                        tool_signal=0.45,
                        correct=True,
                        observed_plan=pattern,
                        source="paper_pattern_prior",
                    )
                )
    return priors


class DuelingNetwork:
    def __init__(self, rng: np.random.Generator) -> None:
        self.parameters = {
            "w1": rng.normal(0.0, np.sqrt(2 / 38), size=(38, 32)),
            "b1": np.zeros(32),
            "w2": rng.normal(0.0, np.sqrt(2 / 32), size=(32, 32)),
            "b2": np.zeros(32),
            "value_w": rng.normal(0.0, 0.08, size=(32, 1)),
            "value_b": np.zeros(1),
            "advantage_w": rng.normal(0.0, 0.08, size=(32, 5)),
            "advantage_b": np.zeros(5),
        }
        self.m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.optimization_step = 0

    def copy_from(self, other: "DuelingNetwork") -> None:
        for name in self.parameters:
            self.parameters[name][...] = other.parameters[name]

    def forward(self, x: np.ndarray, cache: bool = False) -> Any:
        p = self.parameters
        z1 = x @ p["w1"] + p["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ p["w2"] + p["b2"]
        h2 = np.maximum(z2, 0.0)
        value = h2 @ p["value_w"] + p["value_b"]
        advantage = h2 @ p["advantage_w"] + p["advantage_b"]
        q = value + advantage - advantage.mean(axis=1, keepdims=True)
        if cache:
            return q, (x, z1, h1, z2, h2)
        return q

    def update(
        self,
        x: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        q, cache = self.forward(x, cache=True)
        chosen = q[np.arange(len(x)), actions]
        error = chosen - targets
        abs_error = np.abs(error)
        loss = np.where(abs_error <= 1.0, 0.5 * error**2, abs_error - 0.5).mean()
        derivative = np.where(abs_error <= 1.0, error, np.sign(error)) / len(x)
        dq = np.zeros_like(q)
        dq[np.arange(len(x)), actions] = derivative

        x0, z1, h1, z2, h2 = cache
        dv = dq.sum(axis=1, keepdims=True)
        da = dq - dq.mean(axis=1, keepdims=True)
        p = self.parameters
        gradients: Dict[str, np.ndarray] = {}
        gradients["value_w"] = h2.T @ dv
        gradients["value_b"] = dv.sum(axis=0)
        gradients["advantage_w"] = h2.T @ da
        gradients["advantage_b"] = da.sum(axis=0)
        dh2 = dv @ p["value_w"].T + da @ p["advantage_w"].T
        dz2 = dh2 * (z2 > 0.0)
        gradients["w2"] = h1.T @ dz2
        gradients["b2"] = dz2.sum(axis=0)
        dh1 = dz2 @ p["w2"].T
        dz1 = dh1 * (z1 > 0.0)
        gradients["w1"] = x0.T @ dz1
        gradients["b1"] = dz1.sum(axis=0)
        self._adam(gradients, learning_rate)
        return float(loss)

    def _adam(self, gradients: Dict[str, np.ndarray], learning_rate: float) -> None:
        self.optimization_step += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in gradients.items():
            gradient = np.clip(gradient, -5.0, 5.0)
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * gradient
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * gradient**2
            m_hat = self.m[name] / (1.0 - beta1**self.optimization_step)
            v_hat = self.v[name] / (1.0 - beta2**self.optimization_step)
            self.parameters[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def _masked_argmax(q: np.ndarray, valid_actions: Sequence[Tuple[str, ...]]) -> np.ndarray:
    masked = np.full_like(q, -1e9)
    for row, valid in enumerate(valid_actions):
        for action in valid:
            masked[row, ACTION_INDEX[action]] = q[row, ACTION_INDEX[action]]
    return masked.argmax(axis=1)


def train_network(
    transitions: Sequence[Transition],
    *,
    updates: int,
    seed: int,
    batch_size: int,
    gamma: float,
    learning_rate: float,
) -> Tuple[DuelingNetwork, List[float]]:
    rng = np.random.default_rng(seed)
    online = DuelingNetwork(rng)
    target = DuelingNetwork(np.random.default_rng(seed + 1))
    target.copy_from(online)
    states = np.asarray([item.state.vector() for item in transitions], dtype=np.float64)
    next_states = np.asarray([item.next_state.vector() for item in transitions], dtype=np.float64)
    actions = np.asarray([ACTION_INDEX[item.action] for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float64)
    dones = np.asarray([item.done for item in transitions], dtype=np.float64)
    next_valid = [item.next_valid_actions for item in transitions]
    losses = []
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(transitions), size=min(batch_size, len(transitions)))
        next_online_q = online.forward(next_states[indices])
        next_actions = _masked_argmax(next_online_q, [next_valid[index] for index in indices])
        next_target_q = target.forward(next_states[indices])
        bootstrap = next_target_q[np.arange(len(indices)), next_actions]
        targets = rewards[indices] + gamma * (1.0 - dones[indices]) * bootstrap
        loss = online.update(states[indices], actions[indices], targets, learning_rate)
        losses.append(loss)
        if update % 100 == 0:
            target.copy_from(online)
    return online, losses


def _episode_split(episode_id: str) -> str:
    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()[0]
    return "holdout" if digest < 51 else "train"


def evaluate_policy(
    network: DuelingNetwork,
    transitions: Sequence[Transition],
    confidence_margin: float,
) -> Dict[str, Any]:
    if not transitions:
        return {"transitions": 0}
    q = network.forward(np.asarray([item.state.vector() for item in transitions]))
    learned_correct = 0
    rule_correct = 0
    hybrid_correct = 0
    learned_used = 0
    successful_count = 0
    weighted_reward_learned = 0.0
    weighted_reward_rule = 0.0
    for row, item in enumerate(transitions):
        valid_indices = [ACTION_INDEX[action] for action in item.valid_actions]
        ranking = sorted(valid_indices, key=lambda index: q[row, index], reverse=True)
        learned = ACTIONS[ranking[0]]
        margin = float("inf") if len(ranking) == 1 else q[row, ranking[0]] - q[row, ranking[1]]
        hybrid = learned if margin >= confidence_margin else item.heuristic_action
        learned_correct += learned == item.action
        rule_correct += item.heuristic_action == item.action
        hybrid_correct += hybrid == item.action
        learned_used += margin >= confidence_margin
        successful_count += item.successful
        weighted_reward_learned += item.reward * (1.0 if learned == item.action else 0.0)
        weighted_reward_rule += item.reward * (1.0 if item.heuristic_action == item.action else 0.0)
    total = len(transitions)
    return {
        "transitions": total,
        "successful_trace_transitions": successful_count,
        "raw_learned_action_accuracy": round(learned_correct / total, 4),
        "rule_action_accuracy": round(rule_correct / total, 4),
        "hybrid_action_accuracy": round(hybrid_correct / total, 4),
        "learned_policy_coverage": round(learned_used / total, 4),
        "matched_proxy_return_learned": round(weighted_reward_learned, 4),
        "matched_proxy_return_rule": round(weighted_reward_rule, 4),
    }


def select_confidence_margin(
    network: DuelingNetwork,
    transitions: Sequence[Transition],
) -> Tuple[float, Dict[str, Any]]:
    candidates = [
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.12,
        0.18,
        0.25,
        0.35,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        5.0,
    ]
    scored = [(margin, evaluate_policy(network, transitions, margin)) for margin in candidates]
    best = max(
        scored,
        key=lambda item: (
            item[1].get("hybrid_action_accuracy", 0.0),
            item[1].get("learned_policy_coverage", 0.0),
            -item[0],
        ),
    )
    return best


def save_artifact(
    network: DuelingNetwork,
    output: Path,
    *,
    seed: int,
    updates: int,
    confidence_margin: float,
    trace_metadata: Dict[str, Any],
    train_transition_count: int,
    holdout_report: Dict[str, Any],
    final_loss: float,
) -> None:
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "offline_double_dueling_dqn",
        "paper": "RL OF THOUGHTS: NAVIGATING LLM REASONING (ICLR 2026)",
        "actions": list(ACTIONS),
        "input_size": 38,
        "hidden_size": 32,
        "parameter_count": count_parameters(),
        "state": {
            "paper_self_evaluation_dimensions": 7,
            "runtime_context_dimensions": 31,
            "domains": list(DOMAINS),
        },
        "training": {
            "algorithm": "offline Double-Dueling-DQN with target network and Huber loss",
            "updates": updates,
            "seed": seed,
            "train_transitions": train_transition_count,
            "trace_data": trace_metadata,
            "paper_prior_episodes": len(paper_prior_signals()),
            "reward": "final correctness +/-1 + process-potential delta - action cost",
            "prm_note": "Local traces do not contain Math-Shepherd PRM scores; deterministic trace evidence is used as a process-reward proxy.",
            "final_training_loss": round(final_loss, 8),
        },
        "inference": {"confidence_margin": confidence_margin, "fail_open": True},
        "holdout": holdout_report,
        "network": {
            name: np.round(value, 8).tolist() for name, value in network.parameters.items()
        },
    }
    _atomic_write_text(output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NaviMath's offline RLoT navigator.")
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "agent" / "rlot_policy.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "eval_outputs" / "rlot_training_report.json",
    )
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = [path if path.is_absolute() else ROOT / path for path in args.inputs]
    inputs = inputs or sorted((ROOT / "eval_outputs").glob("*.jsonl"))
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    trace_signals, trace_metadata = load_trace_signals(inputs)
    if not trace_signals:
        raise SystemExit("No evaluation traces with route metadata were found.")
    trace_train = [item for item in trace_signals if _episode_split(item.episode_id) == "train"]
    trace_holdout = [item for item in trace_signals if _episode_split(item.episode_id) == "holdout"]
    train_episodes = [*trace_train, *paper_prior_signals()]
    train_transitions = [
        transition
        for episode in train_episodes
        for transition in trace_episode_transitions(episode)
    ]
    holdout_transitions = [
        transition
        for episode in trace_holdout
        for transition in trace_episode_transitions(episode)
    ]
    network, losses = train_network(
        train_transitions,
        updates=max(1, args.updates),
        seed=args.seed,
        batch_size=max(1, args.batch_size),
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )
    confidence_margin, holdout_report = select_confidence_margin(network, holdout_transitions)
    holdout_report.update(
        {
            "episodes": len(trace_holdout),
            "confidence_margin": confidence_margin,
            "warning": "Offline action agreement and proxy return are not end-to-end answer accuracy.",
        }
    )
    save_artifact(
        network,
        output_path,
        seed=args.seed,
        updates=max(1, args.updates),
        confidence_margin=confidence_margin,
        trace_metadata=trace_metadata,
        train_transition_count=len(train_transitions),
        holdout_report=holdout_report,
        final_loss=losses[-1],
    )
    report = {
        "model": _display_path(output_path),
        "parameter_count": count_parameters(),
        "trace_episodes": len(trace_signals),
        "trace_train_episodes": len(trace_train),
        "trace_holdout_episodes": len(trace_holdout),
        "paper_prior_episodes": len(paper_prior_signals()),
        "train_transitions": len(train_transitions),
        "training_updates": max(1, args.updates),
        "initial_loss": round(losses[0], 8),
        "final_loss": round(losses[-1], 8),
        "holdout": holdout_report,
        "limitations": [
            "The local logs contain final correctness but no Math-Shepherd per-step PRM score.",
            "The holdout report measures policy agreement/proxy return, not model-answer accuracy.",
            "New navigator traces should be evaluated end to end and fed back into later retraining.",
        ],
    }
    _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
