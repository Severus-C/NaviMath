import json
import tempfile
import unittest
from pathlib import Path

from agent.rlot_navigator import ACTIONS, RLoTNavigator, RLoTState, count_parameters


class RLoTNavigatorTests(unittest.TestCase):
    def test_paper_state_plus_runtime_context_has_stable_dimension(self) -> None:
        state = RLoTState.from_signals(
            difficulty=8,
            proof_mode=True,
            domain="real_analysis",
            candidate_count=3,
            valid_answer_count=2,
            distinct_answer_count=2,
            agreement=0.5,
            mean_confidence=0.7,
            verifier_signal=-0.5,
            tool_signal=0.75,
            budget_ratio=0.4,
            step_ratio=0.5,
            previous_action="Debate",
        )

        self.assertEqual(len(state.vector()), 38)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in state.vector()))
        self.assertEqual(len(state.as_dict()["self_evaluation"]), 7)

    def test_trained_default_artifact_is_loadable_and_under_3k_parameters(self) -> None:
        navigator = RLoTNavigator()

        self.assertTrue(navigator.loaded, navigator.load_error)
        self.assertEqual(navigator.parameter_count, count_parameters())
        self.assertEqual(navigator.parameter_count, 2502)
        self.assertLess(navigator.parameter_count, 3000)

    def test_corrupt_artifact_fails_open_to_rule_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
            navigator = RLoTNavigator(path)
            state = RLoTState()
            decision = navigator.decide(
                state,
                ["ReasonOneStep", "Decompose"],
                "ReasonOneStep",
            )

        self.assertFalse(navigator.loaded)
        self.assertEqual(decision.action, "ReasonOneStep")
        self.assertEqual(decision.source, "rule_fallback")

    def test_first_action_never_uses_refine(self) -> None:
        navigator = RLoTNavigator()
        state = RLoTState()
        valid = [action for action in ACTIONS if action not in {"Refine", "Terminate"}]
        action = navigator.heuristic_action(state, [], valid)

        self.assertIn(action, valid)
        self.assertNotEqual(action, "Refine")


if __name__ == "__main__":
    unittest.main()
