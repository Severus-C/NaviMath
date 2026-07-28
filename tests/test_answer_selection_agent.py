import unittest

from agent.agent_utils import Candidate, Route, Skill
from agent.answer_selection_agent import AnswerSelectionAgent
from agent.tool_verify import REJECTED, VERIFIED


class AnswerSelectionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = AnswerSelectionAgent()
        self.route = Route(
            subject="algebra",
            skill=Skill("algebra", "algebra", [], "solve"),
            difficulty_score=4,
            proof_mode=False,
            action_plan=[],
        )

    @staticmethod
    def _candidate(role: str, answer: str, *, tool_verdict: str = "UNKNOWN") -> Candidate:
        return Candidate(
            role=role,
            content=f"SOLUTION: checked\nFINAL_ANSWER: {answer}",
            extracted_answer=answer,
            normalized_answer=answer,
            confidence=0.7,
            tool_verdict=tool_verdict,
            tool_confidence=1.0 if tool_verdict == VERIFIED else 0.0,
        )

    def test_selects_only_from_hard_rule_eligible_candidates(self) -> None:
        prompts: list[str] = []

        def chat(prompt: str, **_: object) -> str:
            prompts.append(prompt)
            return """SELECTED_INDEX: 2
TARGET_MATCH: PASS
CONSTRAINTS: PASS
DERIVATION: PASS
UNITS_FORMAT: PASS
EVIDENCE: deterministic evidence supports this candidate"""

        candidates = [
            self._candidate("bad", "1", tool_verdict=REJECTED),
            self._candidate("first", "2"),
            self._candidate("verified", "3", tool_verdict=VERIFIED),
        ]
        decision = self.selector.select(
            problem="Find x.",
            route=self.route,
            candidates=candidates,
            chat=chat,
            max_tokens=500,
        )

        self.assertEqual(decision.selected_index, 2)
        self.assertEqual(decision.source, "selection_subagent")
        self.assertNotIn("CANDIDATE_INDEX: 0", prompts[0])
        self.assertEqual(decision.checks["target_match"], "PASS")

    def test_rejects_out_of_set_or_answer_generating_response(self) -> None:
        candidates = [self._candidate("first", "2"), self._candidate("second", "3")]

        out_of_set = self.selector.select(
            problem="Find x.",
            route=self.route,
            candidates=candidates,
            chat=lambda *args, **kwargs: "SELECTED_INDEX: 99",
            max_tokens=500,
        )
        generated_answer = self.selector.select(
            problem="Find x.",
            route=self.route,
            candidates=candidates,
            chat=lambda *args, **kwargs: "FINAL_ANSWER: 7",
            max_tokens=500,
        )

        self.assertIsNone(out_of_set.selected_index)
        self.assertIsNone(generated_answer.selected_index)


if __name__ == "__main__":
    unittest.main()
