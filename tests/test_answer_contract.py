import unittest

from agent.answer_contract import AnswerContract, has_terminal_answer
from agent.agent_utils import extract_final_answer, is_plausible_final_answer
from agent.answer_selection_agent import AnswerSelectionAgent
from agent.agent_utils import Candidate, Route, Skill
from scripts.eval_agent_on_dataset import is_match


class AnswerContractTests(unittest.TestCase):
    def test_accepts_common_terminal_protocol_variants(self) -> None:
        variants = [
            "FINALANSWER: b+c=10",
            "FINAL ANSWER: b+c=10",
            "**FINAL_ANSWER:** b+c=10",
        ]
        for response in variants:
            with self.subTest(response=response):
                self.assertTrue(has_terminal_answer(response))
                self.assertEqual(extract_final_answer(response), "b+c=10")

    def test_rejects_v3_schema_placeholders(self) -> None:
        for value in ["the condition", "the completion", "just the number/expression"]:
            with self.subTest(value=value):
                self.assertFalse(is_plausible_final_answer(value))

    def test_preserves_embedded_currency_marker(self) -> None:
        self.assertTrue(
            is_match(
                "a loss of $1,000",
                r"\text{loss of}\textdollar1000",
                answer_type="expression",
            )
        )
        self.assertTrue(
            is_match(
                "loss of 1000",
                r"\text{loss of}\textdollar1000",
                answer_type="expression",
            )
        )

    def test_cleans_v3_protocol_narration(self) -> None:
        self.assertEqual(
            extract_final_answer("C=3P+7` contains the answer. The text before it is fine"),
            "C=3P+7",
        )
        self.assertEqual(extract_final_answer('Just the values. "4, -1'), "4, -1")
        self.assertEqual(extract_final_answer('Justthevalues."4,-1'), "4,-1")

    def test_infers_and_enforces_target_shapes(self) -> None:
        ratio = AnswerContract.infer(
            "The area of one square is to the area of another square as:"
        )
        count = AnswerContract.infer("The number of solutions to this problem is:")
        condition = AnswerContract.infer("The identity holds if:")
        area = AnswerContract.infer("Then the area of triangle N_1N_2N_3 is:")

        self.assertEqual(ratio.kind, "ratio")
        self.assertTrue(ratio.accepts("2/5"))
        self.assertFalse(ratio.accepts(r"(\pm s_2/2,s_2)"))
        self.assertTrue(count.accepts("3"))
        self.assertFalse(count.accepts(r"\{24,32\}"))
        self.assertTrue(condition.accepts("b+c=10"))
        self.assertFalse(condition.accepts("1+9,5+5"))
        self.assertFalse(area.accepts(r"N_1=AD\cap CF"))

    def test_rejects_unrequested_pi_approximation(self) -> None:
        contract = AnswerContract.infer(
            "The number of revolutions required for a point on the rim to go one mile is:"
        )
        self.assertTrue(contract.accepts(r"880/\pi", r"5280/(6\pi)=880/\pi"))
        self.assertFalse(contract.accepts("280", "Use pi=22/7 to obtain an integer."))

    def test_selector_applies_contract_before_model_call(self) -> None:
        selector = AnswerSelectionAgent()
        route = Route("geometry", Skill("geometry", "geometry", [], "solve"), 3, False, [])
        candidates = [
            Candidate("bad", "FINAL_ANSWER: (1,2)", "(1,2)", "(1,2)", 1.0),
            Candidate("good", "FINAL_ANSWER: 2/5", "2/5", "2/5", 0.5),
        ]
        decision = selector.select(
            problem="The area of A is to the area of B as:",
            route=route,
            candidates=candidates,
            chat=lambda *args, **kwargs: self.fail("single eligible candidate should not call chat"),
            max_tokens=100,
        )
        self.assertEqual(decision.selected_index, 1)


if __name__ == "__main__":
    unittest.main()
