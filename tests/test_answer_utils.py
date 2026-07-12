import unittest

from agent.answer_normalizer import AnswerContext, AnswerNormalizer
from agent.agent_utils import canonical_key, extract_final_answer, is_plausible_final_answer, parse_verdict
from scripts.eval_agent_on_dataset import is_match


class AnswerExtractionTests(unittest.TestCase):
    def test_rejects_instruction_echo(self) -> None:
        self.assertEqual(extract_final_answer("only the actual final answer"), "")
        self.assertFalse(is_plausible_final_answer("onlytheactualfinalanswer"))

    def test_cleans_simple_assignment(self) -> None:
        self.assertEqual(extract_final_answer("FINAL_ANSWER: x=15"), "15")
        self.assertEqual(extract_final_answer("FINAL_ANSWER: OB^2=26"), "26")

    def test_cleans_trailing_uncertainty_punctuation(self) -> None:
        self.assertEqual(extract_final_answer("FINAL_ANSWER: 20?"), "20")

    def test_rejects_unsolved_polynomial(self) -> None:
        answer = "FINAL_ANSWER: 10s^2-1000s+19240=0"
        self.assertEqual(extract_final_answer(answer), "")

    def test_integer_mode_rejects_symbolic_fragments(self) -> None:
        self.assertEqual(extract_final_answer("$x^2$", integer_only=True), "")
        self.assertEqual(extract_final_answer("FINAL_ANSWER: [Answer]", integer_only=True), "")

    def test_integer_mode_extracts_conclusion(self) -> None:
        response = "After factoring the cubic, the largest real value is 4."
        self.assertEqual(extract_final_answer(response, integer_only=True), "4")
        self.assertEqual(extract_final_answer("**Final Answer:** **4**", integer_only=True), "4")

    def test_integer_mode_does_not_take_decimal_prefix(self) -> None:
        self.assertEqual(extract_final_answer("The probability is 0.375.", integer_only=True), "")
        self.assertEqual(extract_final_answer("The total number is 1750.", integer_only=True), "")

    def test_integer_mode_extracts_requested_target_relations(self) -> None:
        samples = {
            "Total = 216 + 216 = 432.": "432",
            "The sum of the numerator and denominator is 11+46=57.": "57",
            "Thus AB = 65.": "65",
            r"Finally, \frac{n}{15}=592.": "592",
            "The requested volume is V=288.": "288",
            "Therefore the product mn=175.": "175",
        }
        for response, expected in samples.items():
            with self.subTest(response=response):
                self.assertEqual(extract_final_answer(response, integer_only=True), expected)

    def test_problem_aware_aime_targets(self) -> None:
        cases = [
            (
                "The line has slope m=-24.",
                "What is the absolute value of the slope of this line?",
                "24",
            ),
            ("Therefore Mary's score was 119.", "What was Mary's score?", "119"),
            ("The fraction is 7/99, so m+n=106.", "Find m+n.", "106"),
            (
                r"Although \boxed{18} is representable, 38 is the largest even integer with the property.",
                "What is the largest even integer that cannot be represented?",
                "38",
            ),
        ]
        for response, problem, expected in cases:
            with self.subTest(problem=problem):
                self.assertEqual(
                    extract_final_answer(response, integer_only=True, problem=problem),
                    expected,
                )

    def test_verifier_meta_accept_does_not_hide_later_rejection(self) -> None:
        report = "Output format: VERDICT: ACCEPT. Independent check: the candidate is INCORRECT because 18=9+9."
        self.assertEqual(parse_verdict(report), "REJECT")

    def test_recovers_boxed_answer_after_bad_final_line(self) -> None:
        response = "SOLUTION: Thus the requested value is \\boxed{60}.\nFINAL_ANSWER: only the actual final answer"
        self.assertEqual(extract_final_answer(response), "60")

    def test_evaluator_accepts_equivalent_output_forms(self) -> None:
        self.assertTrue(is_match("x=15", "15"))
        self.assertTrue(is_match("20?", "20"))
        self.assertEqual(canonical_key(extract_final_answer("OB^2=26")), "26")


class AnswerNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = AnswerNormalizer()

    def test_normalizes_nested_fraction_and_decimals(self) -> None:
        self.assertEqual(self.normalizer.canonicalize(r"\frac{\frac{1}{2}}{3}"), "1/6")
        self.assertTrue(self.normalizer.equivalent("0.5", r"\frac{2}{4}"))

    def test_sorts_set_members(self) -> None:
        context = AnswerContext(answer_type="set")
        self.assertTrue(self.normalizer.equivalent(r"\{2,1,2\}", "{1,2}", context))

    def test_normalizes_intervals(self) -> None:
        context = AnswerContext(answer_type="interval")
        self.assertEqual(
            self.normalizer.canonicalize(r"\left[0,\infty\right)", context),
            "[0,infinity)",
        )

    def test_normalizes_matrix_and_vector(self) -> None:
        matrix = r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}"
        self.assertEqual(self.normalizer.canonicalize(matrix), "matrix[1,2;3,4]")
        vector_context = AnswerContext(answer_type="vector")
        self.assertEqual(self.normalizer.canonicalize("(1, 2, 3)", vector_context), "(1,2,3)")

    def test_sorts_schema_declared_multiple_answers(self) -> None:
        context = AnswerContext(answer_type="multiple")
        self.assertTrue(self.normalizer.equivalent("3; 1; 2", "1,2,3", context))

    def test_aime_zero_policy(self) -> None:
        self.assertTrue(is_match("038", "38", answer_type="integer", contest="AIME"))
        self.assertEqual(self.normalizer.format_aime("38"), "038")
        self.assertEqual(self.normalizer.format_aime("000"), "000")
        with self.assertRaises(ValueError):
            self.normalizer.format_aime("1000")


if __name__ == "__main__":
    unittest.main()
