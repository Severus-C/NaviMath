import unittest

from agent.tool_verify import REJECTED, UNKNOWN, VERIFIED, ToolVerify


class ToolVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verify = ToolVerify(random_samples=4)
        if not self.verify.available:
            self.skipTest("SymPy is not installed")

    def test_symbolic_and_numeric_equivalence(self) -> None:
        result = self.verify.check_equivalence("(x+1)^2", "x^2+2*x+1")
        self.assertEqual(result.verdict, VERIFIED)
        mismatch = self.verify.check_equivalence("x^2", "x^2+1")
        self.assertEqual(mismatch.verdict, REJECTED)
        self.assertEqual(self.verify.check_equivalence("|x|", "abs(x)").verdict, VERIFIED)

    def test_random_substitution_checks_identity(self) -> None:
        result = self.verify.random_numeric_equivalence("sin(x)^2+cos(x)^2", "1")
        self.assertEqual(result.verdict, VERIFIED)
        self.assertGreaterEqual(result.samples, 4)

    def test_equation_root_validation_checks_completeness(self) -> None:
        correct = self.verify.verify_equation_roots("x^2-5*x+6=0", "{2,3}")
        self.assertEqual(correct.verdict, VERIFIED)
        incomplete = self.verify.verify_equation_roots("x^2-5*x+6=0", "{2}")
        self.assertEqual(incomplete.verdict, REJECTED)
        invalid = self.verify.verify_equation_roots("x^2-5*x+6=0", "{2,4}")
        self.assertEqual(invalid.verdict, REJECTED)

    def test_calculus_checks(self) -> None:
        derivative = self.verify.verify_derivative("x^3+sin(x)", "x", "3*x^2+cos(x)")
        self.assertEqual(derivative.verdict, VERIFIED)
        antiderivative = self.verify.verify_integral("x", "x", "x^2/2+7")
        self.assertEqual(antiderivative.verdict, VERIFIED)
        definite = self.verify.verify_integral("x", "x", "2", lower="0", upper="2")
        self.assertEqual(definite.verdict, VERIFIED)
        limit = self.verify.verify_limit("sin(x)/x", "x", "0", "1")
        self.assertEqual(limit.verdict, VERIFIED)

    def test_residue_check(self) -> None:
        result = self.verify.verify_residue("1/(z*(z-1))", "z", "0", "-1")
        self.assertEqual(result.verdict, VERIFIED)

    def test_auto_detection_is_fail_open(self) -> None:
        derivative = self.verify.verify_candidate(
            "Differentiate x^4-2*x with respect to x.",
            "4*x^3-2",
        )
        self.assertEqual(derivative.verdict, VERIFIED)
        roots = self.verify.verify_candidate("Solve the equation x^2-5*x+6=0 for x.", "{2,3}")
        self.assertEqual(roots.verdict, VERIFIED)
        unknown = self.verify.verify_candidate("How many triangles are possible?", "12")
        self.assertEqual(unknown.verdict, UNKNOWN)

    def test_auto_detection_understands_common_latex_calculus(self) -> None:
        cases = [
            (r"Compute $\frac{d}{dx} x^3$.", "3*x^2", "derivative"),
            (r"Compute $\int_{0}^{2} x\,dx$.", "2", "integral"),
            (r"Find $\lim_{x\to 0} \frac{\sin(x)}{x}$.", "1", "limit"),
            (r"Find $\operatorname{Res}_{z=0} \frac{1}{z(z-1)}$.", "-1", "residue"),
        ]
        for problem, answer, check in cases:
            with self.subTest(check=check):
                result = self.verify.verify_candidate(problem, answer)
                self.assertEqual(result.verdict, VERIFIED)
                self.assertEqual(result.check, check)

    def test_equivalent_groups(self) -> None:
        groups = self.verify.equivalent_groups(["x*(x+1)", "x^2+x", "x^2-x"])
        self.assertEqual(groups, [[0, 1], [2]])

    def test_structured_values_fail_open_in_scalar_equivalence(self) -> None:
        result = self.verify.check_equivalence("(2,0)", "(3,3)")
        self.assertEqual(result.verdict, UNKNOWN)


if __name__ == "__main__":
    unittest.main()
