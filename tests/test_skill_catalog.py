import json
import unittest
from pathlib import Path

from agent.agent_utils import SkillRegistry
from agent.reasoning_agent import ReasoningAgent
from agent.skill_catalog import DistilledSkillCatalog


class SkillCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SkillRegistry()

    def test_catalog_is_distilled_and_contains_no_reference_solutions(self) -> None:
        path = Path(__file__).resolve().parents[1] / "agent" / "skill_catalog.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(payload["templates"]), 25)
        self.assertEqual(len(payload["sources"]), 2)
        for template in payload["templates"]:
            self.assertNotIn("problem", template)
            self.assertNotIn("solution", template)
            self.assertNotIn("answer", template)

    def test_routes_to_fine_grained_ratio_template(self) -> None:
        route = self.registry.route("A total is divided in the ratio 2:3:5. Find the smallest part.")
        self.assertEqual(route.subject, "algebra")
        self.assertIn("prealgebra_ratio_percent", [template.id for template in route.templates])

    def test_routes_circle_template(self) -> None:
        route = self.registry.route("A tangent and a secant meet outside a circle. Find the chord length.")
        self.assertEqual(route.subject, "geometry")
        self.assertIn("geometry_circle_power", [template.id for template in route.templates])

    def test_short_trigger_tokens_respect_word_boundaries(self) -> None:
        catalog = DistilledSkillCatalog.load()
        matches = catalog.match("Using a single substitution, solve the equation.", "algebra")
        self.assertNotIn("precalculus_trigonometry", [template.id for template in matches])

    def test_university_domains_override_harp_router_when_explicit(self) -> None:
        cases = {
            "Let X be a compact topological space. Prove every closed subset is compact.": "topology",
            "Solve this boundary value differential equation for y.": "differential_equations",
            "Compute the Gaussian curvature of this Riemannian manifold.": "differential_geometry",
            "Find the maximum likelihood estimator for the normal distribution.": "statistics",
        }
        for problem, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(self.registry.route(problem).subject, expected)

    def test_solver_and_verifier_prompts_receive_distilled_guidance(self) -> None:
        agent = ReasoningAgent(client=object())
        route = self.registry.route("Find the remainder when 2^100 is divided by 7.")
        role = agent._solver_roles(route)[0]
        solver_prompt = agent._solver_prompt("Find the remainder when 2^100 is divided by 7.", route, role, True)
        self.assertIn("number_theory_modular", solver_prompt)
        self.assertIn("HARP support", solver_prompt)


if __name__ == "__main__":
    unittest.main()
