import unittest

from agent.error_analysis import diagnose_record
from scripts.eval_agent_on_dataset import (
    build_benchmark_quality,
    build_pipeline_metrics,
    detect_benchmark_flags,
)


class EvaluationPipelineTests(unittest.TestCase):
    def test_flags_non_unique_stem_after_choices_are_removed(self) -> None:
        problem = "The points (6,12) and (0,-6) are connected by a line. Another point on this line is:"
        self.assertIn("non_unique_open_response", detect_benchmark_flags(problem))

    def test_reports_strict_and_scorable_accuracy_separately(self) -> None:
        records = [
            {"id": "ambiguous", "is_correct": False, "benchmark_flags": ["non_unique_open_response"]},
            {"id": "clean", "is_correct": True, "benchmark_flags": []},
        ]
        quality = build_benchmark_quality(records)
        self.assertEqual(quality["adjudication_needed"]["count"], 1)
        self.assertEqual(quality["scorable_accuracy"], 1.0)

    def test_pipeline_metrics_expose_selection_gap(self) -> None:
        records = [
            {
                "id": "lost",
                "answer_type": "expression",
                "contest": "AHSME",
                "expected": "2:5",
                "prediction": "(1,2)",
                "is_correct": False,
                "trace": [
                    {
                        "step": "candidate",
                        "content": {
                            "extracted_answer": "2/5",
                            "protocol_compliant": True,
                            "recovered": False,
                        },
                    }
                ],
            }
        ]
        metrics = build_pipeline_metrics(records)
        self.assertEqual(metrics["candidate_oracle_correct"], 1)
        self.assertEqual(metrics["correct_candidate_lost"]["ids"], ["lost"])

    def test_set_braces_and_combinatorics_route_are_not_failures(self) -> None:
        row = {
            "id": "count",
            "contest": "AHSME",
            "answer_type": "expression",
            "subject": ["counting_and_probability"],
            "expected": "3",
            "prediction": r"\{24,32\}",
            "is_correct": False,
            "status": "success",
            "error": None,
            "trace": [{"step": "route", "content": {"subject": "combinatorics"}}],
        }
        diagnosis = diagnose_record(row)
        self.assertNotIn("latex_normalization_failure", diagnosis["root_causes"])
        self.assertNotIn("routing_error", diagnosis["root_causes"])

    def test_ambiguous_benchmark_is_not_labeled_solver_error(self) -> None:
        row = {
            "id": "ambiguous",
            "contest": "AHSME",
            "answer_type": "expression",
            "subject": ["geometry"],
            "expected": "(3,3)",
            "prediction": "(2,0)",
            "is_correct": False,
            "status": "success",
            "error": None,
            "benchmark_flags": ["non_unique_open_response"],
            "trace": [],
        }
        diagnosis = diagnose_record(row)
        self.assertEqual(diagnosis["primary_cause"], "benchmark_ambiguity")
        self.assertNotIn("solver_math_error", diagnosis["root_causes"])


if __name__ == "__main__":
    unittest.main()
