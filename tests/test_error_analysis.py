import unittest

from agent.error_analysis import build_error_report, diagnose_record


class ErrorAnalysisTests(unittest.TestCase):
    def _row(self, **updates):
        row = {
            "id": "sample",
            "contest": "AIME",
            "answer_type": "integer",
            "subject": ["number_theory"],
            "expected": "38",
            "prediction": "18",
            "expected_key": "38",
            "prediction_key": "18",
            "is_correct": False,
            "status": "success",
            "error": None,
            "trace": [],
        }
        row.update(updates)
        return row

    def test_detects_wrong_consensus_and_false_accept(self):
        row = self._row(
            trace=[
                {"step": "route", "content": {"subject": "number_theory"}},
                {"step": "candidate", "content": {"normalized_answer": "18"}},
                {"step": "candidate", "content": {"normalized_answer": "18"}},
                {"step": "consensus_lock", "content": {"answer": "18"}},
                {
                    "step": "attack_verifier",
                    "content": {"role": "checker", "verdict": "ACCEPT", "candidate_answer": "18"},
                },
            ]
        )
        diagnosis = diagnose_record(row)
        self.assertIn("wrong_consensus", diagnosis["root_causes"])
        self.assertIn("verifier_misjudgment", diagnosis["root_causes"])

    def test_detects_final_judge_wrong_selection(self):
        row = self._row(
            trace=[
                {"step": "candidate", "content": {"normalized_answer": "38"}},
                {"step": "candidate", "content": {"normalized_answer": "18"}},
                {"step": "final_judge", "content": {"normalized": "18"}},
            ]
        )
        diagnosis = diagnose_record(row)
        self.assertEqual(diagnosis["primary_cause"], "final_judge_wrong_selection")

    def test_report_aggregates_examples(self):
        report = build_error_report([self._row(prediction="", prediction_key="")])
        self.assertEqual(report["by_primary_cause"], {"format_error": 1})
        self.assertIn("format_error", report["examples"])

    def test_prealgebra_to_algebra_is_not_a_routing_error(self):
        row = self._row(
            subject=["prealgebra"],
            trace=[{"step": "route", "content": {"subject": "algebra"}}],
        )
        diagnosis = diagnose_record(row)
        self.assertNotIn("routing_error", diagnosis["root_causes"])

    def test_equivalent_symbolic_verifier_answer_is_not_misjudged(self):
        row = self._row(
            contest="AHSME",
            answer_type="expression",
            subject=["algebra"],
            expected="(2abc)/(ac+bc-ab)",
            expected_key="(2abc)/(ac+bc-ab)",
            prediction="0",
            prediction_key="0",
            trace=[
                {
                    "step": "attack_verifier",
                    "content": {
                        "role": "checker",
                        "verdict": "ACCEPT",
                        "candidate_answer": "(2abc)/(bc+ca-ab)",
                    },
                }
            ],
        )
        diagnosis = diagnose_record(row)
        self.assertNotIn("verifier_misjudgment", diagnosis["root_causes"])


if __name__ == "__main__":
    unittest.main()
