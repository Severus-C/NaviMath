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


if __name__ == "__main__":
    unittest.main()
