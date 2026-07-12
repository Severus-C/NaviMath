import unittest

from agent.reasoning_agent import ReasoningAgent


class QueueClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, **_: object) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class ReasoningAgentTests(unittest.TestCase):
    def test_unique_consensus_skips_bad_judge_override(self) -> None:
        client = QueueClient(
            [
                "FINAL_ANSWER: 649",
                "FINAL_ANSWER: 649",
                "FINAL_ANSWER: 1",
                "VERDICT: ACCEPT\nCORRECTED_FINAL_ANSWER: 649",
                "VERDICT: ACCEPT\nCORRECTED_FINAL_ANSWER: 649",
                "FINAL_ANSWER: 1",
            ]
        )
        result = ReasoningAgent(client).solve(
            "Find the largest possible integer value.",
            {"contest": "AIME", "answer_type": "integer", "difficulty": 3},
        )

        self.assertEqual(result["final_response"], "649")
        self.assertEqual(client.calls, 5)
        self.assertIn("consensus_lock", [entry["step"] for entry in result["trace"]])


if __name__ == "__main__":
    unittest.main()
