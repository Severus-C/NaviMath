import json
import threading
import unittest
from pathlib import Path

from agent.agent_utils import Candidate
from agent.answer_contract import AnswerContract
from agent.reasoning_agent import AgentConfig, ReasoningAgent


class QueueClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, **_: object) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class ConstantClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, **_: object) -> str:
        self.calls += 1
        return self.response


class ReasoningAgentTests(unittest.TestCase):
    def test_concurrent_solves_keep_contract_and_budget_isolated(self) -> None:
        blocked = threading.Event()
        release = threading.Event()

        class CoordinatedClient:
            def chat(self, **kwargs: object) -> str:
                messages = kwargs["messages"]
                prompt = messages[-1]["content"]
                if "COUNT_PROBLEM" in prompt:
                    blocked.set()
                    if not release.wait(5):
                        raise TimeoutError("test coordination timeout")
                    return "FINAL_ANSWER: 3"
                return "FINAL_ANSWER: 2:5"

        config = AgentConfig(
            rlot_easy_call_budget=2,
            rlot_medium_call_budget=2,
            rlot_hard_call_budget=2,
            rlot_proof_call_budget=2,
            rlot_max_steps=2,
        )
        agent = ReasoningAgent(CoordinatedClient(), config=config)
        results: dict[str, dict] = {}
        errors: list[BaseException] = []

        def solve_count() -> None:
            try:
                results["count"] = agent.solve(
                    "COUNT_PROBLEM: How many objects are there?",
                    {"answer_type": "count", "idx": "count"},
                )
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)

        thread = threading.Thread(target=solve_count)
        thread.start()
        self.assertTrue(blocked.wait(5))
        results["ratio"] = agent.solve(
            "RATIO_PROBLEM: The area of one square is to another as:",
            {"answer_type": "ratio", "idx": "ratio"},
        )
        release.set()
        thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["ratio"]["final_response"], "2:5")
        self.assertEqual(results["count"]["final_response"], "3")

    def test_trace_redacts_credentials_and_absolute_paths(self) -> None:
        class FailingClient:
            def chat(self, **_: object) -> str:
                raise RuntimeError(
                    r"Authorization: Bearer DEMO_SECRET_TOKEN at D:\Users\private\client.json"
                )

        config = AgentConfig(
            rlot_easy_call_budget=2,
            rlot_medium_call_budget=2,
            rlot_hard_call_budget=2,
            rlot_proof_call_budget=2,
            rlot_max_steps=2,
        )
        result = ReasoningAgent(FailingClient(), config=config).solve("Find x.", {})
        payload = json.dumps(result, ensure_ascii=False)

        self.assertNotIn("DEMO_SECRET_TOKEN", payload)
        self.assertNotIn(r"D:\Users", payload)
        self.assertNotIn(str(Path(__file__).resolve().parents[1]), payload)
        self.assertIn("RuntimeError", payload)

    def test_dependent_consensus_skips_bad_judge_override(self) -> None:
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
        self.assertLessEqual(client.calls, 5)
        steps = [entry["step"] for entry in result["trace"]]
        self.assertNotIn("consensus_lock", steps)
        self.assertIn("final_judge_rejected", steps)
        self.assertIn("tool_verify", [entry["step"] for entry in result["trace"]])
        self.assertIn("navigator_summary", [entry["step"] for entry in result["trace"]])

    def test_two_independent_solvers_form_consensus(self) -> None:
        agent = ReasoningAgent(ConstantClient("unused"))
        route = agent.registry.route("Find x.")
        candidates = [
            Candidate("direct_solver", "FINAL_ANSWER: 7", "7", "7", 0.5),
            Candidate("decomposition_solver", "FINAL_ANSWER: 7", "7", "7", 0.5),
        ]

        answer = agent._unique_consensus_answer(candidates, route, False)

        self.assertEqual(answer, "7")

    def test_tool_verify_blocks_wrong_symbolic_candidate(self) -> None:
        client = QueueClient(
            [
                "FINAL_ANSWER: 4*x^2",
                "FINAL_ANSWER: 3*x^2",
                "FINAL_ANSWER: 3*x^2",
                "VERDICT: ACCEPT\nCORRECTED_FINAL_ANSWER: 3*x^2",
                "VERDICT: ACCEPT\nCORRECTED_FINAL_ANSWER: 3*x^2",
            ]
        )
        result = ReasoningAgent(client).solve(
            "Differentiate x^3 with respect to x.",
            {"answer_type": "expression", "subject": "calculus", "difficulty": 1},
        )

        self.assertEqual(result["final_response"], "3*x^2")
        tool_records = [entry["content"] for entry in result["trace"] if entry["step"] == "tool_verify"]
        self.assertIn("REJECTED", [record["verdict"] for record in tool_records])
        self.assertIn("VERIFIED", [record["verdict"] for record in tool_records])

    def test_metadata_forces_proof_mode_and_uses_final_judge(self) -> None:
        proof = "Proof: assume a minimal counterexample, derive a smaller one, and conclude no counterexample exists. " * 2
        client = QueueClient(
            [proof] * 6
            + ["VERDICT: ACCEPT\nCORRECTED_FINAL_ANSWER: UNKNOWN"] * 2
            + ["SELECTED_INDEX: 1"]
        )
        result = ReasoningAgent(client).solve(
            "Determine all integer pairs satisfying the stated relation.",
            {"answer_type": "proof", "subject": "number_theory", "requires_proof": True},
        )

        route = next(entry["content"] for entry in result["trace"] if entry["step"] == "route")
        self.assertTrue(route["proof_mode"])
        self.assertIn("final_judge", [entry["step"] for entry in result["trace"]])
        self.assertNotIn("consensus_lock", [entry["step"] for entry in result["trace"]])

    def test_navigator_enforces_hard_call_budget(self) -> None:
        client = ConstantClient("FINAL_ANSWER: 7\nCONFIDENCE: 80")
        config = AgentConfig(
            rlot_easy_call_budget=2,
            rlot_medium_call_budget=2,
            rlot_hard_call_budget=2,
            rlot_proof_call_budget=2,
            rlot_max_steps=8,
        )
        result = ReasoningAgent(client, config=config).solve(
            "Find the requested integer.",
            {"contest": "AIME", "answer_type": "integer", "difficulty": 2},
        )

        self.assertEqual(result["final_response"], "7")
        self.assertLessEqual(client.calls, 2)
        summary = next(
            entry["content"] for entry in result["trace"] if entry["step"] == "navigator_summary"
        )
        self.assertLessEqual(summary["calls_used"], summary["call_budget"])

    def test_navigator_never_selects_placeholder_candidate(self) -> None:
        client = ConstantClient("FINAL_ANSWER: [content]")
        config = AgentConfig(
            rlot_easy_call_budget=2,
            rlot_medium_call_budget=2,
            rlot_hard_call_budget=2,
            rlot_proof_call_budget=2,
            rlot_max_steps=3,
        )
        result = ReasoningAgent(client, config=config).solve(
            "Find x.",
            {"answer_type": "expression", "difficulty": 2},
        )

        self.assertEqual(result["final_response"], "Unable to determine")
        self.assertNotEqual(result["final_response"], "[content]")
        steps = [entry["step"] for entry in result["trace"]]
        self.assertIn("candidate_rejected", steps)

    def test_final_judge_cannot_invent_an_answer(self) -> None:
        client = ConstantClient("FINAL_ANSWER: 999")
        agent = ReasoningAgent(client)
        route = agent.registry.route("Find x.")
        candidates = [
            Candidate("first", "FINAL_ANSWER: 1", "1", "1", 0.5),
            Candidate("second", "FINAL_ANSWER: 2", "2", "2", 0.5),
        ]
        trace: list[dict] = []

        answer = agent._final_judge("Find x.", route, 0, candidates, trace, False)

        self.assertEqual(answer, "")
        self.assertIn("final_judge_rejected", [entry["step"] for entry in trace])

    def test_refinement_preserves_original_candidate(self) -> None:
        client = ConstantClient("FINAL_ANSWER: 2")
        agent = ReasoningAgent(client)
        route = agent.registry.route("Find x.")
        original = Candidate("direct", "FINAL_ANSWER: 1", "1", "1", 0.5)

        refined = agent._refine_candidate(
            "Find x.", route, original, "VERDICT: REJECT", [], False
        )

        self.assertEqual(original.display_answer(), "1")
        self.assertIsNotNone(refined)
        self.assertEqual(refined.display_answer(), "2")
        self.assertEqual(refined.origin(), "direct")

    def test_long_response_without_protocol_uses_short_recovery(self) -> None:
        client = QueueClient(["FINAL_ANSWER: 3"])
        agent = ReasoningAgent(client)
        problem = "The number of solutions to this problem is:"
        route = agent.registry.route(problem)
        agent._active_contract = AnswerContract.infer(problem)
        response = ("The derivation checks all cases. " * 150) + "There are three cases."
        trace: list[dict] = []

        answer, protocol, recovered, contract_valid = agent._extract_or_recover_answer(
            response=response,
            problem=problem,
            route=route,
            integer_only=False,
            role="direct_solver",
            trace=trace,
        )

        self.assertEqual(answer, "3")
        self.assertTrue(protocol)
        self.assertTrue(recovered)
        self.assertTrue(contract_valid)
        self.assertIn("answer_recovery", [entry["step"] for entry in trace])

    def test_refinement_cannot_replace_ratio_with_coordinates(self) -> None:
        client = ConstantClient(r"FINAL_ANSWER: (\pm s_2/2,s_2)")
        agent = ReasoningAgent(client)
        problem = "The area of one square is to the area of another as:"
        route = agent.registry.route(problem)
        agent._active_contract = AnswerContract.infer(problem)
        original = Candidate("direct", "FINAL_ANSWER: 2/5", "2/5", "2/5", 0.8)
        trace: list[dict] = []

        refined = agent._refine_candidate(
            problem,
            route,
            original,
            "VERDICT: REJECT",
            trace,
            False,
        )

        self.assertIsNone(refined)
        self.assertEqual(original.display_answer(), "2/5")
        rejection = next(entry["content"] for entry in trace if entry["step"] == "refine_rejected")
        self.assertEqual(rejection["reason"], "answer_contract_mismatch")

    def test_debate_echo_does_not_create_false_consensus(self) -> None:
        agent = ReasoningAgent(ConstantClient("unused"))
        route = agent.registry.route("Find x.")
        candidates = [
            Candidate("direct_refined", "FINAL_ANSWER: 9/7", "9/7", "9/7", 0.5, origin_role="direct"),
            Candidate("debate_synthesizer_1", "FINAL_ANSWER: 9/7", "9/7", "9/7", 0.5),
        ]

        answer = agent._unique_consensus_answer(candidates, route, False)

        self.assertEqual(answer, "")

    def test_integer_mod_1000_uses_strict_integer_extraction(self) -> None:
        client = ConstantClient("FINAL_ANSWER: 125, it implies the value is 125 percent")
        config = AgentConfig(
            rlot_easy_call_budget=2,
            rlot_medium_call_budget=2,
            rlot_hard_call_budget=2,
            rlot_proof_call_budget=2,
            rlot_max_steps=2,
        )
        result = ReasoningAgent(client, config=config).solve(
            "Find the requested residue.",
            {"answer_type": "integer_mod_1000", "difficulty": 1},
        )

        self.assertEqual(result["final_response"], "125")


if __name__ == "__main__":
    unittest.main()
