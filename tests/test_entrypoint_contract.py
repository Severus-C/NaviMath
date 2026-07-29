import inspect
import json
import unittest

from user_agent import ReasoningAgent


class EntrypointContractTests(unittest.TestCase):
    def test_exposes_exact_runner_signatures(self) -> None:
        init_parameters = list(inspect.signature(ReasoningAgent.__init__).parameters.values())
        self.assertEqual([parameter.name for parameter in init_parameters], ["self", "client", "args", "kwargs"])
        self.assertEqual(init_parameters[2].kind, inspect.Parameter.VAR_POSITIONAL)
        self.assertEqual(init_parameters[3].kind, inspect.Parameter.VAR_KEYWORD)

        solve_signature = inspect.signature(ReasoningAgent.solve)
        self.assertEqual(list(solve_signature.parameters), ["self", "problem", "metadata"])
        self.assertIs(solve_signature.parameters["problem"].annotation, str)
        self.assertIs(solve_signature.parameters["metadata"].annotation, dict)
        self.assertIs(solve_signature.return_annotation, dict)

    def test_official_client_call_and_result_are_compatible(self) -> None:
        class OfficialClient:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def chat(self, **kwargs: object) -> str:
                self.calls.append(kwargs)
                return "FINAL_ANSWER: 72\nCONFIDENCE: 1.0"

        client = OfficialClient()
        result = ReasoningAgent(client=client).solve(
            "Compute the requested value.",
            {"idx": 0, "answer": "this local-only field must be ignored"},
        )

        self.assertEqual(result["final_response"], "72")
        self.assertIsInstance(result["trace"], list)
        json.dumps(result, ensure_ascii=False)
        self.assertGreater(len(client.calls), 0)
        for call in client.calls:
            self.assertEqual(set(call), {"messages", "temperature", "max_tokens"})
            self.assertIsInstance(call["messages"], list)
            self.assertIsInstance(call["temperature"], float)
            self.assertIsInstance(call["max_tokens"], int)


if __name__ == "__main__":
    unittest.main()
