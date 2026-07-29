"""Competition entrypoint.

The official runner imports ``ReasoningAgent`` from this root-level module.
The implementation lives in ``agent`` so the project can keep a normal
package layout without breaking the competition interface.
"""

from agent.reasoning_agent import AgentConfig, ReasoningAgent as _ReasoningAgent


class ReasoningAgent(_ReasoningAgent):
    """Root-level adapter matching the competition runner contract exactly."""

    def __init__(self, client, *args, **kwargs):
        super().__init__(client, *args, **kwargs)

    def solve(self, problem: str, metadata: dict) -> dict:
        return super().solve(problem, metadata)

__all__ = ["AgentConfig", "ReasoningAgent"]
