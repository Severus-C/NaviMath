"""Competition entrypoint.

The official runner imports ``ReasoningAgent`` from this root-level module.
The implementation lives in ``agent`` so the project can keep a normal
package layout without breaking the competition interface.
"""

from agent.reasoning_agent import AgentConfig, ReasoningAgent

__all__ = ["AgentConfig", "ReasoningAgent"]
