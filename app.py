"""
HuggingFace Spaces FastAPI entry point using OpenEnv library.

Uses the official OpenEnv library to create the HTTP server.
Endpoints are auto-generated:
    POST /reset     → Observation JSON
    POST /step      → StepResponse (observation, reward, done, info)
    GET  /state     → State JSON
    GET  /health    → {"status": "ok"}
    GET  /docs      → Auto-generated API documentation

Architecture notes:
  - openenv creates a NEW env instance per HTTP request (stateless model).
  - FinanceEnv is stateful, so FinanceEnvAdapter delegates to a module-level
    singleton so state (task, step count, scores) persists across reset/step calls.
  - Pass the class (not an instance) to create_fastapi_app — openenv calls it as
    a factory. Each call returns the adapter wrapping the same singleton.
  - Adapter inherits from Environment so openenv's step_async/__func__ check passes.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server import Environment as OpenEnvEnvironment, create_fastapi_app

from finance_env import FinanceEnv
from finance_env.models import Action, Observation, State

# Singleton — persists state across HTTP requests
_singleton = FinanceEnv()


class FinanceEnvAdapter(OpenEnvEnvironment):
    """Thin adapter: conforms to openenv's Environment interface, delegates to singleton."""

    def __init__(self) -> None:
        super().__init__()
        self._env = _singleton

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        return self._env.reset(seed=seed, episode_id=episode_id, task_id=task_id, **kwargs)

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        obs, reward, done, _ = self._env.step(action)
        obs.done = done
        obs.reward = reward.cumulative_score
        return obs

    @property
    def state(self) -> State:
        return self._env.state()


app = create_fastapi_app(
    FinanceEnvAdapter,
    action_cls=Action,
    observation_cls=Observation,
)
