"""
HuggingFace Spaces FastAPI entry point.

Endpoints (per spec Part 7 / Part 8):
    POST /reset  → body: {"task_id": str}   → Observation JSON
    POST /step   → body: Action JSON         → {observation, reward, done, info}
    GET  /state                              → State JSON
    GET  /health                             → {"status": "ok"}
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from finance_env import FinanceEnv
from finance_env.models import Action, Observation, Reward, State

app = FastAPI(title="FinanceEnv", version="1.0.0")
env = FinanceEnv()


class ResetRequest(BaseModel):
    task_id: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = Body(...)) -> Observation:
    try:
        return env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state() -> State:
    return env.state()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
