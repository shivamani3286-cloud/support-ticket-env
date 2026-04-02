"""
server.py — FastAPI server exposing the OpenEnv interface over HTTP
====================================================================
Endpoints:
  GET  /           → health check (returns 200)
  POST /reset      → reset environment, returns Observation
  POST /step       → execute action, returns (obs, reward, done, info)
  GET  /state      → returns current environment state
  GET  /tasks      → list all tasks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import make_env, Action, SupportTicketEnv

app = FastAPI(
    title="Support Ticket Triage — OpenEnv",
    description="Real-world customer support triage environment for AI agents.",
    version="1.0.0",
)

# Global env store (keyed by task_id for simplicity)
_envs: dict[str, SupportTicketEnv] = {}


def get_env(task_id: str) -> SupportTicketEnv:
    if task_id not in _envs:
        _envs[task_id] = make_env(task_id)
    return _envs[task_id]


# ─── Request / Response Schemas ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "classify_priority"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: str = "classify_priority"
    priority: Optional[str] = None
    department: Optional[str] = None
    response_text: Optional[str] = None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "env": "support-ticket-triage", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        env = make_env(req.task_id, seed=req.seed)
        _envs[req.task_id] = env
        obs = env.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{req.task_id}'. Call /reset first."
        )
    action = Action(
        priority=req.priority,
        department=req.department,
        response_text=req.response_text,
    )
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(task_id: str = "classify_priority"):
    env = _envs.get(task_id)
    if env is None:
        return {"task_id": task_id, "initialized": False}
    return env.state()


@app.get("/tasks")
def tasks():
    return {"tasks": SupportTicketEnv.TASKS}


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
