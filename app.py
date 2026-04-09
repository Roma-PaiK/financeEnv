"""
HuggingFace Spaces FastAPI entry point using OpenEnv library.

Uses the official OpenEnv library to create the HTTP server.
Endpoints are auto-generated:
    POST /reset     → Observation JSON
    POST /step      → StepResponse (observation, reward, done, info)
    GET  /state     → State JSON
    GET  /health    → {"status": "ok"}
    GET  /docs      → Auto-generated API documentation
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from finance_env import FinanceEnv

# Create the environment instance
env = FinanceEnv()

# Use OpenEnv library to auto-generate all endpoints
app = create_app(env)
