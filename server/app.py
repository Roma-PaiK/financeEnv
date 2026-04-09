"""
Re-export the FastAPI app for HuggingFace Spaces deployment.

HF Spaces expects the app at server/app.py.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from root app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

__all__ = ["app"]
