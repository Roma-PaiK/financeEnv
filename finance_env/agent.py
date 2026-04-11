"""Entry point for `uv run agent` / `python -m finance_env.agent`."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference import main

if __name__ == "__main__":
    main()
