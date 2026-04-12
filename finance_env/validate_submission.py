"""
validate_submission.py — pre-submission sanity check.

Runs inference.py as a subprocess and verifies the evaluator-required
[START] / [STEP] / [END] markers are present in stdout.
Writes a report to artifacts/submission/pre_submission_report.json.
Exits with code 1 on failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_inference_check() -> Dict[str, Any]:
    root = _project_root()
    env = os.environ.copy()
    env.setdefault("SPACE_URL", "http://localhost:7860")
    proc = subprocess.run(
        [sys.executable, str(root / "finance_env" / "inference.py")],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    has_start = "[START]" in stdout
    has_step  = "[STEP]"  in stdout
    has_end   = "[END]"   in stdout
    passed = proc.returncode == 0 and has_start and has_step and has_end
    return {
        "name": "inference_structured_output",
        "passed": passed,
        "returncode": proc.returncode,
        "has_start": has_start,
        "has_step": has_step,
        "has_end": has_end,
        "stdout_preview": stdout[:4000],
        "stderr_preview": stderr[:2000],
    }


def _write_report(report: Dict[str, Any]) -> None:
    out_dir = _project_root() / "artifacts" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre_submission_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    check = _run_inference_check()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": bool(check["passed"]),
        "checks": [check],
    }
    _write_report(report)
    status = "PASS" if report["overall_passed"] else "FAIL"
    print(f"validate_submission: {status} (start={check['has_start']} step={check['has_step']} end={check['has_end']})", flush=True)
    if not report["overall_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
