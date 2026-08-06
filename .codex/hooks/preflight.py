#!/usr/bin/env python3
"""Lightweight local preflight helper for agent-driven experiment work."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "AGENTS.md",
    "CLAUDE.md",
    "docs/agent/SHARED_CONTEXT.md",
    "docs/PROJECT_STATE.md",
    "docs/RESEARCH_CONTRACT.md",
    "docs/DATA_PROTOCOL.md",
    "docs/METRIC_CONTRACT.md",
    "docs/EXPERIMENT_POLICY.md",
    "docs/AGENT_DEVELOPMENT_GUIDE.md",
    "docs/PAPER_CLAIM_LEDGER.md",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing governance files:\n" + "\n".join(missing))
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    if status.strip():
        print("git_status_not_clean")
        print(status)
    else:
        print("preflight_ok")


if __name__ == "__main__":
    main()
