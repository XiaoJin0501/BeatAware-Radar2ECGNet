#!/usr/bin/env python3
"""Synchronize shared agent skills into Claude and Codex entry directories."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent_skills"
TARGETS = [
    ROOT / ".agents" / "skills",
    ROOT / ".claude" / "skills",
]


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_dir():
            shutil.copytree(item, target / item.name)


def _compare_dirs(a: Path, b: Path) -> list[str]:
    mismatch: list[str] = []
    cmp = filecmp.dircmp(a, b)
    mismatch.extend(str(a / name) for name in cmp.left_only)
    mismatch.extend(str(b / name) for name in cmp.right_only)
    mismatch.extend(str(a / name) for name in cmp.diff_files)
    for name in cmp.common_dirs:
        mismatch.extend(_compare_dirs(a / name, b / name))
    return mismatch


def sync() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source skill directory: {SOURCE}")
    for target in TARGETS:
        _copy_tree(SOURCE, target)


def check() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source skill directory: {SOURCE}")
    for target in TARGETS:
        if not target.exists():
            raise FileNotFoundError(f"Missing synced skill directory: {target}")
        mismatch = _compare_dirs(SOURCE, target)
        if mismatch:
            joined = "\n".join(mismatch[:50])
            raise SystemExit(f"Skill sync mismatch for {target}:\n{joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify synced skill directories.")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        sync()
        check()
        print("skills_synced_ok")


if __name__ == "__main__":
    main()
