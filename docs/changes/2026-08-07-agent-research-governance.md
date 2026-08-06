# Change Note: Agent Research Governance

Date: 2026-08-07

## Why

Claude Code and Codex needed one shared knowledge layer, enforceable research
rules, and a traceable path from code changes to experiment evidence and paper
claims.

## What Changed

- Added thin `AGENTS.md` and `CLAUDE.md` entry points.
- Added shared research context and project state documents.
- Added research, data, metric, experiment, model, and claim contracts.
- Added ADRs for Shape-200 and long-term ECG assembly.
- Added agent skill single-source directory and sync script.
- Added schemas, PR template, issue templates, and CI validation.

## Scientific Impact

This change does not change model behavior or experimental results. It changes
the governance layer so future results are easier to audit.

## Tests

Pending in this branch:

- syntax checks for scripts;
- skill sync check;
- governance CI check.

## Next Agent Task

Implement the radarODE-style long-term ECG assembly evaluator under a new
registered experiment/spec branch.
