# Reversible project cleanup — 2026-11-04

User requested project-wide removal of garbage scripts/files. Historical files were relocated outside the working tree rather than destroyed; only regenerable caches erased. See `docs/CLEANUP_20261104.md` for exact counts, preserved dependencies, tests, archive and recovery command.

Scientific behavior unchanged. Two active helper source files relocated byte-for-byte; import paths updated.105 retained tests pass, exact WSST numeric equality checked,192 selected checkpoints and4571 paper/source/result files SHA-preserved. No training or result re-evaluation.

Existing user-edited metric/roadmap/literature files and registry content retained; pre-cleanup status and binary diff saved externally. Registry updates mark relocated top-level historical experiments Archived; moving resume states does not mark retained experiment families Archived. Generated-data ignore rules keep local outputs on disk, not remove them.
