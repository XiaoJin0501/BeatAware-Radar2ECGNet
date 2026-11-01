# PAPER_FACTS read-only export — 2026-11-01

Author request: A–I manuscript fact inventory for phase_pd seed-base42 and existing adapted baselines.

Added `docs/paper/PAPER_FACTS.md` with code-line/result-path citations and verified/missing status. Included metadata demographics/scenario inventory, actual train/val/test H5 subject/window counts (16 folds × 3 splits; disjointness asserted), executable architecture/loss/selection details, actual phase_pd selected epochs and logged timings, all108 comparison cells, clinical coverage, all architecture/factorial/multitask CSV rows, roundtrip/BA/HR/anchor-robustness/complexity results and provenance limitations.

Read-only derivations: unique-subject age/sampleSD and sex counts; per-scene metadata duration sums; H5 inventory and test record coverage; model-module trainable parameter counts (CPU instantiation only); existing per-epoch seconds sums; matched-reference mean/sampleSD and finite-HR ±5 accuracy from frozen paired_values.csv. No inference, training, new significance test, checkpoint selection or result modification.

Important verified distinctions:
- seed42 means seed_base42, actual initialization42+fold.
- MM training stride2–20s, not universally2s; val/test2s.
- MM phase_pd drop_lastFalse vs adapted baseline True.
- 91 MM records and80/81 CR records occur in the actual test splits; CRRVS_0024 absent, rejection cause/counts missing.
- Existing main localizationCDF, scenario plot and waveform examples still contain legacy dual-head REM, not phase_pd. Their phase_pd replacements are missing; old figure rows retained only as labeled provenance audit.
- Conformer baseline is registered to Liu2024 Cross-Modal Autoencoder (local bib `croos-modalAutoencoder`), while bib `conformer` is the speech paper.
- Clinical baseline inference is now complete. Native per-beat radar contexts differ from REM parent-window feature reuse; do not claim identical generation input support/cost.
- Clinical main-table macro MAE differs from pooled-BA MAE; low coverage and negative ICC retained.
- Missing original metadata/license/search/statistics are not fabricated; complexity-run environment does not prove historical training environment.

Checks:48 split H5s read and pairwise subject disjointness verified;108 main numeric cells and8 factorial rows exported; cited literal file existence audited. No manuscript prose, figures, weights or prior results changed.
