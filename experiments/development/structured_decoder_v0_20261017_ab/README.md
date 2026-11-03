# Retained shared inference dependency

This historical A/B directory remains because current figures, clinical baseline inference
and HRV processing import it. It is not an invitation to restart that development route.

Active helpers: `bootstrap.py`, `rawio.py`, `inference.py`, `layout.py`, `structured.py`,
`unified_metrics.py`, `waveform_metrics_shared.py`, `precompute_wsst8s_mmecg.py`.
`raw_reference/` and `training_statistics/` are required local source data.

The WSST source and shared waveform metrics were relocated byte-for-byte from historical
directories during cleanup; only import paths changed. Old metadata and diagnostics here
remain traceability material, not current phase_pd results. See the project cleanup report
and docs/paper/PAPER_FACTS.md for model-lineage boundaries.
