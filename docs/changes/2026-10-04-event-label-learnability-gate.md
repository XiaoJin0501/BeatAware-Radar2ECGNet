# Event-label learnability Gate 0

The author approved the first stage of the structured event-conditioned REM-Net
rescue route on 2026-10-04. Before any new model training, a fixed-label audit was
registered in `experiments/specs/q1_event_label_learnability_v1.json` and run over
all 16,682 unique MMECG ECG windows.

`scripts/evaluate_event_label_learnability.py` records per-subject Q/R/S/T label
availability, overlap-window repeatability, and Q<R<S<T ordering. The locked gate
passed. This changes development authorization only: a folds 1/5/9 event-only
gate may now be designed. It does not alter frozen benchmark results, validate
Q/S clinically, or authorize 11-fold training.
