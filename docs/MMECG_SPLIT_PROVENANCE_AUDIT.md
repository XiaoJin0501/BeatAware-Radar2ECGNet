# MMECG LOSO Split Provenance Audit

Date: 2026-09-09. Target: `/home/qhh2237/Datasets/MMECG/processed/loso`.
Status: **defect confirmed**. Gate 1 and Gate 2 are paused.

## Finding

Across 11 folds there are only **2 distinct validation subjects**.

| validation subject | folds | val file SHA256 |
|---|---|---|
| S14 | 1, 2, 4, 5, 6 | `6f3df0e7a74fda28…` (byte-identical) |
| S9  | 3, 7, 8, 9, 10, 11 | `255300e8f1684ba1…` (byte-identical) |

## Root cause

`/home/qhh2237/Datasets/MMECG/03B_create_loso_splits.py`, `pick_val_subject()`:

> among the remaining subjects, pick the one whose **record count is closest to the
> median**; tie-break by smallest `subject_id`.

That criterion barely depends on which subject was removed as test, so the same subject
wins in almost every fold. The selection is deterministic and deterministically
degenerate. Output manifest: `metadata/mmecg_loso_splits.csv`.

## What is NOT broken

- **Subjects**: 11 present — 1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30.
- **Test structure is a correct LOSO**: each subject is the test subject exactly once
  (folds 1–11 → S1, S2, S5, S9, S10, S13, S14, S16, S17, S29, S30).
- **Within-fold disjointness holds in all 11 folds**: no subject appears in more than
  one of train/val/test. Verified programmatically.

So a single fold, taken alone, is a valid strict-LOSO fold.

## What IS broken

**1. Validation diversity.** Model selection is calibrated on 2 subjects out of 11.

**2. Cross-fold model-selection bias in the aggregate.**

| subject | validation in folds | test in fold | consequence |
|---|---|---|---|
| S14 | 1, 2, 4, 5, 6 | **7** | selection saw S14 in 5 folds before testing on it |
| S9 | 3, 7, 8, 9, 10, 11 | **4** | selection saw S9 in 6 folds before testing on it |

Each fold is individually clean, but the 11-fold aggregate is not: checkpoints and
hyperparameters were chosen using subjects that later appear as test subjects in other
folds.

**Terminology.** This is a *repeated / non-representative validation assignment* causing
*cross-fold model-selection bias*. It is NOT test leakage: there is no within-fold
train/val/test subject overlap in any of the 11 folds (invariant I1, verified
programmatically). Calling it leakage would overstate the defect and misdescribe the
test partition, which is a correct LOSO.

**3. The recent pilots' "3 folds" are not 3 folds.** Folds 1, 5, 9 → validation
subjects 14, 14, 9. Folds 1 and 5 have **byte-identical** validation data. Every
"3/3 folds" statement from those pilots is really "2 distinct validation subjects, one
counted twice", and must be withdrawn.

## Blocked / invalidated

| experiment | action |
|---|---|
| `q1_continuous_ecg_b0_b2_pilot` | **PROTOCOL-INVALID** — all "3/3 folds" conclusions withdrawn |
| `q1_cgd_reeval_fullsupport` | **PROTOCOL-INVALID** — same folds, same defect |
| any conclusion drawn from folds 1/5/9 validation | **BLOCKED** pending re-run on a repaired split |
| 203 run-configs referencing this directory | flagged for review; test-tier results need separate assessment (see below) |

**Final-test results are affected differently and less severely.** The test partition
itself is a correct LOSO, so per-fold test numbers remain individually valid; what is
compromised is the aggregate, because selection touched S9 and S14 before they were
tested. This must be stated rather than silently carried.

## Repair

Rotation assignment: for fold *i* with test subject `S[i]`, take
`val = S[(i+1) mod 11]`. Each subject is then the test subject exactly once **and** the
validation subject exactly once, and val ≠ test by construction.

Generated to a **new** directory; nothing existing is overwritten or deleted.
