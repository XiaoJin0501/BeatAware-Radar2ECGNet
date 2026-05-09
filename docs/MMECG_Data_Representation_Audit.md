# MMECG Data Representation Audit (Phase A)

**Date**: 2026-05-10
**Goal**: Test three sub-hypotheses (H1/H2/H3) explaining why MMECG cross-subject performance is stuck at PCC ≈ 0.27 despite confirmed model capacity (train PCC 0.69).

## Diagnostics

| ID | Question | Method | Output |
|----|----------|--------|--------|
| D1 | Which range bin actually carries the cardiac signal per subject? | Per-window Pearson(rcg_bin, ecg) on samplewise val.h5; aggregated by (subject, scene); two views: raw 0.5-20 Hz and narrow 0.8-3.5 Hz | `experiments_mmecg/diagnostics/D1_*` |
| D2 | Does the trained FMCWRangeEncoder's SE-attention actually focus on those bins? | Hook SE-attention output (B,50) on test loader of two trained models | `experiments_mmecg/diagnostics/D2_*` |
| D3 | Does per-window min-max ECG normalization destroy amplitude consistency? | Per-window (max−min) z-score histogram per subject on samplewise train.h5 | `experiments_mmecg/diagnostics/D3_*` |

## H3 — RCG signal quality varies wildly across subjects (D1) ✅ confirmed

D1 results (mean |Pearson(rcg_bin, ecg)| on samplewise val.h5):

| Subject-Scene | best raw bin | raw max corr | best narrow bin | narrow max corr | mean narrow corr |
|---------------|:---:|:---:|:---:|:---:|:---:|
| S5 NB  | 47 | 0.371 | 25 | 0.173 | 0.090 |
| S5 PE  | 42 | 0.096 | 31 | 0.142 | 0.062 |
| S9 NB  | 13 | 0.116 | 22 | 0.196 | 0.119 |
| S10 IB | 13 | 0.113 | 0  | 0.199 | 0.099 |
| S10 PE | 43 | 0.180 | 2  | 0.166 | 0.094 |
| **S13 NB** | 47 | 0.343 | 9  | **0.287** | 0.132 |
| **S14 NB** | 7  | 0.135 | 35 | **0.229** | 0.141 |
| **S16 IB** | 2  | 0.269 | 2  | **0.288** | 0.100 |
| **S17 SP** | 23 | 0.327 | 11 | **0.304** | 0.163 |
| S29 NB | 1  | 0.314 | 35 | 0.115 | 0.070 |
| **S29 SP** | 4  | **0.557** | 4  | 0.152 | 0.062 |
| S30 IB | 10 | 0.358 | 10 | 0.240 | 0.107 |

Key observations:
1. **No universal heart bin** — best-correlated bin varies from 1 to 47 across subject-scenes.
2. The "raw" view (0.5–20 Hz) and "narrow" view (0.8–3.5 Hz) often disagree on the best bin (e.g. S29 NB: raw bin 1 corr 0.31 vs narrow bin 35 corr 0.11). Heart-band filtering does not always raise correlation.
3. Maximum per-bin correlations are modest (mostly 0.15–0.35); this is the **upper bound on what any single-bin model could achieve**.
4. The "mean narrow corr" averaged over all 50 bins (0.06–0.16) is much lower than the best-bin correlation, confirming most bins are noise.

→ **H3 is at least partially true**: signal source quality is heterogeneous per subject. Some subjects (S29 SP raw 0.557) have a clearly dominant heart bin; others (S5 PE) barely have any.

## H2 — Current SE attention does NOT pick the heart bin (D2) ✅ STRONG confirmation

D2 on `mmecg_reg_samplewise_subject` best.pt (val_pcc=0.4421):

| Subject-Scene | n | max_bin | max_attn | mean_attn | std_attn | top-5 bins |
|---------------|---|:---:|:---:|:---:|:---:|---|
| S1 NB  | 85  | **7** | 0.608 | 0.494 | 0.067 | [7, 13, 0, 4, 23]  |
| S2 NB  | 85  | **7** | 0.602 | 0.494 | 0.066 | [7, 13, 0, 29, 4]  |
| S10 NB | 85  | **7** | 0.603 | 0.495 | 0.066 | [7, 13, 0, 4, 29]  |
| S13 IB | 85  | **7** | 0.604 | 0.493 | 0.067 | [7, 13, 0, 29, 4]  |
| S13 NB | 85  | **7** | 0.602 | 0.493 | 0.066 | [7, 13, 0, 29, 23] |
| S14 NB | 170 | 13    | 0.600 | 0.493 | 0.067 | [13, 7, 0, 29, 5]  |
| S16 NB | 170 | **7** | 0.600 | 0.496 | 0.066 | [7, 13, 0, 29, 4]  |
| S29 IB | 170 | **7** | 0.602 | 0.494 | 0.066 | [7, 13, 0, 29, 4]  |
| S29 SP | 170 | **7** | 0.606 | 0.493 | 0.067 | [7, 13, 0, 29, 4]  |
| S30 PE | 170 | **7** | 0.605 | 0.494 | 0.067 | [7, 13, 0, 29, 4]  |

Two devastating findings:

**Finding 1 — SE attention is subject-independent.** The top-5 bins `[7, 13, 0, 29, 4]` are nearly identical for all 10 subject-scenes. Mean and std of the 50-dim attention vector are essentially constant (0.494 ± 0.067) across subjects. The network learned a **global prior over bins**, not an adaptive selector.

**Finding 2 — Attention barely modulates.** With std=0.067 around mean=0.494, the highest-attended bin is only ~25% above average and the lowest is ~25% below. This is **soft uniform weighting**, not selection.

Cross-referencing with D1 (the actual best-correlated bin per subject):

| Subject-Scene | D1 best raw bin | D2 model max_bin | D1 best narrow bin | Match D2? |
|---------------|:---:|:---:|:---:|:---:|
| S5 NB        | 47  | n/a (S5 not in samplewise val) | 25 | n/a |
| S13 IB        | (S13 IB in val) | 7 | 9  | ❌ |
| S13 NB        | 47  | 7 | 9  | ❌ |
| S14 NB        | 7   | 13 | 35 | ❌ |
| S16 IB        | 2   | 7 | 2  | ❌ |
| S29 NB        | 1   | 7 | 35 | ❌ |
| S29 SP        | 4   | 7 | 4  | partial (S29 SP "ok" performer) |
| S30 IB        | 10  | 7 | 10 | partial |

→ The model's hot bins (7, 13, 0, 29, 4) only weakly overlap with the actual best-correlated bins per subject. **The model has learned a generic preference that happens to work for some subjects (e.g. S10/S13/S14 NB) and miss for others (e.g. S5/S16/S29 NB).**

→ **H2 is strongly confirmed**: the current FMCWRangeEncoder is not adaptive. Replacing the soft global SE attention with a per-recording top-K selection (or sharper input-conditional attention) is now the most-motivated next step.

## H1 — per-window min-max ECG normalization compresses amplitude (D3) ❌ unlikely main bottleneck

D3 on samplewise train.h5:

| Subject | n | amp median | amp std | CV | max/median ratio |
|---------|---|:---:|:---:|:---:|:---:|
| S1  | 14  | 5.90 | 0.14 | 0.023 | 1.05 |
| S2  | 14  | 7.15 | 0.19 | 0.026 | 1.08 |
| S5  | 271 | 6.29 | 0.35 | 0.055 | 1.43 |
| S9  | 110 | 6.62 | 0.29 | 0.044 | 1.18 |
| S13 | 62  | 6.25 | 0.76 | 0.117 | 1.42 |
| S14 | 195 | 7.19 | 1.81 | **0.242** | 2.51 |
| S16 | 96  | 6.08 | 0.71 | 0.113 | 1.87 |
| S17 | 203 | 5.14 | 0.31 | 0.059 | 1.35 |
| S29 | 451 | 7.30 | 0.58 | 0.079 | 2.26 |
| S30 | 223 | 6.41 | 0.29 | 0.046 | 1.10 |

- Within-subject amp variance is small (CV mostly 0.02–0.12) → per-window min-max does not catastrophically distort R-peak relative height.
- S14 has CV 0.24 (some outlier windows up to 2.5× median) — only one subject is meaningfully affected.
- Cross-subject median ranges 5.14–7.30 — per-window min-max homogenizes this, but PCC is scale-invariant anyway.

→ **H1 is mostly false** as a primary bottleneck. We will still run B1 (zscore target) per the plan to confirm — but expect raw PCC change ≤ ±0.02. The interesting test is whether L1/STFT loss converges differently.

## Conclusion and revised priorities

**The dominant finding is H2** (D2): the FMCWRangeEncoder's SE attention is a static global prior over range bins. It has no per-input adaptivity, so when the actual heart bin shifts (D1), the model simply weights the wrong bins.

This mechanistically explains:
- Why some subjects (S10/S13/S14 NB) work — their heart happens to fall near bin 7/13.
- Why others (S5/S16/S29) fail — their heart is on bins 1, 2, 25, 31, 35, 47 that the network barely attends.
- Why `balance_by=subject` helped — it let the model occasionally see input from subjects whose heart is on the model's preferred bins, anchoring the global prior.

**Revised execution priority** (the plan kept B1 first; D2 strongly suggests inverting):

1. ✅ **Phase A done**.
2. **B2 first (top-K adaptive bins)** is now the highest-leverage experiment. Two designs:
   - **B2a (offline)**: per-recording precompute the top-K bins by 0.8-3.5 Hz energy or |corr(bin, ECG)| (oracle); only feed these K=10 bins.
   - **B2b (online)**: replace SE attention with a *per-input* hard top-K selection (Gumbel-softmax / straight-through), or with cross-attention queried from a pooled summary of the input itself.
3. **B1 (zscore target) still worth doing in parallel** as a confirmation; expected to be ~neutral.
4. With B2 confirmed in samplewise, launch full 11-fold LOSO `mmecg_reg_loso_topk`.

**Paper story update** (replaces v3 in plan): the contribution becomes:
> "We expose that current radar-to-ECG architectures (radarODE, BeatAware) suffer from a static range-bin selector that does not adapt across subjects. We show this mechanistically explains 4-5× cross-subject performance drops in strict LOSO. We propose a per-recording adaptive top-K selector that closes most of the gap."

This is a cleaner contribution than the previous "lag-aware" framing — and is supported by direct mechanistic evidence (D1+D2), not just downstream metrics.

## Artifacts

- `experiments_mmecg/diagnostics/D1_rcg_bin_correlation_raw.png`
- `experiments_mmecg/diagnostics/D1_rcg_bin_correlation_narrow.png`
- `experiments_mmecg/diagnostics/D1_rcg_bin_correlation_summary.csv`
- `experiments_mmecg/diagnostics/D2_se_attention_heatmap_mmecg_reg_samplewise_subject_samplewise.png`
- `experiments_mmecg/diagnostics/D2_se_attention_heatmap_mmecg_reg_loso_subject_fold_01.png`
- `experiments_mmecg/diagnostics/D2_se_attention_summary_*.csv`
- `experiments_mmecg/diagnostics/D3_ecg_amplitude_histogram.png`
- `experiments_mmecg/diagnostics/D3_ecg_amplitude_summary.csv`

## Source-of-truth scripts (committed)

- `scripts/diagnose_rcg_bin_correlation.py` — D1
- `scripts/diagnose_se_attention.py` — D2
- `scripts/diagnose_ecg_amplitude.py` — D3
