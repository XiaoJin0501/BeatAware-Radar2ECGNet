# RCG Provenance and Temporal-Resolution Audit

Date: 2026-09-09. Gate for the B2b dual-path arm. Analysis only; nothing modified.

## Headline: a window-support asymmetry that likely explains the morphology/continuous split

`scripts/precompute_wsst4s_mmecg.py` computes WSST from `rcg[..., 400:1200]` and
records `crop = "center_4s_samples_400_1200"` in the file attributes. The ECG target
is the full 1600 samples.

| | |
|---|---|
| ECG output window | 1600 samples = **8.0 s** |
| WSST radar support | samples 400-1200 = **4.0 s, centre only** |
| output with no radar input support | samples 0-399 and 1200-1599 = **50 % of every window** |
| Shape-200 centre cycle inside the supported region | **170/170 = 100 %** (fold 01 test) |
| full-window target inside the supported region | **50 %** by construction |

**The model is asked to produce 8 s of ECG from 4 s of radar.** The outer 2 s on each
side is unconstrained extrapolation. Shape-200 sits entirely inside the supported
region because the centre cycle is selected around sample 800; the full-window target
does not.

This is a data-pipeline fact, not a physics or capacity result, and it is a strong
candidate explanation for the dissociation the manuscript has attributed to anchors:
Shape-200 ~0.88 against full-window ~0.32. It must be resolved before any further
mechanism claim, and before B2b is interpreted.

## Corrected temporal-resolution statement

My earlier figure of 15 Hz was wrong: it divided 120 bins by 8 s, but the WSST spans
4 s. **Corrected: 120 bins over 4.0 s = 30 Hz effective, one bin = 33.3 ms, and a
~80 ms QRS spans ~2.4 bins.**

This is a **severe temporal-resolution bottleneck relative to a 200 Hz target**, and it
bounds how sharply QRS onset can be placed from this representation alone. It is
**not** a demonstration that QRS recovery from WSST is impossible: the WSST is a
complex time-frequency representation over 50 range bins, and 2.4 bins of support with
learned priors is a hard problem, not a proven-empty one. No impossibility is claimed.

## Provenance

- `rcg` in the H5 is `(N, 50, 1600)` float32, **z-scored per channel** at build time
  (`src/data/mmecg_dataset.py:200`), at the native 200 Hz ECG grid.
- WSST is derived **from** `rcg` (broadband, unfiltered) by `precompute_wsst4s_mmecg.py`,
  which applies no bandpass. The `_wsst4s_` name and the crop attribute are the only
  record of the 4 s restriction.

## Effective bandwidth

Welch PSD, 40 windows, fold 01 test:

| band | MMECG `rcg` | CR_RVS `rcg` |
|---|---|---|
| 0.8-3.5 Hz | 2.3 % | **49.2 %** |
| 3.5-10 Hz | 23.7 % | 4.8 % |
| 10-20 Hz | **70.7 %** | 0.1 % |
| 95 % of power below | **19.1 Hz** | **3.1 Hz** |
| (ECG target, same windows) | 14.8 Hz | — |

**MMECG `rcg` is genuinely broadband** — its 95 % bandwidth (19.1 Hz) exceeds the
ECG's own (14.8 Hz), so it is not resolution-limited relative to the target. Much of
that energy sits at 10-20 Hz, which overlaps the known clutter band and is not
automatically cardiac; bandwidth availability is not the same as cardiac information.

**CR_RVS `rcg` is already effectively band-limited to ~3 Hz.** A native-rate path there
has little high-frequency content to contribute.

## Preprocessing hazard

`configs` for the frozen incumbent: MMECG `narrow_bandpass = True`, CR_RVS `False`.
When enabled, `src/data/mmecg_dataset.py` applies a 4th-order 0.8-3.5 Hz Butterworth to
`rcg` at load time and re-normalizes. That band holds only **2.3 %** of MMECG `rcg`
power, so this discards ~97.7 % of it, including the entire 3.5-20 Hz range where QRS
structure would live. **Any B2b run must declare its `narrow_bandpass` setting; leaving
it True would make the native-rate path largely pointless on MMECG.**

Note this filter is applied to the `rcg` load path. The WSST used by the current model
is precomputed from unfiltered `rcg`, so the incumbent's WSST input is not affected.

## Leakage

`_oracle_corr_per_bin` (`mmecg_dataset.py:113`) selects range bins by correlation with
**ECG ground truth** and is correctly flagged in its own docstring as
non-deployable. It is reachable only when `topk_method == "corr"`.

- Default is `topk_method = "energy"` (input-only, deployable) at every call site.
- `topk_bins` defaults to `None` in all three loader builders.
- The incumbent config carries `n_range_bins = 50` with no `topk_bins` key.

**Assessed clean for the incumbent path.** The `corr` mode must remain prohibited for
any B2 arm, and the B2b spec should assert `topk_method != "corr"` explicitly.

## Synchronization

Radar-envelope to R-peak lag, 60 windows, ±300 ms search: median **-140 ms**,
IQR [-145, -93]. This is consistent with a physiological electromechanical delay of the
expected order and direction, not a gross indexing misalignment. `rcg` and `ecg` share
the same 1600-sample grid and the same `start_idx`/`end_idx`.

## Availability

| | MMECG | CR_RVS |
|---|---|---|
| `rcg` present | yes, `(N, 50, 1600)` | yes, `(N, 1, 1600)` |
| channels | 50 range bins | 1 (CW) |
| usable extra bandwidth vs WSST | substantial | minimal |

## Implications for the pilot

1. **The 4 s / 8 s support asymmetry should be treated as the first-order suspect** for
   the continuous-reconstruction failure, ahead of decoder capacity. B2a and B2b as
   currently framed would both inherit it.
2. **B2b is expected to help MMECG and not CR_RVS**, for a measurable reason. CR_RVS
   must not be read as a refutation of the dual-path design, and must not trigger
   changes to the shared trunk.
3. **B2 arms must declare `narrow_bandpass = False`** for the native-rate path, and
   `topk_method = "energy"`.
4. A fair B2a/B2b comparison requires the input support question to be settled first —
   either by restricting the target to the supported 4 s, or by extending WSST support
   to the full window. That choice is a protocol decision, not an implementation detail.
