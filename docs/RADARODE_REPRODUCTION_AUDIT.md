# radarODE-MTL Reproduction Audit

Date: 2026-09-09. Sources: `reference_papers/Zhang et al. - 2025 - radarODE-MTL ...pdf`,
official repo `/home/qhh2237/Projects/radarODE-MTL` (HEAD `efcbe8c`), local adaptation
`scripts/train_radarode_mtl_ega_mmecg.py`.

## Headline

**Our local 0.728 does not represent official radarODE-MTL performance, and the
claim that REM-Net 0.885 fairly beats radarODE-MTL 0.728 must be withdrawn.**
The local run is a Shape-200-only adaptation with a replaced morphology decoder,
a materially altered PPI target, and no long-term assembly. Separately, the
official repository does not ship the assembly stage that produces the paper's
headline long-term number, so an "official-protocol reproduction" is not
currently possible from released code.

## A. Discrepancies inside the official release (paper vs code)

| # | Finding | Evidence |
|---|---|---|
| A1 | **The ODE embedding is inactive.** `CNNLSTMDecoder.forward` returns `x_temproal`; the `temp+ode` fusion branch is fully commented out, as is the `ode only` branch. The author's own comment on the disabled branch reads "temp+ode, give the best result but time consuming due to the ode solver", and on the active one "temp only, faster but not as good as temp+ode". | `Projects/radarODE_plus/nets/decoder.py:128-151` |
| A2 | `self.ode_solver = ode1_solver()` is commented out in `__init__`. | `nets/decoder.py:65` |
| A3 | The assembled `radarODE_MTL` class is annotated "Only for illustration purposes, not used anywhere in the code". Training builds `backbone` + `shapeDecoder` + `PPI_decoder` + `anchor_decoder` separately via LibMTL. | `nets/model.py:47-49`, `main.py:76-80` |
| A4 | **No long-term assembly or long-record evaluation exists anywhere in the repository.** A repo-wide search for assembly/stitching/overlap-add/long-record terms returns only `Cross_stitch` (an unrelated LibMTL architecture). | repo-wide grep |
| A5 | `num_out_channels['Anchor'] = 200` is declared, but the `PPI_decoder(output_dim=200)` line for Anchor is commented out and `anchor_decoder()` is used instead; its output length is fixed by its conv stack, not by the declared value. | `main.py:76-80` |

**Consequence:** the titular ODE contribution is not exercised by the released
code path, and the paper's long-term median PCC (~0.927) cannot be regenerated
from the repository. Any number we quote as "official" is therefore
*reported-only*, not *official-code reproduction*.

## B. Official task definitions (as actually implemented)

From `Projects/radarODE_plus/spectrum_dataset.py` and `main.py`:

- Inputs are pre-computed `.npy` triples `(sst, ecg, anchor)`; SST generation lives in
  `MMECG_to_SST.m` and the normalization applied there is **not visible from Python**.
- `ECG_shape`: `down_sample(ecg_data, target_len=200)` — a **200-point resampled single
  cycle**. This is the same output family as our Shape-200, not a continuous waveform.
- `PPI`: the **original, un-resampled** cycle right-padded to length 260 with the
  sentinel `-10`. The padding boundary is what encodes cycle duration.
- `Anchor`: loaded from the precomputed `.npy`; construction not reconstructible from
  released Python.
- Training: 200 epochs, batch size 22.

## C. Defects in our local adaptation

| # | Ours | Official | Impact |
|---|---|---|---|
| C1 | `ShapeDecoder200`, a custom deconv stack ending in **`nn.Sigmoid()`** plus an `F.interpolate` fallback to 200 | `CNNLSTMDecoder` (encoder + conv + out_conv), **no sigmoid** | Different capacity and a bounded output range the official model does not impose. Not a faithful decoder. |
| C2 | PPI = `normalize_01(resample_1d(cycle, 260))` when `cycle.size > 260`, else `normalize_01(cycle)` then pad with `-10` | raw cycle padded to 260 with `-10`, **no per-cycle normalization** | Two deviations. (i) We min-max normalize each cycle, which the official does not. (ii) For cycles longer than 260 samples (heart rate below ~46 bpm at 200 Hz) we resample instead of padding, which **destroys the duration signal** the PPI task exists to carry. Rare at normal rates, but the normalization difference applies to every sample. |
| C3 | Anchor = 800-point Gaussian mask, sigma 4, over window samples 400-1200 | precomputed, unknown construction | Not verifiable as equivalent. |
| C4 | `ECG_shape` = `normalize_01(resample_1d(cycle, 200))` | `down_sample(cycle, 200)`, normalization upstream and unknown | Possibly equivalent, unverified. |
| C5 | **No long-term assembly implemented**; final scoring is Shape-200 | paper reports long-term assembled ECG | Our 0.728 measures a different endpoint from the paper's 0.927. |

## D. Evidence tiers — never mix these

1. **Reported-only** — paper's ~0.927 long-term median PCC. Not reproducible from
   released code (A4). Cite as literature context only.
2. **Official-code reproduction** — *currently impossible* for the long-term endpoint;
   possible in principle for `ECG_shape` alone, and even then with the ODE path
   disabled (A1), so it would not test the paper's central claim.
3. **Strict-LOSO reproduction under our protocol** — what `0.728 ± 0.133` actually is,
   subject to defects C1-C5.

**Statistic mixing is prohibited.** The paper reports a *trial median*; we report
*fold-macro mean ± std*. These are not comparable quantities and must never appear in
the same ranking column.

## E. Required manuscript corrections

- Withdraw any head-to-head superiority claim of REM-Net over radarODE-MTL.
- Re-label our radarODE-MTL row as an *adapted Shape-200 reproduction under our
  protocol*, with C1-C5 disclosed.
- State that the official ODE fusion path is disabled in the public release, so no
  reproduction to date tests the paper's ODE contribution.

## F. Open items

- Recover SST preprocessing from `MMECG_to_SST.m` to check input parity.
- Long-term assembly is not an active implementation route. If a future user
  explicitly reopens it, any implementation would be *our*
  protocol, not the official one, and must be labelled as such.
