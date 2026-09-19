# CR_RVS Literature Protocol Audit

Date: 2026-09-19. Sources: `reference_papers/CR_RVS_baselines model/` (full text
read; every number below is quoted from the paper and located by section/table).
Purpose: the CR_RVS **Literature Comparison** block of the TIM manuscript. These are
reported numbers under each paper's own protocol; none is a matched reproduction and
none may share a ranking column with our results (METRIC_CONTRACT, CONTINUOUS
contract §6).

## The four papers

| Paper | Scenario(s) | Split | Windows | Target | Reported |
|---|---|---|---|---|---|
| **Wu et al. 2024**, RSSRnet, IEEE Sensors J. 24(5) | Resting, 25 subjects | (a) five-fold CV, "80 % of the data used for training, the remaining 20 %" for test -- a data-level split; (b) separately, 25-fold leave-one-subject-out ("trained on data from 24 individuals and evaluated on the remaining one") | 8 s at 100 Hz, step 20 % of the length (**80 % overlap**) | min-max per data pair | (a) PCC **0.959 ± 0.039** (Table I); median 0.964 (abstract). (b) per-subject median PCC **ranging 0.39 to 0.90**, "more than half of the individuals … above 0.7" (Sec. IV-F, Fig. 12d) |
| **Xiao et al. 2026**, CA-WTBNet, Bioengineering 13:731 | **Resting only** | subject-independent, **one fixed split**: 25 subjects train/validation (85/15 random within), **5 subjects test** (298 segments); no cross-validation over subjects | 2048 samples at 200 Hz (10.24 s), non-overlapping | ECG un-normalised; radar min-max | pooled segment-level PCC **0.9641 ± 0.0421**, per-subject 0.957-0.974 (Tables 3-4). Table 5 lists RSSRnet 0.9590 "for reference" beside its own subject-independent result |
| **Li et al. 2025**, Radar2ECG, ICASSP 2025 | Resting | "split 8:2 into 29,680 training [samples]"; no subject separation stated | width 4096, step 1024 (**75 % overlap**) | normalised | **cosine similarity 0.987**, R² 0.879 (Table I) -- not PCC |
| **Chowdhury et al. 2024**, MultiResLinkNet, Comput. Biol. Med. 176:108555 | Resting, Valsalva, Apnea, and combined (RVA) | five-fold CV; Sec. 2 states "separate test subjects were used … no data was shared", while Sec. 3.2 describes experiment B as using "subject-wise stratification … instead of random stratification" and both experiments as 80 % / 20 % segment sets -- **internally inconsistent** | overlapping windows | — | average temporal correlation **66.10 ± 19.33 %** (Resting), 60.14 (Valsalva), 61.86 (RVA); spectral correlation 74.7-82.4 % (Tables 2-3) |

## What the literature shows

1. **The same model, the same data, two protocols.** RSSRnet reaches PCC 0.959 under
   a data-level five-fold split of 80 %-overlapping windows, and per-subject medians of
   0.39-0.90 under leave-one-subject-out -- in the authors' own paper. This is the
   cleanest available measurement of how much the split protocol moves the headline.
2. **Overlapping windows plus a data-level split.** Wu 2024 (80 % overlap) and Li 2025
   (75 % overlap) both combine heavily overlapping windows with a split that is not
   stated to be subject-wise. Adjacent windows then share most of their samples across
   train and test.
3. **A subject-independent number is not automatically general.** Xiao 2026 is
   subject-independent, but on a single fixed five-subject test set, resting only, with
   pooled segment statistics and no cross-validation over subjects. Its 0.964 is
   compared in its own Table 5 with RSSRnet's data-level 0.959 as if the protocols
   matched. **No leakage is alleged for Xiao 2026**; the issue is the size and
   selection of the test set and the protocol mix in the comparison.
4. **The one paper describing subject-separated folds reports ~0.6-0.66.** Chowdhury
   2024's temporal correlation is far below the 0.96-0.99 figures, consistent with
   point 1, although its protocol text is inconsistent and it is cited with that caveat.
5. **Metrics are not interchangeable.** Li 2025 reports cosine similarity, Chowdhury
   2024 reports a temporal correlation in percent; neither is placed in a PCC column.

## How the manuscript uses this

- A "Reported literature on CR_RVS" table with a protocol column (scenario, split
  unit, window overlap, target normalisation, metric), kept visually separate from our
  matched results.
- The RSSRnet 0.959 → 0.39-0.90 contrast is quoted as the field's own evidence for
  protocol-driven inflation; our subject-independent CR_RVS results (W3) are reported
  overall and for the Resting scenario alone, so the like-for-like scenario is visible.
- Wording rules: say "data-level split" or "split not stated to be subject-wise", never
  "random split" where a paper does not say random; never state or imply leakage for
  Xiao 2026; quote the per-subject range 0.39-0.90 exactly, not a median the paper does
  not report.

## Matched CR_RVS arms (W3 scope, revised 2026-09-19)

REM-Net, Chen-TCN (native time-domain family) and WaveGRU-Net (MODWT wavelet family,
also the family Xiao 2026 builds its front end on). radarODE-MTL, Attention-DCGAN and
the Conformer are MMECG-only: radarODE-MTL's official squeeze kernel needs a
71-bin time-frequency map, and resampling CR_RVS's 65-bin STFT onto it to fit a fixed
kernel adds compute without adding evidence.
