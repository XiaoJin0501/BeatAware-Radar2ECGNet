# User-authorized continuous-regression screening

New spec q1_event_conditioned_continuous_screen_v1 supersedes prior unexecutable
three-subject Wilcoxon development gate ONLY for this descriptive screening.
M1/M2/M3 fixed before runs; unit tests cover identity,20ms delay and missed beats.
M1 is conditional/R-aligned; matched and scored coverage are explicit. M3 uses
fixed common support and reference-selected lag. Neither replaces strict metrics.

MMECG folds1/2/3 x exact seeds42/43/44 x four unchanged arms=36runs. Seed values
are direct RNG seeds, not base+fold. Shared loss/optimizer/budget as user specified.
M2 selects checkpoints; M1/M3 cannot select. Arm4 is GT oracle, descriptive only.
No test access, no11fold confirmation, no automatic expansion. Screening thresholds
are8of9 positive M2 deltas, mean M2/M1 gains>=.01, endpoint macro MAE guard+5ms.

Old A supplements read saved waves, not network inference, and write only the
new results directory. A judgment and historical tables/figures unchanged.
Use clean isolated committed source worktree to protect user's dirty manuscript.
