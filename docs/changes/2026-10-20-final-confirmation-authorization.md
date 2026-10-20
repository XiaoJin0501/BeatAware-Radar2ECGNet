# Final confirmation authorization, fixed before screening results

The author approved MMECG strict11-fold LOSO and CR_RVS subject-independent
5-fold, main seeds42/43/44. No further architecture design follows the current
27-run screen. Winner priority remains D>C>B>A. This note changes no screening
source, metric, gate, checkpoint or running process. It lives in a separate
worktree/branch so the running source/commit freeze remains valid.

## Conditional execution matrix

|Winner|Main runs|Seed42 controls on all16folds|Total formal runs|
|---|---:|---|---:|
|D|48|A/B/C,48runs|96|
|C|48|A,16runs|64|
|B|48|A,16runs|64|
|A (arm1)|48|None|48|

These are fresh formal runs; development validation scores are not test scores.
The author allows continuation without another reply after the screening
decision summary is reported. This is scope authorization, not a claim that
the formal runner is already implemented or started. Before execution, complete
the new runner/data preflight, registry/manifests and one-time test state machine;
do not alter any frozen screening worktree to implement it. In particular,
CR_RVS train/validation/test native references and aligned STFT/RCG must be
supported and tested; current screening is MMECG-only.

## Test and inference integrity

Training/early stopping/checkpoint selection use validation only, under the
existing frozen criterion. After training terminates, freeze the selected
checkpoint and evaluate test once per run. No test-derived scaling, temporal
compensation, checkpoint reranking or tuning. GT-selected record-optimal PCC
remains a diagnostic metric, not a deployable alignment method. Preserve
explicit test-access state before opening it; a failed/partial test evaluation
stops for audit rather than being silently repeated.

## Subject-level statistics

Datasets are analyzed separately. Average the main's three seed metrics within
subject before pairing against A's seed42 value. The M1×coverage endpoint is
the average of each seed's product, not the product of seed means. Report
two-sided Wilcoxon signed-rank (zero_method=wilcox, method=auto; all-zero p=1),
the median within-subject difference and a paired-subject bootstrap95%CI
(10000resamples, percentile2.5/97.5, dedicated RNGseed20261020). Never resample
seeds or treat seed replicas as independent subjects. Keep this RNG separate
from all training/evaluation RNG. For errors, a negative main-minus-A effect
favors the main; for PCC/coverage, positive does.

The two screening-target endpoints M1×coverage and QRS-region PCC are the
predeclared primary confirmation endpoints; report raw and Holm-adjusted
p-values within each dataset. Other metric comparisons are descriptive or
exploratory. Explicitly report missingness and effective subject counts;
incomplete primary endpoints do not support complete-confirmation claims.
If A wins, it is both the main and comparator: no artificial self-comparison,
Wilcoxon or bootstrap superiority statement is produced.

## Schedule

Finish screen → report its fixed decision → formal main/controls → Chen's nine
development runs. Chen's GPU full-validation timing and any required cadence
freeze still precede Chen training. The earlier plan placing Chen immediately
after the screen is superseded. No additional architecture search is authorized.
