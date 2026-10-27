# Author-requested paired ΔPCC revision

Existing sigma0..80 data unchanged, reference figure archived before replacement.
Y/CI/star are per-subject A4−A0 differences, paired bootstrap10000; zero dashed
line is Context only. Extend nominal sigma128, then fixed256/512/1024 only if
needed for paired statistical crossover. All16 frozen weights, five fixed
repeat seeds, no training or tuning. Same model/targets/error constraints.

Generalize original inference helper's sigma/output-directory arguments without
changing defaults. New caches collect signed endpoint errors for RR-span/common-
mode diagnostics on matched, valid and all-window subsets. Same denominators
for primary star x and span interpretation; no unsupported device-offset claim.
Y-range sharing uses predefined CI-span ratio<=2. New source CSV and REPORT in
anchor_robustness_delta_20261028; original absolute source CSV/report untouched.
