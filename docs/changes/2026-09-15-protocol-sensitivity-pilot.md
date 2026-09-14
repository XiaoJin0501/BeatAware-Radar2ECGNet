# Register subject-overlap protocol sensitivity pilot

This pilot measures the performance effect of reusing subject identity without
mislabeling it as LOSO. The existing MMECG `samplewise` partition is actually
record-disjoint: its training and validation records do not overlap, while most
validation subjects occur in training. The arm is therefore labeled
`record-disjoint, subject-overlap validation`.

The continuous B1 architecture, loss, 80/20 budget, training-only scaler,
checkpoint rule, full-window metrics, and 8-second input support remain fixed.
Only the split protocol changes. The source test partition is unopened.

This is a diagnostic pilot. Its result may quantify protocol sensitivity but
cannot be promoted, mixed into strict-LOSO aggregates, or called LOSO.
