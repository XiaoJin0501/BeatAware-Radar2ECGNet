# Subject-overlap protocol sensitivity result

The fixed B1 continuous model completed on the registered record-disjoint,
subject-overlap MMECG partition. Its validation PCC was 0.3955 and R2 was
0.1213. The repaired `loso_v2` B1 development folds 1/5/9 have fold-macro PCC
0.2231 and R2 -0.1356. The registered numeric threshold therefore passes with
delta PCC +0.1725.

The result does **not** identify subject reuse as the cause. Within the same
validation partition, the eight subjects seen in training pooled to PCC 0.3627,
whereas the sole unseen subject, S10, reached PCC 0.5922. Subject composition,
record/state difficulty, and aggregation differ between the two protocols.

The defensible conclusion is limited to protocol sensitivity and large
subject-level heterogeneity. This pilot cannot be called LOSO, cannot be mixed
with the strict aggregate, and cannot be used to accuse a baseline of leakage.
The source test partition remained unopened. Shape-200 was auxiliary and was
not used for checkpoint selection.
