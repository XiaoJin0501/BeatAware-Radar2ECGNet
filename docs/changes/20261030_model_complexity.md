# Trained seed42 model-complexity measurement

Author authorized inference-only measurements of six methods, MMECG/CR_RVS,
batch1/8s,50 warmups+200 timed iterations on GPU and single-thread CPU.
No training,checkpoint selection or existing results/figures changed.

Entry point:scripts/profile_model_complexity.py. Native first fold01 test radar
inputs; MM WSST50x71x240,CR log-STFT1x65x93; time-domain baselines use nativeRCG.
CR radarODE bilinear resize71x240 included. Strict-load selected full phase_pd
REM-Net and original five adapted dual-head trained checkpoints; SHA/epoch saved.

fvcore unified tracing,2FLOP/MAC plus bias additions. Custom counting covers fused
MHA,GRU,phase grid_sample and interpolation; all final unsupported-op dictionaries
empty. Explicit floating-arithmetic estimates vs zero-cost indexing/control and
untraced NumPy peak decoding described inREPORT. Timing includes that CPUdecoder,
D2Htransfer,REM phase resampling/internal decoder,final external event decoder.
FP32,TF32off,eval/no_grad,synchronized CUDA wall time,median200. All raw timings
retained; no timing change after count-only fused-bias supplementation.

Allocated trainable parameters include dormant original baseline heads; executed
FLOPs/latency exclude dormant paths. Do not conflate these with published native
network sizes or active-only parameter counts. Same seed42/base-seed lineage,
not a repeat over random inputs,records,folds or seeds. No model quality claims.

Outputs local docs/paper/sections/complexity_20261030/:complexity.csv,MMECG
single-column table_complexity.tex,CR table_complexity_crrvs.tex,REPORT.md,
measurements.json,profile_audit.json,environment.json. LaTeX usesbooktabs,
graphicx,xcolor[table];resizebox columnwidth ensures single-column fit. NoLaTeX
engine available locally; source/layout width specified,not compiled here.

Validation:12 strict checkpoint loads,input shapes verified,400 timing samples
per cell (GPU+CPU200 each),zero unsupported tensor operators,4 analytic tests
pass(conv2FLOP/MAC,bidirectional GRU,bilinear grid sampling,fused MHA). Local
fvcore dependencies installed only under/tmp/rem_complexity_tools.
