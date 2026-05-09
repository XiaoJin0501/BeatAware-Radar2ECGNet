# Current MMECG Scripts

Only the active MMECG paper/debugging pipeline is kept in this directory.

| Script | Role |
|---|---|
| `train_mmecg.py` | Train MMECG models under LOSO or samplewise protocols. |
| `test_mmecg.py` | Run full waveform, peak, QMR, fiducial, and interval evaluation. |
| `analyze_mmecg_lag_adjusted_pcc.py` | Diagnose raw-vs-shifted PCC timing failures. |
| `analyze_mmecg_input_lag.py` | Diagnose input RCG-vs-ECG lag before model reconstruction. |
| `plot_mmecg_failure_cases.py` | Generate representative waveforms, subject bars, and subject-scene heatmaps. |
| `probe_mmecg_output_lag_head.py` | Inspect scalar output-lag head behavior during alignment experiments. |

Removed obsolete Schellenberger / old ablation scripts:

- `train.py`
- `test.py`
- `run_ablation.sh`
- `run_ablation_ac.sh`
- `run_ablation_mmecg.sh`
- `run_model_d.sh`
- `summarize_ablation.py`
- `plot_subject_metrics.py`
- `plot_paper_figures.py`
- `plot_loso_predictions.py`
- `plot_training_curves.py`

