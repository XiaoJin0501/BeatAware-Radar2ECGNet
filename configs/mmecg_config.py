"""MMECG configuration for the ICONIP2026 BeatAware-Radar2ECGNet runs."""


class MMECGConfig:
    # Data: 11-fold strict LOSO, 8 s windows at 200 Hz.
    loso_h5_dir: str = "/home/qhh2237/Datasets/MMECG/processed/loso"
    fs: int = 200
    signal_len: int = 1600
    input_type: str = "fmcw"
    n_range_bins: int = 50
    n_folds: int = 11

    # ICONIP2026 main model: KI16 + lightweight TCN + CP + PA.
    model_arch: str = "beataware"
    backbone_type: str = "tcn"
    use_pam: bool = True
    use_emd: bool = True
    fmcw_selector: str = "se"
    fmcw_topk: int = 10
    fmcw_out_channels: int = 16
    fmcw_tau_init: float = 1.0

    # Training.
    batch_size: int = 16
    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    early_stop_patience: int = 20  # epochs without validation improvement

    # Data loading.
    num_workers: int = 4
    balanced_sampling: bool = True
    balance_by: str = "subject"
    narrow_bandpass: bool = True
    target_norm: str = "minmax"
    wsst_suffix: str = "_wsst.h5"
    beta_peak: float = 1.0
    use_shape200_loss: bool = False
    lambda_shape200: float = 0.2
    shape200_pcc_weight: float = 0.2

    # Output.
    exp_dir: str = "experiments_mmecg"

    # Model dimensions.
    C: int = 64
    d_state: int = 16  # Legacy PAM arg kept for config/checkpoint compatibility
    emd_max_delay: int = 20
    dropout: float = 0.1

    # Compatibility defaults used by optional CLI paths and old checkpoints.
    # These are off in the ICONIP2026 main run.
    samplewise_h5_dir: str = "/home/qhh2237/Datasets/MMECG/processed/samplewise"
    baseline_hidden: int = 128
    topk_bins: int = 0
    topk_method: str = "energy"
    fmcw_tau_final: float = 0.1
    fmcw_tau_anneal: str = "linear"
    use_diffusion: bool = False
    diff_T: int = 1000
    diff_ddim_steps: int = 50
    diff_hidden: int = 256
    diff_n_blocks: int = 8
    use_lag_aware_loss: bool = False
    lag_max_ms         : float = 100.0
    lambda_lag_pcc     : float = 0.2
    lambda_lag_l1      : float = 0.05
    lambda_zero_pcc    : float = 0.0
    lambda_lag_penalty : float = 0.0
    lag_softmax_tau    : float = 0.05

    use_output_lag_align: bool = False
    output_lag_max_ms    : float = 200.0
    lambda_output_lag_l1 : float = 0.0

    # Hybrid electromechanical alignment (rem_align_hybrid_mtl). See
    # docs/ALIGNMENT_DESIGN.md. Toggle delta_sys / residual for the B0-B4 ablation.
    use_delta_sys         : bool  = True
    use_delta_residual    : bool  = True
    delta_sys_knots       : int   = 5
    delta_sys_max_ms      : float = 150.0
    delta_residual_eps_ms : float = 30.0
    align_space           : str   = "latent"  # "latent" | "output"

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_") and not callable(v):
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# 单例，可直接 import
mmecg_cfg = MMECGConfig()

if __name__ == "__main__":
    print(mmecg_cfg)
