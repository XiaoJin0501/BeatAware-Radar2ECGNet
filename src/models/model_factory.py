"""Model factory shared by MMECG training and testing scripts."""

from __future__ import annotations

from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.models.remalign import REMAlignMTLNet, REMAlignMTLRefineNet, REMAlignNet, REMEventNet


def build_mmecg_model(cfg):
    """Build the configured MMECG model.

    cfg.model_arch:
      - "beataware": frozen ICONIP BeatAwareRadar2ECGNet family
      - "tcn", "ki3_tcn", "ki8_tcn", "ki16_tcn", "stft_tcn", "wsst_tcn":
        runnable simple continuous baselines
      - "rem_event", "rem_align", "rem_align_mtl", "rem_align_mtl_refine":
        current REM-AlignNet family
    """
    arch = getattr(cfg, "model_arch", "beataware").lower()
    if arch == "beataware":
        return BeatAwareRadar2ECGNet(
            input_type="fmcw",
            n_range_bins=cfg.n_range_bins,
            C=cfg.C,
            d_state=cfg.d_state,
            dropout=cfg.dropout,
            use_pam=cfg.use_pam,
            use_emd=cfg.use_emd,
            emd_max_delay=cfg.emd_max_delay,
            use_diffusion=cfg.use_diffusion,
            diff_T=cfg.diff_T,
            diff_ddim_steps=cfg.diff_ddim_steps,
            diff_hidden=cfg.diff_hidden,
            diff_n_blocks=cfg.diff_n_blocks,
            use_output_lag_align=getattr(cfg, "use_output_lag_align", False),
            output_lag_max_samples=int(round(
                getattr(cfg, "output_lag_max_ms", 200.0)
                / 1000.0
                * getattr(cfg, "fs", 200)
            )),
            fmcw_selector=getattr(cfg, "fmcw_selector", "se"),
            fmcw_topk=getattr(cfg, "fmcw_topk", 10),
            fmcw_out_channels=getattr(cfg, "fmcw_out_channels", 16),
            fmcw_tau=getattr(cfg, "fmcw_tau_init", 1.0),
            backbone_type=getattr(cfg, "backbone_type", "tcn"),
        )

    if arch in ("rem_event", "remalign_event"):
        return REMEventNet(
            in_channels=getattr(cfg, "n_range_bins", 50),
            hidden=getattr(cfg, "baseline_hidden", 128),
            dropout=getattr(cfg, "dropout", 0.1),
        )

    if arch in ("rem_align", "remalign", "remalign_net"):
        return REMAlignNet(
            in_channels=getattr(cfg, "n_range_bins", 50),
            hidden=getattr(cfg, "baseline_hidden", 128),
            dropout=getattr(cfg, "dropout", 0.1),
            max_delay=getattr(cfg, "align_max_delay", 8.0),
        )

    if arch in ("rem_align_mtl", "remalign_mtl", "remalign_net_mtl"):
        return REMAlignMTLNet(
            in_channels=getattr(cfg, "n_range_bins", 50),
            hidden=getattr(cfg, "baseline_hidden", 128),
            dropout=getattr(cfg, "dropout", 0.1),
            max_delay=getattr(cfg, "align_max_delay", 8.0),
        )

    if arch in ("rem_align_mtl_refine", "remalign_mtl_refine", "remalign_net_mtl_refine"):
        return REMAlignMTLRefineNet(
            in_channels=getattr(cfg, "n_range_bins", 50),
            hidden=getattr(cfg, "baseline_hidden", 128),
            dropout=getattr(cfg, "dropout", 0.1),
            max_delay=getattr(cfg, "align_max_delay", 8.0),
        )

    try:
        from src.models.baselines.simple_temporal import build_simple_baseline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Simple baseline models require src/models/baselines/simple_temporal.py. "
            "The BeatAware model is still available with model_arch='beataware'."
        ) from exc

    return build_simple_baseline(
        arch=arch,
        n_range_bins=cfg.n_range_bins,
        hidden=getattr(cfg, "baseline_hidden", 128),
        dropout=cfg.dropout,
    )
