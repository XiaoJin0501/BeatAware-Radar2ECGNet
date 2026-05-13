"""
mmecg_config.py — MMECG 数据集训练配置

继承并覆盖 config.py 中的通用参数，适配：
  - FMCW 50 通道输入 (input_type='fmcw')
  - LOSO 11 折交叉验证
  - dataset_mmecg/ 目录结构
"""

from pathlib import Path


class MMECGConfig:
    # ── 数据（新流程：直接读取预构建 H5）────────────────────────────────────────
    loso_h5_dir       : str  = "/home/qhh2237/Datasets/MMECG/processed/loso"
    samplewise_h5_dir : str  = "/home/qhh2237/Datasets/MMECG/processed/samplewise"
    protocol          : str  = "loso"    # "loso" | "samplewise"

    # ── 数据（旧流程保留，不再使用）──────────────────────────────────────────────
    dataset_dir   : str  = "dataset_mmecg"
    h5_path       : str  = "/home/qhh2237/Datasets/MMECG/MMECG.h5"
    fs            : int  = 200
    signal_len    : int  = 1600     # samples (8 s @ 200 Hz)

    # ── 输入类型 ───────────────────────────────────────────────────────────────
    input_type    : str  = "fmcw"
    n_range_bins  : int  = 50

    # ── 交叉验证 ───────────────────────────────────────────────────────────────
    cv_strategy   : str  = "loso"   # leave-one-subject-out
    n_folds       : int  = 11       # 11 个受试者，每折留 1 人

    # ── 模型开关（消融实验用） ───────────────────────────────────────────────────
    use_pam       : bool = True
    use_emd       : bool = True

    # ── 训练超参 ───────────────────────────────────────────────────────────────
    batch_size    : int  = 16
    epochs        : int  = 150
    lr            : float= 1e-4
    weight_decay  : float= 1e-4
    grad_clip     : float= 1.0
    warmup_epochs : int  = 5        # 前 5 epoch 只训练 L_recon
    scheduler     : str  = "cosine"
    early_stop_patience: int = 20  # epochs without validation improvement

    # ── 时移鲁棒波形损失（诊断/改进实验用，默认关闭）────────────────────────────
    use_lag_aware_loss: bool = False # True → 在小范围时移内优化 PCC/L1
    lag_max_ms         : float = 100.0
    lambda_lag_pcc     : float = 0.2
    lambda_lag_l1      : float = 0.05
    lambda_zero_pcc    : float = 0.0
    lambda_lag_penalty : float = 0.0
    lag_softmax_tau    : float = 0.05

    # ── 显式输出时延校正（per-segment scalar lag head，默认关闭）──────────────
    use_output_lag_align: bool = False
    output_lag_max_ms    : float = 200.0
    lambda_output_lag_l1 : float = 0.0

    # ── 数据加载 ───────────────────────────────────────────────────────────────
    num_workers      : int  = 4
    balanced_sampling: bool = True           # WeightedRandomSampler
    use_class_balanced_sampling: bool = True # 按 physistatus 类别均衡（旧开关，保留兼容性）
    balance_by       : str  = "subject"      # "subject" | "class"
    narrow_bandpass  : bool = True           # RCG 0.8-3.5 Hz 心跳协带

    # ── 输入表示（B2a 实验） ───────────────────────────────────────────────────
    topk_bins        : int  = 0              # 0 = 用全部 50 bin；>0 = per-(subject,scene) top-K bin selection
    topk_method      : str  = "energy"       # "energy" 0.8-3.5Hz band energy（可部署）|"corr" |Pearson(bin,ECG)| oracle 上界
    target_norm      : str  = "minmax"       # "minmax" → [0,1] sigmoid 输出；"zscore" → z-score 直接预测

    # ── B2b 在线 learnable hard top-K（FMCWRangeEncoder 选择器）─────────────
    fmcw_selector       : str   = "se"         # "se" 软注意力 | "gumbel_topk" 硬选择
    fmcw_topk           : int   = 10           # selector="gumbel_topk" 时的 K
    fmcw_tau_init       : float = 1.0          # Gumbel softmax 起始温度
    fmcw_tau_final      : float = 0.1          # 训练末期温度
    fmcw_tau_anneal     : str   = "linear"     # "linear" | "exp" | "none"

    # ── 实验输出 ───────────────────────────────────────────────────────────────
    exp_dir       : str  = "experiments_mmecg"

    # ── 模型结构 ───────────────────────────────────────────────────────────────
    C             : int  = 64       # encoder base channels
    d_state       : int  = 16       # PAM internal SSM state dim
    emd_max_delay : int  = 20       # EMD FIR 最大延迟（100ms @ 200Hz）
    dropout       : float= 0.1

    # ── 条件扩散解码器 ─────────────────────────────────────────────────────────
    use_diffusion  : bool = False   # True → BeatAwareDiffusionDecoder
    diff_T         : int  = 1000    # 扩散步数 T（v1=100 质量差，v2 改为 1000）
    diff_ddim_steps: int  = 50      # DDIM 采样步数（T=1000 下 50 步质量/速度平衡）
    diff_hidden    : int  = 256     # ResBlock 隐通道数（v1=128 容量不足，v2 翻倍）
    diff_n_blocks  : int  = 8       # ResBlock 层数（v1=6，v2 加深）

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
