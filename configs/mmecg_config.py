"""
mmecg_config.py — MMECG 数据集训练配置

继承并覆盖 config.py 中的通用参数，适配：
  - FMCW 50 通道输入 (input_type='fmcw')
  - LOSO 11 折交叉验证
  - dataset_mmecg/ 目录结构
"""

from pathlib import Path


class MMECGConfig:
    # ── 数据 ──────────────────────────────────────────────────────────────────
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
    early_stop_patience: int = 20

    # ── 数据加载 ───────────────────────────────────────────────────────────────
    num_workers   : int  = 4
    balanced_sampling: bool = True  # WeightedRandomSampler 均衡受试者

    # ── 实验输出 ───────────────────────────────────────────────────────────────
    exp_dir       : str  = "experiments_mmecg"

    # ── 模型结构（与 Schellenberger 版本一致） ──────────────────────────────────
    C             : int  = 64       # encoder base channels
    d_state       : int  = 16       # Mamba SSM state dim
    emd_max_delay : int  = 20       # EMD FIR 最大延迟（100ms @ 200Hz）
    dropout       : float= 0.1

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
