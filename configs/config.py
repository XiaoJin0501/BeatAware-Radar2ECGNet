"""
config.py — BeatAware-Radar2ECGNet 实验配置

用法：
  from configs.config import Config, get_config
  cfg = get_config()                           # 从命令行解析
  cfg = Config(exp_tag="debug", epochs=5)      # 直接构造
"""

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────
    dataset_dir: str  = "dataset"
    input_type:  str  = "phase"          # 'raw' | 'phase' | 'spec'
    scenarios:   list = field(default_factory=lambda: ["resting", "valsalva", "apnea"])

    # ── 模型 ──────────────────────────────────────────────────────────────
    C:             int   = 64
    d_state:       int   = 16
    use_pam:       bool  = True
    use_emd:       bool  = True   # Phase C EMD 对齐层（False = Model A/B）
    emd_max_delay: int   = 20     # EMD 最大时移（采样点，20 = 100ms @ 200Hz）
    dropout:       float = 0.1

    # ── Loss 权重（V2 自适应加权，alpha/beta 由 log_vars 取代）─────────────
    alpha_stft:     float = 0.1  # STFT loss 在 L_recon 内的固定权重
    warmup_epochs:  int   = 5    # 前 N epoch 只训练 L_recon，log_vars 解冻后开始自适应

    # ── 训练 ──────────────────────────────────────────────────────────────
    epochs:       int   = 150
    batch_size:   int   = 32
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    num_workers:  int   = 4

    # ── Cross-Validation ──────────────────────────────────────────────────
    n_folds:     int  = 5
    fold_idx:    int  = -1       # -1 表示运行全部 5 folds

    # ── 随机种子 ──────────────────────────────────────────────────────────
    seed:        int  = 42

    # ── 实验管理 ──────────────────────────────────────────────────────────
    exp_tag:     str  = "default"
    exp_dir:     str  = "experiments"
    device:      str  = "cuda"

    # ── 日志频率 ──────────────────────────────────────────────────────────
    log_every:   int  = 10       # 每 N step 打一次 train loss
    val_every:            int  = 1        # 每 N epoch 做一次 val
    f1_every:             int  = 10       # 每 N epoch 计算一次 R 峰 F1（耗时）
    early_stop_patience:  int  = 20       # val_pcc 连续 N epoch 无提升则停止训练

    # ── 路径（运行时自动推导，无需手动设置）──────────────────────────────
    @property
    def exp_root(self) -> Path:
        """experiments/<exp_tag>/"""
        return Path(self.exp_dir) / self.exp_tag

    def fold_dir(self, fold: int) -> Path:
        """experiments/<exp_tag>/fold_<N>/"""
        return self.exp_root / f"fold_{fold}"

    def ckpt_dir(self, fold: int) -> Path:
        return self.fold_dir(fold) / "checkpoints"

    def log_dir(self, fold: int) -> Path:
        return self.fold_dir(fold) / "logs"

    def result_dir(self, fold: int) -> Path:
        return self.fold_dir(fold) / "results"


def get_config() -> Config:
    """
    从命令行参数解析 Config，支持所有字段覆盖。

    用法示例：
        python scripts/train.py --exp_tag ExpB_phase --input_type phase --epochs 150
    """
    cfg = Config()
    parser = argparse.ArgumentParser(description="BeatAware-Radar2ECGNet 训练配置")

    for f in fields(cfg):
        val = getattr(cfg, f.name)
        t   = type(val)

        if isinstance(val, bool):
            # bool 需要特殊处理（argparse 把字符串当 bool）
            parser.add_argument(
                f"--{f.name}",
                type=lambda x: x.lower() in ("true", "1", "yes"),
                default=val,
                metavar="BOOL",
            )
        elif isinstance(val, list):
            parser.add_argument(
                f"--{f.name}",
                nargs="+",
                default=val,
            )
        else:
            parser.add_argument(f"--{f.name}", type=t, default=val)

    args = parser.parse_args()
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(args, f.name))
    return cfg
