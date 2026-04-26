"""
fmcw_encoder.py — FMCWRangeEncoder（时域方案，配合 0.8-3.5Hz 窄带预处理）

设计背景：
  FMCW range-time 矩阵的主功率集中在 13–16 Hz（静态杂波拍频），
  心脏成分（0.8–3.5 Hz）是叠加在上面幅度约 1/10 的调制信号。
  预处理已使用 0.8–3.5 Hz 窄带滤波将 RCG-ECG 相关性从 0.06 提升至 0.63，
  本编码器在此基础上进行空间聚合：

    (B, 50, L)  — 预处理后的窄带 range-time 信号
      → 时域滤波（大核 k=61，~305ms，进一步聚焦心脏频段）
      → SE 空间注意力（选取心脏信号最强的 range bins）
      → 1×1 卷积投影 50 → 3

  激活函数使用 GELU（非 ReLU），保留心脏 AC 信号的负半周期信息。
  输出 (B, 3, L) 直接替代原来 raw/phase 路径的 KI 输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMCWRangeEncoder(nn.Module):
    """
    FMCW 50-channel → 3-channel cardiac signal aggregation.

    Parameters
    ----------
    n_range   : int  range bin 数量（默认 50，与 MMECG 一致）
    L         : int  输入长度（仅用于 shape 断言）
    reduction : int  SE bottleneck 压缩比（默认 8）
    """

    def __init__(self, n_range: int = 50, L: int = 1600, reduction: int = 8):
        super().__init__()
        self.n_range = n_range

        # ── 逐通道时域滤波（感受野 61 × 5ms = 305ms ≈ 1 个心动周期）──────
        # 学习在 0.8-3.5 Hz 窄带内进一步聚焦，抑制带内残余噪声
        self.temporal_filter = nn.Conv1d(
            n_range, n_range,
            kernel_size=61, padding=30,
            groups=n_range, bias=False
        )
        self.temporal_bn = nn.BatchNorm1d(n_range)

        # ── SE 空间注意力（对 50 range bins 加权）──────────────────────────
        # 自适应识别胸壁回波最强的 range bins
        se_mid = max(n_range // reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1  = nn.Linear(n_range, se_mid)
        self.se_fc2  = nn.Linear(se_mid, n_range)

        # ── 投影 50 → 3（学习 3 种互补的 range bin 加权组合）──────────────
        self.proj    = nn.Conv1d(n_range, 3, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(3)

        self._init_weights()

    def _init_weights(self):
        # temporal_filter：初始化为近似恒等（中心 tap=1，其余=0）
        nn.init.zeros_(self.temporal_filter.weight)
        center = self.temporal_filter.kernel_size[0] // 2
        with torch.no_grad():
            self.temporal_filter.weight[:, 0, center] = 1.0

        # proj：均匀平均初始化
        nn.init.constant_(self.proj.weight, 1.0 / self.n_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 50, L)  — 预处理后的 range-time 信号（0.8-3.5 Hz + z-score）

        Returns
        -------
        out : (B, 3, L)  — 3 通道心脏信号聚合
        """
        B, R, L = x.shape
        assert R == self.n_range, f"Expected {self.n_range} range bins, got {R}"

        # ── Step 1: 逐通道时域滤波 ──────────────────────────────────────
        # GELU 保留负半周期（心脏 AC 信号），不同于原来的 ReLU
        x_filt = F.gelu(self.temporal_bn(self.temporal_filter(x)))  # (B, 50, L)

        # ── Step 2: SE 空间注意力 ────────────────────────────────────────
        attn = self.se_pool(x_filt).squeeze(-1)          # (B, 50)
        attn = F.relu(self.se_fc1(attn))                 # (B, se_mid)
        attn = torch.sigmoid(self.se_fc2(attn))          # (B, 50)
        x_att = x_filt * attn.unsqueeze(-1)              # (B, 50, L)

        # ── Step 3: 投影 50 → 3 ─────────────────────────────────────────
        out = self.proj_bn(self.proj(x_att))             # (B, 3, L)  无激活函数

        return out
