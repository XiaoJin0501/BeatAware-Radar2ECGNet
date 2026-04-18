"""
peak_module.py — Peak Auxiliary Module (PAM)

核心作用：
  1. 显式检测 QRS/P/T 波位置 → 3 路峰值 Mask（监督信号：高斯软标签）
  2. 提取节律向量 → rhythm_vec [B, PAM_DIM]（传给 TFiLMGenerator 驱动主干调制）

V2 改动（Phase B）：
  - Head1 从 1 路（QRS）扩展为 3 路：Head_QRS / Head_P / Head_T
  - 返回值改为 (qrs_mask, p_mask, t_mask) 元组 + rhythm_vec

结构（1D 输入路径）：
    [B, 3, L]  （V2：原始 + 速度 + 加速度）
    → Multi-scale Conv1d (k=7, 15, 31, 各 pam_channels=32 通道) + BN + ReLU
    → Concat → [B, 96, L]
    → VSSSBlock1D × 2
    → LayerNorm
       ├── Head_QRS: Conv1d(96→1) + Sigmoid → qrs_mask [B, 1, L]
       ├── Head_P:   Conv1d(96→1) + Sigmoid → p_mask   [B, 1, L]
       ├── Head_T:   Conv1d(96→1) + Sigmoid → t_mask   [B, 1, L]
       └── Head2:    AdaptiveMaxPool1d(1) → Flatten → rhythm_vec [B, 96]

Spec 输入路径（input_type='spec'）：
    [B, 1, F_spec, T_spec]（spec 路径不做 diff，保持原始 1 通道输入）
    → Conv2d(1, 96, (F_spec, 1)) → squeeze(2) → [B, 96, T]
    → interpolate(size=L) → [B, 96, L]
    → VSSSBlock1D × 2 → LayerNorm → Head_QRS/P/T / Head2（同上）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.ssm import VSSSBlock1D


class PeakAuxiliaryModule(nn.Module):
    """
    PAM — 峰值检测辅助模块。

    Parameters
    ----------
    input_type    : str  '1d' (radar_raw/phase) | 'spec' (radar_spec_input)
    pam_channels  : int  每路多尺度分支通道数（默认32，3路共96）
    signal_len    : int  目标信号长度；spec 输入时插值到此维度（默认1600）
    spec_freq_bins: int  spec 输入的频率 bins 数（默认33，对应 nperseg=64）
    d_state       : int  VSSSBlock1D 的 SSM 状态维度（默认16）
    """

    def __init__(
        self,
        input_type:     str = "1d",
        pam_channels:   int = 32,
        signal_len:     int = 1600,
        spec_freq_bins: int = 33,
        d_state:        int = 16,
    ):
        super().__init__()
        self.input_type = input_type
        self.signal_len = signal_len
        pam_total = pam_channels * 3    # 96

        if input_type == "spec":
            # 频率轴压缩：(1, F, T) → (96, 1, T) → squeeze → (96, T) → interpolate
            self.spec_proj = nn.Conv2d(1, pam_total, kernel_size=(spec_freq_bins, 1), bias=False)
            self.spec_bn   = nn.BatchNorm2d(pam_total)
        else:
            # 多尺度卷积（k=7, 15, 31，padding=k//2 保持长度）
            # V2: in_channels=3（原始 + 速度 + 加速度三通道导数输入）
            kernels = [7, 15, 31]
            self.ms_convs = nn.ModuleList([
                nn.Conv1d(3, pam_channels, kernel_size=k, padding=k // 2, bias=True)
                for k in kernels
            ])
            self.ms_bns = nn.ModuleList([
                nn.BatchNorm1d(pam_channels) for _ in kernels
            ])

        # SSM 序列建模（×2）
        self.ssm1 = VSSSBlock1D(pam_total, d_state=d_state)
        self.ssm2 = VSSSBlock1D(pam_total, d_state=d_state)
        self.norm = nn.LayerNorm(pam_total)

        # V2 三路峰值 Mask 检测头
        # Head_QRS：R 峰（QRS 复合波）检测，σ=5（25ms）
        # Head_P：P 波检测，σ=10（50ms）
        # Head_T：T 波检测，σ=15（75ms）
        def _make_head():
            return nn.Sequential(nn.Conv1d(pam_total, 1, kernel_size=1), nn.Sigmoid())

        self.head_qrs = _make_head()
        self.head_p   = _make_head()
        self.head_t   = _make_head()

        # Head2：节律向量（全局最大池化）
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor
            '1d'  : (B, 3, L)  — [原始, 速度, 加速度] 三通道（V2 derivative encoder）
            'spec': (B, 1, F_spec, T_spec)

        Returns
        -------
        peak_masks : tuple (qrs_mask, p_mask, t_mask)
                     各 Tensor (B, 1, L)，值域 [0, 1]
        rhythm_vec : Tensor, (B, 96)    —— 节律特征向量（传给 TFiLMGenerator）
        """
        if self.input_type == "spec":
            # (B, 1, F, T) → Conv2d → (B, 96, 1, T) → BN → squeeze → (B, 96, T)
            feat = F.relu(self.spec_bn(self.spec_proj(x)))  # (B, 96, 1, T)
            feat = feat.squeeze(2)                           # (B, 96, T)
            # 插值到目标信号长度
            feat = F.interpolate(feat, size=self.signal_len, mode="linear", align_corners=False)
        else:
            # 三路多尺度卷积 + BN + ReLU，各输出 (B, 32, L)
            outs = [F.relu(bn(conv(x))) for conv, bn in zip(self.ms_convs, self.ms_bns)]
            feat = torch.cat(outs, dim=1)    # (B, 96, L)

        # VSSSBlock1D × 2
        feat = self.ssm1(feat)
        feat = self.ssm2(feat)

        # LayerNorm（沿通道维）
        feat = self.norm(feat.transpose(1, 2)).transpose(1, 2)   # (B, 96, L)

        # 三路峰值 Mask
        qrs_mask = self.head_qrs(feat)   # (B, 1, L)
        p_mask   = self.head_p(feat)     # (B, 1, L)
        t_mask   = self.head_t(feat)     # (B, 1, L)

        # Head2：节律向量
        rhythm_vec = self.pool(feat).squeeze(-1)   # (B, 96)

        return (qrs_mask, p_mask, t_mask), rhythm_vec
