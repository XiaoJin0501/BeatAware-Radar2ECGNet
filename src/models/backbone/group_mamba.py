"""
group_mamba.py — GroupMambaBlock

结构：
    LayerNorm → 分 num_groups 组 → 各组独立 VSSSBlock1D → Concat
    → CAM（通道注意力门控）→ 投影 → 残差

CAM (Channel Affine Modulation)：
    AvgPool → Conv1d(1×1, C//4) → ReLU → Conv1d(1×1, C) → Sigmoid
"""

import torch
import torch.nn as nn

from .ssm import VSSSBlock1D


class GroupMambaBlock(nn.Module):
    """
    GroupMambaBlock：通道分组后独立 SSM 序列建模，CAM 调制后汇聚。

    Parameters
    ----------
    d_model    : int   输入/输出通道数（必须能被 num_groups 整除）
    num_groups : int   分组数（默认4）
    d_state    : int   VSSSBlock1D 的 SSM 状态维度（默认16）
    """

    def __init__(self, d_model: int, num_groups: int = 4, d_state: int = 16):
        super().__init__()
        assert d_model % num_groups == 0, (
            f"d_model ({d_model}) must be divisible by num_groups ({num_groups})"
        )
        self.num_groups = num_groups
        self.group_dim  = d_model // num_groups

        self.norm = nn.LayerNorm(d_model)

        self.ssm_blocks = nn.ModuleList([
            VSSSBlock1D(self.group_dim, d_state=d_state)
            for _ in range(num_groups)
        ])

        # Channel Affine Modulation
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 4, d_model, kernel_size=1),
            nn.Sigmoid(),
        )

        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, d_model, L)

        Returns
        -------
        Tensor, shape (B, d_model, L)
        """
        residual = x

        # LayerNorm（转置到最后一维处理）
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)   # (B, d_model, L)

        # CAM 权重（在 SSM 之前计算，用输入特征）
        w = self.cam(x)    # (B, d_model, 1)

        # 分组 SSM
        groups = torch.chunk(x, self.num_groups, dim=1)    # num_groups × (B, group_dim, L)
        outs   = [block(g) for block, g in zip(self.ssm_blocks, groups)]
        x = torch.cat(outs, dim=1)   # (B, d_model, L)

        # CAM 调制
        x = x * w

        # 投影 + 残差
        return self.proj(x) + residual
