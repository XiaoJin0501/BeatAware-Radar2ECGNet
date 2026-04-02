"""
tfilm.py — TFiLMGenerator

将来自 PAM Head2 的节律向量映射为 gamma / beta 调制参数，
用于 Multi-scale Encoder 的 TFiLM 特征调制。

初始化为 Identity（gamma=0, beta=0），确保训练初期不干扰主干梯度流。
"""

import torch
import torch.nn as nn


class TFiLMGenerator(nn.Module):
    """
    TFiLM 参数生成器。

    节律向量 rhythm_vec [B, input_dim]
    → gamma [B, output_channels]   (调制增益偏移)
    → beta  [B, output_channels]   (调制偏置)

    调制公式（对特征 f: [B, C, L]）：
        f = f * (1 + gamma[:, :, None]) + beta[:, :, None]

    Parameters
    ----------
    input_dim      : int  节律向量维度（来自 PAM Head2，通常为 96 = 3 × 32）
    output_channels: int  调制目标通道数（通常为 4*C，覆盖 Encoder 全部分支）
    """

    def __init__(self, input_dim: int, output_channels: int):
        super().__init__()
        self.fc_gamma = nn.Linear(input_dim, output_channels)
        self.fc_beta  = nn.Linear(input_dim, output_channels)

        # Identity 初始化：训练开始时 TFiLM 不改变任何特征
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, rhythm_vec: torch.Tensor):
        """
        Parameters
        ----------
        rhythm_vec : Tensor, shape (B, input_dim)

        Returns
        -------
        gamma : Tensor, shape (B, output_channels)
        beta  : Tensor, shape (B, output_channels)
        """
        return self.fc_gamma(rhythm_vec), self.fc_beta(rhythm_vec)
