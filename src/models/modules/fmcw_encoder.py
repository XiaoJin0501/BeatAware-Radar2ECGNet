"""
fmcw_encoder.py — FMCWRangeEncoder（时域方案，配合 0.8-3.5Hz 窄带预处理）

设计背景：
  FMCW range-time 矩阵的主功率集中在 13–16 Hz（静态杂波拍频），
  心脏成分（0.8–3.5 Hz）是叠加在上面幅度约 1/10 的调制信号。
  预处理已使用 0.8–3.5 Hz 窄带滤波将 RCG-ECG 相关性从 0.06 提升至 0.63，
  本编码器在此基础上进行空间聚合：

    (B, 50, L)  — 预处理后的窄带 range-time 信号
      → 时域滤波（大核 k=61，~305ms，进一步聚焦心脏频段）
      → 空间选择器（SE 软注意力 / Gumbel 硬 top-K，由 selector 控制）
      → 1×1 卷积投影 50 → 3

  激活函数使用 GELU（非 ReLU），保留心脏 AC 信号的负半周期信息。
  输出 (B, 3, L) 直接替代原来 raw/phase 路径的 KI 输出。

Selector modes:
  - "se" (默认)        : 现有 SE soft attention，input-conditional 但权重密集
  - "gumbel_topk"      : Gumbel-softmax + straight-through K-hot mask，
                         forward 只用 K 个 bin、backward 梯度传所有 50 个 logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMCWRangeEncoder(nn.Module):
    """
    FMCW n-channel → 3-channel cardiac signal aggregation.

    Parameters
    ----------
    n_range   : int   range bin 数量（默认 50；offline top-K 时为 K）
    L         : int   输入长度（仅用于 shape 断言）
    reduction : int   SE bottleneck 压缩比（默认 8）
    selector  : str   "se" | "gumbel_topk"
    topk      : int   selector="gumbel_topk" 时的 K 值（默认 10）
    tau       : float Gumbel softmax 温度（训练循环里按 epoch 衰减；初始默认 1.0）
    """

    def __init__(
        self,
        n_range: int = 50,
        L: int = 1600,
        reduction: int = 8,
        selector: str = "se",
        topk: int = 10,
        tau: float = 1.0,
    ):
        super().__init__()
        if selector not in ("se", "gumbel_topk"):
            raise ValueError(f"Unknown selector: {selector!r}")
        if selector == "gumbel_topk" and topk >= n_range:
            raise ValueError(
                f"gumbel_topk requires topk < n_range, got topk={topk} n_range={n_range}"
            )

        self.n_range = n_range
        self.selector = selector
        self.topk = topk
        # tau 是 buffer 而不是 Parameter，训练循环可直接赋值
        self.register_buffer("tau", torch.tensor(float(tau)))

        # ── 逐通道时域滤波（感受野 61 × 5ms = 305ms ≈ 1 个心动周期）──────
        # 学习在 0.8-3.5 Hz 窄带内进一步聚焦，抑制带内残余噪声
        self.temporal_filter = nn.Conv1d(
            n_range, n_range,
            kernel_size=61, padding=30,
            groups=n_range, bias=False
        )
        self.temporal_bn = nn.BatchNorm1d(n_range)

        # ── 空间选择器分支 ──────────────────────────────────────────────
        se_mid = max(n_range // reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        if selector == "se":
            # SE soft attention（保留原行为）
            self.se_fc1  = nn.Linear(n_range, se_mid)
            self.se_fc2  = nn.Linear(se_mid, n_range)
        else:
            # gumbel_topk: 2-layer logit MLP
            self.logit_fc1 = nn.Linear(n_range, se_mid)
            self.logit_fc2 = nn.Linear(se_mid, n_range)

        # ── 投影 n_range → 3（学习 3 种互补的 range bin 加权组合）─────────
        self.proj    = nn.Conv1d(n_range, 3, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(3)

        self._init_weights()

    def _init_weights(self):
        # temporal_filter：初始化为近似恒等（中心 tap=1，其余=0）
        nn.init.zeros_(self.temporal_filter.weight)
        center = self.temporal_filter.kernel_size[0] // 2
        with torch.no_grad():
            self.temporal_filter.weight[:, 0, center] = 1.0

        # logit MLP：小权重初始化，让初始 logits ≈ 0 ⇒ 初始选择接近 uniform
        if self.selector == "gumbel_topk":
            nn.init.xavier_uniform_(self.logit_fc1.weight, gain=0.1)
            nn.init.zeros_(self.logit_fc1.bias)
            nn.init.xavier_uniform_(self.logit_fc2.weight, gain=0.1)
            nn.init.zeros_(self.logit_fc2.bias)

        # proj：均匀平均初始化
        nn.init.constant_(self.proj.weight, 1.0 / self.n_range)

    def set_tau(self, tau: float) -> None:
        """训练循环用：每 epoch 调用以退火 tau。"""
        self.tau.fill_(float(tau))

    def _gumbel_topk_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, R) → mask: (B, R)，forward hard K-hot，backward soft 梯度可传。

        Eval 模式下不加 Gumbel 噪声，纯 deterministic top-K。
        """
        if self.training:
            # 加 Gumbel(0, 1) 噪声做 stochastic top-K
            # g = -log(-log(U)) where U ~ Uniform(0, 1); clamp U away from 0 and 1.
            u = torch.rand_like(logits).clamp(1e-9, 1.0 - 1e-9)
            gumbel = -torch.log(-torch.log(u))
            scores = (logits + gumbel) / self.tau
        else:
            # 推理：确定性，不加噪
            scores = logits / self.tau
        soft = torch.softmax(scores, dim=-1)
        _, top_idx = soft.topk(self.topk, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1, top_idx, 1.0)
        # Straight-through: forward hard, backward soft
        return (hard - soft).detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_range, L)  — 预处理后的 range-time 信号（0.8-3.5 Hz + z-score）

        Returns
        -------
        out : (B, 3, L)  — 3 通道心脏信号聚合
        """
        B, R, L = x.shape
        assert R == self.n_range, f"Expected {self.n_range} range bins, got {R}"

        # ── Step 1: 逐通道时域滤波 ──────────────────────────────────────
        # GELU 保留负半周期（心脏 AC 信号），不同于原来的 ReLU
        x_filt = F.gelu(self.temporal_bn(self.temporal_filter(x)))  # (B, n_range, L)

        # ── Step 2: 空间选择 ─────────────────────────────────────────────
        pooled = self.se_pool(x_filt).squeeze(-1)        # (B, n_range)
        if self.selector == "se":
            attn = F.relu(self.se_fc1(pooled))           # (B, se_mid)
            mask = torch.sigmoid(self.se_fc2(attn))      # (B, n_range)
        else:  # gumbel_topk
            logits = self.logit_fc2(F.relu(self.logit_fc1(pooled)))  # (B, n_range)
            mask = self._gumbel_topk_mask(logits)        # (B, n_range)
        x_att = x_filt * mask.unsqueeze(-1)              # (B, n_range, L)

        # ── Step 3: 投影 n_range → 3 ────────────────────────────────────
        out = self.proj_bn(self.proj(x_att))             # (B, 3, L)  无激活函数

        return out
