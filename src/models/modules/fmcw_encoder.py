"""
fmcw_encoder.py — FMCWRangeEncoder（STFT + 共享 2D Conv 时频前端）

设计背景：
  FMCW range-time 矩阵中，50 个 range bin 各自包含不同距离处的回波。
  胸壁所在的 1-3 个 bin 包含心脏信号，其余为杂波。直接使用 1D bandpass
  时域方案存在两个问题：
    (1) bandpass 为线性定频滤波器，对 FMCW 混杂杂波去噪能力有限；
    (2) 50 通道 z-score 后 SE 注意力需要区分"心脏振荡"与"杂波振荡"，
        在时域上两者频谱有重叠，难以分离。

  本版本改用 STFT 时频前端：
    (B, 50, L)
      → torch.stft  [n_fft=64, hop=1]  →  幅度谱 (B*50, 33, L)
      → 共享 2D Conv（所有 range bin 用同一组权重）  →  (B*50, 1, 33, L)
      → 可学习频域 collapse  →  (B*50, 1, 1, L)  →  (B, 50, L)
      → SE 注意力 + 投影 50→3  →  (B, 3, L)

  核心优势：
    - 2D Conv 能识别频-时联合模式（如心跳的频率谱峰随时间的演化），
      比 1D bandpass + temporal conv 的表达能力更强；
    - 共享权重：所有 range bin 使用同一套 2D 核，参数极少（~1.6K）；
    - freq_collapse 是可学习频域加权，训练时自动聚焦心脏频段（0.8–3.5 Hz）；
    - 与 SOTA radarODE 的 SST 路线在设计思路上对齐，但无需额外预处理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMCWRangeEncoder(nn.Module):
    """
    FMCW 50-channel → 3-channel: STFT + 共享 2D Conv 时频前端。

    Parameters
    ----------
    n_range    : int   range bin 数量（默认 50）
    L          : int   输入长度（用于 shape 断言，默认 1600）
    reduction  : int   SE bottleneck 压缩比（默认 8）
    n_fft      : int   STFT FFT 点数（默认 64 → 33 个频率 bin，分辨率 3.125 Hz）
    hop_length : int   STFT 步长（默认 4 → T'=400，2D Conv 在 1/4 时间维度上计算，
                       再 interpolate 回 L；比 hop=1 快 ~4x，20ms 分辨率足够捕捉
                       心脏信号时频特征）
    """

    def __init__(
        self,
        n_range: int = 50,
        L: int = 1600,
        reduction: int = 8,
        n_fft: int = 64,
        hop_length: int = 4,
    ):
        super().__init__()
        self.n_range    = n_range
        self.n_fft      = n_fft
        self.hop_length = hop_length
        n_freq = n_fft // 2 + 1   # 33

        # ── 共享 2D Conv Block ─────────────────────────────────────────
        # 所有 range bin 使用同一组卷积核（物理上合理：心脏信号有相同时频结构）
        # k=(9,7): 覆盖 9 个 freq bins (~28 Hz) × 7 个时间步 (~35ms @ 200Hz)
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=(9, 7), padding=(4, 3), bias=False)
        self.bn2d_1   = nn.BatchNorm2d(8)
        self.conv2d_2 = nn.Conv2d(8, 1, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.bn2d_2   = nn.BatchNorm2d(1)

        # ── 可学习频域 collapse: 33 freq bins → 1 ────────────────────
        # Conv2d(1,1, k=(n_freq,1)) 等价于对 33 个 freq bin 做加权求和
        self.freq_collapse = nn.Conv2d(1, 1, kernel_size=(n_freq, 1), bias=True)

        # ── BN over range bins ───────────────────────────────────────
        self.range_bn = nn.BatchNorm1d(n_range)

        # ── SE 空间注意力（选取心脏信号最强的 range bins）────────────
        se_mid = max(n_range // reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1  = nn.Linear(n_range, se_mid)
        self.se_fc2  = nn.Linear(se_mid, n_range)

        # ── 投影 50 → 3 ──────────────────────────────────────────────
        self.proj    = nn.Conv1d(n_range, 3, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(3)

        self._init_weights()

    def _init_weights(self):
        # freq_collapse：均匀初始化（训练初期对所有频段平等加权）
        nn.init.constant_(self.freq_collapse.weight, 1.0 / (self.n_fft // 2 + 1))
        nn.init.zeros_(self.freq_collapse.bias)
        # proj：均匀平均初始化（所有 range bin 初始贡献相同）
        nn.init.constant_(self.proj.weight, 1.0 / self.n_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 50, L)  — 预处理后的 range-time 信号（0.5-40 Hz bandpass + z-score）

        Returns
        -------
        out : (B, 3, L)  — 3 通道心脏信号聚合
        """
        B, R, L = x.shape
        assert R == self.n_range, f"Expected {self.n_range} range bins, got {R}"

        # ── Step 1: Batched STFT ────────────────────────────────────
        # (B, 50, L) → flatten → (B*50, L) → stft → (B*50, n_freq, T')
        x_flat = x.reshape(B * R, L)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec   = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )                                   # (B*R, n_freq, T')
        mag    = spec.abs()                 # (B*R, n_freq, T')
        T_stft = mag.size(-1)

        # ── Step 2: 共享 2D Conv ────────────────────────────────────
        feat = mag.unsqueeze(1)                                # (B*R, 1, n_freq, T')
        feat = F.gelu(self.bn2d_1(self.conv2d_1(feat)))       # (B*R, 8, n_freq, T')
        feat = F.gelu(self.bn2d_2(self.conv2d_2(feat)))       # (B*R, 1, n_freq, T')

        # ── Step 3: 可学习频域 collapse ─────────────────────────────
        feat = self.freq_collapse(feat)                        # (B*R, 1, 1, T')
        feat = feat.squeeze(1).squeeze(1)                      # (B*R, T')

        # ── Step 4: Reshape + upsample to L ──────────────────────────
        feat = feat.reshape(B, R, T_stft)                     # (B, 50, T')
        if T_stft != L:
            # 线性插值回输入长度（hop_length>1 时 T' < L）
            feat = F.interpolate(feat, size=L, mode='linear', align_corners=False)

        feat = F.gelu(self.range_bn(feat))                    # (B, 50, L)

        # ── Step 5: SE 空间注意力 ────────────────────────────────────
        attn = self.se_pool(feat).squeeze(-1)                 # (B, 50)
        attn = F.relu(self.se_fc1(attn))
        attn = torch.sigmoid(self.se_fc2(attn))
        feat = feat * attn.unsqueeze(-1)                      # (B, 50, L)

        # ── Step 6: 投影 50 → 3 ─────────────────────────────────────
        out  = self.proj_bn(self.proj(feat))                  # (B, 3, L)
        return out
