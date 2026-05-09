"""
diffusion_decoder.py — BeatAware Conditional Diffusion Decoder

条件信号：
  h_enc      (B, 256, 400)  EMD 对齐后的编码器特征
  rhythm_vec (B, 96)        PAM 全局节律向量（use_pam=True 时）
  peak_masks (B, 3, 1600)   QRS/P/T 峰值空间掩码（use_pam=True 时）

扩散目标：预测噪声 ε（类似 DDPM，x_0 ∈ [-1,1]）
推理路径：DDIM 确定性采样（eta=0），输出转换回 [0,1]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 正弦时步嵌入
# =============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """t (B,) long → (B, embed_dim) via sinusoidal encoding + 2-layer MLP."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )                                           # (half,)
        args  = t.float()[:, None] * freqs[None]   # (B, half)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return self.mlp(emb)


# =============================================================================
# 扩散 ResBlock（FiLM 条件化）
# =============================================================================

class DiffusionResBlock(nn.Module):
    """
    1-D residual block conditioned on `cond` via FiLM.

    in_ch  → GroupNorm → GELU → Conv(k=3) → GroupNorm → FiLM → GELU → Conv(k=3) → + skip
    """

    def __init__(self, in_channels: int, hidden: int = 128, cond_dim: int = 224):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.film  = nn.Linear(cond_dim, hidden * 2)   # → gamma, beta
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.skip  = (
            nn.Conv1d(in_channels, hidden, kernel_size=1)
            if in_channels != hidden else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(F.gelu(self.norm1(x))))
        h = self.norm2(h)
        gamma, beta = self.film(cond).chunk(2, dim=-1)    # (B, hidden) each
        h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        h = self.conv2(F.gelu(h))
        return h + self.skip(x)


# =============================================================================
# BeatAware Diffusion Decoder
# =============================================================================

class BeatAwareDiffusionDecoder(nn.Module):
    """
    Replaces the 2-layer ConvTranspose regression decoder.

    Architecture
    ------------
    upsample : ConvTranspose1d(256, 256, k=8, s=4, p=2) + BN + GELU
               → (B, 256, 1600)   [(400-1)*4 - 4 + 8 = 1600 ✓]
    in_proj  : Conv1d(in_ch, 128, k=1)
               in_ch = 1 + 256 + (3 if use_pam else 0)
    blocks   : n_blocks × DiffusionResBlock(128, 128, cond_dim=224)
               cond = cat(t_emb(128), rhythm_vec(96)) → (B, 224)
               when use_pam=False, rhythm_vec is replaced by zeros
    out_proj : Conv1d(128, 1, k=1)
    """

    def __init__(
        self,
        h_enc_channels: int = 256,
        hidden: int = 128,
        n_blocks: int = 6,
        T: int = 100,
        ddim_steps: int = 20,
        use_pam: bool = True,
    ):
        super().__init__()
        self.T          = T
        self.ddim_steps = ddim_steps
        self.use_pam    = use_pam
        self._hidden    = hidden

        # ── 上采样：(B,256,400) → (B,256,1600) ────────────────────────────────
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(h_enc_channels, h_enc_channels,
                               kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(h_enc_channels),
            nn.GELU(),
        )

        # ── 噪声预测网络 ──────────────────────────────────────────────────────
        self.t_embed  = SinusoidalTimestepEmbedding(hidden)
        cond_dim      = hidden + 96                                  # t_emb + rhythm_vec
        in_ch         = 1 + h_enc_channels + (3 if use_pam else 0)  # x_t + h_up + peaks

        self.in_proj  = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.blocks   = nn.ModuleList(
            [DiffusionResBlock(hidden, hidden, cond_dim) for _ in range(n_blocks)]
        )
        self.out_norm = nn.GroupNorm(8, hidden)
        self.out_proj = nn.Conv1d(hidden, 1, kernel_size=1)

        # ── 噪声调度（余弦，T 步）────────────────────────────────────────────
        self._register_cosine_schedule(T)

    # ── 噪声调度 ──────────────────────────────────────────────────────────────

    def _register_cosine_schedule(self, T: int) -> None:
        """Cosine noise schedule (Nichol & Dhariwal 2021)."""
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-8)                          # (T+1,)
        alpha_bar = alpha_bar[1:]                                        # (T,)
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        betas = torch.cat([torch.tensor([1 - alpha_bar[0].item()]), betas])

        self.register_buffer("betas",           betas.float())
        self.register_buffer("alpha_bar",       alpha_bar.float())
        self.register_buffer("sqrt_alpha_bar",  alpha_bar.sqrt().float())
        self.register_buffer("sqrt_1m_alpha_bar", (1 - alpha_bar).sqrt().float())

    # ── 前向扩散（训练用）────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x0 ∈ [-1,1], t (B,) long → (x_t, eps)."""
        if eps is None:
            eps = torch.randn_like(x0)
        a  = self.sqrt_alpha_bar[t][:, None, None]
        sa = self.sqrt_1m_alpha_bar[t][:, None, None]
        return a * x0 + sa * eps, eps

    # ── 噪声预测网络 forward ──────────────────────────────────────────────────

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_enc: torch.Tensor,
        rhythm_vec: torch.Tensor | None,
        peak_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        x_t        (B, 1, 1600)
        t          (B,) long
        h_enc      (B, 256, 400)
        rhythm_vec (B, 96)  or None
        peak_masks (B, 3, 1600) or None
        → eps_pred (B, 1, 1600)
        """
        h_up = self.upsample(h_enc)                           # (B, 256, 1600)

        parts = [x_t, h_up]
        if self.use_pam and peak_masks is not None:
            parts.append(peak_masks)
        x_in = torch.cat(parts, dim=1)                        # (B, in_ch, 1600)

        h = self.in_proj(x_in)                                # (B, hidden, 1600)

        t_emb = self.t_embed(t)                               # (B, hidden)
        if rhythm_vec is None:
            rhythm_vec = torch.zeros(t.shape[0], 96, device=t.device)
        cond = torch.cat([t_emb, rhythm_vec], dim=-1)         # (B, 224)

        for blk in self.blocks:
            h = blk(h, cond)

        return self.out_proj(F.gelu(self.out_norm(h)))        # (B, 1, 1600)

    # ── 训练步骤 ──────────────────────────────────────────────────────────────

    def training_step(
        self,
        h_enc: torch.Tensor,
        rhythm_vec: torch.Tensor | None,
        peak_masks: torch.Tensor | None,
        ecg_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ecg_gt (B, 1, 1600) in [0,1]
        → (epsilon_pred, epsilon_true)  both (B, 1, 1600)
        """
        B = ecg_gt.shape[0]
        x0 = ecg_gt * 2.0 - 1.0                               # [0,1] → [-1,1]
        t  = torch.randint(0, self.T, (B,), device=ecg_gt.device)
        x_t, eps = self.q_sample(x0, t)
        eps_pred = self.predict_noise(x_t, t, h_enc, rhythm_vec, peak_masks)
        return eps_pred, eps

    # ── DDIM 推理 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        h_enc: torch.Tensor,
        rhythm_vec: torch.Tensor | None,
        peak_masks: torch.Tensor | None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        → ecg_pred (B, 1, 1600) in [0,1]
        """
        B, _, _ = h_enc.shape
        device  = h_enc.device

        # 均匀选 ddim_steps 个时步（降序）
        c = self.T // self.ddim_steps
        timesteps = torch.arange(self.T - 1, -1, -c, device=device)[:self.ddim_steps]

        x = torch.randn(B, 1, 1600, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(B)
            ab_t    = self.alpha_bar[t_batch][:, None, None]

            eps     = self.predict_noise(x, t_batch, h_enc, rhythm_vec, peak_masks)
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(timesteps):
                t_prev  = timesteps[i + 1]
                ab_prev = self.alpha_bar[t_prev.expand(B)][:, None, None]
                x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            else:
                x = x0_pred

        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)             # [-1,1] → [0,1]
