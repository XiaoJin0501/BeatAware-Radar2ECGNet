"""
RSSRnet (Wu et al. 2024, IEEE Sensors J.) — paper-faithful reproduction, ECG-only.

Reference:
    Wu et al. "Contactless reconstruction of ECG and respiration signals with
    mmWave radar based on RSSRnet," IEEE Sensors J., vol. 25, no. 14, 2024.

Architecture matches paper Section III + IV.D (Implement Details):
    Encoder : 4 layers, C=32 (first filter count), kernel (5,2), stride (2,1)
              channels: 32 → 64 → 128 → 256 (×2 per layer)
              freq:     64 → 32 → 16  → 8   (/2 per layer except layer-1 which keeps freq)
              Each block = Conv2D + BN + CA + PReLU (paper Fig. 4)
    Separator: Conv1D(2048→D=512) → 4× TransformerBlock (heads=4, FFN=1024) → Conv1D(D→2048)
    Decoder : 4 layers (mirror of encoder), with multi-scale skip connections via CCA
              Each block = CCA(skip, dec_in) → ConvTranspose2D + BN + PReLU
    Output  : 1×1 Conv2D → 2 channels (real+imag STFT)

ECG-only: paper has dual-branch (ECG + RSP), we drop RSP branch.

Plan §9 deviation notes:
    Kernel (5,2) replaced with (5,3) for clean time-dim preservation (see W8).
    Other deviations W1-W7 already documented in plan.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Channel Attention (CA) — paper Fig. 5, Eq. 4
# =============================================================================

class ChannelAttention(nn.Module):
    """CA: avg-pool + max-pool → shared MLP → sigmoid → channel scale."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.PReLU(),
            nn.Linear(hidden, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx  = self.mlp(self.max_pool(x).view(b, c))
        w = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * w


# =============================================================================
# Cross-Channel Attention (CCA) — paper Fig. 6
# =============================================================================

class CrossChannelAttention(nn.Module):
    """CCA: separate attention weights for skip + decoder input, then fuse.

    Per paper Section III.C / Fig. 6:
        w_skip = MLP(maxpool(skip)) + MLP(avgpool(skip))
        w_dec  = MLP(maxpool(dec))  + MLP(avgpool(dec))
        out    = sigmoid(w_skip + w_dec) * (skip + dec)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_skip = nn.Sequential(
            nn.Linear(channels, hidden), nn.PReLU(), nn.Linear(hidden, channels),
        )
        self.mlp_dec = nn.Sequential(
            nn.Linear(channels, hidden), nn.PReLU(), nn.Linear(hidden, channels),
        )

    def forward(self, skip: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        # skip, dec_in: (B, C, F, T) — same shape required
        b, c, _, _ = skip.shape
        w_skip = self.mlp_skip(self.avg_pool(skip).view(b, c)) \
               + self.mlp_skip(self.max_pool(skip).view(b, c))
        w_dec  = self.mlp_dec(self.avg_pool(dec_in).view(b, c)) \
               + self.mlp_dec(self.max_pool(dec_in).view(b, c))
        w = torch.sigmoid(w_skip + w_dec).view(b, c, 1, 1)
        return w * (skip + dec_in)


# =============================================================================
# Encoder block: Conv2D + BN + CA + PReLU (paper Fig. 4 left)
# =============================================================================

class EncoderBlock(nn.Module):
    """Encoder block: 2D conv (kernel (5,3), padding (2,1)) with freq downsample option."""

    def __init__(self, in_ch: int, out_ch: int, freq_downsample: bool = True):
        super().__init__()
        stride = (2, 1) if freq_downsample else (1, 1)
        # Kernel (5,3) with padding (2,1): keeps time exact; freq /2 if stride=2 else preserved
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(5, 3), stride=stride, padding=(2, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.ca = ChannelAttention(out_ch, reduction=8)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.ca(x)
        return self.act(x)


# =============================================================================
# Decoder block: CCA(skip, dec_in) → ConvTranspose2D + BN + PReLU (paper Fig. 4 right)
# =============================================================================

class DecoderBlock(nn.Module):
    """Decoder block: CCA(skip, dec) → ConvTranspose2D + BN + PReLU."""

    def __init__(self, in_ch: int, out_ch: int, freq_upsample: bool = True):
        super().__init__()
        self.cca = CrossChannelAttention(in_ch, reduction=8)
        stride = (2, 1) if freq_upsample else (1, 1)
        # output_padding=(1,0) when upsampling freq: ConvTranspose formula gives 2(F-1)-4+5=2F-1 (odd),
        # need +1 to recover 2F (the encoder's pre-downsample freq).
        op_f = 1 if freq_upsample else 0
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(5, 3), stride=stride,
            padding=(2, 1), output_padding=(op_f, 0),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, dec_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # dec_in and skip must have the same shape (B, in_ch, F, T)
        x = self.cca(skip, dec_in)
        x = self.deconv(x)
        x = self.bn(x)
        return self.act(x)


# =============================================================================
# Transformer block — paper Eqs. 6-7 (pre-norm)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 4, ffn_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h


# =============================================================================
# Main model: paper-faithful RSSRnet (ECG-only)
# =============================================================================

class RSSRnetECG(nn.Module):
    """RSSRnet ECG-only branch, matching paper Section IV.D specs.

    Args (paper defaults):
        in_channels   : 2 (STFT real+imag)
        freq_bins     : 64 (paper: first 64 freq bins of n_fft=256 STFT)
        C             : 32 (first encoder filter count)
        n_enc_layers  : 4
        d_model       : 512 (separator self-attn dim D)
        n_heads       : 4
        ffn_dim       : 1024
        sep_depth     : 4 (separator transformer blocks)
        out_channels  : 2 (STFT real+imag of reconstructed ECG)

    Forward I/O:
        in : (B, 2, 64, T)  STFT real+imag of radar
        out: (B, 2, 64, T)  STFT real+imag of ECG
    """

    def __init__(
        self,
        in_channels: int = 2,
        freq_bins: int = 64,
        C: int = 32,
        n_enc_layers: int = 4,
        d_model: int = 512,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        sep_depth: int = 4,
        out_channels: int = 2,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.C = C
        self.n_enc_layers = n_enc_layers
        # Encoder channel/freq schedule (paper formula: O_i ∈ R^(C·2^(i-1) × F/2^(i-1) × T)):
        #   Layer 1: 2 → C (no freq downsample, keeps freq=F)
        #   Layer 2: C → 2C (freq /2)
        #   Layer 3: 2C → 4C (freq /2)
        #   Layer 4: 4C → 8C (freq /2)
        # Final bottleneck: (8C, F/8, T) = (256, 8, T) for C=32, F=64.
        enc_channels = [in_channels] + [C * (2 ** i) for i in range(n_enc_layers)]
        # → [2, 32, 64, 128, 256]
        self.encoder = nn.ModuleList()
        for i in range(n_enc_layers):
            # Layer 1 does NOT downsample freq (only sets up channels)
            freq_ds = (i > 0)
            self.encoder.append(EncoderBlock(enc_channels[i], enc_channels[i + 1], freq_downsample=freq_ds))

        # Bottleneck shape
        self.enc_out_ch = enc_channels[-1]                            # 256
        self.enc_out_freq = freq_bins // (2 ** (n_enc_layers - 1))    # 8

        # ── Separator: Conv1D → D=512 → 4× transformer → Conv1D → original dim
        flat_dim = self.enc_out_ch * self.enc_out_freq                # 2048
        self.sep_in = nn.Conv1d(flat_dim, d_model, kernel_size=1)
        self.separator = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(sep_depth)]
        )
        self.sep_out = nn.Conv1d(d_model, flat_dim, kernel_size=1)

        # ── Decoder: 4 layers (mirror of encoder), each with CCA-based skip fusion
        # In channels follow encoder reversed: [256, 128, 64, 32]
        # Out channels keep encoder ch pattern but last block holds 32 (head does the 32→2):
        #   dec[0]: 256 → 128   (freq 8 → 16)
        #   dec[1]: 128 → 64    (freq 16 → 32)
        #   dec[2]: 64 → 32     (freq 32 → 64)
        #   dec[3]: 32 → 32     (freq 64, no upsample; mirrors encoder layer 1)
        dec_in_channels  = list(reversed(enc_channels))[:n_enc_layers]   # [256, 128, 64, 32]
        dec_out_channels = list(reversed(enc_channels))[1:n_enc_layers]  # [128, 64, 32]
        dec_out_channels.append(C)                                        # [128, 64, 32, 32]
        self.decoder = nn.ModuleList()
        for i in range(n_enc_layers):
            # Last decoder layer (i = n_enc_layers - 1) does NOT upsample freq (mirrors enc layer 1).
            freq_us = (i < n_enc_layers - 1)
            self.decoder.append(DecoderBlock(dec_in_channels[i], dec_out_channels[i], freq_upsample=freq_us))

        # ── Final 1×1 conv to 2 channels (paper: "1×1 2-D convolution to compress to 2")
        self.head = nn.Conv2d(C, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 64, T)
        # ── Encode + collect skips ───────────────────────────────────
        skips = []
        h = x
        for blk in self.encoder:
            h = blk(h)
            skips.append(h)              # skips[0..3] from shallow → deep
        # h: (B, 256, 8, T)

        # ── Separator: flatten ch×freq, apply transformer over time
        B, C, Fh, T = h.shape
        tokens = h.reshape(B, C * Fh, T)                  # (B, 2048, T)
        tokens = self.sep_in(tokens)                       # (B, 512, T)
        tokens = tokens.permute(0, 2, 1)                   # (B, T, 512)
        for block in self.separator:
            tokens = block(tokens)
        tokens = tokens.permute(0, 2, 1)                   # (B, 512, T)
        tokens = self.sep_out(tokens)                      # (B, 2048, T)
        h = tokens.reshape(B, C, Fh, T)                    # (B, 256, 8, T)

        # ── Decode with multi-scale skips (reverse order: deep → shallow)
        # decoder[0] uses skips[-1] (deepest), decoder[3] uses skips[0] (shallowest)
        for i, dec_blk in enumerate(self.decoder):
            skip = skips[self.n_enc_layers - 1 - i]
            h = dec_blk(h, skip)

        # ── Final 1×1 conv → 2 channels (real+imag of ECG STFT)
        out = self.head(h)                                 # (B, 2, 64, T)
        return out


# =============================================================================
# Losses (paper Eqs. 12-14, unchanged from v1)
# =============================================================================

def spectral_magnitude_loss(
    pred_time: torch.Tensor, gt_time: torch.Tensor,
    n_fft: int = 256, hop: int = 10, win: int = 40,
) -> torch.Tensor:
    """L_spec = (1/T·F) Σ ||E(u,f)| − |Ê(u,f)|| (paper Eq. 14)."""
    window = torch.hann_window(win, device=pred_time.device, dtype=pred_time.dtype)
    pred_spec = torch.stft(
        pred_time.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    )
    gt_spec = torch.stft(
        gt_time.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    )
    return (pred_spec.abs() - gt_spec.abs()).abs().mean()


def rssrnet_loss(
    pred_time: torch.Tensor, gt_time: torch.Tensor,
    alpha_wave: float = 0.9, alpha_spec: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """L = α₁·L_wave (L1) + α₂·L_spec (paper Eq. 12)."""
    l_wave = F.l1_loss(pred_time, gt_time)
    l_spec = spectral_magnitude_loss(pred_time, gt_time)
    total = alpha_wave * l_wave + alpha_spec * l_spec
    return total, {"l_wave": float(l_wave.item()), "l_spec": float(l_spec.item())}
