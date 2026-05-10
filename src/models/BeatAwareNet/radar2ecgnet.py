"""
radar2ecgnet.py — BeatAwareRadar2ECGNet 主模型

架构（Input Adapter 方案A，按输入表征切换前端，Backbone 统一）：

    输入 [B, 1, L] 或 [B, 1, F, T]
           │
           ├──────────────────────────────────────┐
           ▼                                      ▼
      PAM (峰值辅助模块)                     Input Adapter
        peak_mask [B,1,L]                   → [B, 4C, L//4]
        rhythm_vec [B, 96]                         │  (TFiLM 调制)
           │                                       │
    TFiLMGenerator                           GroupMambaBlock × 2
    gamma, beta [B, 4C]                      ConformerFusionBlock
                                             EMDAlignLayer
                                             Decoder
                                             → ECG [B, 1, L]

参数：C=64（已确认），4C=256，L=1600，L_enc=400

输入类型（input_type）：
    'raw'  : radar_raw   [B, 1, L]    → Multi-scale Conv1d Encoder (TFiLM)
    'phase': radar_phase [B, 1, L]    → Multi-scale Conv1d Encoder (TFiLM)
    'spec' : radar_spec  [B, 1, F, T] → Spec Adapter → TFiLM 调制 → Backbone
    'fmcw' : radar_rcg   [B, 50, L]   → FMCWRangeEncoder → Multi-scale Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.group_mamba import GroupMambaBlock
from ..modules.diffusion_decoder import BeatAwareDiffusionDecoder
from ..modules.fmcw_encoder import FMCWRangeEncoder
from ..modules.peak_module import PeakAuxiliaryModule
from ..modules.tfilm import TFiLMGenerator


# =============================================================================
# ConformerFusionBlock
# =============================================================================

class ConformerFusionBlock(nn.Module):
    """
    ConformerFusionBlock：MHSA + DepthwiseConv1d + FFN 融合模块。

    设计动机：Mamba 做顺序扫描，缺乏对称的跨位置对齐能力；
    MHSA 专责全局跨位置注意力，与 Mamba 功能互补。
    （D4 消融实验验证其增益：删除后 val PCC 0.4421 → 0.1287，−0.31）

    Parameters
    ----------
    d_model   : int  特征维度（输入/输出一致）
    num_heads : int  多头注意力头数（默认4）
    dropout   : float Dropout（默认0.1）
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.mhsa   = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm2   = nn.LayerNorm(d_model)
        self.dw_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=31, padding=15, groups=d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, L)"""
        # MHSA（需要 [B, L, d_model]）
        xt = x.transpose(1, 2)                          # (B, L, d_model)
        normed = self.norm1(xt)
        attn_out, _ = self.mhsa(normed, normed, normed)
        xt = xt + attn_out                              # 残差

        # DepthwiseConv1d（需要 [B, d_model, L]）
        xc = xt.transpose(1, 2)
        xc = xc + self.dw_conv(self.norm2(xt).transpose(1, 2))

        # FFN
        xt = xc.transpose(1, 2)
        xt = xt + self.ffn(self.norm3(xt))

        return xt.transpose(1, 2)   # (B, d_model, L)


# =============================================================================
# EMDAlignLayer — 电-机械延迟物理对齐层（Phase C）
# =============================================================================

class EMDAlignLayer(nn.Module):
    """
    电-机械延迟（EMD）物理对齐层。

    物理背景：ECG（电信号）触发领先于心脏机械运动 50-150ms。
    雷达感知的是机械运动，因此特征图需要在时间轴上"提前"以对齐 ECG。

    实现：逐通道（depthwise）可学习 FIR 滤波器。
      - 初始化为 Dirac delta（单位冲激）= 零延迟恒等映射
      - 训练中自发学习 ±max_delay 范围内的每通道时移

    插入位置：ConformerFusionBlock 之后、Decoder 之前
    特征 shape：(B, 4C, L_enc) = (B, 256, 400)

    Parameters
    ----------
    channels  : int  特征通道数（= 4C）
    max_delay : int  最大时移范围（采样点），默认 20 = 100ms @ 200Hz
                     覆盖典型 EMD 范围 50-150ms (10-30 samples)
    """

    def __init__(self, channels: int, max_delay: int = 20):
        super().__init__()
        kernel_size = 2 * max_delay + 1   # 41-tap FIR
        self.delay_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=max_delay,             # 保持时间轴长度 L_enc 不变
            groups=channels,               # 逐通道独立滤波（depthwise）
            bias=False,
        )
        # 手动 Dirac delta 初始化：每个通道的中心 tap=1，其余=0 → 恒等映射。
        # 不用 nn.init.dirac_，因为 depthwise conv 权重形状 (C, 1, K) 中
        # dirac_ 仅初始化 min(C, 1)=1 个通道，其余通道保持 0。
        with torch.no_grad():
            self.delay_conv.weight.zero_()
            center = kernel_size // 2   # = max_delay
            self.delay_conv.weight[:, :, center] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, C, L)"""
        return self.delay_conv(x)


# =============================================================================
# Spec Adapter（radar_spec 表征专用前端）
# =============================================================================

class SpecAdapter(nn.Module):
    """
    将 radar_spec_input [B, 1, F, T] 适配为 [B, 4C, L_enc]。

    Conv2d(1, C, (F_spec, 1)) 压缩频率轴
    → squeeze → Conv1d(C, 4C, 1) 扩展通道
    → BN → interpolate(size=L_enc) 对齐时间维度

    Parameters
    ----------
    in_freq : int  频率 bins 数（默认33）
    C       : int  基础通道数（默认64）
    L_enc   : int  目标时间帧数（默认400 = 1600//4）
    """

    def __init__(self, in_freq: int = 33, C: int = 64, L_enc: int = 400):
        super().__init__()
        self.L_enc  = L_enc
        self.conv2d = nn.Conv2d(1, C, kernel_size=(in_freq, 1), bias=False)
        self.bn2d   = nn.BatchNorm2d(C)
        self.conv1d = nn.Conv1d(C, 4 * C, kernel_size=1, bias=False)
        self.bn1d   = nn.BatchNorm1d(4 * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, F, T) → (B, 4C, L_enc)"""
        x = F.relu(self.bn2d(self.conv2d(x)))   # (B, C, 1, T)
        x = x.squeeze(2)                          # (B, C, T)
        x = F.relu(self.bn1d(self.conv1d(x)))    # (B, 4C, T)
        x = F.interpolate(x, size=self.L_enc, mode="linear", align_corners=False)
        return x                                  # (B, 4C, L_enc)


# =============================================================================
# BeatAwareRadar2ECGNet — 主模型
# =============================================================================

class BeatAwareRadar2ECGNet(nn.Module):
    """
    BeatAware-Radar2ECGNet：雷达信号 → ECG 重建。

    Parameters
    ----------
    input_type     : str    'raw' | 'phase' | 'spec' | 'fmcw'
    C              : int    Backbone 基础通道数（默认64）
    signal_len     : int    输入信号长度（默认1600）
    spec_freq_bins : int    radar_spec_input 的频率 bins（默认33）
    d_state        : int    VSSSBlock1D 的 SSM 状态维度（默认16）
    dropout        : float  Dropout（默认0.1）
    use_pam        : bool   是否使用 PAM + TFiLM（False = Exp A 基线）
    use_emd        : bool   是否使用 EMD 物理对齐层（Phase C，False = Model A/B）
    emd_max_delay  : int    EMD 最大时移范围（采样点，默认20 = 100ms @ 200Hz）
    use_diffusion  : bool   True → 条件扩散解码器替换 ConvTranspose 回归解码器
    diff_T         : int    扩散步数 T（默认100）
    diff_ddim_steps: int    DDIM 推理步数（默认20）
    diff_hidden    : int    扩散 ResBlock 隐通道数（默认128）
    diff_n_blocks  : int    扩散 ResBlock 层数（默认6）
    use_output_lag_align : bool  True → predict per-segment scalar lag and
                                differentiably shift the reconstructed ECG
    output_lag_max_samples : int max absolute output shift in raw samples
    """

    def __init__(
        self,
        input_type:      str   = "phase",
        C:               int   = 64,
        signal_len:      int   = 1600,
        spec_freq_bins:  int   = 33,
        d_state:         int   = 16,
        dropout:         float = 0.1,
        use_pam:         bool  = True,
        use_emd:         bool  = True,
        use_mamba:       bool  = True,
        emd_max_delay:   int   = 20,
        n_range_bins:    int   = 50,
        use_diffusion:   bool  = False,
        diff_T:          int   = 100,
        diff_ddim_steps: int   = 20,
        diff_hidden:     int   = 128,
        diff_n_blocks:   int   = 6,
        use_output_lag_align: bool = False,
        output_lag_max_samples: int = 40,
        fmcw_selector:   str   = "se",
        fmcw_topk:       int   = 10,
        fmcw_tau:        float = 1.0,
    ):
        super().__init__()
        self.input_type    = input_type
        self.C             = C
        self.signal_len    = signal_len
        self.use_pam       = use_pam
        self.use_emd       = use_emd
        self.use_mamba     = use_mamba
        self.use_diffusion = use_diffusion
        self.use_output_lag_align = use_output_lag_align
        self.output_lag_max_samples = float(output_lag_max_samples)
        self.last_output_lag_samples: torch.Tensor | None = None
        L_enc              = signal_len // 4   # 1600 → 400
        self.L_enc         = L_enc
        PAM_DIM            = 96   # 3路 × 32

        # ── FMCW 空间聚合前端（仅 input_type='fmcw' 时启用）────────
        # 将 (B, 50, L) 50通道距离-时间信号聚合为 (B, 3, L) 运动学表征，
        # 之后完全复用现有 Encoder/Backbone/PAM/TFiLM/EMD/Decoder。
        if input_type == "fmcw":
            self.fmcw_enc = FMCWRangeEncoder(
                n_range=n_range_bins,
                L=signal_len,
                selector=fmcw_selector,
                topk=fmcw_topk,
                tau=fmcw_tau,
            )

        # ── PAM + TFiLM（Exp A 基线时禁用，use_pam=False）────────
        if use_pam:
            self.pam = PeakAuxiliaryModule(
                input_type="spec" if input_type == "spec" else "1d",
                pam_channels=32,
                signal_len=signal_len,
                spec_freq_bins=spec_freq_bins,
                d_state=d_state,
            )
            self.tfilm_gen = TFiLMGenerator(input_dim=PAM_DIM, output_channels=4 * C)

        # ── Encoder / Adapter ─────────────────────────────────────
        if input_type in ("raw", "phase", "fmcw"):
            # Multi-scale Conv1d (k=3,5,7,9, stride=4)
            # V2: in_channels=3（原始 + 速度 + 加速度）
            # 1600 → 400（padding=k//2 保证各核输出等长）
            self.enc_convs = nn.ModuleList([
                nn.Conv1d(3, C, kernel_size=k, padding=k // 2, stride=4, bias=False)
                for k in [3, 5, 7, 9]
            ])
            self.enc_bns = nn.ModuleList([nn.BatchNorm1d(C) for _ in range(4)])
        else:
            self.spec_adapter = SpecAdapter(in_freq=spec_freq_bins, C=C, L_enc=L_enc)

        # ── Bottleneck（GroupMamba × 2）───────────────────────────
        # use_mamba=False 时跳过整个 SSM bottleneck，直接进 Conformer
        if use_mamba:
            self.mamba1 = GroupMambaBlock(4 * C, num_groups=4, d_state=d_state)
            self.mamba2 = GroupMambaBlock(4 * C, num_groups=4, d_state=d_state)

        # ── Fusion ────────────────────────────────────────────────
        self.fusion = ConformerFusionBlock(4 * C, num_heads=4, dropout=dropout)

        # ── EMD 物理对齐层（Phase C，PA 消融开关）──────────────────
        # 插入 Fusion 之后、Decoder 之前；use_emd=False 时为恒等映射
        if use_emd:
            self.emd = EMDAlignLayer(4 * C, max_delay=emd_max_delay)

        if use_output_lag_align:
            self.output_lag_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(4 * C, 2 * C),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * C, 1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.output_lag_head[-2].weight)
            nn.init.zeros_(self.output_lag_head[-2].bias)

        # ── Decoder ──────────────────────────────────────────────────
        if use_diffusion:
            self.diff_decoder = BeatAwareDiffusionDecoder(
                h_enc_channels=4 * C,
                hidden=diff_hidden,
                n_blocks=diff_n_blocks,
                T=diff_T,
                ddim_steps=diff_ddim_steps,
                use_pam=use_pam,
            )
        else:
            # 回归解码器（stride=2 上采样 × 2：400 → 800 → 1600）
            self.up1   = nn.ConvTranspose1d(4 * C, 2 * C, kernel_size=4, stride=2, padding=1)
            self.up2   = nn.ConvTranspose1d(2 * C,     C, kernel_size=4, stride=2, padding=1)
            self.final = nn.Conv1d(C, 1, kernel_size=1)

    @staticmethod
    def _fractional_shift_1d(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Differentiable per-sample temporal shift with linear interpolation.

        x:     (B, 1, L)
        shift: (B,) in samples. Positive shift delays the signal:
               y[t] = x[t - shift].
        """
        B, C, L = x.shape
        base = torch.arange(L, device=x.device, dtype=x.dtype)[None, :]
        src = (base - shift[:, None]).clamp(0.0, float(L - 1))
        left = src.floor().long()
        right = (left + 1).clamp(max=L - 1)
        weight = (src - left.to(src.dtype)).unsqueeze(1)
        left_v = x.gather(-1, left.unsqueeze(1).expand(B, C, L))
        right_v = x.gather(-1, right.unsqueeze(1).expand(B, C, L))
        return left_v * (1.0 - weight) + right_v * weight

    # ------------------------------------------------------------------
    def _encode_1d(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-scale Conv1d Encoder with per-branch TFiLM injection.

        x     : (B, 3, L)  — [原始, 速度, 加速度] 三通道（V2 derivative encoder）
        gamma : (B, 4C) → reshape → (B, 4, C, 1)
        beta  : (B, 4C) → reshape → (B, 4, C, 1)
        Returns: (B, 4C, L_enc)
        """
        B = x.size(0)
        gamma4 = gamma.view(B, 4, self.C, 1)
        beta4  = beta.view(B, 4, self.C, 1)

        feats = []
        for i, (conv, bn) in enumerate(zip(self.enc_convs, self.enc_bns)):
            f = bn(conv(x))                              # (B, C, L_enc)
            f = f * (1.0 + gamma4[:, i]) + beta4[:, i]  # TFiLM
            feats.append(F.relu(f))
        return torch.cat(feats, dim=1)                   # (B, 4C, L_enc)

    def _encode_spec(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Spec Adapter with full-channel TFiLM injection.

        x     : (B, 1, F, T)
        gamma : (B, 4C)
        beta  : (B, 4C)
        Returns: (B, 4C, L_enc)
        """
        f = self.spec_adapter(x)                              # (B, 4C, L_enc)
        f = f * (1.0 + gamma[..., None]) + beta[..., None]   # TFiLM
        return F.relu(f)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, ecg_gt: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x      : Tensor
            input_type='raw'/'phase'/'fmcw' : (B, n_ch, L)
            input_type='spec'               : (B, 1, F, T)
        ecg_gt : Tensor | None
            (B, 1, L) ground-truth ECG in [0,1].  Required during training
            when use_diffusion=True; ignored otherwise.

        Returns
        -------
        Regression decoder (use_diffusion=False):
            ecg_pred   : Tensor, (B, 1, L) in [0,1]
            peak_masks : tuple (qrs,p,t) each (B,1,L) | None

        Diffusion decoder (use_diffusion=True):
            training   : (eps_pred, eps_true) both (B,1,L), peak_masks
            inference  : ecg_pred (B,1,L) in [0,1],         peak_masks
        """
        # ── 输入预处理 ─────────────────────────────────────────────
        if self.input_type == "fmcw":
            x_input = self.fmcw_enc(x)                        # (B, 3, L)
        elif self.input_type in ("raw", "phase"):
            v = torch.diff(x, dim=-1, prepend=x[:, :, :1])   # (B, 1, L)
            a = torch.diff(v, dim=-1, prepend=v[:, :, :1])   # (B, 1, L)
            x_input = torch.cat([x, v, a], dim=1)            # (B, 3, L)
        else:
            x_input = x   # spec 路径不变

        # ── PAM + TFiLM ────────────────────────────────────────────
        if self.use_pam:
            peak_masks, rhythm_vec = self.pam(x_input)    # (qrs,p,t), (B,96)
            gamma, beta = self.tfilm_gen(rhythm_vec)       # each (B, 4C)
        else:
            peak_masks  = None
            rhythm_vec  = None
            gamma = x.new_zeros(x.size(0), 4 * self.C)
            beta  = x.new_zeros(x.size(0), 4 * self.C)

        # ── Encoder ────────────────────────────────────────────────
        if self.input_type in ("raw", "phase", "fmcw"):
            enc = self._encode_1d(x_input, gamma, beta)   # (B, 4C, L_enc)
        else:
            enc = self._encode_spec(x_input, gamma, beta) # (B, 4C, L_enc)

        # ── Bottleneck + Fusion ────────────────────────────────────
        if self.use_mamba:
            h = self.mamba1(enc)
            h = self.mamba2(h)
        else:
            h = enc
        h = self.fusion(h)

        # ── EMD 物理对齐 ───────────────────────────────────────────
        if self.use_emd:
            h = self.emd(h)

        if self.use_output_lag_align:
            output_lag = self.output_lag_head(h).squeeze(-1) * self.output_lag_max_samples
        else:
            output_lag = None

        # ── Decoder ────────────────────────────────────────────────
        if self.use_diffusion:
            # peak_masks: (qrs, p, t) → cat to (B, 3, L)，or None
            peak_masks_cat = (
                torch.cat(list(peak_masks), dim=1)  # (B, 3, 1600)
                if peak_masks is not None else None
            )
            if self.training and ecg_gt is not None:
                out = self.diff_decoder.training_step(
                    h, rhythm_vec, peak_masks_cat, ecg_gt
                )
                return out, peak_masks   # out = (eps_pred, eps_true)
            else:
                ecg_pred = self.diff_decoder.ddim_sample(
                    h, rhythm_vec, peak_masks_cat
                )
                if output_lag is not None:
                    ecg_pred = self._fractional_shift_1d(ecg_pred, output_lag)
                    self.last_output_lag_samples = output_lag.detach()
                return ecg_pred, peak_masks
        else:
            h = F.relu(self.up1(h))                       # (B, 2C, 800)
            h = F.relu(self.up2(h))                       # (B,  C, 1600)
            ecg_pred = torch.sigmoid(self.final(h))       # (B, 1, 1600)
            if output_lag is not None:
                ecg_pred = self._fractional_shift_1d(ecg_pred, output_lag)
                self.last_output_lag_samples = output_lag.detach()
            return ecg_pred, peak_masks


# =============================================================================
# 参数量统计工具
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # use_pam=True（完整模型）
    for input_type in ["raw", "phase", "spec"]:
        model = BeatAwareRadar2ECGNet(input_type=input_type, C=64, use_pam=True).to(device)
        x = torch.randn(2, 1, 33, 196).to(device) if input_type == "spec" else torch.randn(2, 1, 1600).to(device)
        ecg_pred, peak_masks = model(x)
        qrs, p, t = peak_masks
        assert ecg_pred.shape == (2, 1, 1600)
        assert qrs.shape == p.shape == t.shape == (2, 1, 1600)
        assert ecg_pred.min() >= 0.0 and ecg_pred.max() <= 1.0
        print(f"  [PAM=True , {input_type}] ECG={ecg_pred.shape}, QRS/P/T={qrs.shape}, "
              f"Params={count_parameters(model):,}")

    # use_pam=False（Exp A 基线）
    for input_type in ["raw", "phase", "spec"]:
        model = BeatAwareRadar2ECGNet(input_type=input_type, C=64, use_pam=False).to(device)
        x = torch.randn(2, 1, 33, 196).to(device) if input_type == "spec" else torch.randn(2, 1, 1600).to(device)
        ecg_pred, peak_masks = model(x)
        assert ecg_pred.shape == (2, 1, 1600)
        assert peak_masks is None, "use_pam=False 时 peak_masks 应为 None"
        assert ecg_pred.min() >= 0.0 and ecg_pred.max() <= 1.0
        print(f"  [PAM=False, {input_type}] ECG={ecg_pred.shape}, Masks=None, "
              f"Params={count_parameters(model):,}")

    print("All shape checks passed.")
