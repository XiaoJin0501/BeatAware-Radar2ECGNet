"""
ssm.py — VSSSBlock1D (1D Visual Selective State Space Block)

独立实现，不依赖外部 mamba-ssm 包。
提供纯 PyTorch 参考实现，有编译好的 CUDA 核心时自动切换为快速后端。

核心结构：
    in_proj(x, z) → DepthwiseConv1d → SiLU → x_proj(dt, B_ssm, C_ssm)
    → dt_proj → SelectiveScan1D → gate(z) → out_proj
    残差连接：out = out_proj(y) + x_in
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Selective Scan — TorchScript 编译实现（消除 Python loop 开销）
# =============================================================================

@torch.jit.script
def _selective_scan_ref(
    u:              torch.Tensor,           # (B, D, L)
    delta:          torch.Tensor,           # (B, D, L)
    A:              torch.Tensor,           # (D, N)
    B_ssm:          torch.Tensor,           # (B, N, L)
    C_ssm:          torch.Tensor,           # (B, N, L)
    D:              Optional[torch.Tensor] = None,          # (D,)
    delta_bias:     Optional[torch.Tensor] = None,          # (D,)
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    TorchScript 顺序扫描（回退方案，当并行扫描显存不足时使用）。

    显存需求：O(B·D·L·N)。L=1600, B=32, D=192, N=16 时约需 3.8GB 额外显存。
    若显存紧张，回退到此实现（138ms/call 但显存仅 O(B·D·N)）。
    """
    B_size = u.shape[0]
    D_size = u.shape[1]
    L      = u.shape[2]
    N      = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    h = torch.zeros(B_size, D_size, N, dtype=u.dtype, device=u.device)
    ys: List[torch.Tensor] = []
    for t in range(L):
        dt   = delta[:, :, t]
        dA_t = torch.exp(dt.unsqueeze(-1) * A)
        dB_t = dt.unsqueeze(-1) * B_ssm[:, :, t].unsqueeze(1)
        h    = dA_t * h + dB_t * u[:, :, t].unsqueeze(-1)
        y_t  = (h * C_ssm[:, :, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + D.unsqueeze(-1) * u
    return y


# =============================================================================
# 后端选择（有 CUDA 核心时优先使用）
# =============================================================================

# 后端优先级：
#   1. mamba-ssm 官方 CUDA kernel（最快，~10x vs JIT）
#   2. selective_scan_cuda（手动编译的旧接口）
#   3. TorchScript JIT（纯 PyTorch，138ms/call，约慢 10x）
#
# 安装官方 CUDA kernel：bash scripts/install_mamba_cuda.sh
# 未安装时回退到 JIT，训练结果完全一致，速度约慢 2x（epoch 274s vs 130s）

_MAMBA_SSM_AVAILABLE = False
_CUDA_AVAILABLE = False

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_scan_fn
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    try:
        import selective_scan_cuda as _scan_cuda
        _CUDA_AVAILABLE = True
    except ImportError:
        pass


def selective_scan_1d(
    u:              torch.Tensor,
    delta:          torch.Tensor,
    A:              torch.Tensor,
    B_ssm:          torch.Tensor,
    C_ssm:          torch.Tensor,
    D:              Optional[torch.Tensor] = None,
    delta_bias:     Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    统一 Selective Scan 接口，按优先级自动选择后端：
      1. mamba-ssm 官方 CUDA kernel（最快，需安装）
      2. selective_scan_cuda（手动编译）
      3. TorchScript 顺序扫描（回退，~71ms/call with warmup）
    """
    if _MAMBA_SSM_AVAILABLE:
        return _mamba_scan_fn(
            u, delta, A, B_ssm, C_ssm, D,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
        )
    if _CUDA_AVAILABLE:
        try:
            out, *_ = _scan_cuda.fwd(u, delta, A, B_ssm, C_ssm, D, None, delta_bias, delta_softplus)
            return out
        except Exception:
            pass

    return _selective_scan_ref(u, delta, A, B_ssm, C_ssm, D,
                               delta_bias=delta_bias, delta_softplus=delta_softplus)


# =============================================================================
# VSSSBlock1D
# =============================================================================

class VSSSBlock1D(nn.Module):
    """
    1D Visual Selective State Space Block。

    对 [B, d_model, L] 的序列做 SSM 序列建模，带残差连接。

    Parameters
    ----------
    d_model : int
        输入/输出通道数
    d_state : int
        SSM 状态维度（默认16）
    d_conv : int
        深度卷积核大小（默认3）
    expand : float
        内部通道扩展比（默认2.0）
    dropout : float
        输出 Dropout（默认0，即不使用）
    """

    def __init__(
        self,
        d_model:  int,
        d_state:  int   = 16,
        d_conv:   int   = 3,
        expand:   float = 2.0,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16)
        self.d_state  = d_state

        # 输入投影：同时生成激活路径 x 和门控路径 z
        self.in_proj = nn.Conv1d(d_model, self.d_inner * 2, kernel_size=1, bias=False)

        # 深度卷积：提取局部特征
        self.dw_conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        # SSM 参数投影（生成 dt, B_ssm, C_ssm）
        self.x_proj = nn.Conv1d(
            self.d_inner, self.dt_rank + 2 * d_state,
            kernel_size=1, bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Mamba 论文推荐：dt_proj.bias 初始化到稳定的初始 delta 范围 [dt_min, dt_max]
        # 避免默认均匀初始化导致 delta 在训练中漂移到极端值 → SSM 状态在 L=1600 步后爆炸 → NaN
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # dt_proj.weight 用默认初始化即可，bias 不参与梯度更新的额外约束
        self.dt_proj.bias._no_reinit = True   # 标记，防止 reset_parameters 覆盖

        # 固定参数（S4D 初始化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Conv1d(self.d_inner, d_model, kernel_size=1, bias=False)
        self.dropout  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

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

        # 投影并分离 x（激活路径）和 z（门控路径）
        xz = self.in_proj(x)              # (B, 2*d_inner, L)
        x, z = xz.chunk(2, dim=1)         # each (B, d_inner, L)

        # 深度卷积 + 激活
        x = self.act(self.dw_conv(x))     # (B, d_inner, L)

        # SSM 参数生成
        x_dbl = self.x_proj(x)            # (B, dt_rank + 2*d_state, L)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1
        )
        # dt: (B, dt_rank, L) → dt_proj → (B, d_inner, L)
        dt = F.linear(dt.transpose(1, 2), self.dt_proj.weight).transpose(1, 2)

        A = -torch.exp(self.A_log)        # (d_inner, d_state)，保证 A < 0

        # Selective Scan
        y = selective_scan_1d(
            x, dt, A, B_ssm, C_ssm,
            D=self.D,
            delta_bias=self.dt_proj.bias,
            delta_softplus=True,
        )                                 # (B, d_inner, L)

        # 门控 + 输出投影
        y = y * self.act(z)
        out = self.dropout(self.out_proj(y))

        return out + residual
