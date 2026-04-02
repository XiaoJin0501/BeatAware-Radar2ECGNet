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

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Selective Scan — 纯 PyTorch 参考实现
# =============================================================================

def _selective_scan_ref(
    u:            torch.Tensor,   # (B, D, L)
    delta:        torch.Tensor,   # (B, D, L)
    A:            torch.Tensor,   # (D, N)
    B_ssm:        torch.Tensor,   # (B, N, L)
    C_ssm:        torch.Tensor,   # (B, N, L)
    D:            torch.Tensor | None = None,   # (D,)
    delta_bias:   torch.Tensor | None = None,   # (D,)
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    纯 PyTorch 实现的 Selective Scan (SSM 前向传播)。

    数学等价于：
        h[t] = exp(delta[t]*A) * h[t-1] + delta[t]*B[t] * u[t]
        y[t] = C[t]^T * h[t]  +  D * u[t]

    Returns: (B, D, L)
    """
    B_size, D_size, L = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    # 离散化（ZOH 近似）
    # dA: (B, D, L, N) — 状态转移
    # dB: (B, D, L, N) — 输入矩阵
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    dB = torch.einsum("bdl,bnl->bdln", delta, B_ssm)

    # 顺序扫描
    h = u.new_zeros(B_size, D_size, N)
    u_exp = u.unsqueeze(-1)   # (B, D, L, 1)
    ys = []
    for t in range(L):
        h = dA[:, :, t] * h + dB[:, :, t] * u_exp[:, :, t]
        y_t = torch.einsum("bdn,bn->bd", h, C_ssm[:, :, t])
        ys.append(y_t)

    y = torch.stack(ys, dim=2)   # (B, D, L)
    if D is not None:
        y = y + D[..., None] * u
    return y


# =============================================================================
# 后端选择（有 CUDA 核心时优先使用）
# =============================================================================

try:
    import selective_scan_cuda as _scan_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


def selective_scan_1d(
    u:            torch.Tensor,
    delta:        torch.Tensor,
    A:            torch.Tensor,
    B_ssm:        torch.Tensor,
    C_ssm:        torch.Tensor,
    D:            torch.Tensor | None = None,
    delta_bias:   torch.Tensor | None = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """统一接口：有编译的 CUDA 核心时使用，否则回退到纯 PyTorch。"""
    if _CUDA_AVAILABLE:
        try:
            out, *_ = _scan_cuda.fwd(u, delta, A, B_ssm, C_ssm, D, None, delta_bias, delta_softplus)
            return out
        except Exception:
            pass
    return _selective_scan_ref(u, delta, A, B_ssm, C_ssm, D, delta_bias, delta_softplus)


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
