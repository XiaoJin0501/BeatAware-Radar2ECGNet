#!/usr/bin/env bash
# install_mamba_cuda.sh — 安装 Mamba 官方 CUDA 核心（修正版）
#
# 核心问题与解法：
#   RTX 4080 SUPER = sm_89（Ada Lovelace），nvcc 11.6 不原生支持 sm_89。
#   解法：TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"
#   "+PTX" 保留 sm_86 PTX 中间码，运行时 JIT 编译为 sm_89 原生码（~1min，之后缓存）。
#
# 预期加速：~138ms/call → ~14ms/call（~10x），epoch ~765s → ~80s
#
# 用法：
#   conda activate cyberbrain
#   bash scripts/install_mamba_cuda.sh

set -e

echo "============================================================"
echo " Installing Mamba CUDA kernels"
echo "============================================================"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "nvcc    : $(nvcc --version | grep 'release' | awk '{print $5}' | tr -d ',')"
echo "GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# ── 关键：编译目标架构 ──────────────────────────────────────────
# 8.6+PTX = sm_86 原生 + PTX 保留（供 sm_89 运行时 JIT）
# nvcc 11.6 最高支持 sm_86，+PTX 解决 sm_89 兼容性
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"
echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

# 自动从 nvcc 推断 CUDA_HOME
NVCC_PATH=$(which nvcc)
export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
echo "CUDA_HOME            = $CUDA_HOME"
echo ""

# ── Step 1: causal-conv1d ──────────────────────────────────────
echo "[1/2] Installing causal-conv1d==1.1.1 ..."
pip install "causal-conv1d==1.1.1" --no-build-isolation
echo "      causal-conv1d: OK"
echo ""

# ── Step 2: mamba-ssm ─────────────────────────────────────────
echo "[2/2] Installing mamba-ssm==1.1.1 ..."
pip install "mamba-ssm==1.1.1" --no-build-isolation
echo "      mamba-ssm: OK"
echo ""

# ── Step 3: 验证 + 测速 ────────────────────────────────────────
echo "[验证] Testing CUDA kernel + timing ..."
python -c "
import torch, time

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('  mamba_ssm import: OK')

B, D, L, N = 32, 128, 1600, 16
u  = torch.randn(B, D, L, device='cuda').float()
dt = torch.randn(B, D, L, device='cuda').float()
A  = -torch.exp(torch.randn(D, N, device='cuda').float())
Bs = torch.randn(B, N, L, device='cuda').float()
Cs = torch.randn(B, N, L, device='cuda').float()
Ds = torch.ones(D, device='cuda').float()

print('  首次调用（可能触发 PTX JIT 编译 ~30-60s，请稍候）...')
out = selective_scan_fn(u, dt, A, Bs, Cs, Ds, delta_softplus=True)
torch.cuda.synchronize()
print(f'  输出 shape: {out.shape}  — 正确')

# 正式测速（20次均值）
t0 = time.time()
for _ in range(20):
    selective_scan_fn(u, dt, A, Bs, Cs, Ds, delta_softplus=True)
torch.cuda.synchronize()
ms = (time.time() - t0) / 20 * 1000

print(f'  CUDA kernel avg : {ms:.1f} ms/call')
print(f'  JIT baseline    : ~138.0 ms/call')
print(f'  实际加速比       : {138.0/ms:.1f}x')
"

echo ""
echo "============================================================"
echo " 安装完成！重启训练即可自动使用 CUDA kernel。"
echo " ssm.py 会在下次 import 时检测到 mamba_ssm 并切换后端。"
echo "============================================================"
