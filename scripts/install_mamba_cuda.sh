#!/usr/bin/env bash
# install_mamba_cuda.sh — 尝试安装 Mamba 官方 CUDA 核心
#
# 背景：
#   当前 _selective_scan_ref 使用 TorchScript（JIT）实现，
#   实测 138ms/call（B=32, D=128, L=1600），占训练时间约 60%。
#   官方 CUDA 核心（mamba-ssm）约快 10x，可将 epoch 时间从 274s 降至 ~130s。
#
# 运行前提：
#   - conda activate cyberbrain
#   - GPU 驱动 + nvcc 可用（已确认：nvcc 11.6）
#   - PyTorch 1.13.1+cu113（注意与 nvcc 11.6 存在 minor mismatch，可能编译失败）
#
# 用法：
#   bash scripts/install_mamba_cuda.sh
#
# 若失败：保持现有 JIT 实现，训练速度慢约 2x，但结果完全正确。

set -e

echo "============================================================"
echo " Installing Mamba CUDA kernels"
echo " Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo " PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo " nvcc: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "============================================================"

# Step 1: causal-conv1d（mamba-ssm 依赖）
echo ""
echo "[1/2] Installing causal-conv1d ..."
pip install causal-conv1d --no-build-isolation 2>&1 | tail -5

# Step 2: mamba-ssm
# 指定与 PyTorch 1.13.x 兼容的版本
echo ""
echo "[2/2] Installing mamba-ssm ..."
pip install mamba-ssm==1.1.1 --no-build-isolation 2>&1 | tail -5

# Step 3: 验证
echo ""
echo "[验证] Testing CUDA kernel ..."
python -c "
import torch
from mamba_ssm import selective_scan_fn
print('mamba_ssm CUDA kernel: OK')

# 测速对比
import time
B, D, L, N = 32, 128, 1600, 16
u     = torch.randn(B, D, L, device='cuda').float()
delta = torch.randn(B, D, L, device='cuda').float()
A     = -torch.exp(torch.randn(D, N, device='cuda').float())
B_ssm = torch.randn(B, N, L, device='cuda').float()
C_ssm = torch.randn(B, N, L, device='cuda').float()
D_ssm = torch.ones(D, device='cuda').float()

# warmup
selective_scan_fn(u, delta, A, B_ssm, C_ssm, D_ssm, delta_softplus=True)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    selective_scan_fn(u, delta, A, B_ssm, C_ssm, D_ssm, delta_softplus=True)
torch.cuda.synchronize()
print(f'CUDA kernel avg: {(time.time()-t0)/10*1000:.1f} ms/call')
print('Expected speedup: ~5-15x vs JIT 138ms')
"

echo ""
echo "============================================================"
echo " Done! Restart training to use CUDA kernels."
echo " ssm.py will auto-detect mamba_ssm on next import."
echo "============================================================"
