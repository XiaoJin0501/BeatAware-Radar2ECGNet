# CLAUDE.md — BeatAware-Radar2ECGNet 项目指南

本文件为 Claude Code 提供项目级开发上下文。代码从零独立实现，**不直接复用任何已有项目代码**。完整架构设计见 `docs/ARCHITECTURE.md`，本文件侧重快速上下文恢复与开发规范约束。

---

## 实验状态（2026-05-07）

| 实验 | 状态 | 说明 |
|------|------|------|
| `mmecg_v1` samplewise（回归，旧架构 baseline） | ✅ 已测试 | test PCC=0.211, F1@150ms=0.841, QMR=79.6%；论文 Table 1 baseline |
| `mmecg_diff_v2` samplewise（扩散） | 🔄 **运行中** | T=1000/hidden=256/n_blocks=8，300 epochs；`screen -r diff_v2` |
| `mmecg_reg_clean` samplewise（回归对照） | 🔄 **运行中** | balance_by=class + narrow_bandpass=false，验证预处理假设；`screen -r reg_clean` |
| MMECG LOSO 全 11 折 | ⏳ 待定 | samplewise 验证后启动 |
| 消融 A/B/C/D（扩散版） | ⏳ 待定 | `scripts/run_ablation_mmecg.sh` 已更新为 v2 配置 |
| Schellenberger 实验 | ⏳ MMECG 完成后 | Phase 4，cross-dataset 验证（dataset/ 目录待预处理）|

**已删除的失败实验**（信息留作教训）：
- `mmecg_diff_v1`（T=100）：test PCC=0.0513，R²=−4.19，T 太小完全无法生成有效 ECG
- `mmecg_reg_improved`（subject + narrow_bandpass）：val PCC=0.1548 远差于 v1 的 0.211，怀疑预处理拖累

**查看当前训练进度**：
```bash
tail -f experiments_mmecg/mmecg_diff_v2_sw.log
tail -f experiments_mmecg/mmecg_reg_clean_sw.log
# 或进入 screen session：
screen -r diff_v2     # 扩散主训练
screen -r reg_clean   # 回归对照
```

---

## 项目目标与核心创新点

从非接触式雷达信号重建高保真 ECG 信号。三个核心创新（KI/PA/CP）必须保留，是论文主线：

| 缩写 | 模块 | 实现位置 | 说明 |
|------|------|---------|------|
| **KI** | FMCWRangeEncoder | `src/models/modules/fmcw_encoder.py` | 50 range bin → 3 ch 运动学特征（GELU 关键，不能换 ReLU）|
| **PA** | EMDAlignLayer | `src/models/BeatAwareNet/radar2ecgnet.py` | 可学习 41-tap depthwise FIR，自动补偿雷达-ECG 物理时延 50-150ms |
| **CP** | PeakAuxiliaryModule | `src/models/modules/peak_module.py` | 3路 QRS/P/T 检测 + rhythm_vec(B,96) → TFiLM 调制；扩散模式下峰值掩码还作为空间引导 |

**论文定位**：
- Table 1（主线）：BeatAware-Regression vs 对比方法（radarODE-MTL 等）
- Table 2（进阶）：BeatAware-Diffusion，展示扩散解码器与 KI/PA/CP encoder 的兼容性

---

## 模型架构

### 整体数据流

```
MMECG:  RCG [B,50,L] ──► FMCWRangeEncoder ──► [B,3,L]
                                                    │
                    ┌───────────────────────────────┤
                    ▼                               ▼
              PAM（峰值辅助）               Multi-scale Encoder(×4)
              QRS/P/T masks                    + TFiLM(rhythm_vec)
              rhythm_vec[B,96]             GroupMambaBlock × 2
                    │                      ConformerFusionBlock
                    └──── γ,β ────────►    EMDAlignLayer（PA）
                                               │
                          ┌────────────────────┘
                          ▼
              [use_diffusion=False]  ConvTranspose × 2 + Sigmoid → ECG [B,1,L]
              [use_diffusion=True]   BeatAwareDiffusionDecoder（DDIM 20步）→ ECG [B,1,L]
```

### 关键模型参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `C=64` | 64 | encoder base channels（4C=256 为主干宽度）|
| `d_state=16` | 16 | Mamba SSM 状态维度 |
| `emd_max_delay=20` | 20 | 41-tap FIR，覆盖 ±100ms 延迟 |
| `use_pam=True` | True | PAM + TFiLM；False = 消融 A/B |
| `use_emd=True` | True | EMD 对齐；False = 消融 A/C |
| `use_diffusion=False` | False | True = 扩散解码器（BeatAwareDiffusionDecoder）|

### 扩散解码器（BeatAwareDiffusionDecoder）

- **文件**：`src/models/modules/diffusion_decoder.py`
- **参数量**：约 1,400,000（替换回归解码器的 164,097）
- **总模型参数**：3,098,974（vs 原来 1,698,270）
- 余弦噪声调度，T=100；DDIM 确定性采样 20 步（推理加速 5×）
- ECG 值域：H5 存储 `[0,1]`，扩散内部需 `x0 = ecg_gt*2-1` → `[-1,1]`，输出再 `(x+1)/2`
- 条件信号：`h_enc(B,256,400)` + `rhythm_vec(B,96)` + `peak_masks(B,3,1600)`（use_pam=True 时）

### 消融实验定义

| 模型 | `use_pam` | `use_emd` | 意义 |
|------|:---:|:---:|------|
| A | False | False | 纯扩散基线（无 KI CP/PA 创新）|
| B | False | True  | +EMD 物理对齐条件化 |
| C | True  | False | +PAM 峰值引导扩散 |
| D | True  | True  | 完整 BeatAware-Diffusion（Full）|

---

## 数据集

### MMECG（主实验）

- **原始路径**：`/home/qhh2237/Datasets/MMECG/MMECG.h5`
- **预构建 H5**：`/home/qhh2237/Datasets/MMECG/processed/`
- 11 名受试者，77GHz FMCW，50 range bins，200Hz，8s 窗口（1600点），50% 重叠
- 生理状态：NB / IB / SP / PE

**重要数据不均衡问题**（根因之一）：
- Sub 1/2（中老年女性）samplewise 训练集仅 14 个样本（0.85%）
- 当前训练用 `balance_by=subject`（`subject_weights()` 按 1/count 赋权）解决

**H5 key 说明**：

| Key | Shape | 说明 |
|-----|-------|------|
| `rcg` | [N,50,1600] float32 | per-channel z-score（H5 中已 0.5-20Hz 带通）|
| `ecg` | [N,1,1600] float32 | z-score（loader 内再 min-max → [0,1]）|
| `rpeak_indices` | vlen int32 | R峰 → loader 转为 Gaussian mask σ=5 |
| `q/s/tpeak_indices` | vlen int32 | Q/S/T 峰（-1=missing）|
| `delineation_valid` | uint8 [N] | Level 3/4 评估有效标志 |
| `subject_id` | int32 [N] | 受试者编号 1~11 |
| `physistatus` | bytes [N] | b"NB"/b"IB"/b"SP"/b"PE" |

**`narrow_bandpass=True`**：loader 加载时对 RCG 额外做 0.8-3.5 Hz 窄带滤波（心跳频段），再次 z-score 归一化。消除 3.5-20 Hz 主杂波，提高心跳分量 SNR。

### Schellenberger（次实验，Phase 4）

- 30 名受试者，24GHz CW，2000Hz，使用 Resting/Valsalva/Apnea 场景
- 预处理输出：`dataset/` 目录，NPY 格式
- 5-Fold CV（每折 6 人，seed=42）
- 当前暂缓，MMECG 实验完成后执行

---

## 损失函数

| 模式 | 类 | 公式 |
|------|-----|------|
| 回归 | `TotalLoss` | L1 + α·MultiResSTFT + β·QRS-BCE |
| 扩散 | `DiffusionLoss` | MSE(ε_pred, ε) + β·QRS-BCE |

两者均在 `src/losses/losses.py`。训练时通过 `cfg.use_diffusion` 自动选择。

---

## 常用命令

```bash
# ── 环境激活 ────────────────────────────────────────────────────────
conda activate cyberbrain   # PyTorch, CUDA, mamba-ssm 等

# ── 查看当前运行实验 ───────────────────────────────────────────────
screen -ls
tail -f experiments_mmecg/mmecg_diff_v2_sw.log

# ── MMECG 训练（扩散 v2，T=1000）──────────────────────
# samplewise
python scripts/train_mmecg.py \
    --exp_tag mmecg_diff_v2 --protocol samplewise --epochs 300 \
    --use_diffusion true --balance_by subject --narrow_bandpass true

# LOSO 全 11 折
python scripts/train_mmecg.py \
    --exp_tag mmecg_diff_v2 --protocol loso --fold_idx -1 \
    --use_diffusion true --balance_by subject --narrow_bandpass true

# LOSO 单折调试
python scripts/train_mmecg.py \
    --exp_tag mmecg_diff_v2 --protocol loso --fold_idx 1 \
    --use_diffusion true --balance_by subject --narrow_bandpass true

# ── MMECG 测试 ─────────────────────────────────────────────────────
python scripts/test_mmecg.py --exp_tag mmecg_diff_v2 --fold_idx -1
python scripts/test_mmecg.py --exp_tag mmecg_diff_v2 --protocol samplewise

# ── 消融实验（已内置 --use_diffusion true，自动用 v2 超参）
bash scripts/run_ablation_mmecg.sh

# ── 数据诊断 ───────────────────────────────────────────────────────
python tests/visualize_mmecg.py --fold 1 --split test --no-plot
python tests/visualize_mmecg.py --protocol samplewise --split val --n 3

# ── 单元测试：扩散解码器形状 ────────────────────────────────────────
python -c "
import torch
from src.models.modules.diffusion_decoder import BeatAwareDiffusionDecoder
dec = BeatAwareDiffusionDecoder()
h = torch.randn(2, 256, 400)
rv = torch.randn(2, 96)
pm = torch.randn(2, 3, 1600)
ecg = torch.rand(2, 1, 1600)
dec.train(); ep, et = dec.training_step(h, rv, pm, ecg)
assert ep.shape == (2, 1, 1600)
dec.eval(); out = dec.ddim_sample(h, rv, pm)
assert out.shape == (2, 1, 1600) and out.min() >= 0 and out.max() <= 1
print('OK')
"
```

---

## 评估指标体系（4级协议）

| Level | 指标 | 主要用途 |
|-------|------|---------|
| **L1 波形** | PCC, RMSE_norm, MAE_norm, R² | 所有论文通用 |
| **L2 峰值定时** | R/Q/S/T peak error(ms), RR error, QMR, MDR | 与 radarODE-MTL / AirECG 比较 |
| **L3 Fiducial F1** | Rpeak/Pon/Toff precision/recall/F1 @150ms | 与 Cao et al. 比较 |
| **L4 临床间期** | PR, QRS, QT, QTc 误差(ms) | 本文扩展指标 |

**测试输出文件**（per fold）：
```
experiments_mmecg/<exp_tag>/<run_label>/results/
    segment_metrics.csv    每行 = 1 个 8s window
    beat_metrics.csv       每行 = 1 个匹配 beat
    subject_summary.csv    按 (subject_id, scene) 聚合
    global_summary.json    全局 mean/median/std/IQR

experiments_mmecg/<exp_tag>/loso_summary/   ← 仅 --fold_idx -1 时生成
```

---

## 文件结构

```
BeatAware-Radar2ECGNet/
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── ssm.py              # VSSSBlock1D + SelectiveScan1D
│   │   │   └── group_mamba.py      # GroupMambaBlock
│   │   ├── modules/
│   │   │   ├── fmcw_encoder.py     # FMCWRangeEncoder: (B,50,L)→(B,3,L)，GELU 关键
│   │   │   ├── peak_module.py      # PAM: 3路 QRS/P/T + rhythm_vec(B,96)
│   │   │   ├── tfilm.py            # TFiLMGenerator
│   │   │   └── diffusion_decoder.py # BeatAwareDiffusionDecoder（新）
│   │   └── BeatAwareNet/
│   │       └── radar2ecgnet.py     # 主模型（支持 use_diffusion 开关）
│   ├── data/
│   │   ├── mmecg_dataset.py        # H5 loader，subject_weights，narrow_bandpass
│   │   └── dataset.py              # Schellenberger NPY loader
│   ├── losses/
│   │   └── losses.py               # TotalLoss（回归）+ DiffusionLoss（扩散）
│   └── utils/
│       ├── metrics.py              # 4级评估协议
│       └── logger.py / seeding.py
├── configs/
│   ├── mmecg_config.py             # MMECG 配置（含 use_diffusion / balance_by 等）
│   └── config.py                   # Schellenberger 配置
├── scripts/
│   ├── train_mmecg.py              # MMECG 训练（支持 --use_diffusion, --balance_by）
│   ├── test_mmecg.py               # MMECG 测试（自动从 config.json 恢复扩散超参）
│   ├── run_ablation_mmecg.sh       # 消融批量脚本（待更新 --use_diffusion true）
│   └── plot_subject_metrics.py / plot_paper_figures.py / summarize_ablation.py
├── tests/
│   └── visualize_mmecg.py          # H5 数据诊断可视化
├── experiments_mmecg/              # 实验输出（不提交到 git）
│   ├── mmecg_v1/                   # 回归 baseline（已完成，PCC=0.211）
│   ├── mmecg_diff_v2/              # 扩散主训练（T=1000, hidden=256，运行中）
│   └── mmecg_reg_clean/            # 回归对照（class+全频带，运行中）
├── docs/                           # ARCHITECTURE.md 等文档
└── CLAUDE.md                       # 本文件
```

---

## 环境与性能

- **conda 环境**：`cyberbrain`（PyTorch 1.13.1, CUDA 11.3, RTX 4080 SUPER 16GB）
- **GPU**：RTX 4080 SUPER 16GB
- **训练速度（扩散模式）**：约 100s/epoch（samplewise，batch_size=16）
- **训练速度（回归模式）**：约 30s/epoch
- val_every=5（回归）；扩散模式建议 val_every=10（DDIM 20步 × val 集约 75 batch，慢约 3-4×）

SSM 后端自动降级为 TorchScript JIT（约 138ms/call）。安装 CUDA kernel：
```bash
bash scripts/install_mamba_cuda.sh
```

---

## 开发约束

- **不复用已有项目代码**，所有模块从零实现
- `docs/ARCHITECTURE.md` 是架构权威来源；若本文件与其冲突，以 ARCHITECTURE.md 为准
- 每个新模块写完后必须用单元测试验证 tensor shape（`python -c "..."` 快速验证）
- 实验输出统一写入 `experiments_mmecg/<EXP_TAG>/`，不污染代码目录
- config.json 随每次训练自动保存到 `<run_dir>/config.json`；test 脚本从此文件恢复所有超参

---

## 已知问题 / 注意事项

1. **FMCWRangeEncoder 必须用 GELU**：BatchNorm 输出均值零，ReLU 截断心脏 AC 信号负半周期。已修复。
2. **ECG 值域变换**：H5 中 `[0,1]`，扩散内部需 `x0 = ecg_gt*2-1`，DDIM 输出 `(x+1)/2` 还原。
3. **peak_masks 是元组**：`forward()` 中传给扩散解码器前需 `torch.cat(list(peak_masks), dim=1)` → `(B,3,1600)`。
4. **use_pam=False 时输入通道数不同**：`in_ch = 257`（无峰值掩码）vs `260`，`diffusion_decoder.py` 中 `in_proj` 已按 `use_pam` 动态设置。
5. **samplewise vs LOSO PCC 差异**：samplewise 也是 record-level 划分（非 segment-level），Sub 10 仅出现在 val/test，本质上也是部分跨受试者评估。
