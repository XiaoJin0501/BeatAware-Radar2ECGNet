# CLAUDE.md — BeatAware-Radar2ECGNet 项目指南

本文件为 Claude Code 提供项目级开发上下文。代码从零独立实现，**不直接复用任何已有项目代码**。完整架构设计见 `docs/ARCHITECTURE.md`，本文件侧重快速上下文恢复与开发规范约束。

---

## 项目目标

从非接触式 **24GHz 连续波雷达**信号重建高保真 **ECG 信号**，核心创新：
1. **导数感知编码器（KI）**：将雷达信号与其一阶/二阶差分拼接为3通道输入，显式提供速度/加速度信息
2. **多头峰值辅助模块（PAM + CP）**：同时检测 QRS/P/T 三类波群，节律特征向量驱动 TFiLM 调制
3. **EMD 物理对齐层（PA）**：可学习深度卷积 FIR 滤波器，自动补偿雷达-ECG 物理时延
4. **自适应多任务损失（Kendall & Gal 2018）**：4项任务的同方差不确定性权重，消除手动超参调优

**环境**：`conda activate cyberbrain`（PyTorch 1.13.1, CUDA 11.3, RTX 4080 SUPER 16GB）

---

## 当前状态（2026-04-19）

| 阶段 | 状态 |
|------|------|
| Phase 1：数据预处理（Step 1-4）| ✅ 完成 — 30受试者，12,807段，5-Fold CV |
| Phase 1b：P/T波标注（Step 2b）| ⏳ 待运行 — `step2b_delineate.py` 已实现，需在实际数据上执行 |
| Phase 2：V2 模型代码 | ✅ 完成 — 所有模块已实现，forward+backward 验证通过 |
| Phase 3：训练框架 | ✅ 完成 — train.py / test.py / run_ablation.sh 已实现 |
| Phase 4：消融实验训练 | ⏳ 待执行 — step2b 完成后启动 Model A/B/C/D |

---

## SSM 训练速度说明

`src/models/backbone/ssm.py` 的 SelectiveScan 后端按以下优先级自动选择：

| 优先级 | 后端 | 速度 | 安装状态 |
|--------|------|------|---------|
| 1 | `mamba-ssm` 官方 CUDA kernel | ~14ms/call | **未安装** |
| 2 | `selective_scan_cuda`（手动编译）| ~14ms/call | 未安装 |
| 3 | TorchScript JIT（当前使用）| **138ms/call** | 自动回退 |

**当前 JIT 速度**：~274s/epoch（含 backward），训练150 epochs约 11.4h/fold，早停后约5-6h/fold。

**安装 CUDA kernel（可选）**：`bash scripts/install_mamba_cuda.sh`

---

## 数据集

**Schellenberger et al., *Scientific Data* 7:291 (2020)**

| 项目 | 内容 |
|------|------|
| 受试者 | 30名健康成人（GDN0001–GDN0030） |
| 雷达信号 | `radar_i`, `radar_q`（小写），2000 Hz |
| ECG 导联 | `tfm_ecg2`（**导联II**，R波最明显），**2000 Hz** |
| 使用场景 | **Resting（全30人）/ Valsalva（27人）/ Apnea（24人）** |
| 忽略场景 | TiltUp / TiltDown |

**Valsalva 缺失**：GDN0015, GDN0024, GDN0026
**Apnea 缺失**：GDN0001–0003, GDN0015, GDN0024, GDN0026

**数据集统计（已完成预处理）**：

| 场景 | 受试者数 | 总分段数 |
|------|---------|---------|
| Resting | 30 | 4,714 |
| Valsalva | 27 | 6,953 |
| Apnea | 24 | 1,140 |
| **合计** | 30 | **12,807** |

**5-Fold CV 划分**（每折6人，seed=42）：

| Fold | 验证集受试者 |
|------|------------|
| fold_0 | GDN0009, GDN0010, GDN0016, GDN0018, GDN0024, GDN0028 |
| fold_1 | GDN0001, GDN0005, GDN0013, GDN0017, GDN0025, GDN0029 |
| fold_2 | GDN0002, GDN0003, GDN0006, GDN0012, GDN0014, GDN0023 |
| fold_3 | GDN0004, GDN0019, GDN0022, GDN0026, GDN0027, GDN0030 |
| fold_4 | GDN0007, GDN0008, GDN0011, GDN0015, GDN0020, GDN0021 |

---

## 整体架构（V2）

```
雷达信号 [B,1,L]
   │
   ├─ diff → [B,3,L]（原始+速度+加速度）
   │       │
   │       ├────────────────────────────────────────┐
   │       ▼                                        ▼
   │  ┌─────────────────────┐          ┌────────────────────────────────┐
   │  │  PAM（多头峰值检测）  │  γ,β     │  主干 Backbone                  │
   │  │  Multi-scale Conv    │ ──────►  │  Multi-scale Encoder(3→C,×4)   │
   │  │  (k=7,15,31; 3ch)    │          │  + TFiLM 节律注入               │
   │  │  + VSSSBlock1D ×2    │          │                                │
   │  │  ┌───┬───┬───┐      │          │  GroupMambaBlock ×2             │
   │  │  QRS  P   T  │      │          │  ConformerFusionBlock           │
   │  │  └───┴───┘   │      │          │  EMD 物理对齐层（可选）           │
   │  │  节律向量→TFiLM│      │          │  Decoder（×4上采样）             │
   │  └─────┬─────────┘      │          └──────────────┬─────────────────┘
   │        │ L_peak          │                         │
   │        ▼                 │                         ▼
   │    L_peak Loss           │               ECG重建输出 [B,1,L]
   └──────────────────────────┘
```

**4项任务自适应加权损失（V2）**：
```
L_total = Σ_i [ 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i ]
任务: [L_recon, L_peak, L_der, L_interval]
log_vars = nn.Parameter(zeros(4))  — 与模型参数共同训练
```

---

## 模块详细设计（V2）

### 1. 导数感知编码器（KI — Kinematic Input）

**位置**：`forward()` 入口，适用于 `input_type='raw'` 或 `'phase'`

```python
v = torch.diff(x, dim=-1, prepend=x[:, :, :1])   # 速度 [B,1,L]
a = torch.diff(v, dim=-1, prepend=v[:, :, :1])   # 加速度 [B,1,L]
x_input = torch.cat([x, v, a], dim=1)             # [B,3,L]
```

`spec` 输入路径不做差分，保持单通道输入。

### 2. 峰值检测辅助模块（PAM — V2 多头）

```
输入: [B,3,L]（1D路径）或 [B,1,F,T]（spec路径）
  → Multi-scale Conv1d(3→32, k=7/15/31) × 3 路 → Concat → [B,96,L]
  → VSSSBlock1D × 2 → LayerNorm
     ├── Head_QRS: Conv1d(96→1)+Sigmoid → qrs_mask [B,1,L]，σ=5（25ms）
     ├── Head_P:   Conv1d(96→1)+Sigmoid → p_mask   [B,1,L]，σ=10（50ms）
     ├── Head_T:   Conv1d(96→1)+Sigmoid → t_mask   [B,1,L]，σ=15（75ms）
     └── Head2:    AdaptiveMaxPool1d(1) → rhythm_vec [B,96] → TFiLM
```

P/T 波监督需先运行 `step2b_delineate.py` 生成标注；失败样本置 `valid=False`，对应 BCE 项自动屏蔽。

### 3. TFiLM 节律注入

```python
gamma, beta = TFiLMGenerator(rhythm_vec)  # rhythm_vec 来自 PAM Head2: [B,96]
f_i = BN(conv_i(x_input))
f_i = f_i * (1 + gamma_i[:, :, None]) + beta_i[:, :, None]
f_i = ReLU(f_i)
```

### 4. EMD 物理对齐层（PA — Phase Alignment，可选）

**位置**：ConformerFusionBlock 之后、Decoder 之前

```python
class EMDAlignLayer(nn.Module):
    def __init__(self, channels, max_delay=20):  # max_delay=20 → 100ms@200Hz
        # 逐通道 FIR：Conv1d(C,C, k=41, groups=C)
        # 手动初始化为 Dirac delta（零延迟）
        # 训练中自发学习物理时延补偿
```

`use_emd=False` 时跳过（Model A / Model B 消融配置）。

### 5. V2 损失函数

| 任务 | 公式 | 说明 |
|------|------|------|
| `L_recon` | L1 + 0.1·MultiResSTFT（SC+log-mag）| 时域+频域波形重建 |
| `L_peak` | BCE(QRS) + masked BCE(P) + masked BCE(T) | 三路峰值定位 |
| `L_der` | L1(diff1) + L1(diff2) | QRS 边界锐化 |
| `L_interval` | soft-argmax PR 间期惩罚（正常范围 120-200ms）| 房室传导约束 |

**训练策略**：前 5 epoch warm-up 只训练 `L_recon`，之后解锁全部4项。

---

## 消融实验框架（V2，三维 KI/PA/CP）

| 变体 | KI（3通道diff）| PA（EMD层）| CP（多头PAM）| 参数量 |
|------|:---:|:---:|:---:|------|
| **Model A** — Baseline | ✓ | | | 1.49M |
| **Model B** — +PA | ✓ | ✓ | | 1.50M |
| **Model C** — +PA+CP | ✓ | ✓ | ✓ | 1.69M |

> **注**：KI（torch.diff）在 `radar_phase` 输入下始终激活，无需单独开关。Model A/B 的差异仅在 PAM/EMD；若需严格隔离 KI 增益可补充 `--use_ki false` 参数。

其余消融：
| 实验 | 改动 | 状态 |
|------|------|------|
| D4 | 去除 ConformerFusionBlock | 待模型支持 `--use_conformer` |
| D5 | 仅 resting 训练 → 全场景泛化测试 | 待训练 |

---

## 评估指标

**波形重建**：MAE / RMSE / PCC / PRD
**峰值检测**（重建ECG上重新运行 NeuroKit2）：F1（R峰）
**临床生理**：RR 间期误差 / QRS 时限误差 / PR 间期误差 / QT 间期误差

---

## 数据文件结构

```
dataset/
    GDN0001/
        resting/
            radar_raw.npy         # (L_200,)  全段，椭圆校正后相位
            radar_phase.npy       # (L_200,)  带通滤波相位
            radar_spec_input.npy  # (1,33,T)  细粒度STFT，模型输入用
            ecg_clean.npy         # (L_200,)  NeuroKit2清洗后ECG
            rpeak_indices.npy     # (M,)      R峰全局索引@200Hz
            pwave_indices.npy     # (M_p,)    P波索引@200Hz（step2b生成）
            twave_indices.npy     # (M_t,)    T波索引@200Hz（step2b生成）
            segments/
                radar_raw.npy         # [N,1,1600]
                radar_phase.npy       # [N,1,1600]
                radar_spec_input.npy  # [N,1,33,~193]
                ecg.npy               # [N,1,1600]  per-segment归一化[0,1]
                rpeak.npy             # [N,1,1600]  高斯软标签，σ=5
                pwave.npy             # [N,1,1600]  σ=10（step2b+step4生成）
                twave.npy             # [N,1,1600]  σ=15
                pwave_valid.npy       # [N,]  bool
                twave_valid.npy       # [N,]  bool
    ...
    metadata.json
    qc_report.json
```

---

## 文件结构

```
BeatAware-Radar2ECGNet/
├── data_preprocessing/
│   ├── step1_radar_processing.py    # 椭圆校正 + 相位解调 + STFT
│   ├── step2_ecg_processing.py      # NeuroKit2 清洗 + R峰检测 + 降采样
│   ├── step2b_delineate.py          # P/T 波标注（nk.ecg_delineate）← V2 新增
│   ├── step3_qc.py                  # 质量控制 + QC 报告
│   ├── step4_segment_save.py        # 对齐 + 分段 + 保存 NPY（含 pwave/twave）
│   ├── verify_dataset.py            # 数据集完整性校验
│   ├── PREPROCESSING_LOG.md         # 预处理运行记录
│   └── utils/
│       ├── ellipse_correction.py
│       ├── gaussian_mask.py
│       └── mat_loader.py
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── ssm.py               # VSSSBlock1D + SelectiveScan1D
│   │   │   └── group_mamba.py       # GroupMambaBlock
│   │   ├── modules/
│   │   │   ├── peak_module.py       # PAM（V2：3路输出 QRS/P/T）
│   │   │   └── tfilm.py             # TFiLMGenerator
│   │   └── BeatAwareNet/
│   │       └── radar2ecgnet.py      # 主模型（含 EMDAlignLayer）
│   ├── data/
│   │   └── dataset.py               # RadarECGDataset（支持 pwave/twave）
│   ├── losses/
│   │   └── losses.py                # TotalLoss（V2：4任务自适应加权）
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── seeding.py
├── configs/
│   └── config.py
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── run_ablation.sh              # V2 Model A/B/C/D 批量脚本
│   ├── plot_training_curves.py
│   ├── plot_subject_metrics.py
│   ├── summarize_ablation.py
│   └── install_mamba_cuda.sh
├── tests/
│   └── visualize_dataset.py
├── docs/
│   ├── ARCHITECTURE.md              # 完整架构设计文档（权威来源）
│   └── RESULTS_GUIDE.md             # 实验结果解读指南
├── papers/                          # 参考文献 PDF
├── IMPLEMENTATION_SPEC_V1.md        # V2 优化方案设计规格
└── CLAUDE.md                        # 本文件
```

---

## 常用命令

```bash
# 训练单个变体（fold 0 快速验证）
python scripts/train.py --exp_tag ModelD_full --input_type phase \
    --use_pam true --use_emd true --fold_idx 0 --epochs 150

# 完整消融实验（全部 folds）
bash scripts/run_ablation.sh

# 运行 P/T 波标注（step2b，需先完成 step1-4）
conda activate cyberbrain
python data_preprocessing/step2b_delineate.py

# 生成 pwave/twave 分段（重新运行 step4）
python data_preprocessing/step4_segment_save.py
```

---

## 开发约束

- **不复用已有项目代码**，所有模块从零实现
- `docs/ARCHITECTURE.md` 是架构权威来源；若本文件与其冲突，以 ARCHITECTURE.md 为准
- 每个新模块写完后必须用单元测试验证 tensor shape
- 实验输出统一写入 `experiments/<EXP_TAG>/`，不污染代码目录
