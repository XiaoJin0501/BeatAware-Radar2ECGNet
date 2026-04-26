# CLAUDE.md — BeatAware-Radar2ECGNet 项目指南

本文件为 Claude Code 提供项目级开发上下文。代码从零独立实现，**不直接复用任何已有项目代码**。完整架构设计见 `docs/ARCHITECTURE.md`，本文件侧重快速上下文恢复与开发规范约束。

---

## 项目目标与双数据集策略

从非接触式雷达信号重建高保真 ECG 信号，核心创新：
1. **导数感知编码器（KI）**：将雷达信号与其一阶/二阶差分拼接为3通道输入，显式提供速度/加速度信息
2. **多头峰值辅助模块（PAM + CP）**：同时检测 QRS/P/T 三类波群，节律特征向量驱动 TFiLM 调制
3. **EMD 物理对齐层（PA）**：可学习深度卷积 FIR 滤波器，自动补偿雷达-ECG 物理时延
4. **自适应多任务损失（Kendall & Gal 2018）**：4项任务的同方差不确定性权重，消除手动超参调优
5. **FMCWRangeEncoder**：针对 77GHz FMCW 50通道 range-time 矩阵的轻量前端编码器（GELU关键）

**双数据集评估策略**（面向期刊论文）：
- **MMECG**（主）：77GHz FMCW，11受试者，LOSO 11折，当前主要实验
- **Schellenberger**（次）：24GHz CW，30受试者，5-Fold CV，后续跨数据集验证

**环境**：`conda activate cyberbrain`（PyTorch 1.13.1, CUDA 11.3, RTX 4080 SUPER 16GB）

---

## 当前状态（2026-04-26）

| 阶段 | 状态 |
|------|------|
| **MMECG 预处理** | ✅ 完成 — `scripts/preprocess_mmecg.py`，dataset_mmecg/ 已生成 |
| **MMECG 训练（mmecg_G）** | 🔄 进行中 — 11折 LOSO，0.5-40Hz宽带 + GELU修复，~5-6h |
| **FMCWRangeEncoder 修复** | ✅ 完成 — `F.relu` → `F.gelu`，保留心脏AC信号负半周期 |
| Phase 1（Schellenberger 预处理）| ✅ 完成 — 30受试者，12,807段，5-Fold CV |
| Phase 1b：P/T波标注（Step 2b）| ⏳ 待运行 — `step2b_delineate.py` 已实现，需在实际数据上执行 |
| Phase 2：V2 模型代码 | ✅ 完成 — 所有模块已实现，forward+backward 验证通过 |
| Phase 3：训练框架 | ✅ 完成 — train.py / test.py / run_ablation.sh / train_mmecg.py / test_mmecg.py |
| Phase 4：Schellenberger 消融 | ⏳ MMECG 实验完成后执行 Model A/B/C |

---

## MMECG 数据集

**来源**：MMECG.h5（77GHz FMCW毫米波雷达）
**路径**：`/home/qhh2237/Datasets/MMECG/MMECG.h5`
**预处理输出**：`dataset_mmecg/`

| 项目 | 内容 |
|------|------|
| 受试者数量 | 11名（subject_1 ~ subject_11）|
| 雷达信号 | RCG，50个range bin，shape: (35505, 50)，200Hz |
| ECG 信号 | shape: (35505, 1)，200Hz |
| 生理状态 | NB(0)=正常呼吸 / IB(1)=不规则呼吸 / SP(2) / PE(3)=运动后 |
| 分段窗口 | 1600点（8s），步长800点（50%重叠）|
| 数据集划分 | LOSO 11折（fold_i 留出 subject_ids[i] 作测试集）|

**预处理关键参数**：
- RCG：0.5-40Hz 宽带带通 + 逐通道 z-score（保留 QRS 高频形态，使 SE 注意力按质量选bin）
- ECG：0.5-40Hz 带通 + per-window min-max [0,1]

**输出结构**：
```
dataset_mmecg/
    subject_1/
        rcg.npy     [N, 50, 1600]  float32
        ecg.npy     [N,  1, 1600]  float32
        rpeak.npy   [N,  1, 1600]  float32
        meta.npy    [N,  2]        int32  (subject_id, state_code)
    ...
    subject_11/
    metadata_mmecg.json
```

---

## Schellenberger 数据集

**来源**：Schellenberger et al., *Scientific Data* 7:291 (2020)

| 项目 | 内容 |
|------|------|
| 受试者 | 30名健康成人（GDN0001–GDN0030） |
| 雷达信号 | `radar_i`, `radar_q`（小写），2000 Hz |
| ECG 导联 | `tfm_ecg2`（**导联II**，R波最明显），**2000 Hz** |
| 使用场景 | **Resting（全30人）/ Valsalva（27人）/ Apnea（24人）** |
| 忽略场景 | TiltUp / TiltDown |

**数据集统计（已完成预处理）**：

| 场景 | 受试者数 | 总分段数 |
|------|---------|---------|
| Resting | 30 | 4,714 |
| Valsalva | 27 | 6,953 |
| Apnea | 24 | 1,140 |
| **合计** | 30 | **12,807** |

**5-Fold CV 划分**（每折6人，seed=42）：

| Fold | 测试集受试者 |
|------|------------|
| fold_0 | GDN0009, GDN0010, GDN0016, GDN0018, GDN0024, GDN0028 |
| fold_1 | GDN0001, GDN0005, GDN0013, GDN0017, GDN0025, GDN0029 |
| fold_2 | GDN0002, GDN0003, GDN0006, GDN0012, GDN0014, GDN0023 |
| fold_3 | GDN0004, GDN0019, GDN0022, GDN0026, GDN0027, GDN0030 |
| fold_4 | GDN0007, GDN0008, GDN0011, GDN0015, GDN0020, GDN0021 |

---

## FMCWRangeEncoder 设计

**文件**：`src/models/modules/fmcw_encoder.py`
**接口**：`(B, 50, L) → (B, 3, L)`

```
(B, 50, L)
  ↓ DepthwiseConv1d(50, k=61, groups=50) + BN + GELU   ← 关键：GELU非ReLU
  ↓ SE attention: AdaptiveAvgPool → Linear(50→6) + ReLU → Linear(6→50) + Sigmoid
  ↓ Conv1d(50→3, k=1) + BN
(B, 3, L)
```

**为什么是 GELU**：BatchNorm 输出均值为零，ReLU 截断负值 → 心脏 AC 信号 ~50% 信息丢失。GELU 在负值区间平滑衰减，保留完整心脏波形。这是模型性能的关键修复。

---

## 整体架构（V2）

```
MMECG 路径:  RCG [B,50,L] → FMCWRangeEncoder → [B,3,L] ─┐
                                                          ▼
Schellenberger:  [B,1,L] → diff → [B,3,L] ────────────►  [B,3,L]
                                          │
                    ┌─────────────────────┼──────────────────────────┐
                    ▼                     │                           ▼
              PAM（峰值检测）              │                    主干 Backbone
              Multi-scale Conv            │              Multi-scale Encoder(×4)
              + VSSSBlock1D ×2            │                  + TFiLM 节律注入
              → QRS/P/T mask + rhythm_vec │              GroupMambaBlock ×2
                    │                     │              ConformerFusionBlock
                    ▼  L_peak             │              EMD 对齐层
               BCE Loss                  └──── γ,β ────► Decoder → ECG [B,1,L]
```

**4项任务自适应加权损失（V2）**：
```
L_total = Σ_i [ 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i ]
任务: [L_recon, L_peak, L_der, L_interval]
```

---

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **PCC** | Pearson 相关系数 | 越大越好，>0.85 视为良好 |
| **MAE** | 时域绝对误差 | 越小越好 |
| **RMSE** | 均方根误差 | 越小越好 |
| **PRD** | Percent Root-mean-square Difference | 越小越好，<10% 视为良好 |
| **F1（R峰）** | R峰检测 F1 Score | 越大越好，>0.90 临床可用 |

**MMECG 专项**：按状态细分（NB / IB / SP / PE）+ LOSO 11折均值 ± 标准差

---

## SSM 训练速度说明

`src/models/backbone/ssm.py` 的 SelectiveScan 后端按以下优先级自动选择：

| 优先级 | 后端 | 速度 | 安装状态 |
|--------|------|------|---------|
| 1 | `mamba-ssm` 官方 CUDA kernel | ~14ms/call | **未安装** |
| 2 | `selective_scan_cuda`（手动编译）| ~14ms/call | 未安装 |
| 3 | TorchScript JIT（当前使用）| **138ms/call** | 自动回退 |

**MMECG 训练速度**：~30s/epoch（batch_size=32，11折 × 150 epochs，早停后约 5-6h 完成）
**Schellenberger 训练速度**：~274s/epoch（含 backward），训练150 epochs约 11.4h/fold

**安装 CUDA kernel（可选）**：`bash scripts/install_mamba_cuda.sh`

---

## 常用命令

```bash
# ── MMECG 相关 ─────────────────────────────────────────────────────
# 预处理（仅需一次，已完成）
conda run -n cyberbrain python scripts/preprocess_mmecg.py

# 训练单折验证
conda run -n cyberbrain python scripts/train_mmecg.py \
    --exp_tag mmecg_test --fold_idx 0 --epochs 5

# 训练全部11折（后台）
nohup conda run -n cyberbrain python scripts/train_mmecg.py \
    --exp_tag mmecg_G --fold_idx -1 --epochs 150 \
    > experiments_mmecg/mmecg_G_train.log 2>&1 &

# 测试（所有折）
conda run -n cyberbrain python scripts/test_mmecg.py --exp_tag mmecg_G

# 测试（单折）
conda run -n cyberbrain python scripts/test_mmecg.py --exp_tag mmecg_G --fold_idx 0

# ── Schellenberger 相关 ────────────────────────────────────────────
# 训练单个变体（fold 0 快速验证）
python scripts/train.py --exp_tag ModelD_full --input_type phase \
    --use_pam true --use_emd true --fold_idx 0 --epochs 150

# 完整消融实验（全部 folds）
bash scripts/run_ablation.sh

# 运行 P/T 波标注（step2b，需先完成 step1-4）
conda activate cyberbrain
python data_preprocessing/step2b_delineate.py

# ── 监控训练 ────────────────────────────────────────────────────────
# 查看 MMECG 训练日志
tail -f experiments_mmecg/mmecg_G_train.log

# 查看进程
nvidia-smi  # GPU占用
ps aux | grep train_mmecg  # 训练进程
```

---

## 数据文件结构

```
dataset/                              # Schellenberger 预处理输出
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

dataset_mmecg/                        # MMECG 预处理输出
    subject_1/
        rcg.npy     [N, 50, 1600]
        ecg.npy     [N,  1, 1600]
        rpeak.npy   [N,  1, 1600]
        meta.npy    [N,  2]
    ...
    subject_11/
    metadata_mmecg.json
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
│   │   │   ├── tfilm.py             # TFiLMGenerator
│   │   │   └── fmcw_encoder.py      # FMCWRangeEncoder（MMECG专用，GELU关键）
│   │   └── BeatAwareNet/
│   │       └── radar2ecgnet.py      # 主模型（含 EMDAlignLayer，支持 fmcw 输入）
│   ├── data/
│   │   ├── dataset.py               # RadarECGDataset（Schellenberger）
│   │   └── mmecg_dataset.py         # MMECGDataset + build_loso_loaders
│   ├── losses/
│   │   └── losses.py                # TotalLoss（V2：4任务自适应加权）
│   └── utils/
│       ├── logger.py
│       ├── metrics.py               # compute_all_metrics（含 F1/PRD）
│       └── seeding.py
├── configs/
│   ├── config.py                    # Schellenberger 配置
│   └── mmecg_config.py              # MMECG 配置（input_type='fmcw', n_folds=11）
├── scripts/
│   ├── train.py                     # Schellenberger 训练
│   ├── test.py                      # Schellenberger 测试
│   ├── train_mmecg.py               # MMECG 训练（LOSO 11折）
│   ├── test_mmecg.py                # MMECG 测试（per-state 细分）
│   ├── preprocess_mmecg.py          # MMECG 预处理（H5 → NPY）
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
├── IMPLEMENTATION_SPEC_V1.md        # V2 优化方案设计规格（历史文档）
└── CLAUDE.md                        # 本文件
```

---

## 开发约束

- **不复用已有项目代码**，所有模块从零实现
- `docs/ARCHITECTURE.md` 是架构权威来源；若本文件与其冲突，以 ARCHITECTURE.md 为准
- 每个新模块写完后必须用单元测试验证 tensor shape
- 实验输出统一写入 `experiments/<EXP_TAG>/`（Schellenberger）或 `experiments_mmecg/<EXP_TAG>/`（MMECG），不污染代码目录
- 设计变更必须同时更新 `docs/ARCHITECTURE.md` 和 `CLAUDE.md`
