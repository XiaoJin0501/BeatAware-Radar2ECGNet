# BeatAware-Radar2ECGNet — 架构设计文档

> 记录实验动机、模块设计、Loss 策略、数据预处理方案和实验计划。
> 本项目从零开始独立实现，不直接复用任何已有代码。

---

## 1. 项目目标

从非接触式 24GHz 连续波雷达信号重建高保真 ECG 信号，核心挑战：

1. 雷达信号与 ECG 之间存在复杂的非线性映射，噪声多
2. ECG 波形精度要求高——P波、QRS波群、T波的位置与形态都需要准确重建
3. 心率节律信息（RR间期）需要被显式利用，而非让网络隐式学习

---

## 2. 数据集信息

**来源**：Schellenberger et al., *Scientific Data* 7:291 (2020)
DOI: 10.1038/s41597-020-00629-5 | Figshare: 10.6084/m9.figshare.12186516

### 2.1 基本信息

| 项目 | 内容 |
|------|------|
| 受试者数量 | 30名健康成人（GDN0001–GDN0030），16男14女 |
| 平均年龄 | 30.7 ± 9.9 岁 |
| 雷达系统 | 24GHz 六端口连续波雷达（Six-Port CW Radar）|
| 参考系统 | Task Force Monitor 3040i（ECG + ICG + BP 同步采集）|
| 原始文件格式 | MATLAB .mat（每受试者每场景一个文件）|
| 总记录时长 | ~24小时 (86459 秒) |

### 2.2 信号采样率

| 信号 | 字段名 | 采样率 | 备注 |
|------|-------|-------|------|
| 雷达 I/Q 原始信号 | `radar_i`, `radar_q`（**小写**） | 2000 Hz | |
| ECG（导联1、2） | `tfm_ecg2`, `tfm_ecg2` | **2000 Hz**（非1000Hz） | 与雷达等长，已天然对齐 |
| ICG | `tfm_icg` | 1000 Hz | |
| 血压 | `tfm_bp` | 200 Hz | |

> **注意**：`tfm_ecg2` 采样率为 **2000 Hz**（与雷达相同），两者天然等长、无需额外对齐。降采样因子均为 10（2000→200Hz）。

### 2.3 实验场景与数据可用性

本项目仅使用以下三个场景（不使用 TiltUp / TiltDown）：

| 场景 | 描述 | 平均时长 | 缺失受试者 |
|------|------|---------|----------|
| **Resting** | 平静仰卧呼吸，至少10分钟 | ~634s | 无（全部30人） |
| **Valsalva** | 瓦尔萨尔瓦动作，3次×20s，共~16分钟 | ~1000s | GDN0015, GDN0024, GDN0026 |
| **Apnea** | 屏气（吸气/呼气各一次） | ~170s | GDN0001, GDN0002, GDN0003, GDN0015, GDN0024, GDN0026 |

**缺失数据处理原则**：对应场景目录直接不创建，不填充零值。

---

## 3. 整体架构

```
雷达信号输入 [B, 1, L]
        │
        ├─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 ▼
┌───────────────────────┐                   ┌────────────────────────────┐
│  峰值检测辅助模块       │                   │  主干网络 (Backbone)         │
│  (Peak Auxiliary      │                   │                            │
│   Module, PAM)        │  gamma, beta       │  Multi-scale Encoder       │
│                       │ ─────────────────► │  (k=3,5,7,9) + TFiLM注入  │
│  Multi-scale Conv     │                   │                            │
│  (k=7,15,31)          │                   │  GroupMambaBlock x 2       │
│  + VSSSBlock1D x 2    │                   │                            │
│         │             │                   │  ConformerFusionBlock      │
│    ┌────┴────┐        │                   │                            │
│    │         │        │                   │  Decoder (上采样)           │
│    ▼         ▼        │                   └────────────┬───────────────┘
│  Head1     Head2      │                                │
│  峰值Mask  节律向量    │                                ▼
│  预测      → TFiLM    │                        ECG重建输出 [B, 1, L]
│  [B,1,L]              │
└──────┬────────────────┘
       │
       ▼
  L_peak Loss
```

**网络输入的三种雷达表征**（训练阶段分别对比，最终选择最优）：

| 表征 | 来源 | 特点 |
|------|------|------|
| `radar_raw` | 原始 I/Q 相位（弧度）时序 | 直接物理量，未做任何心脏信号分离 |
| `radar_phase` | 椭圆校正 + 反正切解调后的相位信号 | 线性位移，呼吸+心跳混合 |
| `radar_spec` | 雷达相位信号的短时傅里叶变换谱图 | 频域表征，心跳频率分量清晰 |

---

## 4. 各模块详细设计

### 4.1 峰值检测辅助模块 (PAM)

**作用**：显式引导网络学习 ECG 的节律位置（QRS波群），并将节律特征传递给 TFiLM。

**结构**：
```
输入: [B, 1, L]
  → Multi-scale Conv1d (k=7, 15, 31, padding=same, 各32通道)
  → Concat → [B, 96, L]
  → VSSSBlock1D x 2  (1D Mamba序列建模)
  → LayerNorm
     ├── Head1: Conv1d(96→1) + Sigmoid → 峰值Mask预测 [B, 1, L]
     └── Head2: AdaptiveMaxPool1d(1) → Flatten → [B, 96]
                → TFiLMGenerator → (gamma [B,C], beta [B,C])
```

**多尺度卷积感受野设计（雷达信号以200Hz处理为例）**：

| 卷积核 | 感受野 | 捕捉目标 |
|-------|-------|---------|
| k=7 | ~35ms | QRS 波群的尖锐峰值 |
| k=15 | ~75ms | P波和T波的缓慢形态 |
| k=31 | ~155ms | RR 间期的整体节律 |

> 实际卷积核大小根据预处理后的目标采样率调整。

**监督信号**：以 R 峰为中心的高斯加权软标签（由预处理步骤3生成的 `rpeak.npy`）

### 4.2 TFiLM 节律注入

**作用**：将 PAM 提取的节律向量动态调制主干网络 Encoder 的每一路特征。

**注入方式**：
```
gamma, beta = TFiLMGenerator(rhythm_vec)   # rhythm_vec 来自 PAM Head2: [B, C]

# 对 Encoder 第 i 路特征 f_i: [B, C, L]
f_i = BN(conv_i(x))
f_i = f_i * (1 + gamma_i[:, :, None]) + beta_i[:, :, None]
f_i = ReLU(f_i)
```

**与参考项目的关键区别**：
- 参考项目：节律向量直接从原始雷达信号的平行分支提取
- 本项目：节律向量来自 PAM Head2，**先经过峰值感知的 Mamba 序列建模**，携带了心跳节律的高层语义

### 4.3 主干 Backbone

**Multi-scale Encoder（TFiLM注入）**：
```
4路并行 Conv1d:
  Conv1d(1, C, k=3, stride=4) → BN → TFiLM → ReLU
  Conv1d(1, C, k=5, stride=4) → BN → TFiLM → ReLU
  Conv1d(1, C, k=7, stride=4) → BN → TFiLM → ReLU
  Conv1d(1, C, k=9, stride=4) → BN → TFiLM → ReLU
  → Concat → [B, 4C, L/4]
```

**Bottleneck — GroupMambaBlock**：
```
GroupMambaBlock(4C, num_groups=4) x 2
  结构: LayerNorm → 分4组 → 各组独立 VSSSBlock1D → Concat → CAM调制 → 投影
  CAM (Channel Affine Modulation): AvgPool → FC → ReLU → FC → Sigmoid
```

**VSSSBlock1D（1D Mamba核心）**：
```
in_proj → split(x, z) → DepthwiseConv1d → SiLU → x_proj
→ split(dt, B, C) → dt_proj → SelectiveScan1D → gate(z) → out_proj
```

**ConformerFusionBlock**：
```
MHSA → DepthwiseConv1d(k=31) + BN + SiLU → FFN
```

**Decoder**：
```
ConvTranspose1d(4C → 2C, stride=2) → ReLU
ConvTranspose1d(2C → C,  stride=2) → ReLU
Conv1d(C → 1) → Sigmoid → ECG重建输出 [B, 1, L]
```

---

## 5. Loss 函数设计

```
L_total = L_time + α·L_freq + β·L_peak
```

| 项 | 公式 | 作用 | 建议初始权重 |
|----|------|------|------------|
| `L_time` | MAE(pred_ecg, gt_ecg) | 时域波形重建 | 1.0（基准，固定不调） |
| `L_freq` | Multi-resolution STFT Loss | 频域结构一致性（QRS高频 + RR低频） | α：待消融实验确定 |
| `L_peak` | BCE(pred_mask, gt_mask) | 峰值定位监督（use_pam=False 时不计算） | β：待消融实验确定 |

**Multi-resolution STFT 参数（目标ECG采样率200Hz）**：
```python
FFT_SIZES   = [128, 256, 512]   # 覆盖 QRS(~40ms) / P-T波(~150ms) / RR间期(~600ms)
HOP_SIZES   = [64,  128, 256]
WIN_LENGTHS = [16,  32,  64]
```

---

## 6. 数据预处理方案

### 6.1 原始数据结构（.mat格式）

```
raw_data/
    GDN0001/
        GDN0001_Resting.mat
        GDN0001_Valsalva.mat
        GDN0001_Apnea.mat       ← 不存在（该受试者无此场景）
        ...
    GDN0002/
        ...
    additional_data.xlsx        ← 元数据（采样率、场景时长等）
```

**每个 .mat 文件中的关键字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `radar_I` | 2000Hz 时序 | 雷达 I 路原始信号（mV）|
| `radar_Q` | 2000Hz 时序 | 雷达 Q 路原始信号（mV）|
| `tfm_ecg2` | 1000Hz 时序 | ECG 导联1（mV）|
| `tfm_ecg2` | 1000Hz 时序 | ECG 导联2（mV）|
| `fs_radar` | 标量 | 雷达实际采样率（标称2000Hz）|

### 6.2 预处理流程

```
原始 .mat 文件
    │
    ├─ Step 1: 雷达信号处理（生成3种表征）
    │       ├── 椭圆校正（补偿 I/Q 幅度/相位不平衡）
    │       ├── 反正切解调 → radar_phase（弧度，线性位移）
    │       ├── 带通滤波：0.5~10Hz（保留心跳+呼吸频段）
    │       └── STFT → radar_spec（时频谱图）
    │               窗长: 256点, 步长: 32点, FFT: 512点（基于2000Hz）
    │
    ├─ Step 2: ECG 处理（使用 NeuroKit2）
    │       ├── 加载 tfm_ecg2（导联1作为主导联）
    │       ├── nk.ecg_clean()：带通滤波（0.5~40Hz）+ 基线校正
    │       ├── nk.ecg_peaks()：R峰检测（Pan-Tompkins算法）
    │       ├── 下采样至目标采样率（如 200Hz）
    │       └── per-segment min-max 归一化至 [0, 1]
    │
    ├─ Step 3: 峰值标注（高斯Mask生成）
    │       ├── 将 R峰位置（1000Hz坐标）映射到目标采样率
    │       └── 以每个R峰为中心生成高斯软标签：
    │               σ = 5个采样点（@200Hz = 25ms，覆盖QRS波群）
    │               mask[t] = Σ exp(-0.5 * ((t - r_i) / σ)²)
    │               clip(mask, 0, 1)
    │
    ├─ Step 4: 质量控制（QC）
    │       ├── 剔除 NeuroKit2 R峰检测置信度 < 阈值的片段
    │       ├── 剔除 RR间期异常的片段（心率超出 30~200 BPM 范围）
    │       ├── 剔除 ECG 信噪比过低的片段（平均振幅 < 阈值）
    │       ├── 剔除 雷达信号中含有明显运动伪影的片段（检测异常大幅度跳变）
    │       └── 生成 QC 报告（每个受试者的合格率统计）
    │
    ├─ Step 5: 时间对齐与分段
    │       ├── 雷达与ECG已由数据集提供者通过互相关同步，直接使用
    │       ├── 将雷达（2000Hz）和ECG（1000Hz）统一下采样到目标采样率（TBD）
    │       └── 滑动窗口分段（窗口长度和步长TBD，取决于目标采样率）
    │
    └─ Step 6: 保存（NPY格式，按受试者+场景组织）
```

### 6.3 输出数据结构

```
dataset/
    GDN0001/
        resting/
            radar_raw.npy        # 原始相位信号（椭圆校正后，未滤波）[N, 1, L]
            radar_phase.npy      # 带通滤波后的相位时序 [N, 1, L]
            radar_spec.npy       # STFT时频谱图 [N, F, T]
            ecg.npy              # 处理后ECG，归一化到[0,1] [N, 1, L]
            rpeak.npy            # 高斯R峰软标签 [N, 1, L]
        valsalva/
            radar_raw.npy
            radar_phase.npy
            radar_spec.npy
            ecg.npy
            rpeak.npy
        # apnea/ 不存在（GDN0001无此场景数据，不创建目录）
    GDN0002/
        resting/
        valsalva/
        # apnea/ 不存在
    GDN0004/
        resting/
        valsalva/
        apnea/
    ...
    GDN0030/
        resting/
        valsalva/
        apnea/
    metadata.json
```

**metadata.json 内容**：
```json
{
  "dataset_name": "GUARDIAN_Radar_ECG",
  "source_paper": "Schellenberger et al., Scientific Data 2020",
  "radar_fs_raw": 2000,
  "ecg_fs_raw": 1000,
  "target_fs": null,
  "segment_length": null,
  "segment_stride": null,
  "ecg_tool": "NeuroKit2",
  "rpeak_gaussian_sigma": 5,
  "normalization": "per_segment_minmax",
  "scenarios": ["resting", "valsalva", "apnea"],
  "subjects_total": 30,
  "subjects_with_apnea": 24,
  "subjects_with_valsalva": 27,
  "subjects_missing_apnea": ["GDN0001", "GDN0002", "GDN0003", "GDN0015", "GDN0024", "GDN0026"],
  "subjects_missing_valsalva": ["GDN0015", "GDN0024", "GDN0026"],
  "qc_params": {}
}
```

### 6.4 目标采样率（已确认）

雷达和ECG统一下采样到 **200 Hz**：
- 与现有文献（BeatAware_R-M2Net等）一致
- 8s窗口 = 1600采样点，计算量适中
- 分段步长50%重叠 = 4s = 800点

`metadata.json` 中：`target_fs=200`，`segment_length=1600`，`segment_stride=800`

---

## 7. 参考项目分析

调研了以下工作（仅作设计参考，不复用代码）：

| 项目 | 输入格式 | 峰值/节律监督 | 条件注入方式 | 主要不足 |
|------|---------|-------------|------------|---------|
| **BeatAware_R-M2Net** | 1D Radar | anchor branch → R峰Mask | TFiLM 注入 Encoder | 代码质量较低、架构有误，仅设计思路参考；峰值分支与TFiLM平行独立 |
| **radarODE-MTL** | 2D时频谱图 | anchor_decoder + PPI估计 | 无显式注入，多任务独立 | 节律估计与波形重建未耦合 |
| **AirECG** | 2D时频谱图 | 无峰值监督 | Diffusion + Cross-Attention | 推理慢，无峰值约束 |

**本项目核心创新**：将峰值检测模块与 TFiLM 节律注入**串联**，峰值模块提取的节律特征直接驱动 TFiLM，形成显式的信息流依赖。

---

## 8. 实验计划

### Exp A — 无峰值监督基线
- 模型：仅 Backbone（无 PAM，无 TFiLM）
- 输入：radar_phase（单一表征）
- Loss：L_time + α·L_freq
- 目的：纯重建能力基线

### Exp B — 雷达表征对比
- 模型：完整模型（PAM + TFiLM + Backbone）
- 分别用 radar_raw / radar_phase / radar_spec 作输入
- 目的：确定最优雷达输入表征

### Exp C — 本项目完整模型（最优表征）
- 模型：BeatAware-Radar2ECGNet
- Loss：α=0.05, β=1.0, γ=0.5（超参需消融验证）

### Exp D — 消融实验

| 实验 | 改动 | 目的 |
|------|------|------|
| D1 | 去除 PAM，用二值 R峰Mask 监督 | 高斯软标签 vs 二值标签 |
| D2 | 保留 PAM，TFiLM 改为平行（节律向量从原始雷达提取）| PAM→TFiLM 串联的增益 |
| D3 | PAM 用单一卷积核 k=15，去除多尺度 | 多尺度感受野的增益 |
| D4 | 去除 ConformerFusionBlock | Conformer 融合的必要性 |
| D5 | 仅用 resting 场景训练，测试跨场景泛化 | 场景多样性的增益 |

### 评估指标

| 指标 | 说明 |
|------|------|
| MAE | 时域绝对误差均值 |
| RMSE | 均方根误差 |
| PCC | Pearson 相关系数（整体波形相似度）|
| PRD | Percent Root-mean-square Difference（ECG领域标准指标）|
| F1 (R峰) | 在重建ECG上用 Pan-Tompkins 检测R峰，计算 F1 Score |

---

## 9. 文件结构规划

```
BeatAware-Radar2ECGNet/
├── data_preprocessing/
│   ├── step1_radar_processing.py    # 椭圆校正 + 相位解调 + STFT → radar_raw/phase/spec
│   ├── step2_ecg_processing.py      # NeuroKit2 清洗 + R峰检测 + 高斯Mask生成
│   ├── step3_qc.py                  # 质量控制：剔除低质量片段，生成QC报告
│   ├── step4_segment_save.py        # 对齐 + 分段 + 保存为NPY（按受试者/场景）
│   ├── verify_dataset.py            # 数据集完整性校验（shape、值域、缺失率统计）
│   └── utils/
│       ├── ellipse_correction.py    # I/Q椭圆校正工具函数
│       ├── gaussian_mask.py         # 高斯Mask生成
│       └── mat_loader.py            # .mat文件读取与字段解析
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── ssm.py               # VSSSBlock1D + SelectiveScan1D
│   │   │   └── group_mamba.py       # GroupMambaBlock
│   │   ├── modules/
│   │   │   ├── peak_module.py       # 峰值检测辅助模块 (PAM)
│   │   │   └── tfilm.py             # TFiLMGenerator
│   │   ├── morphradarnet/
│   │   │   └── radar2ecgnet.py      # 主模型 BeatAwareRadar2ECGNet
│   │   └── __init__.py
│   ├── data/
│   │   └── dataset.py               # RadarECGDataset（NPY格式加载，支持多表征）
│   ├── losses/
│   │   └── losses.py                # TotalLoss
│   └── utils/
│       ├── logger.py
│       ├── seeding.py
│       └── metrics.py               # MAE / RMSE / PCC / PRD / F1
├── configs/
│   └── config.py
├── scripts/
│   ├── train.py
│   ├── test.py
│   └── run_ablation.sh
├── experiments/                     # 实验输出（自动生成）
├── docs/
│   └── ARCHITECTURE.md              # 本文件
└── CLAUDE.md
```

---

## 10. 开发顺序

**阶段一：数据准备**（收到原始 .mat 数据后执行）

1. `data_preprocessing/utils/mat_loader.py` — .mat读取与字段解析
2. `data_preprocessing/utils/ellipse_correction.py` — I/Q椭圆校正
3. `data_preprocessing/utils/gaussian_mask.py` — 高斯Mask生成
4. `data_preprocessing/step1_radar_processing.py` — 3种雷达表征生成
5. `data_preprocessing/step2_ecg_processing.py` — ECG清洗 + R峰标注
6. `data_preprocessing/step3_qc.py` — 质量控制
7. `data_preprocessing/step4_segment_save.py` — 分段保存NPY
8. `data_preprocessing/verify_dataset.py` — 数据验证

**阶段二：模型搭建**（每步验证后进入下一步）

9.  `src/models/backbone/ssm.py` — VSSSBlock1D
10. `src/models/backbone/group_mamba.py` — GroupMambaBlock
11. `src/models/modules/tfilm.py` — TFiLMGenerator
12. `src/models/modules/peak_module.py` — PAM 峰值检测模块
13. `src/models/morphradarnet/radar2ecgnet.py` — 主模型整合
14. `src/losses/losses.py` — TotalLoss

**阶段三：训练与评估**

15. `src/data/dataset.py` — NPY数据加载器（支持多场景、多表征切换）
16. `configs/config.py` — 配置管理
17. `scripts/train.py` — 训练脚本
18. `scripts/test.py` — 测试 + 指标计算
19. `scripts/run_ablation.sh` — 消融实验批量脚本

---

## 11. 参数状态

| 参数 | 状态 | 值 |
|------|------|-----|
| 目标采样率 | **已确认** | **200 Hz** |
| 分段窗口长度 | **已确认** | **8s = 1600 采样点** |
| 分段步长/重叠率 | **已确认** | **50% = 4s = 800 点** |
| 使用ECG导联 | **已确认** | **tfm_ecg2（导联II，R波最明显）** |
| 高斯Mask σ | **已确认** | **5 采样点 = 25ms @ 200Hz** |
| 数据划分方式 | **已确认** | **5-Fold CV（按受试者），seed=42** |
| 使用场景 | **已确认** | **Resting / Valsalva / Apnea（3种，忽略TiltUp/TiltDown）** |
| Backbone C | **已确认** | **C=64，4C=256** |
| 输入适配方案 | **已确认** | **方案A：Input Adapter，Backbone统一；1D走Multi-scale Conv1d，spec走Spec Adapter** |
| radar_spec_input | **已确认** | **nperseg=64, noverlap=56 @200Hz → shape [1,33,196]，模型输入用** |
| radar_spec_loss | **已确认** | **多分辨率 nperseg=[128,256,512]，STFT Loss专用，3种表征实验统一** |
| QC阈值 | 待定 | 由数据探索后参数化确定 |
