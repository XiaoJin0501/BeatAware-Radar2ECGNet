# CLAUDE.md — BeatAware-Radar2ECGNet 项目指南

本文件为 Claude Code 提供项目级开发上下文。代码从零独立实现，**不直接复用任何已有项目代码**。完整架构设计见 `docs/ARCHITECTURE.md`，本文件侧重快速上下文恢复与开发规范约束。

---

## 项目目标

从非接触式 **24GHz 连续波雷达**信号重建高保真 **ECG 信号**，核心创新是将峰值检测模块（PAM）与 TFiLM 节律注入**串联**：PAM 提取的节律特征直接驱动 TFiLM，形成显式信息流依赖，而非平行独立分支。

**环境**：`conda activate cyberbrain`（PyTorch 1.13.1, CUDA 11.3, RTX 4080 SUPER 16GB）

---

## SSM 训练速度说明

`src/models/backbone/ssm.py` 的 SelectiveScan 后端按以下优先级自动选择：

| 优先级 | 后端 | 速度 | 安装状态 |
|--------|------|------|---------|
| 1 | `mamba-ssm` 官方 CUDA kernel | ~14ms/call | **未安装**（需手动安装）|
| 2 | `selective_scan_cuda`（手动编译）| ~14ms/call | 未安装 |
| 3 | TorchScript JIT（当前使用）| **138ms/call** | 自动回退 |

**当前 JIT 速度影响**：~274s/epoch（含 backward），约为 CUDA kernel 的 2x 慢。
训练 150 epochs 约 11.4h/fold，早停后实际约 5-6h/fold。

**安装 CUDA kernel（可选，需用户确认）**：
```bash
conda activate cyberbrain
bash scripts/install_mamba_cuda.sh
```
> ⚠️ PyTorch 1.13.1+cu113 与 nvcc 11.6 存在 minor version mismatch，安装可能失败。
> 若失败则保持 JIT 回退，训练结果完全一致。

**Claude Code 行为约定**：
- **不自动安装** `mamba-ssm` 或 `causal-conv1d`，须用户显式运行安装脚本
- 安装成功后代码自动切换到 CUDA kernel，无需修改任何训练代码
- 若用户要求安装，运行 `bash scripts/install_mamba_cuda.sh` 后验证速度

---

## 数据集

**Schellenberger et al., *Scientific Data* 7:291 (2020)**
DOI: 10.1038/s41597-020-00629-5 | Figshare: 10.6084/m9.figshare.12186516

| 项目 | 内容 |
|------|------|
| 受试者 | 30 名健康成人（GDN0001–GDN0030） |
| 雷达系统 | 24GHz 六端口连续波雷达 |
| 雷达原始信号 | `radar_i`, `radar_q`（**小写**），2000 Hz |
| ECG | `tfm_ecg2`（**导联II**，R波最明显，便于特征提取），**2000 Hz**（与雷达等长，已天然对齐） |
| 场景 | **Resting（全30人）/ Valsalva（27人）/ Apnea（24人）**，共3种场景用于训练评估 |
| 忽略场景 | TiltUp / TiltDown（原始数据中存在，但本项目不使用） |
| 缺失场景 | 对应目录直接不创建，不填充零值 |

**缺失受试者**：
- Valsalva 缺失：GDN0015, GDN0024, GDN0026
- Apnea 缺失：GDN0001, GDN0002, GDN0003, GDN0015, GDN0024, GDN0026

**三种雷达输入表征**（训练阶段对比，选最优）：

| 表征 | 文件 | 形状 | 描述 |
|------|------|------|------|
| `radar_raw` | `radar_raw.npy` | `[N,1,1600]` | 椭圆校正后相位时序，未滤波 |
| `radar_phase` | `radar_phase.npy` | `[N,1,1600]` | 椭圆校正 + 反正切解调 + 0.5~10Hz 带通滤波 |
| `radar_spec` | `radar_spec_input.npy` | `[N,1,33,196]` | 细粒度 STFT 时频图（nperseg=64, noverlap=56, @200Hz），用于模型输入 |

> **区分两种 spec**：
> - `radar_spec_input.npy`：细粒度单分辨率，形状 `[N,1,33,196]`，**模型输入用**
> - `radar_spec_loss.npy`：多分辨率（3组参数），形状 `[N,3,F,T]`，**STFT Loss 计算用**，所有3种表征实验统一使用

---

## 整体架构

```
雷达信号输入 [B, 1, L]
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌─────────────────────┐            ┌──────────────────────────┐
│  峰值检测辅助模块      │            │  主干 Backbone             │
│  (PAM)              │  γ, β      │                          │
│                     │ ─────────► │  Multi-scale Encoder     │
│  Multi-scale Conv   │            │  (k=3,5,7,9) + TFiLM注入 │
│  (k=7,15,31)        │            │                          │
│  + VSSSBlock1D x2   │            │  GroupMambaBlock x2      │
│         │           │            │                          │
│    ┌────┴────┐      │            │  ConformerFusionBlock    │
│  Head1    Head2     │            │                          │
│  峰值Mask  节律向量  │            │  Decoder（上采样）         │
│  [B,1,L]  → TFiLM  │            └──────────┬───────────────┘
└─────┬───────────────┘                       │
      │                                       ▼
  L_peak Loss                       ECG重建输出 [B, 1, L]
```

---

## 模块详细设计

### 1. 峰值检测辅助模块（PAM）

```
输入: [B, 1, L]
  → Multi-scale Conv1d (k=7, 15, 31, padding=same, 各32通道)
  → Concat → [B, 96, L]
  → VSSSBlock1D x2  (1D Mamba序列建模)
  → LayerNorm
     ├── Head1: Conv1d(96→1) + Sigmoid → 峰值Mask [B, 1, L]
     └── Head2: AdaptiveMaxPool1d(1) → Flatten → [B, 96]
                → TFiLMGenerator → (gamma [B,C], beta [B,C])
```

多尺度感受野（@200Hz）：k=7 → ~35ms（QRS尖峰），k=15 → ~75ms（P/T波），k=31 → ~155ms（RR节律）

### 2. TFiLM 节律注入

```python
gamma, beta = TFiLMGenerator(rhythm_vec)   # rhythm_vec 来自 PAM Head2: [B, C]

# 对 Encoder 第 i 路特征 f_i: [B, C, L]
f_i = BN(conv_i(x))
f_i = f_i * (1 + gamma_i[:, :, None]) + beta_i[:, :, None]
f_i = ReLU(f_i)
```

节律向量来自 PAM Head2（经过峰值感知 Mamba 建模），不是直接从原始雷达提取——这是与参考项目的核心区别。

### 3. 主干 Backbone

**C=64（已确认）**：Backbone 基础通道数，Multi-scale Encoder 输出 4C=256 通道。

**Input Adapter（方案A，按输入表征切换前端，Backbone 完全统一）**：

| 输入表征 | Adapter | 输出形状 |
|---------|---------|---------|
| `radar_raw` / `radar_phase` | Multi-scale Conv1d(1, C, k=3/5/7/9, stride=4) + TFiLM | `[B, 4C, L/4]` = `[B, 256, 400]` |
| `radar_spec_input` | Spec Adapter（Conv2d 压缩 F 轴 → 插值对齐 T=400） | `[B, 4C, 400]` = `[B, 256, 400]` |

> PAM 也随输入切换前端：1D 输入走 Conv1d；spec 输入走 Conv2d 投影到 1D 后再进 VSSSBlock1D。

**Multi-scale Encoder（1D 表征，TFiLM注入）**：
```
4路并行 Conv1d(1, 64, k=3/5/7/9, stride=4) → BN → TFiLM → ReLU
→ Concat → [B, 256, 400]
```

**Spec Adapter（radar_spec 表征）**：
```
radar_spec_input [B, 1, 33, 196]
→ Conv2d(1, 64, kernel=(33,1)) → [B, 64, 1, 196] → squeeze → [B, 64, 196]
→ Conv1d(64, 256, k=1)                            → [B, 256, 196]
→ F.interpolate(size=400)                          → [B, 256, 400]
```

**GroupMambaBlock**：
```
LayerNorm → 分4组 → 各组独立 VSSSBlock1D → Concat → CAM调制 → 投影
CAM: AvgPool → FC → ReLU → FC → Sigmoid
```

**VSSSBlock1D（1D Mamba核心）**：
```
in_proj → split(x, z) → DepthwiseConv1d → SiLU → x_proj
→ split(dt, B, C) → dt_proj → SelectiveScan1D → gate(z) → out_proj
```

**ConformerFusionBlock**：`MHSA → DepthwiseConv1d(k=31) + BN + SiLU → FFN`
> 保留原因：Mamba 做序列扫描，无法做对称的跨尺度注意力；MHSA 专责跨尺度特征对齐，与 Mamba 功能互补。D4 消融验证其增益。

**Decoder**：
```
ConvTranspose1d(4C→2C, stride=2) → ReLU
ConvTranspose1d(2C→C,  stride=2) → ReLU
Conv1d(C→1) → Sigmoid → [B, 1, L]
```

---

## Loss 函数

```
L_total = L_time + α·L_freq + β·L_peak
```

| 项 | 公式 | 权重 |
|----|------|------|
| `L_time` | MAE(pred_ecg, gt_ecg) | 1.0（参考锚点，不调）|
| `L_freq` | Multi-resolution STFT Loss | α：待实验确定 |
| `L_peak` | BCE(pred_mask, gt_mask) | β：待实验确定 |

> `L_time` 固定为 1.0 作为参考锚点——Loss 权重只有相对比值有意义，α 和 β 均定义为相对于 L_time 的比例。自由超参只有 α 和 β。

**α、β 及 STFT 参数均须通过实验验证后确定，不预设固定值。**

**STFT 参数（@目标采样率，以下为候选值，待确认）**：
```python
FFT_SIZES   = [128, 256, 512]   # 覆盖 QRS / P-T波 / RR间期
HOP_SIZES   = [64,  128, 256]
WIN_LENGTHS = [16,  32,  64]
```

---

## 数据预处理流程

```
原始 .mat → Step1 雷达处理 → Step2 ECG处理(NeuroKit2) →
Step3 高斯Mask生成 → Step4 质量控制(QC) →
Step5 时间对齐+分段 → Step6 保存NPY
```

**输出结构**：
```
dataset/
    GDN0001/
        resting/
            radar_raw.npy    # [N, 1, L]
            radar_phase.npy  # [N, 1, L]
            radar_spec.npy   # [N, F, T]
            ecg.npy          # [N, 1, L]，归一化到[0,1]
            rpeak.npy        # [N, 1, L]，高斯R峰软标签
        valsalva/
            ...
        # apnea/ 不存在，直接不创建
    ...
    metadata.json
```

**高斯R峰软标签**：以R峰为中心，σ=5采样点（@200Hz=25ms），clip到[0,1]

**已确认参数**：

| 参数 | 值 |
|------|-----|
| 目标采样率 | **200 Hz**（雷达和ECG统一降采样） |
| 分段窗口长度 | **8s = 1600 采样点** |
| 分段步长 | **50% 重叠 = 4s = 800 采样点** |
| 使用ECG导联 | **tfm_ecg2（导联II）作为唯一 ground truth**（R波更明显） |
| 高斯Mask σ | **5 采样点 = 25ms @ 200Hz** |
| 数据划分方式 | **5-Fold Cross-Validation（按受试者划分）** |
| 随机种子 | **42**（固定，保证可复现） |
| QC 阈值 | 待定（由数据探索后确定，代码支持参数化配置） |

**5-Fold CV 划分逻辑**：
1. 先执行QC，剔除信号质量不合格的受试者（严重运动伪影 / 传感器掉落 / ECG基线漂移过大）
2. 对剩余受试者用 `KFold(n_splits=5, shuffle=True, random_state=42)` 划分
3. 不足5整除时近似均分（如27人→5+5+5+6+6），fold分配写入 `metadata.json`
4. 论文中需建表明确报告：初始人数、剔除人数及原因、各场景各fold片段数量

---

## 文件结构

```
BeatAware-Radar2ECGNet/
├── data_preprocessing/
│   ├── step1_radar_processing.py    # 椭圆校正 + 相位解调 + STFT
│   ├── step2_ecg_processing.py      # NeuroKit2 清洗 + R峰检测 + 高斯Mask
│   ├── step3_qc.py                  # 质量控制 + QC报告
│   ├── step4_segment_save.py        # 对齐 + 分段 + 保存NPY
│   ├── verify_dataset.py            # 数据集完整性校验
│   └── utils/
│       ├── ellipse_correction.py    # I/Q椭圆校正
│       ├── gaussian_mask.py         # 高斯Mask生成
│       └── mat_loader.py            # .mat文件读取与字段解析
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── ssm.py               # VSSSBlock1D + SelectiveScan1D
│   │   │   └── group_mamba.py       # GroupMambaBlock
│   │   ├── modules/
│   │   │   ├── peak_module.py       # PAM 峰值检测辅助模块
│   │   │   └── tfilm.py             # TFiLMGenerator
│   │   └── BeatAwareNet/
│   │       └── radar2ecgnet.py      # 主模型 BeatAwareRadar2ECGNet
│   ├── data/
│   │   └── dataset.py               # RadarECGDataset（NPY，支持多表征）
│   ├── losses/
│   │   └── losses.py                # TotalLoss
│   └── utils/
│       ├── logger.py
│       ├── seeding.py
│       └── metrics.py               # MAE / RMSE / PCC / PRD / DTW / F1 / RR / QRS / PR / QT interval error
├── configs/
│   └── config.py
├── scripts/
│   ├── train.py
│   ├── test.py
│   └── run_ablation.sh
├── experiments/                     # 实验输出（自动生成）
├── docs/
│   └── ARCHITECTURE.md              # 完整架构设计文档（权威来源）
└── CLAUDE.md                        # 本文件
```

---

## 开发顺序

**阶段一：数据预处理**（需要原始 .mat 数据）

1. `data_preprocessing/utils/mat_loader.py`
2. `data_preprocessing/utils/ellipse_correction.py`
3. `data_preprocessing/utils/gaussian_mask.py`
4. `data_preprocessing/step1_radar_processing.py`
5. `data_preprocessing/step2_ecg_processing.py`
6. `data_preprocessing/step3_qc.py`
7. `data_preprocessing/step4_segment_save.py`
8. `data_preprocessing/verify_dataset.py`

**阶段二：模型搭建**（每步写完验证 shape 后再进入下一步）

9.  `src/models/backbone/ssm.py` — VSSSBlock1D
10. `src/models/backbone/group_mamba.py` — GroupMambaBlock
11. `src/models/modules/tfilm.py` — TFiLMGenerator
12. `src/models/modules/peak_module.py` — PAM
13. `src/models/BeatAwareNet/radar2ecgnet.py` — 主模型整合
14. `src/losses/losses.py` — TotalLoss

**阶段三：训练与评估**

15. `src/data/dataset.py`
16. `configs/config.py`
17. `scripts/train.py`
18. `scripts/test.py`
19. `scripts/run_ablation.sh`

**当前状态**：设计完成，代码尚未实现（src/ 目录均为空）。

---

## 实验计划

| 实验 | 描述 |
|------|------|
| Exp A | 基线：仅 Backbone，无 PAM/TFiLM，输入 radar_phase |
| Exp B | 完整模型 × 三种雷达表征对比（radar_raw / radar_phase / radar_spec） |
| Exp C | 完整模型（最优表征），超参消融 |
| D1 | PAM 高斯软标签 vs 二值 R峰标签 |
| D2 | PAM→TFiLM 串联 vs 平行（节律从原始雷达提取）|
| D3 | PAM 多尺度 vs 单卷积核 k=15 |
| D4 | 去除 ConformerFusionBlock |
| D5 | 仅 resting 训练 → 跨场景泛化测试 |

**评估指标**

波形重建质量：

| 指标 | 说明 |
|------|------|
| MAE | 时域绝对误差均值 |
| RMSE | 均方根误差 |
| PCC | Pearson 相关系数（整体波形相似度）|
| PRD | Percent Root-mean-square Difference（ECG 领域标准）|
| DTW | Dynamic Time Warping（波形形态相似度，对局部时间偏移鲁棒）|

临床生理指标（在重建 ECG 上用 Pan-Tompkins 检测波群后计算）：

| 指标 | 说明 |
|------|------|
| F1（R峰） | R峰检测 F1 Score |
| RR Interval Error | RR 间期误差（心率节律精度）|
| QRS Interval Error | QRS 波群时限误差（心室除极持续时间）|
| PR Interval Error | PR 间期误差（房室传导时间）|
| QT Interval Error | QT 间期误差（心室复极时间，与心律失常相关）|

**所有超参和消融实验配置均须经实验验证后确定，不预设固定值。**

---

## 参考项目（仅设计参考，不复用代码）

| 项目 | 路径 | 参考点 |
|------|------|-------|
| BeatAware_R-M2Net | `/home/qhh2237/Projects/BeatAware_R-M2Net` | TFiLM 注入思路，GroupMamba 结构；但代码质量低、架构有误 |
| M3ANet | `/home/qhh2237/Projects/M3ANet` | Multi-scale conv + GroupMamba backbone 来源 |
| radarODE-MTL | `/home/qhh2237/Projects/radarODE-MTL` | 多任务设计参考 |

---

## 开发约束

- **不复用已有项目代码**，所有模块从零实现
- 每个新模块写完后必须用单元测试验证 tensor shape
- `docs/ARCHITECTURE.md` 是架构权威来源；若本文件与其冲突，以 ARCHITECTURE.md 为准
- 实验输出统一写入 `experiments/<EXP_TAG>/`，不污染代码目录
