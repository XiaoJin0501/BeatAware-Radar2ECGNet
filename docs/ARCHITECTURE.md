# BeatAware-Radar2ECGNet — 架构设计文档（V2）

> 记录实验动机、模块设计、Loss 策略、数据预处理方案和实验计划。
> 本项目从零开始独立实现，不直接复用任何已有代码。

---

## 1. 项目目标与双数据集评估策略

从非接触式雷达信号重建高保真 ECG 信号，核心挑战：

1. 雷达信号与 ECG 之间存在复杂的非线性映射，噪声多
2. ECG 波形精度要求高——P波、QRS波群、T波的位置与形态都需要准确重建
3. 心率节律信息（RR间期）需要被显式利用，而非让网络隐式学习
4. 雷达与 ECG 之间存在物理时延（胸壁运动超前于心电活动），需显式补偿

**V2 核心创新**：
- **KI（Kinematic Input）**：3通道差分输入（位移+速度+加速度），显式提供运动学信息
- **CP（Comprehensive Peaks）**：多头 PAM 同时检测 QRS/P/T 三类波群
- **PA（Phase Alignment）**：EMD 物理对齐层，可学习 FIR 滤波器自动补偿时延
- **自适应损失权重**（Kendall & Gal 2018）：4项任务同方差不确定性加权，消除手动超参
- **FMCWRangeEncoder**：针对77GHz FMCW雷达50通道range-time矩阵的轻量前端编码器

### 1.1 双数据集评估策略（面向期刊论文）

| 数据集 | 雷达类型 | 受试者 | 评估框架 | 用途 |
|--------|---------|--------|---------|------|
| **MMECG**（主要） | 77GHz FMCW，50通道range-time | 11人，4状态 | LOSO 11折 | 主要实验，FMCW多通道架构验证 |
| **Schellenberger**（次要） | 24GHz CW，单通道相位 | 30人，3场景 | 5-Fold CV | 跨数据集泛化，与文献直接对比 |

双数据集策略的目的：拓宽期刊论文的深度和广度——一个数据集验证核心架构创新（MMECG），另一个提供与现有文献的直接横向对比（Schellenberger）。

---

## 2. 数据集信息

### 2.1 MMECG 数据集（主要实验数据集）

**来源**：MMECG.h5，77GHz FMCW毫米波雷达
**特点**：多场景、多状态、FMCW range-time矩阵（50个range bin），提供更丰富的空间信息

| 项目 | 内容 |
|------|------|
| 受试者数量 | 11名（subject_1 ~ subject_11）|
| 录音总数 | ~364条（11受试者 × ~33录音/人）|
| 雷达系统 | 77GHz FMCW，50个range bin，range-time矩阵 |
| 采样率 | 200 Hz |
| 原始数据形状 | RCG: (35505, 50)，ECG: (35505, 1) |
| 生理状态 | NB（正常呼吸）/ IB（不规则呼吸）/ SP（某坐位状态）/ PE（体力运动后）|

**预处理参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| RCG 带通滤波 | 0.5–40 Hz，4阶Butterworth | 保留心脏波形完整形态，含高频 QRS 形态信息 |
| RCG 归一化 | 逐通道 z-score | 消除各 range bin 间 ~5× 的幅度差异 |
| ECG 带通滤波 | 0.5–40 Hz | 与 RCG 一致 |
| ECG 归一化 | per-window min-max → [0,1] | 每个分段独立归一化 |
| R峰检测 | NeuroKit2 Pan-Tompkins | Gaussian 软标签 σ=5点（25ms）|
| 分段窗口 | 1600 点（8s）| — |
| 分段步长 | 800 点（50% 重叠）| — |
| 数据集划分 | LOSO 11折 | fold_i 留出第 i+1 位受试者作测试集 |

**滤波策略设计决策**：
- 选用 **0.5–40 Hz 宽带**（而非 0.8–3.5 Hz 窄带）：窄带虽能提升低频相关性，但会去除 QRS 波群的高频形态信息，导致模型无法重建正确的 ECG 形态。
- z-score 归一化的作用：使 FMCWRangeEncoder 的 SE 注意力按信号质量（而非幅度）选择 range bin。

**分段统计**：

| 受试者 | 窗口数（估计）|
|--------|-------------|
| 总计（11人）| ~7,200+ 窗口（依预处理结果）|

**输出文件结构**：
```
dataset_mmecg/
    subject_1/
        rcg.npy     [N, 50, 1600]  float32  — z-score归一化后的range-time矩阵
        ecg.npy     [N, 1,  1600]  float32  — per-window min-max [0,1]
        rpeak.npy   [N, 1,  1600]  float32  — Gaussian软标签，σ=5
        meta.npy    [N, 2]         int32    — (subject_id, state_code)
    ...
    subject_11/
    metadata_mmecg.json  — 受试者列表、LOSO fold 分配、窗口统计
```

**LOSO fold 结构**（fold_i 留出 subject_ids[i] 作测试集）：
```json
"loso_folds": {
    "0": {"test": 1, "train": [2,3,4,5,6,7,8,9,10,11]},
    "1": {"test": 2, "train": [1,3,4,5,6,7,8,9,10,11]},
    ...
}
```

### 2.2 Schellenberger 数据集（次要，跨数据集验证）

**来源**：Schellenberger et al., *Scientific Data* 7:291 (2020)
DOI: 10.1038/s41597-020-00629-5 | Figshare: 10.6084/m9.figshare.12186516

| 项目 | 内容 |
|------|------|
| 受试者数量 | 30名健康成人（GDN0001–GDN0030），16男14女 |
| 平均年龄 | 30.7 ± 9.9 岁 |
| 雷达系统 | 24GHz 六端口连续波雷达（Six-Port CW Radar）|
| 参考系统 | Task Force Monitor 3040i（ECG + ICG + BP 同步采集）|
| 原始文件格式 | MATLAB .mat（每受试者每场景一个文件）|

| 信号 | 字段名 | 采样率 | 备注 |
|------|-------|-------|------|
| 雷达 I/Q | `radar_i`, `radar_q`（**小写**）| 2000 Hz | |
| ECG 导联II | `tfm_ecg2` | **2000 Hz** | 与雷达等长，天然对齐，降采样因子=10 |

| 场景 | 描述 | 缺失受试者 |
|------|------|----------|
| **Resting** | 平静仰卧呼吸，~634s/人 | 无（全30人）|
| **Valsalva** | 瓦尔萨尔瓦动作，~1000s/人 | GDN0015, GDN0024, GDN0026 |
| **Apnea** | 屏气，~170s/人 | GDN0001–0003, GDN0015, GDN0024, GDN0026 |

| 参数 | 值 |
|------|-----|
| 目标采样率 | 200 Hz（因子=10）|
| 分段窗口 | 8s = 1600 采样点 |
| 分段步长 | 4s = 800 采样点（50% 重叠）|
| 总分段数 | 12,807（Resting 4714 + Valsalva 6953 + Apnea 1140）|
| 数据划分 | 5-Fold CV，按受试者，KFold seed=42，每折6人 |

---

## 3. 整体架构（V2）

### 3.1 MMECG 输入路径（input_type='fmcw'）

```
RCG [B, 50, 1600]
       │
       ▼
┌──────────────────────────────┐
│  FMCWRangeEncoder            │
│                              │
│  DepthwiseConv1d(50,k=61)    │
│  + BN + GELU                 │
│  (时域滤波，~305ms 感受野)     │
│                              │
│  SE 空间注意力（50→6→50）      │
│  (选择心脏信号最强的range bin) │
│                              │
│  Conv1d(50→3, k=1) + BN      │
│  (学习3种互补range bin组合)    │
└───────────────┬──────────────┘
                │ [B, 3, 1600]
                ▼
         ←与下方 CW 路径合流→
```

### 3.2 Schellenberger 输入路径（input_type='phase'）

```
雷达相位 [B, 1, L]
       │
       ▼  diff → [B, 3, L]（位移 + 速度 + 加速度）
       │
       ├────────────────────────────────────────────┐
       ▼                                            ▼
┌──────────────────────────┐         ┌─────────────────────────────────┐
│  PAM（多头峰值检测）       │  γ, β   │  主干 Backbone                   │
│                          │ ──────► │                                 │
│  Multi-scale Conv(3→32)  │         │  Multi-scale Encoder            │
│  k=7/15/31，×3路         │         │  Conv1d(3→C, k=3/5/7/9, s=4)   │
│  → [B, 96, L]            │         │  × 4路 + TFiLM注入              │
│  VSSSBlock1D × 2         │         │  → Concat [B, 4C, L/4]         │
│  LayerNorm               │         │                                 │
│  ┌────┬────┬────┐        │         │  GroupMambaBlock × 2            │
│  QRS  P    T  Head2     │         │  ConformerFusionBlock           │
│  mask mask mask rhythm  │         │                                 │
│  [B,1,L] × 3  [B,96]   │         │  EMD 对齐层（use_emd=True时）     │
└───┬──────────────────────┘         │                                 │
    │                                │  Decoder                        │
    ▼                                │  ConvTranspose(4C→2C, s=2)      │
 L_peak Loss                         │  ConvTranspose(2C→C,  s=2)      │
 (QRS + P + T BCE)                   │  Conv(C→1) + Sigmoid            │
                                     └───────────────┬─────────────────┘
                                                     ▼
                                             ECG输出 [B, 1, L]
```

**FMCW 路径说明**：`input_type='fmcw'` 时，FMCWRangeEncoder 将 (B,50,L) 压缩为 (B,3,L) 后直接进入 Backbone，PAM 和 TFiLM 与 CW 路径完全相同（通过 `use_pam` 控制）。

**`use_pam=False` 时**：跳过 PAM 和 TFiLM，Encoder 直接无调制运行，`peak_preds=None`，`L_peak/L_interval=0`。

---

## 4. 各模块详细设计

### 4.1 FMCWRangeEncoder（MMECG 专用前端）

**文件**：`src/models/modules/fmcw_encoder.py`
**接口**：`(B, 50, L) → (B, 3, L)`，完全替代 CW 路径的 KI（diff）操作

**设计背景**：
FMCW range-time 矩阵的主功率集中在 13–16 Hz（静态杂波拍频），心脏成分（0.8–3.5 Hz）是叠加在上面幅度约 1/100 的调制信号。50个range bin之间信号质量差异很大，需要自适应聚合。

**架构**：
```
(B, 50, L)
  ↓ DepthwiseConv1d(50→50, k=61, groups=50, padding=30)  ← 逐通道时域滤波
     + BatchNorm1d(50)
     + GELU  ← 关键：GELU而非ReLU，保留心脏AC信号的负半周期
  (B, 50, L)

  ↓ SE 空间注意力
     AdaptiveAvgPool1d(1) → (B, 50)
     → Linear(50→6) + ReLU
     → Linear(6→50) + Sigmoid
     → x_att = x_filt * attn.unsqueeze(-1)
  (B, 50, L)

  ↓ Conv1d(50→3, k=1) + BatchNorm1d(3)  ← 无激活函数，保留全范围输出
  (B, 3, L)
```

**关键设计决策 — GELU vs ReLU**：
- BatchNorm 输出均值为零，即信号有正有负
- ReLU 会截断所有负值（≈50% 信息丢失），对于心脏 AC 信号是致命的
- GELU 在负值区间平滑衰减（而非硬截断），完整保留心脏信号的负半周期
- 这是 FMCWRangeEncoder 的关键修复，对最终 PCC 有显著影响

**参数量**：~3.7K（极轻量，相对于主干网络可忽略）

**权重初始化**：
- `temporal_filter`：恒等初始化（中心 tap=1，其余=0），训练开始即提供稳定梯度
- `proj`：均匀平均初始化（1/50），训练初期 3 通道输出等于 50 路等权平均

### 4.2 导数感知编码器（KI）

**文件**：`src/models/BeatAwareNet/radar2ecgnet.py`，`forward()` 入口

```python
if self.input_type in ("raw", "phase"):
    v = torch.diff(x, dim=-1, prepend=x[:, :, :1])   # 速度
    a = torch.diff(v, dim=-1, prepend=v[:, :, :1])   # 加速度
    x_input = torch.cat([x, v, a], dim=1)             # [B, 3, L]
elif self.input_type == "fmcw":
    x_input = self.fmcw_enc(x)                        # [B, 3, L]
else:  # spec
    x_input = x  # [B, 1, F, T]，保持原始2D形式
```

`radar_phase` 经带通滤波（0.5-10Hz），二阶差分不会放大噪声。

### 4.3 峰值检测辅助模块（PAM — V2 三路输出）

**文件**：`src/models/modules/peak_module.py`

```
1D 输入路径（raw/phase/fmcw，经过 KI 或 FMCWRangeEncoder 后均为 [B,3,L]）：
  [B, 3, L]
  → Multi-scale Conv1d(3→32, k=7/15/31, padding=k//2) + BN + ReLU × 3路
  → Concat → [B, 96, L]
  → VSSSBlock1D × 2 → LayerNorm

输出头：
  Head_QRS: Conv1d(96→1) + Sigmoid → [B, 1, L]，监督 σ=5（25ms）
  Head_P:   Conv1d(96→1) + Sigmoid → [B, 1, L]，监督 σ=10（50ms）
  Head_T:   Conv1d(96→1) + Sigmoid → [B, 1, L]，监督 σ=15（75ms）
  Head2:    AdaptiveMaxPool1d(1) → [B, 96]  — 节律向量，输入 TFiLM
```

返回：`(qrs_mask, p_mask, t_mask), rhythm_vec`

**多尺度感受野**：k=7 → 35ms（QRS尖峰）/ k=15 → 75ms（P/T波）/ k=31 → 155ms（RR节律）

### 4.4 TFiLM 节律注入

**文件**：`src/models/modules/tfilm.py`

```python
gamma, beta = TFiLMGenerator(rhythm_vec)  # rhythm_vec: [B, 96] → [B, C]

# 对 Encoder 第 i 路特征 f_i: [B, C, L/4]
f_i = BN(conv_i(x_input))  # conv_i: in_channels=3, stride=4
f_i = f_i * (1 + gamma_i[:, :, None]) + beta_i[:, :, None]
f_i = ReLU(f_i)
```

Identity 初始化：TFiLMGenerator 权重初始化为零，偏置初始化为零（γ=0, β=0 → 等效于无调制，稳定训练初期）。

### 4.5 主干 Backbone

**Multi-scale Encoder**（`_encode_1d`）：
```
4路并行 Conv1d(3, C, k=3/5/7/9, stride=4) + BN + TFiLM + ReLU
→ Concat → [B, 4C, L/4]  = [B, 256, 400]
```

**GroupMambaBlock** × 2：
```
LayerNorm → 分4组 → 各组独立 VSSSBlock1D → Concat → CAM → 投影
CAM: AvgPool → FC → ReLU → FC → Sigmoid（通道注意力）
```

**ConformerFusionBlock**：
```
MHSA → DepthwiseConv1d(k=31) + BN + SiLU → FFN
```
保留原因：Mamba 是单向序列扫描，MHSA 负责对称的跨尺度特征对齐，两者功能互补。D4 消融验证其增益。

**Decoder**：
```
ConvTranspose1d(4C→2C, stride=2) → ReLU
ConvTranspose1d(2C→C,  stride=2) → ReLU
Conv1d(C→1) → Sigmoid → [B, 1, L]
```

### 4.6 EMD 物理对齐层（PA）

**文件**：`src/models/BeatAwareNet/radar2ecgnet.py`，`EMDAlignLayer`

```python
class EMDAlignLayer(nn.Module):
    def __init__(self, channels: int, max_delay: int = 20):
        # max_delay=20 → 100ms@200Hz，覆盖典型胸壁-心电时延范围
        # 逐通道 FIR：Conv1d(channels, channels, kernel_size=41, groups=channels)
        # 初始化为 Dirac delta（zero_() + weight[:,:,center]=1.0）
        # ⚠️ 不能用 nn.init.dirac_（对 depthwise shape 只初始化1个通道）
```

插入位置：`ConformerFusionBlock` 之后、`Decoder` 之前。`use_emd=False` 时直接跳过。

---

## 5. Loss 函数（V2）

### 5.1 总体结构

**4项任务自适应加权**（Kendall & Gal 2018，同方差不确定性）：
```
L_total = Σ_i [ 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i ]
```

| 索引 | 任务 | 公式 |
|------|------|------|
| 0 | `L_recon` | L_time + 0.1·L_freq |
| 1 | `L_peak` | BCE(QRS) + masked BCE(P) + masked BCE(T) |
| 2 | `L_der` | L1(diff1) + L1(diff2) |
| 3 | `L_interval` | soft-argmax PR 间期惩罚 |

`log_vars = nn.Parameter(zeros(4))`，与模型参数共同训练（纳入同一 optimizer）。

**训练策略**：前 `warmup_epochs=5` epoch 只计算 `L_recon`（热身），之后解锁全部4项。

### 5.2 各子损失

**L_recon**：
```python
L_time = F.l1_loss(ecg_pred, ecg_gt)
L_freq = MultiResolutionSTFTLoss(ecg_pred, ecg_gt)  # SC + log-mag
L_recon = L_time + 0.1 * L_freq
```

**MultiResolutionSTFTLoss**（V2，替代 V1 的 plain L1 STFT）：
```python
# 对每组 (n_fft, hop, win_length)：
SC   = ||gt_mag - pred_mag||_F / (||gt_mag||_F + 1e-8)   # 谱收敛
logM = mean |log(pred_mag+1e-7) - log(gt_mag+1e-7)|      # 对数幅度
stft_i = (SC + logM) / 2
# 最终取三组分辨率均值
```
参数：`FFT=[128,256,512]`，`HOP=[64,128,256]`，`WIN=[16,32,64]`（@200Hz）

**L_der**（QRS 边界锐化）：
```python
L_der = F.l1_loss(diff(pred,1), diff(gt,1)) + F.l1_loss(diff(pred,2), diff(gt,2))
```

**L_peak**（三路 BCE，P/T 按 valid mask 屏蔽）：
```python
L_peak = BCE(qrs_pred, qrs_gt)
if p_valid.any(): L_peak += BCE(p_pred[p_valid], p_gt[p_valid])
if t_valid.any(): L_peak += BCE(t_pred[t_valid], t_gt[t_valid])
```

**L_interval**（soft-argmax PR 间期约束）：
```python
def soft_argmax(mask, tau=0.1):
    w = F.softmax(mask.squeeze(1) / tau, dim=-1)
    pos = torch.arange(L, device=mask.device).float()
    return (w * pos).sum(-1)  # [B]

pr_samples = soft_argmax(qrs_pred) - soft_argmax(p_pred)
pr_ms = pr_samples * 5.0   # @200Hz，1sample=5ms
L_interval = mean( ReLU(120 - pr_ms) + ReLU(pr_ms - 200) )
# 正常 PR 范围 120-200ms 内无惩罚
```
仅对 `p_valid=True` 的样本计算。

---

## 6. 数据预处理方案

### 6.1 MMECG 预处理流程（脚本：`scripts/preprocess_mmecg.py`）

```
MMECG.h5
    │
    ├─ 读取每条录音（group）：RCG (35505,50)，ECG (35505,1)
    │
    ├─ RCG 处理：
    │     转置 → (50, T)
    │     → 0.5-40 Hz 宽带带通（4阶Butterworth）
    │     → 逐通道 z-score 归一化
    │
    ├─ ECG 处理：
    │     → 0.5-40 Hz 带通
    │     → 滑窗切片后 per-window min-max → [0,1]
    │     → NeuroKit2 R峰检测 → Gaussian软标签 σ=5
    │
    ├─ 滑窗切片（window=1600, stride=800）
    │     → rcg: (N, 50, 1600)
    │     → ecg: (N,  1, 1600)
    │     → rpeak: (N, 1, 1600)
    │     → meta: (N, 2)  [subject_id, state_code]
    │
    └─ 按受试者聚合，保存 NPY + metadata_mmecg.json
```

**state_code 映射**：`{"NB": 0, "IB": 1, "SP": 2, "PE": 3}`

### 6.2 Schellenberger 预处理流程（脚本：`data_preprocessing/step1-4.py`）

```
原始 .mat 文件（2000Hz）
    │
    ├─ Step 1: 雷达处理
    │     椭圆校正(I/Q) → 反正切解调 → radar_raw(200Hz)
    │     → 带通滤波(0.5-10Hz) → radar_phase(200Hz)
    │     → 细粒度STFT → radar_spec_input(1,33,T@200Hz)
    │
    ├─ Step 2: ECG 处理
    │     nk.ecg_clean() → R峰检测 → 降采样200Hz → rpeak_indices
    │
    ├─ Step 2b: P/T 波标注（V2 新增）
    │     nk.ecg_delineate(method='dwt') → pwave_indices, twave_indices
    │     失败样本保存空数组（step4 中 valid=False）
    │
    ├─ Step 3: QC
    │     信噪比/相位跳变/基线漂移/R峰检测失败率 → qc_report.json
    │     全30人通过（Run #1）
    │
    └─ Step 4: 分段保存
          滑窗(1600, stride=800) → per-segment ECG归一化 → 高斯Mask
          输出：radar_raw/phase/spec + ecg + rpeak + pwave + twave + valid
```

---

## 7. 实验计划

### 7.1 MMECG 实验（主要，当前执行中）

**当前实验**：mmecg_G
- 配置：0.5-40Hz 宽带 + GELU 修复 + BeatAwareRadar2ECGNet (input_type='fmcw', use_pam=True, use_emd=True)
- 训练框架：11折 LOSO，150 epochs/fold，早停机制（基于 val_loss + val_pcc）
- 状态：训练中，预计 ~5-6 小时完成

**MMECG 消融计划**（mmecg_G 结果出来后执行）：

| 实验 | 配置 | 目的 |
|------|------|------|
| mmecg_G | fmcw + pam + emd（完整模型）| 基准性能 |
| mmecg_A | fmcw，无 pam，无 emd | FMCWRangeEncoder 单独性能 |
| mmecg_B | fmcw + pam，无 emd | PAM 增益 |
| mmecg_C | fmcw，无 pam，emd | EMD 增益 |

### 7.2 Schellenberger 消融实验（次要，MMECG 之后执行）

**主消融框架（三维 KI/PA/CP）**：

| 变体 | KI | PA | CP | 目的 |
|------|:--:|:--:|:--:|------|
| **Model A** — Baseline | ✓ | | | 纯 Backbone 重建能力下界 |
| **Model B** — +PA | ✓ | ✓ | | EMD 对齐层的增益 |
| **Model C** — Full | ✓ | ✓ | ✓ | 完整 V2 模型性能上界 |

其他消融：

| 实验 | 改动 | 目的 |
|------|------|------|
| D4 | 去除 ConformerFusionBlock | Conformer 必要性 |
| D5 | 仅 resting 训练 → 全场景测试 | 跨场景泛化能力 |

### 7.3 评估指标

**波形重建**：

| 指标 | 说明 |
|------|------|
| MAE | 时域绝对误差 |
| RMSE | 均方根误差 |
| PCC | Pearson 相关系数 |
| PRD | Percent Root-mean-square Difference（ECG领域标准）|
| F1（R峰）| R峰检测 F1 Score |

**MMECG 专项（per-state 细分）**：NB / IB / SP / PE 各状态的指标分别上报，反映不同生理状态下的重建鲁棒性。

**临床生理**（在重建 ECG 上用 NeuroKit2 检测后计算）：

| 指标 | 说明 |
|------|------|
| RR Interval Error | 心率节律精度 |
| QRS Duration Error | 心室除极时限 |
| PR Interval Error | 房室传导时间 |
| QT Interval Error | 心室复极时间（与心律失常相关）|

### 7.4 SOTA 对比参考

| 方法 | 数据集 | PCC | 架构特点 |
|------|--------|-----|---------|
| radarODE | MMECG | ~0.89 | SST时频变换 + Neural ODE，最高复杂度 |
| SS-S2SA | FMCW | — | 多头锚点检测 |
| **BeatAwareNet（本研究）** | MMECG + Schellenberger | TBD | 物理驱动+双数据集验证 |

---

## 8. 参考项目分析

| 项目 | 输入 | 峰值/节律监督 | 条件注入 | 主要不足 |
|------|------|------------|---------|---------|
| **BeatAware_R-M2Net** | 1D Radar | anchor branch → R峰Mask | TFiLM 注入 Encoder | 峰值分支与TFiLM平行独立，无信息流依赖 |
| **radarODE-MTL** | 2D时频谱图（SST）| anchor_decoder | 无显式注入 | 节律估计与波形重建未耦合 |
| **AirECG** | 2D时频谱图 | 无峰值监督 | Diffusion + Cross-Attention | 推理慢，无峰值约束 |

**本项目核心创新**：
1. **FMCWRangeEncoder**：针对FMCW 50通道range-time矩阵的轻量自适应聚合（GELU关键）
2. **PAM → TFiLM 串联**：先检测峰值，再用峰值节律驱动重建（显式信息流）
3. **双数据集验证**：MMECG（77GHz FMCW，LOSO）+ Schellenberger（24GHz CW，5-Fold CV）

---

## 9. 文件结构

```
BeatAware-Radar2ECGNet/
├── data_preprocessing/
│   ├── step1_radar_processing.py
│   ├── step2_ecg_processing.py
│   ├── step2b_delineate.py          ← V2 新增：P/T 波标注
│   ├── step3_qc.py
│   ├── step4_segment_save.py        ← V2 更新：输出 pwave/twave/valid
│   ├── verify_dataset.py
│   ├── PREPROCESSING_LOG.md
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
│   │   │   └── fmcw_encoder.py      # FMCWRangeEncoder（MMECG专用）
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
│   ├── train.py                     # Schellenberger 训练（5-Fold CV）
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
│   ├── ARCHITECTURE.md              ← 本文件（权威来源）
│   └── RESULTS_GUIDE.md             # 实验结果解读指南
├── papers/
├── IMPLEMENTATION_SPEC_V1.md        # V2 优化方案设计规格（历史文档）
└── CLAUDE.md
```
