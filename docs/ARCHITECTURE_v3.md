# ARCHITECTURE v3 — BeatAware-Radar2ECGNet

> 论文 Method 章节底稿。本文档与 `CLAUDE.md` 模型架构章节交叉对照，与代码当前实现完全一致。
> 实验结果章节（§ 11）占位，待 `mmecg_reg_loso` 11 折跑完后填入。

---

## § 1. Overview

### Task Definition
非接触式 FMCW 毫米波雷达 → 单导联 ECG 重建。输入是雷达接收到的 range-time 矩阵 $X \in \mathbb{R}^{50\times 1600}$（50 range bins × 8 s @ 200 Hz），输出是同步的归一化 ECG 波形 $\hat{y} \in [0,1]^{1600}$。

### Clinical Motivation
- 长期监护场景中 wet-electrode ECG 不可持续（皮肤刺激、依从性差）
- 睡眠呼吸暂停、新生儿监护、驾驶疲劳检测等接触受限场景
- 与现有可穿戴方案（PPG）相比，雷达可保留 QRS 形态学和心律不齐细节

### Technical Challenges
1. **(C1) Multi-bin Aggregation**：50 个距离通道中只有 ~3-5 bins 真正落在胸壁回波上，需要可学习地选择并融合
2. **(C2) Electromechanical Delay**：心电活动领先于胸壁机械运动 50–150 ms，模型需主动对齐
3. **(C3) QRS Morphology Fidelity**：扩散/回归直接学波形容易丢失 fiducial points，需引入显式峰值先验

### Our Three Innovations

| 缩写 | 模块 | 解决的挑战 |
|------|------|-----------|
| **KI** Kinematic Inversion | `FMCWRangeEncoder` | C1 |
| **PA** Physical Alignment | `EMDAlignLayer` | C2 |
| **CP** Cardiac Priors     | `PeakAuxiliaryModule` + `TFiLMGenerator` | C3 |

---

## § 2. Architecture Overview

### Data Flow（FMCW input path, `use_diffusion=False`）

```
RCG  X ∈ ℝ^{B×50×1600}
   │
   ▼
[KI] FMCWRangeEncoder ─────────────────────► (B, 3, 1600)
   │   depthwise temporal Conv (k=61) + SE attention + 1×1 proj
   │
   ├─────────────────────────────────────────┐
   │                                         ▼
   │                            [CP-1] PeakAuxiliaryModule (PAM)
   │                              ms-Conv (k=7,15,31) + VSSSBlock×2
   │                              ├─► QRS / P / T masks   (B, 1, 1600) ×3
   │                              └─► rhythm_vec          (B, 96)
   │                                         │
   │                            [CP-2] TFiLMGenerator
   │                              fc(96 → 256) ×2 → γ, β  (B, 256) ×2
   │                                         │
   ▼                                         │ γ, β
Multi-scale Encoder ◄────────────────────────┘
  4× Conv1d(3 → 64, k={3,5,7,9}, stride=4) + per-branch TFiLM
  → concat → (B, 256, 400)
   │
   ▼
ConformerFusionBlock                         ─► (B, 256, 400)
   │
   ▼
[PA] EMDAlignLayer                           ─► (B, 256, 400)
   depthwise FIR k=41 (max_delay=20 ≈ 100 ms @ 200 Hz)
   │
   ▼
Regression Decoder
  ConvTranspose1d(256→128, k=4, s=2)         ─► (B, 128, 800)
  ConvTranspose1d(128→64,  k=4, s=2)         ─► (B,  64, 1600)
  Conv1d(64→1, k=1) + Sigmoid                ─► ŷ (B, 1, 1600) ∈ [0,1]
```

### Decoder Switch

| 模式 | 解码器 | Table | 参数总量 |
|------|--------|-------|----------|
| `use_diffusion=False` | ConvTranspose×2 + Sigmoid | **Table 1** 主线 | **1,239,390** |
| `use_diffusion=True`  | `BeatAwareDiffusionDecoder` (DDPM, T=1000, DDIM 50步) | Table 2 进阶 | ~3,100,000 |

LOSO 实验用回归路径（参数量已与训练日志对齐：`Model params: 1,698,270`）。

代码入口：`src/models/BeatAwareNet/radar2ecgnet.py:186-413`。

---

## § 3. KI — FMCWRangeEncoder

### Problem
原始 RCG $X \in \mathbb{R}^{50 \times 1600}$ 中：
- 主功率集中在 13–16 Hz（静态杂波拍频，与心脏无关）
- 心脏分量仅 0.8–3.5 Hz，幅度约为杂波的 1/10
- 50 个 range bin 中只有 3–5 个真正落在胸壁回波，其余是干扰

### Module Structure
代码：`src/models/modules/fmcw_encoder.py:24-97`

```
X ∈ ℝ^{B×50×1600}
 │
 ├─► depthwise Conv1d(50, 50, k=61, padding=30, groups=50, bias=False)   [L41-45]
 │     初始化为 Dirac delta（中心 tap=1，其余=0）= 恒等映射             [L62-66]
 │     感受野 61 × 5 ms = 305 ms ≈ 1 个心动周期
 │
 ├─► BatchNorm1d(50)                                                      [L46]
 ├─► GELU （非 ReLU；保留心脏 AC 信号的负半周期）                          [L86]
 │
 ├─► SE attention                                                         [L51-53, 88-92]
 │   AdaptiveAvgPool1d(1) → Linear(50, 50/8=6) → ReLU → Linear(6, 50) → Sigmoid
 │   → 50 维 channel-wise 权重，乘回特征 (B, 50, 1600)
 │
 ├─► Conv1d(50, 3, k=1, bias=False)（投影到 3 通道）                       [L56]
 │     初始化为均匀平均 1/50                                               [L69]
 │
 └─► BatchNorm1d(3)（无激活）                                              [L57, L95]

输出 (B, 3, 1600)
```

### Math
- 时域滤波：$\tilde{x}_i[n] = \sum_{k=-30}^{30} h_i[k] \cdot x_i[n-k]$，每通道独立学习 $h_i$
- SE 注意力：$w = \sigma(W_2 \text{ReLU}(W_1 \text{Pool}(\tilde{x})))$，给每个 range bin 一个 [0,1] 权重
- 投影：$z[c, n] = \sum_{i=1}^{50} P_{c,i} \cdot (w_i \tilde{x}_i[n])$，学习 3 种互补的 range bin 加权组合

### Why GELU, not ReLU
心脏机械运动是双向的（收缩 + 舒张），AC 信号经 BatchNorm 后均值为 0。ReLU 会截断 50% 的负半周期能量，使心跳信号严重失真。GELU 的负侧渗漏保留这部分信息（实测 ReLU 版本 val PCC < 0.10，GELU 版本 > 0.30）。

### vs Time-Frequency Representation
radarODE-MTL 用 SST 谱图作输入（`MMECG_to_SST.m` 预生成 71×120 时频图）。我们选时域：
- 端到端可学习滤波（vs 固定 SST kernel）
- 计算量小（1D conv vs 2D spectrogram）
- 与 EMD 时间对齐层语义一致（PA 也是时域）

---

## § 4. PA — EMDAlignLayer

### Problem
心脏电活动（ECG R 峰）→ 心肌等容收缩 → 胸壁位移峰值，物理时延约 50–150 ms（Electromechanical Delay, EMD），与受试者年龄、心脏负荷有关。雷达观测的是位移，目标是 ECG，**模型必须在时间轴上向"提前"方向对齐**。

### Module Structure
代码：`src/models/BeatAwareNet/radar2ecgnet.py:102-143`

```python
class EMDAlignLayer(nn.Module):
    def __init__(self, channels=256, max_delay=20):
        kernel_size = 2 * max_delay + 1   # 41-tap
        self.delay_conv = Conv1d(
            channels, channels,
            kernel_size=41, padding=20,
            groups=channels,        # depthwise
            bias=False,
        )
        # Dirac delta 初始化（恒等映射）
        weight.zero_()
        weight[:, :, 20] = 1.0      # 中心 tap
```

### Math
$$y_c[n] = \sum_{k=-20}^{20} h_c[k] \cdot x_c[n-k], \quad c=1,\ldots,256$$

每通道独立学习 41-tap FIR 滤波器 $h_c$，覆盖 $\pm 100$ ms @ 200 Hz。Dirac 初始化保证训练初期是恒等映射，梯度才能驱动 $h$ 自发偏移到正确的 EMD 滞后值。

### Why depthwise (groups=channels)
- 每个 backbone 通道编码不同的运动学/形态学特征，应有独立时延
- 全连接 conv 会引入 256² 个参数；depthwise 仅 256×41 ≈ 10K，避免过拟合

### vs CFT-RFcardi 的 Beamforming
| 维度 | CFT (空间对齐) | EMD (时间对齐) |
|------|----------------|----------------|
| 物理量 | 雷达-心脏几何（3D 体素位置） | 电-机械时延（标量 lag）|
| 实现 | 数值优化（NOMAD 黑盒搜索） | 端到端可学习 FIR |
| 输入 | raw IQ 12 通道 | 已聚合的 256 通道特征 |

两者互补：CFT 对齐空间，PA 对齐时间。我们的 KI（FMCWRangeEncoder）已隐式做了空间聚合，故 PA 单独负责时间维度足够。

---

## § 5. CP — PAM + TFiLM (Cardiac Priors)

### Problem
直接回归 ECG 波形容易丢失 fiducial points（QRS / P / T peaks），尤其在低 SNR 段。需要引入显式峰值监督 + 节律调制信号。

### CP-1: PeakAuxiliaryModule (PAM)
代码：`src/models/modules/peak_module.py:37-138`

```
输入 (B, 3, 1600)  ◄── 来自 KI 输出
   │
   ├─► 三路多尺度 Conv1d(3, 32, k=7/15/31) + BN + ReLU         [L70-77, L120]
   │   → concat → (B, 96, 1600)
   │
   ├─► VSSSBlock1D × 2                                          [L80-81, L124-125]
   ├─► LayerNorm                                                [L82, L128]
   │
   ├─► Head_QRS:  Conv1d(96, 1, k=1) + Sigmoid → (B,1,1600)    [L91, L131]
   ├─► Head_P:    同上                            → (B,1,1600)    [L92, L132]
   ├─► Head_T:    同上                            → (B,1,1600)    [L93, L133]
   │
   └─► Head_rhythm:  AdaptiveMaxPool1d(1) → squeeze → (B, 96)   [L96, L136]
```

### Supervision (Gaussian soft labels)
- QRS：σ=5（25 ms @ 200 Hz）—— 严格监督
- P/T：σ=10/15 —— 当前**仅 QRS 进入 loss**（P/T GT 质量不稳定，引入噪声）
- BCE 损失：$\mathcal{L}_{\text{peak}} = \text{BCE}(\hat{m}_{\text{QRS}}, m_{\text{QRS}}^{\text{soft}})$

### CP-2: TFiLMGenerator
代码：`src/models/modules/tfilm.py:14-53`

$$\gamma = W_\gamma \cdot \text{rhythm\_vec}, \quad \beta = W_\beta \cdot \text{rhythm\_vec}$$
$$\gamma, \beta \in \mathbb{R}^{B \times 256}$$

权重和偏置都初始化为 0（identity init），训练初期 TFiLM 不干扰主干梯度。

### TFiLM Modulation in Encoder
代码：`src/models/BeatAwareNet/radar2ecgnet.py:298-318`

每个 multi-scale 分支 $i \in \{1,2,3,4\}$ 独立调制：
$$f_i' = f_i \cdot (1 + \gamma_i) + \beta_i, \quad \gamma_i, \beta_i \in \mathbb{R}^{C}$$

将 $\gamma, \beta \in \mathbb{R}^{4C}$ 拆成 4 段，每段对应一路 64 通道的 conv 输出。

### vs radarODE-MTL's PPI/Anchor MTL
- radarODE 把 PPI（脉搏间期 260 维分类）和 Anchor（R 峰位置 800 维稀疏）作**平权 MTL 任务**，需 LibMTL EGA 调度
- 我们把峰值检测当**辅助监督**，并用其内部表征做调制（节律 → FiLM）。三路 head 共享底层特征，无 EGA，损失权重固定 1.0
- 论证：PAM/TFiLM 总参数 ~260K，比独立 PPI/Anchor 解码器（>1M）轻量；不依赖外部 MTL 框架

---

## § 6. Backbone (Multi-scale + Conformer)

### Multi-scale Encoder
代码：`src/models/BeatAwareNet/radar2ecgnet.py:261-265, 298-318`

```
输入 (B, 3, 1600)  ◄── KI 输出
4 个并行分支：
  Conv1d(3, 64, k=3, padding=1, stride=4) + BN + TFiLM(γ_1,β_1) + ReLU → (B, 64, 400)
  Conv1d(3, 64, k=5, padding=2, stride=4) + BN + TFiLM(γ_2,β_2) + ReLU → (B, 64, 400)
  Conv1d(3, 64, k=7, padding=3, stride=4) + BN + TFiLM(γ_3,β_3) + ReLU → (B, 64, 400)
  Conv1d(3, 64, k=9, padding=4, stride=4) + BN + TFiLM(γ_4,β_4) + ReLU → (B, 64, 400)
Concat → (B, 256, 400)
```

stride=4 在 conv 内完成下采样，1600 → 400。多尺度 kernel 覆盖不同时间感受野（15ms / 25ms / 35ms / 45ms @ 200Hz）。

### ConformerFusionBlock
代码：`src/models/BeatAwareNet/radar2ecgnet.py:42-95`

```
x (B, 256, 400)
 │
 ├─► LayerNorm → MultiHeadSelfAttention(dim=256, heads=4) → +residual
 │   全局跨位置对齐
 │
 ├─► LayerNorm → DepthwiseConv1d(256, 256, k=31, groups=256) + BN + SiLU + Conv1d(k=1) → +residual
 │   局部上下文（k=31 覆盖 155 ms）
 │
 └─► LayerNorm → FFN(256 → 1024 → 256) + Dropout → +residual
```

### Design Trade-off
| 选择 | 取代方案 | 理由 |
|------|---------|------|
| Multi-scale Conv | 单一 kernel | 8 s 窗口含多种节律（HRV 50–150 BPM），多尺度 receptive field 覆盖更全 |
| Conformer Fusion | 纯卷积 | 400 token 的编码序列可承受 MHSA；全局节律上下文对 ECG morphology reconstruction 必要 |

---

## § 7. Decoder Heads

### § 7.1 Regression Decoder（LOSO 实验启用）
代码：`src/models/BeatAwareNet/radar2ecgnet.py:293-295, 410-412`

```
h ∈ ℝ^{B×256×400}（EMD 对齐后）
 │
 ├─► ConvTranspose1d(256, 128, k=4, s=2, p=1) + ReLU  → (B, 128, 800)
 ├─► ConvTranspose1d(128,  64, k=4, s=2, p=1) + ReLU  → (B,  64, 1600)
 ├─► Conv1d(64, 1, k=1)                                → (B,   1, 1600)
 └─► Sigmoid                                           → ŷ ∈ [0, 1]
```

简单两级 stride-2 上采样恰好把 400 还原为 1600。Sigmoid 限定输出值域以匹配 H5 中存储的 min-max 归一化 ECG。

### § 7.2 Diffusion Decoder
代码：`src/models/modules/diffusion_decoder.py:83-262`

#### Architecture
```
Upsample: ConvTranspose1d(256, 256, k=8, s=4, p=2) + BN + GELU
          → (B, 256, 1600)

In-projection:
  in_ch = 1 + 256 + (3 if use_pam else 0)        # x_t + h_up + peak_masks
  Conv1d(in_ch, 256, k=1)                        # hidden=256

Conditioning:
  t_emb = SinusoidalEmbedding(t) → MLP → (B, 256)
  cond  = concat(t_emb, rhythm_vec) ∈ ℝ^{B × (256+96)}
  use_pam=False 时 rhythm_vec ← 0

DiffusionResBlock × 8：
  GroupNorm → GELU → Conv(k=3) → GroupNorm → FiLM(cond → γ,β) → GELU → Conv(k=3) → +skip

Out-projection:
  GroupNorm → GELU → Conv1d(256, 1, k=1)          → ε̂ ∈ ℝ^{B×1×1600}
```

#### Diffusion Process

**Forward (noising)**：余弦 β 调度，$T=1000$（Nichol & Dhariwal 2021）
$$\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s), \quad \beta_t \in (0, 0.999]$$
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

ECG 值域转换：$x_0 = 2y - 1 \in [-1, 1]$（输入），$\hat{y} = (\hat{x}_0 + 1)/2$（输出）。

**Reverse (denoising)**：DDIM 50 步确定性采样（$\eta=0$）
$$\hat{x}_0^{(t)} = (x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta) / \sqrt{\bar{\alpha}_t}$$
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0^{(t)} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

#### Conditioning Routes
1. **h_enc** $\in \mathbb{R}^{B \times 256 \times 400}$：通过 upsample → 拼接到输入通道
2. **rhythm_vec** $\in \mathbb{R}^{B \times 96}$：拼接到 $t_{\text{emb}}$，FiLM 调制每个 ResBlock
3. **peak_masks** $\in \mathbb{R}^{B \times 3 \times 1600}$：直接拼接到输入通道（空间引导扩散）

---

## § 8. Loss Functions

### § 8.1 TotalLoss (Regression, LOSO)
代码：`src/losses/losses.py:109-185`

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta_{\text{peak}} \cdot \mathcal{L}_{\text{peak}}$$

$$\mathcal{L}_{\text{recon}} = \underbrace{\|\hat{y} - y\|_1}_{\mathcal{L}_{\text{time}}} + \alpha_{\text{stft}} \cdot \underbrace{\mathcal{L}_{\text{STFT}}}_{\text{multi-resolution}}$$

$$\mathcal{L}_{\text{STFT}} = \frac{1}{3} \sum_{i=1}^{3} \frac{\text{SC}_i + \text{LogMag}_i}{2}$$

参数：$\alpha_{\text{stft}} = 0.05$，$\beta_{\text{peak}} = 1.0$。

| Resolution | n_fft | hop | win |
|-----------|-------|-----|-----|
| coarse | 128 | 64 | 16 |
| mid    | 256 | 128 | 32 |
| fine   | 512 | 256 | 64 |

- **SC** (Spectral Convergence)：$\|S_y - S_{\hat{y}}\|_F / \|S_y\|_F$
- **LogMag**：$\text{L1}(\log S_{\hat{y}}, \log S_y)$

$$\mathcal{L}_{\text{peak}} = \text{BCE}(\hat{m}_{\text{QRS}}, m_{\text{QRS}}^{\text{soft}}), \quad \sigma_{\text{QRS}}=5 \text{ samples}$$

仅 QRS 进入损失。P/T head 训练但不监督（保留以便未来扩展）。

### § 8.2 DiffusionLoss
代码：`src/losses/losses.py:192-243`

$$\mathcal{L}_{\text{total}} = \text{MSE}(\epsilon_\theta(x_t, t, c), \epsilon) + \beta_{\text{peak}} \cdot \mathcal{L}_{\text{peak}}$$

STFT 不显式计算（扩散目标已隐式覆盖频域）。

### Why fixed weights, not adaptive?
Schellenberger 早期实验用 `log_vars` 自适应权重，发现模型会"套利"——把大权重的 loss 项学好，小权重的故意学坏。改用固定权重后训练稳定且性能更好。

---

## § 9. Comparison with Reference Works

| 维度 | AirECG | radarODE-MTL | CFT-RFcardi | **Ours** |
|------|--------|--------------|-------------|----------|
| 输入表示 | mmWave 8ch raw, 1024 pts @ 1 kHz | SST 谱图 [50,71,120] @ 200 Hz | raw 50ch IQ + CFT | **FMCW raw 50ch + KI** |
| 数据集 | 自建 mmWave-ECG | MMECG | MMECG (transfer) | MMECG |
| 主干 | DiT Transformer | DCN ResNet | DCN ResNet (复用 radarODE) | **Multi-scale + Conformer** |
| 时间对齐 | — | — | — | **EMDAlignLayer (PA)** |
| 空间压缩 | — | SST 物理（固定）| **CFT beamforming（数值优化）** | **FMCWRangeEncoder (KI)（端到端学习）** |
| 节律先验 | reference ECG (y2) | PPI MTL | PPI MTL + SSL | **PAM + TFiLM (CP)** |
| 解码器 | DiT Cross-Attention 扩散 | 多任务回归 | 多任务回归 | **回归 / 扩散双轨** |
| 训练目标 | EMA + VB+MSE | EGA MTL | SSL+迁移 | L1+STFT+BCE / MSE+BCE |
| 已发表 PCC | 0.955 (private) | ~0.85 (MMECG) | ~0.88 (MMECG transfer) | **TBD (LOSO 跑中)** |

代码引用：
- AirECG：`/home/qhh2237/Projects/AirECG/{models.py, train.py, diffusion/gaussian_diffusion.py}`
- radarODE-MTL：`/home/qhh2237/Projects/radarODE-MTL/Projects/radarODE_plus/nets/`
- CFT-RFcardi：`/home/qhh2237/Projects/CFT-RFcardi/Cardio_Tracking_and_Focusing/`

### Key Differentiators
1. **唯一同时做时间对齐 (PA) + 空间聚合 (KI)**（不是 SST/CFT 的固定/数值方法，而是端到端学习）
2. **轻量节律先验**（PAM+TFiLM 260K params vs radarODE PPI 解码器 >1M）
3. **回归/扩散双解码器共享 encoder**（同一 backbone 在 Table 1 和 Table 2 复用）

---

## § 10. Implementation Details

### Data
- **Dataset**：MMECG (Wang et al. 2023)
- **Subjects**：11
- **Modalities**：77 GHz FMCW radar (50 range bins) + 12-lead ECG, 同步 200 Hz
- **Windows**：8 s (1600 samples), 50% overlap
- **Physiological states**：NB / IB / SP / PE
- **Preprocessing pipeline**（H5 预构建）：
  1. 雷达：椭圆校正 → arctan 解调 → 相位展开 → 0.5–10 Hz Butterworth → 200 Hz 降采样 → per-channel z-score
  2. ECG：NeuroKit2 cleaning → Pan-Tompkins R 峰检测 → 200 Hz → per-window min-max → [0,1]
  3. 滑窗 → 高斯 R 峰 mask (σ=5)
- **Loader-time augmentation**：`narrow_bandpass=False`（默认开启的 0.8–3.5 Hz 二次滤波已禁用，因实测损害 QRS 形态）

### Train/Eval Protocol
- **Primary**：LOSO 11-fold（subject-level isolated）
  - 划分逻辑：`/home/qhh2237/Datasets/MMECG/03B_create_loso_splits.py`
  - 每折：1 subject test + 1 subject val (records 数最接近中位数) + 9 subjects train
- **Secondary（存在 OOD 污染，仅作消融）**：samplewise 70/15/15 record-level split
  - 划分逻辑：`/home/qhh2237/Datasets/MMECG/03A_create_samplewise_splits.py`
  - 已确认 test 集 21% 来自 train 完全不见或欠采样的受试者，详见 § 12

### Hyperparameters
| 参数 | 值 |
|------|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | Cosine, warmup 5 epochs |
| Epochs | 150 (回归) / 300 (扩散) |
| Batch size | 16 |
| Grad clip | 1.0 |
| Early stop patience | 20 |
| `balance_by` | `class`（按 physistatus）|
| `narrow_bandpass` | `False` |

### Hardware
- RTX 4080 SUPER 16 GB
- 回归 LOSO 单折 ~3.5 h × 11 ≈ 40 h
- 扩散 samplewise 300 epoch ~13 h

---

## § 11. Experimental Results

> ⚠️ 占位章节。`mmecg_reg_loso` 11 折预计 2026-05-10 上午跑完，届时填入下方表格。

### 11.0 Metric Notes（避免混淆）

- **`RMSE_norm` / `MAE_norm` / `MSE_norm`** 都在 H5 存储的 [0,1] 归一化空间上算（`src/utils/metrics.py:506-549`）。MSE 不单独输出但 `MSE = RMSE²` 可秒算。**论文 Table 1 用 `_norm` 数字**，与 radarODE-MTL / AirECG / CFT-RFcardi 同行报告口径一致。
- **`RMSE_mV` / `MAE_mV` 永远是 NaN**（`metrics.py:547-548` 硬编码）。原因：H5 ECG 是 per-window min-max 归一化到 [0,1]，原始毫伏 (min, max) 在归一化时被丢弃，无法反算回 mV。修复需重写 `05A/B_build_*.py` 加 H5 字段 + 重生成所有 H5（约 1 天工作量），临床精度看 **PR / QRS / QT 间期误差（毫秒）** 即可，不必修。

### Table 1. Main Results (LOSO 11-fold, BeatAware-Regression)

| Subject (test) | PCC | RMSE_norm | R² | QMR (%) | R-peak err (ms) | F1@150ms |
|---|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| 30 | TBD | TBD | TBD | TBD | TBD | TBD |
| **Mean ± Std** | TBD | TBD | TBD | TBD | TBD | TBD |
| **Median (IQR)** | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 2. Comparison with Baselines

| Method | PCC | QMR (%) | F1@150ms | PR err (ms) | QRS err (ms) | QT err (ms) |
|--------|-----|---------|----------|-------------|--------------|-------------|
| radarODE-MTL (paper) | TBD | TBD | TBD | — | — | — |
| AirECG (paper, different dataset) | 0.955 | — | — | — | — | — |
| **BeatAware-Regression (Ours)** | TBD | TBD | TBD | TBD | TBD | TBD |
| **BeatAware-Diffusion (Ours)** | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 3. Ablation (LOSO mean PCC)

| Config | use_pam | use_emd | PCC | Δ vs Full |
|--------|:-:|:-:|---|---|
| A baseline | ✗ | ✗ | TBD | — |
| B +EMD     | ✗ | ✓ | TBD | TBD |
| C +PAM     | ✓ | ✗ | TBD | TBD |
| D Full     | ✓ | ✓ | TBD | — |

---

## § 12. Discussion

### 12.1 Why LOSO is the Right Protocol
LOSO 严格按受试者隔离，是真正的 cross-subject 泛化评估。每折训练集完全没见过 test subject，因此 PCC 是在临床部署场景（新患者）下的真实下界。

### 12.2 The samplewise OOD Pollution Issue
`03A_create_samplewise_splits.py` 是 record-level 70/15/15 随机分层。实测：
- train subjects: {1, 2, 5, 9, 13, 14, 16, 17, 29, 30}（其中 sub 1, 2 仅各 14 segments）
- val subjects: {5, 9, **10**, 13, 14, 16, 17, 29, 30}（sub 10 不在 train）
- test subjects: {**1, 2, 10**, 13, 14, 16, 29, 30}（sub 1/2 严重欠采样，sub 10 完全 OOD）

约 21% test 样本来自 train 不见或欠采样的受试者。这是为什么所有 samplewise 实验（v1, reg_clean, diff_v2）的 test PCC 都收敛到 0.18–0.21 的天花板，与架构无关。**论文主指标必须用 LOSO**；samplewise 仅作架构消融的快速代理。

### 12.3 Future Directions
1. **Diffusion LOSO**：等回归 LOSO 出结果，按 plan 决策树判断扩散是否值得 +40 h
2. **Reference ECG conditioning (y2)**：参考 AirECG 用历史窗 ECG 作为额外条件，可能突破单向 radar→ECG 的信息瓶颈
3. **EMA + VB loss (Plan A)**：~3 天工程量，预期扩散稳定性提升 +0.05 PCC
4. **SSL pretraining**：参考 CFT-RFcardi，用 PPI/Anchor 作自监督信号，先在大规模无标 radar 数据上预训练 backbone

---

## Appendix A. Module Parameter Counts

按 `count_parameters` (`src/models/BeatAwareNet/radar2ecgnet.py:420-422`)：

| Module | Params | % of total |
|--------|-------:|-----------:|
| FMCWRangeEncoder (KI) | 3,962 | 0.3% |
| PeakAuxiliaryModule (CP-1) | 141,795 | 11.4% |
| TFiLMGenerator (CP-2) | 49,664 | 4.0% |
| Multi-scale Encoder (4 conv + BN) | 5,120 | 0.4% |
| ConformerFusionBlock | 864,256 | 69.7% |
| EMDAlignLayer (PA) | 10,496 | 0.8% |
| Regression Decoder | 164,097 | 13.2% |
| **Total (regression)** | **1,239,390** | **100%** |
| BeatAwareDiffusionDecoder (T=1000, hidden=256, n_blocks=8) | ~1,400,000 | — |
| **Total (diffusion)** | **~3,100,000** | — |

---

## Appendix B. Quick Verification Commands

```bash
# 模型参数统计（应输出 1,698,270）
python -c "
import torch
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet, count_parameters
m = BeatAwareRadar2ECGNet(input_type='fmcw', use_pam=True, use_emd=True, use_diffusion=False)
print(f'Regression params: {count_parameters(m):,}')
"

# Forward 形状测试
python -c "
import torch
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
m = BeatAwareRadar2ECGNet(input_type='fmcw', use_pam=True, use_emd=True).eval()
x = torch.randn(2, 50, 1600)
y, masks = m(x)
assert y.shape == (2, 1, 1600) and 0 <= y.min() and y.max() <= 1
qrs, p, t = masks
assert qrs.shape == (2, 1, 1600)
print('OK')
"

# 当前 LOSO 进度
tail -20 experiments_mmecg/mmecg_reg_loso.log
```

---

*Last updated: 2026-05-08 (LOSO fold_01 in progress, results to be filled).*
