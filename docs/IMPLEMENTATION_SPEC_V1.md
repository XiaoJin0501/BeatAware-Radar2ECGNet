# BeatAware-Radar2ECGNet: 实现规格文档（V2）

> 本文档记录核心设计决策、模块优化逻辑和消融实验框架。
> 当前实现状态请参考 `CLAUDE.md`；完整架构描述见 `docs/ARCHITECTURE.md`。

---

## 1. 核心目标：解决"电-机"解耦、形态丢失与多通道聚合

当前架构（V2）在 Schellenberger（24GHz CW）数据集上通过多头 PAM + TFiLM + EMD 解决了电-机延迟和 P/T 波形态问题，并针对 MMECG（77GHz FMCW）新增了多通道 range-time 矩阵聚合能力（FMCWRangeEncoder）。

---

## 2. 核心难点与改进逻辑

| 场景难点 | 物理/生理本质 | 改进方案（已实现）|
| :--- | :--- | :--- |
| **PCC 增长受限** | 呼吸噪声主导，心跳微弱 | 多尺度导数增强输入（突出加速度变化）|
| **R 峰对齐但波形失真** | 电信号与机械运动存在延迟 | EMD (电-机械延迟) 对齐层 ✓ |
| **P/T 波被平滑** | 信号能量极低，MAE 损失钝化 | 多头 PAM 锚点 + 导数损失 ✓ |
| **间期逻辑错误 (PR/QT)** | 缺乏解剖学硬约束 | 临床间期一致性损失 ✓ |
| **FMCW 50通道聚合** | range bin 信噪比差异大，静态杂波强 | FMCWRangeEncoder（GELU关键）✓ |

---

## 3. 核心模块实现细节

### 3.1 导数感知 Encoder（KI — Kinematic Input）

**实现文件**：`src/models/BeatAwareNet/radar2ecgnet.py`

```python
# 适用于 input_type='raw' 或 'phase'
v = torch.diff(x, dim=-1, prepend=x[:, :, :1])  # 速度 [B,1,L]
a = torch.diff(v, dim=-1, prepend=v[:, :, :1])  # 加速度 [B,1,L]
x_input = torch.cat([x, v, a], dim=1)           # [B,3,L]
```

二阶导数（加速度）在 QRS 复合波的 R 峰、Q 波起始和 S 波终点处表现为强冲激响应，有助于捕捉高频形态细节。

### 3.2 FMCWRangeEncoder（MMECG专用，input_type='fmcw'）

**实现文件**：`src/models/modules/fmcw_encoder.py`

```
(B, 50, L)
  ↓ DepthwiseConv1d(k=61, groups=50) + BN + GELU  ← GELU: 保留AC信号负半周期
  ↓ SE attention (50→6→50) + Sigmoid              ← 自适应选择强心脏信号的range bin
  ↓ Conv1d(50→3, k=1) + BN                        ← 学习3种互补组合
(B, 3, L)
```

**关键：GELU 而非 ReLU**
- BatchNorm 输出均值为零 → 心脏 AC 信号有正有负
- ReLU 截断负值 → 约 50% 心脏信号信息丢失 → PCC 停滞在 ~0.49
- GELU 平滑保留负值 → 完整 AC 信号 → PCC 预期提升至 0.65+

此模块输出 (B, 3, L)，与 CW 路径的 KI 输出接口完全相同，主干网络无需任何修改。

### 3.3 多头 PAM 模块（V2 三路输出）

**实现文件**：`src/models/modules/peak_module.py`

输入 [B, 3, L] → Multi-scale Conv1d(3→32, k=7/15/31) × 3 路 → Concat [B, 96, L]
→ VSSSBlock1D × 2 → LayerNorm
→ Head_QRS: (σ=5, 25ms), Head_P: (σ=10, 50ms), Head_T: (σ=15, 75ms)
→ Head2: AdaptiveMaxPool1d(1) → rhythm_vec [B, 96] → TFiLM

### 3.4 EMD 物理对齐层（PA）

**实现文件**：`src/models/BeatAwareNet/radar2ecgnet.py`

```python
class EMDAlignLayer(nn.Module):
    def __init__(self, channels: int, max_delay: int = 20):
        # 逐通道 FIR：Conv1d(C, C, k=41, groups=C)
        # max_delay=20 → 100ms@200Hz，覆盖典型胸壁-心电时延
        # 初始化为 Dirac delta（零延迟），训练中自发学习物理时延补偿
```

插入位置：ConformerFusionBlock 之后、Decoder 之前。

### 3.5 损失函数（4项任务自适应加权）

**实现文件**：`src/losses/losses.py`

```python
L_total = Σ_i [ 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i ]
# 任务: L_recon / L_peak / L_der / L_interval
# log_vars = nn.Parameter(zeros(4))，与模型参数共同训练
```

| 任务 | 公式 | 目的 |
|------|------|------|
| `L_recon` | L1 + 0.1·MultiResSTFT | 时域+频域波形重建基线 |
| `L_peak` | BCE(QRS) + masked BCE(P/T) | 多头峰值定位，P/T 需先运行 step2b |
| `L_der` | L1(diff1) + L1(diff2) | 锐化 QRS 边界，防止平滑 |
| `L_interval` | soft-argmax PR 间期约束（120-200ms）| 房室传导生理约束 |

**热身策略**：前 5 epoch 只训练 L_recon，避免早期不稳定的峰值/间期 loss 破坏梯度。

---

## 4. 消融实验框架

### 4.1 MMECG 消融（三维：FMCWEncoder / PAM / EMD）

| 模型变体 | FMCWEncoder | PAM+TFiLM | EMD层 | 实验标签 |
|---------|:-----------:|:---------:|:------:|---------|
| 完整模型（Ours）| ✓ | ✓ | ✓ | mmecg_G |
| w/o PAM & EMD | ✓ | | | mmecg_A |
| w/o EMD | ✓ | ✓ | | mmecg_B |
| w/o PAM | ✓ | | ✓ | mmecg_C |

### 4.2 Schellenberger 消融（三维：KI / PA / CP）

| 模型变体 | KI（diff）| PA（EMD）| CP（多头PAM）| 实验标签 |
|---------|:--------:|:--------:|:-----------:|---------|
| **Model A** — Baseline | ✓ | | | ModelA |
| **Model B** — +PA | ✓ | ✓ | | ModelB |
| **Model C** — Full | ✓ | ✓ | ✓ | ModelC |

> KI（torch.diff）在 `input_type='phase'` 下始终激活，是所有变体的基础配置。

### 4.3 消融验证目标（论文 Table）

**表 1：MMECG LOSO 11折消融结果**（目标填充实验数据）

| 模型 | PCC（均值±标准差）| MAE | F1 | PRD |
|------|-----------------|-----|----|-----|
| mmecg_G（完整）| TBD | TBD | TBD | TBD |
| mmecg_A（无PAM/EMD）| TBD | TBD | TBD | TBD |
| mmecg_B（无EMD）| TBD | TBD | TBD | TBD |
| mmecg_C（无PAM）| TBD | TBD | TBD | TBD |
| radarODE（SOTA）| ~0.89 | — | — | — |

**表 2：Schellenberger 5-Fold CV 消融结果**（目标填充实验数据）

| 模型 | KI | PA | CP | PCC↑ | R峰F1↑ | PRD↓ |
|------|:--:|:--:|:--:|------|--------|------|
| Model A（基线）| ✓ | | | TBD | TBD | TBD |
| Model B（+PA）| ✓ | ✓ | | TBD | TBD | TBD |
| Model C（完整）| ✓ | ✓ | ✓ | TBD | TBD | TBD |

---

## 5. 损失函数解耦与权重轨迹分析

### 5.1 自适应权重的作用

引入动态权重后，消融实验无需手动调超参，转变为：**在全程开启自适应权重的公平环境下，逐步增加约束组件，验证其对特定临床指标的突破。**

### 5.2 预期学习动力学

| 阶段 | Epoch | 预期 L_recon 权重 | L_peak 权重 | L_der 权重 | L_interval 权重 |
|------|-------|:------------:|:--------:|:--------:|:------------:|
| 轮廓期 | 0-5 | 最高（warmup only）| 0 | 0 | 0 |
| 建立期 | 5-30 | 高 | 升高 | 低 | 低 |
| 雕琢期 | 30-80 | 中 | 稳定 | 升高 | 低 |
| 审查期 | 80+ | 中 | 稳定 | 稳定 | 升高 |

这一动力学证明模型自发展现"由宏观到微观（Coarse-to-Fine）"的学习过程——无人工干预下自动对齐医学认知。

---

## 6. 实现状态总结（2026-04-26）

| 模块 | 文件 | 状态 |
|------|------|------|
| FMCWRangeEncoder（含 GELU 修复）| `src/models/modules/fmcw_encoder.py` | ✅ 完成 |
| BeatAwareRadar2ECGNet（fmcw 路径）| `src/models/BeatAwareNet/radar2ecgnet.py` | ✅ 完成 |
| MMECGDataset + LOSO Loaders | `src/data/mmecg_dataset.py` | ✅ 完成 |
| MMECG 配置 | `configs/mmecg_config.py` | ✅ 完成 |
| MMECG 训练脚本（含早停）| `scripts/train_mmecg.py` | ✅ 完成 |
| MMECG 测试脚本（per-state 细分）| `scripts/test_mmecg.py` | ✅ 完成 |
| MMECG 预处理 | `scripts/preprocess_mmecg.py` | ✅ 完成，已执行 |
| PAM（多头 QRS/P/T）| `src/models/modules/peak_module.py` | ✅ 完成 |
| TFiLM | `src/models/modules/tfilm.py` | ✅ 完成 |
| EMD 对齐层 | `radar2ecgnet.py` 内嵌 | ✅ 完成 |
| 4任务自适应损失 | `src/losses/losses.py` | ✅ 完成 |
| Schellenberger 预处理（Step 1-4）| `data_preprocessing/` | ✅ 完成 |
| P/T 波标注（Step 2b）| `data_preprocessing/step2b_delineate.py` | ⏳ 待执行 |
| Schellenberger 训练/测试 | `scripts/train.py` / `test.py` | ✅ 完成，待执行消融 |
| MMECG 训练（mmecg_G，11折）| — | 🔄 进行中 |
| MMECG 消融（mmecg_A/B/C）| — | ⏳ mmecg_G 完成后 |
| Schellenberger 消融（A/B/C）| — | ⏳ MMECG 完成后 |
