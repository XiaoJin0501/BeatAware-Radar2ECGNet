# Phase H — 性能突破路线图：B (Domain Adaptation) + D (Neural ODE) 双轨

**创建日期**：2026-05-10
**目标**：把 LOSO mean PCC 从 ~0.27 拉到 ≥ 0.55，达到 IEEE J-BHI / TBME Q1 论文级别。

---

## 1. Context

### 1.1 当前性能瓶颈

经过完整的消去法 ablation（Phase A-G），已确认：

| 方向 | 验证方法 | 结论 |
|------|---------|------|
| Lag-aware loss | lag100 / output_align_reg | ❌ 不涨 raw PCC |
| Narrow bandpass | reg_improved 实验 | ❌ 性能下降 |
| Diffusion 解码器 | T=100 / T=1000 | ❌ 完全失败 |
| Bin selection (top-K) | Phase D D4/E-oracle + Phase F N1/N2 | ❌ 不是瓶颈 |
| Conformer 必要性 | slim ablation = 0.40+ | ✅ 作为主干核心保留 |
| Subject-balanced sampling | LOSO fold_01: 0.254→0.405 | ✅ 唯一明确正收益 |

**当前最佳数字**：
- legacy full baseline (150 ep): samplewise val_pcc = **0.4421**
- legacy full baseline @ ep 80: samplewise val_pcc = **0.4285**
- **slim Conformer baseline (ep 80): samplewise val_pcc = 0.4038**
- LOSO 严格 mean PCC: ~0.27（旧 baseline，跨人泛化差）
- Train PCC 0.69 → val PCC 0.44 = **0.25 generalization gap**

### 1.2 为何 0.40 不够 Q1

- IEEE J-BHI / TBME 通常要求方法贡献 + 性能提升明显
- 现有 SOTA 数字 0.85+（虽然是 cheating splits）会被审稿人拿来比较
- **必须在严格 LOSO 下达到 0.55+** 才有竞争力

### 1.3 真正的瓶颈：跨受试者泛化 + 解码器自由度

- 0.25 generalization gap 表明模型学了 subject-specific 特征
- Free-form ConvTranspose decoder 没有 ECG 形态约束 → 跨人输出形态崩

### 1.4 Phase H 设计理念

针对两个真实瓶颈各上一种方法：

| 瓶颈 | 方法 | 期望增益 |
|------|------|---------|
| 跨受试者泛化 | **B: Domain Adaptation (DANN)** — 移除 encoder 特征中的 subject identity | +0.05~0.13 LOSO |
| 解码器无形态约束 | **D: Neural ODE (PQRST 5 高斯)** — 强制输出物理合理 ECG | +0.05~0.15 |
| 联合（部分协同）| B + D | +0.18~0.33 |

**目标命中**：LOSO 0.27 → 0.45-0.60 → 论文级别。

---

## 2. Path B: Domain Adaptation (DANN)

### 2.1 核心思路

Encoder 输出特征 `h_enc` 不应包含 subject identity。加一个 subject classifier auxiliary head，用 GRL (Gradient Reversal Layer) 反向梯度，让 encoder 学到 **subject-invariant** 特征。

### 2.2 架构

```
RCG → FMCWEnc → Multi-scale Encoder → ConformerFusionBlock → h_enc [B, 256, 400]
                                                                  │
                                                       ┌──────────┴──────────┐
                                                       ▼                     ▼
                                                   EMD + Decoder      GRL × −λ
                                                  (reconstruction)         │
                                                       │                   ▼
                                                       │         SubjectClassifier
                                                       │         (1×1 conv → AvgPool → Linear)
                                                       │                   │
                                                       ▼                   ▼
                                                   ECG output      subject_logits [B, 11]
                                                       │                   │
                                                       ▼                   ▼
                                                   L_recon            L_subj (CE)
                                                       │                   │
                                                       └─────────┬─────────┘
                                                                 ▼
                                              L_total = L_recon + λ·L_subj
                                              (反向梯度: encoder 收到 −λ·∇L_subj，
                                               所以 encoder 试图让 subject 不可分)
```

### 2.3 GRL 实现

```python
# src/models/modules/grl.py
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambda_init=0.0):
        super().__init__()
        self.lambda_ = lambda_init

    def set_lambda(self, lam: float):
        self.lambda_ = float(lam)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
```

### 2.4 SubjectClassifier 设计

```python
# 在 BeatAwareRadar2ECGNet 中加：
self.grl = GRL(lambda_init=0.0)
self.subject_classifier = nn.Sequential(
    nn.Conv1d(4*C, 4*C, kernel_size=3, padding=1),  # mix 时间维特征
    nn.SiLU(),
    nn.AdaptiveAvgPool1d(1),                          # [B, 4C, 1]
    nn.Flatten(),                                     # [B, 4C]
    nn.Dropout(0.3),
    nn.Linear(4*C, n_subjects),                       # [B, 11]
)

# forward 中:
h_enc = self.fusion(...)  # [B, 256, 400]
ecg = self.decoder(h_enc, ...)
if self.use_dann:
    h_for_subj = self.grl(h_enc)
    subject_logits = self.subject_classifier(h_for_subj)
    return ecg, subject_logits
return ecg, None
```

### 2.5 λ Sigmoid Annealing

经典 DANN 论文 (Ganin 2015) schedule：

```python
def dann_lambda(epoch, total_epochs, init=0.0, final=1.0):
    p = epoch / total_epochs
    return init + (final - init) * (2.0 / (1.0 + np.exp(-10*p)) - 1.0)

# 训练循环：
for epoch in range(epochs):
    lam = dann_lambda(epoch, epochs)
    model.grl.set_lambda(lam)
    ...
```

### 2.6 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/models/modules/grl.py` (新) | GRL Function + Module |
| `src/models/BeatAwareNet/radar2ecgnet.py` | __init__ 加 use_dann/n_subjects；forward 返回 (ecg, subj_logits) |
| `src/data/mmecg_dataset.py` | __getitem__ 返回 subject_id (从 H5 metadata) |
| `src/losses/losses.py` | 加 SubjectAdvLoss = F.cross_entropy(subj_logits, subj_id) |
| `configs/mmecg_config.py` | use_dann/lambda_dann_init/lambda_dann_final/dann_anneal/n_subjects=11 |
| `scripts/train_mmecg.py` | 透传 + per-epoch λ 更新 + L_subj 加权 + log subj_acc |
| `scripts/test_mmecg.py` | 测试时 subject_classifier 不参与（只用 reconstruction head）|

### 2.7 训练监控指标

- `subj_acc`: subject classifier 在 train 上的 accuracy
  - 期望：训练初期 ~1/11=9% chance level → 中期升到 50% → λ 上升后 encoder 抗衡 → 收敛回 ~10-20%
  - 如果一直 >80%：encoder 没学到 invariant 特征 → 需要加大 λ 或加深 classifier
  - 如果一直 ~9%：classifier 太弱或 GRL 反向太强 → 减小 λ
- `val_pcc`: 主指标，目标 LOSO mean ↑ 0.05-0.13

### 2.8 实现时间

- Day 1: GRL 模块 + radar2ecgnet 集成 + smoke test
- Day 2: train script 加 λ schedule + log subj_acc
- Day 3-4: samplewise 跑 + 调多个 λ_final ∈ {0.5, 1.0, 2.0}
- Day 5-6: LOSO 11 折跑 (~14-17h)
- Day 7: 分析结果 + 决策

---

## 3. Path D: Neural ODE 物理先验

### 3.1 核心思路

替换 free-form ConvTranspose decoder 为 **ECG dynamics generator** — 网络只预测 PQRST 5 个高斯峰的参数（峰位置 / 幅度 / 宽度），ECG 由参数合成。物理先验强约束输出形态，抑制跨人形态崩塌。

### 3.2 ECG 5-Gaussian Model (McSharry 2003)

每个心拍的 ECG 由 5 个 Gaussian 叠加：

```
ECG_beat(t) = Σ_{i ∈ {P,Q,R,S,T}} A_i · exp(-(t - μ_i)² / (2σ_i²))
```

约束：
- μ_R = 0（以 R 峰为中心）
- μ_P ∈ [-200, -50] ms 相对 R
- μ_Q ∈ [-30, -10] ms 相对 R
- μ_S ∈ [10, 30] ms 相对 R
- μ_T ∈ [50, 350] ms 相对 R
- A_P > 0, A_Q < 0, A_R > 0, A_S < 0, A_T > 0
- σ ∈ [5, 50] samples（@200Hz = 25-250 ms）

### 3.3 架构

```
encoder 输出 h_enc [B, 256, 400]
   │
   ▼
ECGDynamicsHead (网络预测每拍参数):
   1. 上采样 to [B, 256, 1600] (匹配 R 峰位置精度)
   2. R 峰 mask 来自 PAM 输出 (B, 1, 1600)
   3. 在每个 R 峰位置抽取 [B, 256] 局部特征
   4. MLP 投到 [B, 5×3=15] (5 峰 × 3 参数)
   5. Sigmoid + 范围约束（per-peak 不同）
   │
   ▼
ECGSynthesizer (per-beat synthesis):
   for each R peak in window:
       segment = beat 区间 [-300ms, +500ms] 相对 R
       ECG_segment = sum_{i=1..5} A_i · gaussian(t, μ_i, σ_i)
       将 segments 拼回完整 8s 窗口
   │
   ▼
ECG output [B, 1, 1600]
```

### 3.4 关键设计要点

1. **依赖 PAM 检测的 R 峰位置**：训练初期 PAM 可能不准 → 用 GT R 峰 mask（H5 已有）训练 ODE，推理时用 PAM 输出
2. **可微合成**：所有操作（Gaussian, sum, interpolation）都可微 → 端到端 backprop
3. **Loss 设计**：
   ```
   L_ode = L_recon (waveform L1) 
         + α · L_param (param 范围 hinge)
         + β · L_recon_per_beat (每拍单独 L1，权重大)
   ```
4. **初始化**：关键。第一次 forward 时所有 σ 初始化在 20-30 区间，A 初始化在 0.5（normalized scale），避免 NaN

### 3.5 实现风险与缓解

| 风险 | 缓解 |
|------|------|
| 高斯参数 NaN 或梯度爆炸 | 输出过 sigmoid + bounded scaling；σ 不直接预测，改预测 log(σ) |
| Per-beat segmentation 错误（R 峰漏检）| 训练用 GT R-peak（H5 已有）；推理 fallback：PAM 失败时用 fixed grid 1Hz 切分 |
| 5 高斯模型过简化（PVC/异常拍无法 fit）| 加 residual connection: ECG = ODE_synthesis + small_conv_residual |
| 训练初期网络发散 | Warmup: 前 5 epoch 只训练 reconstruction（free decoder），后切换 ODE |

### 3.6 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/models/modules/ecg_ode_decoder.py` (新) | ECGDynamicsHead + ECGSynthesizer + 参数约束 |
| `src/models/BeatAwareNet/radar2ecgnet.py` | 加 decoder_type: "free"\|"ode"，forward 路由 |
| `src/losses/losses.py` | 加 ECGOdeLoss（reconstruction + param hinge + per-beat）|
| `configs/mmecg_config.py` | decoder_type: str = "free"; ode_warmup_epochs: int = 5 |
| `scripts/train_mmecg.py` | warmup logic + 透传 |

### 3.7 参考实现

- `~/Projects/radarODE-MTL/` — 用户已 clone 的参考项目（用 Neural ODE 做 radar→ECG）
- `~/Projects/CFT-RFcardi/` — 类似思路
- McSharry et al. 2003 *"A Dynamical Model for Generating Synthetic ECG Signals"* — 经典数学模型

### 3.8 实现时间

- Day 1-3: ECGSynthesizer + per-beat segmentation + 单元测试
- Day 4-7: ECGDynamicsHead 网络设计 + 参数约束 + smoke test
- Day 8-10: 集成 radar2ecgnet + warmup logic + samplewise 训练
- Day 11-14: 调参 + LOSO 跑

---

## 4. B + D 联合训练

### 4.1 最终架构

```
RCG → FMCWEnc → Multi-scale → Conformer → h_enc [B, 256, 400]
                                  │
                        ┌─────────┼─────────────┐
                        ▼         ▼             ▼
                    GRL+SubjCls  EMD     ECGDynamicsHead
                    (Path B)    (PA)     (Path D)
                                  │             │
                                  └──────┬──────┘
                                         ▼
                                  ECGSynthesizer
                                  (5 高斯 PQRST/beat)
                                         │
                                         ▼
                                  ECG [B, 1, 1600]
```

### 4.2 联合 Loss

```
L_total = L_recon (waveform L1+STFT)             # 重建主 loss
        + α · L_qrs (BCE on R-peak mask)         # PAM aux loss
        + β · L_subj_adv (CE × −λ via GRL)       # Path B
        + γ · L_ode_param (param range hinge)    # Path D
        + δ · L_per_beat (per-beat L1)           # Path D 强约束
```

权重默认：α=1.0, β=1.0 (with λ schedule), γ=0.1, δ=0.5

### 4.3 训练顺序

**保险方案（推荐）**：分阶段训练
1. Stage 1 (Day 1-7): 仅 B (DA + free decoder) → 基线 0.40 → 期望 0.45-0.50
2. Stage 2 (Day 8-14): 加 D，从 stage 1 checkpoint warm start → 期望 0.50-0.60

**激进方案（备用）**：从头联合训练
- 从 random init 直接训练所有组件
- 风险：D 高斯参数发散 + DA λ 调不好叠加

---

## 5. 时间表（Day-by-Day）

```
Day 0 (今天 2026-05-10)
  ✅ slim Conformer baseline 完成 (val_pcc=0.4038)
  ✅ top-K/oracle 分支已关闭
  📝 写本文档

Day 1
  ├─ [GPU] LOSO 11 折 slim Conformer baseline (80 ep, ~14-17h)
  └─ [CPU] 实现 src/models/modules/grl.py + radar2ecgnet 加 DANN head + smoke test

Day 2-3
  ├─ [GPU] DANN samplewise 验证（多 λ_final 调参）
  └─ [CPU] 调研 radarODE-MTL 源码 + 设计 ECGSynthesizer 数学

Day 4-7
  ├─ [GPU] DANN LOSO 11 折 (~14-17h)
  └─ [CPU] 实现 ECGDynamicsHead + ECGSynthesizer + smoke test (Day 4-5)
            实现 ECGOdeLoss + warmup logic (Day 6-7)

Day 8-10
  ├─ [GPU] D 单独 samplewise 验证（先无 DANN，看 ODE 能否 work）
  └─ [CPU] Path A 论文写作（Method §3 已可写）

Day 11-14
  ├─ [GPU] B+D 联合 samplewise → LOSO 11 折
  └─ [CPU] Result + Discussion 章节

Day 15-21
  ├─ [GPU] LOSO 主表 finalize（多 seed 平均，3 seeds）
  └─ [CPU] Figures + table 完善

Day 22-28
  ├─ 论文 final pass
  └─ 投稿 IEEE J-BHI

总：4 周到投稿（理想）；高风险 5-6 周
```

---

## 6. 失败兜底（Plan H-fallback）

如 Path D (Neural ODE) 实现 2 周仍训不出来：

1. 退到 **Path A + B**（Honest LOSO Benchmark + DANN）
2. D 实现进度作为 future work 写进 discussion
3. 不影响投稿时间窗（A+B 已能投 J-BHI Q1）

最坏情况性能预期：
- A only: LOSO 0.27 (honest baseline framing)
- A+B: LOSO 0.27 → 0.32-0.40 (DANN +0.05-0.13)
- A+B+D 成功：LOSO 0.27 → 0.45-0.60 (+0.18-0.33)

---

## 7. 论文叙事（双 / 三 contribution）

> **Contribution 1 (Path A framing)**: First strict LOSO benchmark for radar-to-ECG with 4-level evaluation protocol (waveform PCC / peak timing / fiducial F1 / clinical intervals), exposing that prior work using samplewise/subject-overlapping splits reports inflated 0.85+ PCC vs an honest 0.27 baseline under strict subject-disjoint evaluation.
>
> **Contribution 2 (Path B method)**: Domain-adversarial training (Gradient Reversal Layer + subject classifier with sigmoid λ annealing) to remove subject identity from learned features, lifting LOSO mean PCC from 0.27 to ~0.40 while training subject-invariant representations.
>
> **Contribution 3 (Path D method)**: Physics-prior decoder using parametric PQRST Gaussian dynamics (5 peaks × 3 params per beat) with end-to-end differentiable ECG synthesis, constraining output to physiologically valid morphology and improving cross-subject robustness, lifting LOSO mean PCC further to ~0.55+.

---

## 8. 关键风险与监控

| 风险 | 监控指标 | 缓解 |
|------|---------|------|
| DANN λ 调不好（subj_acc ≠ chance level） | subj_acc per epoch | sigmoid anneal 0→1 over 80 epochs；多 λ_final 调参 |
| DANN encoder 学不到 reconstruction | val_pcc | warmup: 前 10 epoch λ=0；监控 train PCC |
| Neural ODE 高斯参数 NaN | param histogram | sigmoid + 范围约束；初始化 σ=20，A=0.5 |
| Per-beat seg 错误 | R-peak miss rate | 训练用 GT R 峰，推理 PAM fallback |
| B+D 联合不稳定 | total loss 波动 | warm start: stage 1 (B only) → stage 2 (B+D) |

---

## 9. 关联文档

- `~/.claude/plans/mmecg-diff-v1-kind-lynx.md` — Session plan（详细对话记录）
- `docs/ARCHITECTURE_v3.md` — 架构权威源（Phase H 完成后需更新到 v4）
- `docs/MMECG_Data_Representation_Audit.md` — Phase A/D/F 诊断证据
- `docs/Phase_H_Optimization_Roadmap.md` — **本文档**

---

## 10. 当前状态（2026-05-10 18:21）

- ✅ Phase A-G 完成
- ✅ slim Conformer backbone 已设为主线
- 🔄 `mmecg_reg_loso_slim` 已启动：11-fold strict LOSO, 150 epochs, subject-balanced sampling
- ⏳ 等 fold-level LOSO baseline 完成后，启动 Path B subject-adversarial generalization
- ✅ 已加入 `loso_calib` 协议：可运行 `LOSO + supervised subject calibration`，
  默认按 held-out subject 的 40/10/50 拆分：40% labeled windows 加入训练，
  10% labeled windows 用于 target-subject validation/early stopping，剩余 50% 测试。
  该协议用于 personalization / calibration upper-bound，不作为 strict LOSO 主结果。
