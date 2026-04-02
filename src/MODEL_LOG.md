# BeatAware-Radar2ECGNet — 模型实现日志（Phase 2）

> **用途**：记录模型搭建阶段的实现细节、设计决策、验证结果和已知问题，供论文写作快速检索。
> **负责人**：Claude Code 辅助实现
> **完成日期**：2026-04-03

---

## 1. 概览

| 项目 | 内容 |
|------|------|
| 实现阶段 | Phase 2 — 模型搭建 |
| 实现方式 | 从零独立实现（不复用任何已有项目代码） |
| 参考文献 | BeatAware_R-M2Net（同作者前代，仅设计参考）；M3ANet（GroupMamba 结构参考） |
| 实现完成状态 | ✅ 全部通过 forward + backward 验证 |
| 环境 | cyberbrain（PyTorch 1.13.1, CUDA 11.3, RTX 4080 SUPER 16GB） |

---

## 2. 模型架构参数（已确认）

| 参数 | 值 | 说明 |
|------|-----|------|
| C（基础通道） | **64** | Encoder 每分支通道数，4C=256 为 Backbone 通道 |
| input_type | 'raw' / 'phase' / 'spec' | 三种实验表征，通过参数切换 |
| use_pam | True / False | False 时为 Exp A 基线（无 PAM，无 TFiLM） |
| signal_len | 1600 | 8s @ 200Hz |
| L_enc | 400 | Encoder 输出时间帧数（stride=4 下采样） |
| d_state | 16 | VSSSBlock1D SSM 状态维度 |
| PAM 通道 | 3×32=96 | k=7/15/31 三路多尺度卷积各32通道 |
| TFiLM 输出 | 4C=256 | gamma/beta 共256维，分4份各64维对应4个 Encoder 分支 |

---

## 3. 模块参数量统计

| 配置 | 参数量 | 备注 |
|------|-------|------|
| raw / phase，use_pam=True | **1,677,154** | 完整模型，1D输入 |
| spec，use_pam=True | **1,695,618** | 含 SpecAdapter（+18,464） |
| raw / phase，use_pam=False | **1,489,281** | Exp A 基线，无 PAM/TFiLM |
| spec，use_pam=False | **1,506,369** | Exp A 基线，spec 输入 |

> PAM + TFiLM 贡献约 187,873 参数（约占完整模型 11.2%）。

---

## 4. 实现文件清单

```
src/
├── models/
│   ├── backbone/
│   │   ├── ssm.py            VSSSBlock1D + SelectiveScan1D（自动 CUDA/PyTorch 后端切换）
│   │   └── group_mamba.py    GroupMambaBlock（LayerNorm → 分组 SSM → CAM → 投影）
│   ├── modules/
│   │   ├── tfilm.py          TFiLMGenerator（Identity 初始化）
│   │   └── peak_module.py    PeakAuxiliaryModule，支持 1D/spec 前端切换
│   └── BeatAwareNet/
│       └── radar2ecgnet.py   BeatAwareRadar2ECGNet（含 ConformerFusionBlock, SpecAdapter）
└── losses/
    └── losses.py             TotalLoss（MultiResolutionSTFTLoss + BCE + MAE）
```

---

## 5. 各模块设计要点

### 5.1 VSSSBlock1D（`backbone/ssm.py`）

- **接口**：输入/输出均为 `(B, d_model, L)`，带残差连接
- **SSM 参数**：S4D 初始化（`A[i,n] = n+1`），`A_log` 参数化保证 A < 0（稳定性）
- **dt 初始化**：通过 `dt_proj = Linear(dt_rank, d_inner)` 学习，`delta_softplus=True` 保证 delta > 0
- **后端**：尝试 import `selective_scan_cuda`；失败时自动切换纯 PyTorch 循环（慢但可用）
- **扩展比**：`expand=2.0`（d_inner = 2×d_model），dt_rank = ceil(d_model/16)

### 5.2 GroupMambaBlock（`backbone/group_mamba.py`）

- **分组**：d_model // 4 = 64（每组64通道）→ 4个独立 VSSSBlock1D
- **CAM**：AvgPool(1) → Conv1d(d_model, d_model//4) → ReLU → Conv1d(d_model//4, d_model) → Sigmoid
  - CAM 权重在分组 SSM **之前**计算（用输入特征），调制**之后**的 SSM 输出
  - 这样 CAM 做的是软通道选择，不依赖 SSM 输出，梯度流独立
- **残差**：`x = proj(x * CAM_weight) + residual`

### 5.3 TFiLMGenerator（`modules/tfilm.py`）

- **Identity 初始化**：`fc_gamma.weight/bias = 0`，`fc_beta.weight/bias = 0`
  - 训练开始时 TFiLM 完全不改变 Encoder 特征（gamma=0 → 系数=1，beta=0 → 偏置=0）
  - 避免训练初期 PAM 不稳定时破坏 Encoder 梯度流
- **输入**：rhythm_vec `(B, 96)` → `(B, 256)` gamma + `(B, 256)` beta
- **使用方式**：在 `_encode_1d` 中 reshape 为 `(B, 4, 64, 1)` 后逐分支调制

### 5.4 PeakAuxiliaryModule（`modules/peak_module.py`）

- **1D 路径**：k=7/15/31 感受野分别对应 QRS尖峰(35ms) / P-T波(75ms) / RR节律(155ms) @200Hz
- **Spec 路径**：Conv2d(1, 96, (33,1)) 将 F=33 压缩为1 → squeeze → interpolate 到 L=1600
  - 插值因子 ≈ 8×（从 T≈200 插值到 1600），Mamba 在插值后序列上建模
- **Head2 用 AdaptiveMaxPool**（非 AvgPool）：R峰是局部尖峰事件，最大值比均值更能保留节律信息

### 5.5 BeatAwareRadar2ECGNet（主模型）

**关键实现细节**：

1. **TFiLM 应用时机**：BN 之后、ReLU 之前（与 CLAUDE.md 一致）
   ```python
   f = bn(conv(x))                              # BN 先做
   f = f * (1 + gamma4[:, i]) + beta4[:, i]     # TFiLM 调制
   f = F.relu(f)                                # 激活最后
   ```
   > 参考项目(BeatAware_R-M2Net)的 bug：在 TFiLM 后又做了第二次 BN，本实现已修正。

2. **Encoder 输出对齐**：4个分支（k=3/5/7/9，stride=4，padding=k//2）均精确输出 400 帧
   - 验证：`floor((1600 + 2*(k//2) - k) / 4) + 1 = 400` 对 k=3,5,7,9 全部成立

3. **Decoder 输出对齐**：`ConvTranspose1d(kernel=4, stride=2, padding=1)`
   - 400 → `(400-1)*2 - 2 + 4 = 800` → `(800-1)*2 - 2 + 4 = 1600` ✓

4. **use_pam=False**：gamma/beta 置零 → TFiLM 变为恒等映射，Encoder 行为等价于无调制

### 5.6 TotalLoss（`losses/losses.py`）

- **L_time**：`F.l1_loss`（MAE）
- **L_freq**：MultiResolutionSTFTLoss，3 分辨率（nperseg=128/256/512 @200Hz），在线计算（对 pred_ecg/gt_ecg）
  - 归一化 L1：`|S_pred - S_gt|.sum() / |S_gt|.sum()`，各分辨率取均值
  - 覆盖场景：nperseg=128 → QRS(~40ms 40/1000*200=8 samples → 4 FFT bins)；512 → RR间期(~600ms)
- **L_peak**：`F.binary_cross_entropy`（BCE），use_pam=False 时返回0并不加入总loss
- **`radar_spec_loss.npy`**：数据集中存储了雷达的多分辨率STFT，此版本 L_freq 不使用它（在ECG域在线计算），该文件留作备用

---

## 6. 验证结果

### 6.1 Shape Verification（2026-04-03 运行）

```
Device: cuda
  [PAM=True , raw]   ECG=torch.Size([2, 1, 1600]), Mask=torch.Size([2, 1, 1600]), Params=1,677,154
  [PAM=True , phase] ECG=torch.Size([2, 1, 1600]), Mask=torch.Size([2, 1, 1600]), Params=1,677,154
  [PAM=True , spec]  ECG=torch.Size([2, 1, 1600]), Mask=torch.Size([2, 1, 1600]), Params=1,695,618
  [PAM=False, raw]   ECG=torch.Size([2, 1, 1600]), Mask=None,                    Params=1,489,281
  [PAM=False, phase] ECG=torch.Size([2, 1, 1600]), Mask=None,                    Params=1,489,281
  [PAM=False, spec]  ECG=torch.Size([2, 1, 1600]), Mask=None,                    Params=1,506,369
All shape checks passed.
```

### 6.2 End-to-End Forward + Backward（input_type='phase'，use_pam=True）

```
Loss keys: ['total', 'time', 'freq', 'peak']
  total: 1.018880
  time:  0.255166
  freq:  0.607040
  peak:  0.733362
Backward OK
Grad norm (total): 0.7432...
```

### 6.3 值域验证

- `ecg_pred`：Sigmoid 激活，min ≥ 0.0，max ≤ 1.0 ✓
- `peak_mask`：Sigmoid 激活，min ≥ 0.0，max ≤ 1.0 ✓

---

## 7. 设计决策记录

### D-M1：Identity TFiLM 初始化

**决策**：TFiLMGenerator 的 fc_gamma 和 fc_beta 权重/偏置全部初始化为零。

**原因**：训练初期 PAM 尚未收敛，若 TFiLM 以随机初始化的参数调制 Encoder，会引入混乱的噪声梯度，破坏主干网络的收敛稳定性。Identity 初始化确保训练开始时主干行为等同于无条件注入，PAM 先独立收敛后再逐渐引导 TFiLM。

**参考**：FiLM、AdaIN 相关工作均采用类似初始化策略。

---

### D-M2：CAM 权重基于输入特征（非 SSM 输出）

**决策**：CAM 通道注意力权重在 GroupMambaBlock 中对输入 x 计算，而非对 SSM 输出计算。

**原因**：若 CAM 基于 SSM 输出，则 CAM 和 SSM 梯度相互依赖，训练不稳定。基于输入计算的 CAM 作为独立的软通道门控，梯度流与 SSM 解耦。

---

### D-M3：PAM Head2 使用 AdaptiveMaxPool（非 AvgPool）

**决策**：节律向量从 PAM 特征的全局最大池化中提取。

**原因**：R 峰是局部尖峰事件（在 1600 点序列中约有5~10个尖峰），最大池化能更好地保留这些稀疏高激活位置的信息；平均池化会被大量非峰值位置稀释。

---

### D-M4：删除 L_smooth

**决策**：移除 ARCHITECTURE.md 早期设计中的 L_smooth 项（γ·L_smooth）。

**原因**：
1. STFT Loss（L_freq）中的低分辨率分量（nperseg=512，覆盖 RR 间期）已对低频平滑提供约束
2. L_smooth 与 L_freq 存在功能重叠，引入额外超参 γ 增加调参复杂度
3. CLAUDE.md 在经过详细讨论后已确认只用3项 Loss，以实际代码为准

**文档同步**：已同步修正 ARCHITECTURE.md，删除 γ·L_smooth 和 D5 消融行。

---

### D-M5：TFiLM 在 BN 之后、ReLU 之前注入

**决策**：Encoder 每分支的操作顺序为 `BN(conv(x)) → TFiLM → ReLU`。

**原因**：
- 若 TFiLM 在 BN 之前：BN 会将 TFiLM 的调制效果归一化掉（均值/方差被重置）
- 若 TFiLM 在 ReLU 之后：负值已被截断，gamma/beta 的加法调制效果受限

**已修正的参考代码 bug**：BeatAware_R-M2Net 中 TFiLM 后又做了第二次 BN（`feats.append(F.relu(bn(f)))`），本实现已修正。

---

## 8. 已知问题与修正

### Bug M-1：参考代码 BN 应用两次
- **发现**：BeatAware_R-M2Net `BA_M2Net.py` 第79行：`feats.append(F.relu(bn(f)))` — 在 TFiLM 调制之后又调用了 `bn(f)`，导致 BN 被应用两次（第一次在 TFiLM 前，第二次在 TFiLM 后）。
- **影响**：TFiLM 的调制效果被第二次 BN 大幅削减。
- **本项目修正**：`feats.append(F.relu(f))`，BN 只在 TFiLM 前调用一次。

### Fix M-1：增加 use_pam 参数（2026-04-03）
- **背景**：消融实验 Exp A 需要「无PAM、无TFiLM」的基线模型。
- **修正**：在 `BeatAwareRadar2ECGNet` 添加 `use_pam: bool = True`。
  - `use_pam=False` 时：不创建 PAM 和 TFiLMGenerator 模块，gamma/beta 置零（TFiLM 退化为恒等映射），`forward` 返回 `(ecg_pred, None)`。
  - `TotalLoss` 同步修正：`peak_pred=None` 时跳过 L_peak，L_total = L_time + α·L_freq。
- **参数量对比**：use_pam=True → 1,677,154；use_pam=False → 1,489,281（相差 187,873，约11.2%）。

### Note M-1：radar_spec_input.npy 实际 T=200（非文档所写196/193）
- **现象**：实际分段文件中 `radar_spec_input.npy` 形状为 (N, 1, 33, 200)，文档写的是 T≈196。
- **影响**：SpecAdapter 使用 `F.interpolate(size=L_enc=400)` 处理 T 维，T 值无关紧要，模型正常工作。
- **根因**：step4 分段时 STFT 边界处理方式（`boundary=None, padded=False`）导致实际帧数与理论估算有偏差。不影响功能，无需修改。

---

## 9. Phase 3 待实现

| 文件 | 优先级 | 说明 |
|------|-------|------|
| `src/data/dataset.py` | 高 | RadarECGDataset，fold 切换，惰性加载 |
| `src/utils/metrics.py` | 高 | MAE/RMSE/PCC/PRD/F1 |
| `src/utils/logger.py` | 中 | TensorBoard 封装 |
| `src/utils/seeding.py` | 中 | 随机种子固定 |
| `configs/config.py` | 高 | 超参集中管理（@dataclass） |
| `scripts/train.py` | 高 | 5-Fold CV 训练主脚本 |
| `scripts/test.py` | 中 | 评估脚本 |
| `scripts/run_ablation.sh` | 低 | 消融实验批处理 |
