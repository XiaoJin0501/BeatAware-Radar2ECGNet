# 数据预处理实验记录

**项目**：BeatAware-Radar2ECGNet
**数据集**：(1) Schellenberger et al., *Scientific Data* 7:291 (2020)；(2) MMECG（77GHz FMCW）
**记录目的**：追踪每次预处理的参数配置、质量控制结果和最终数据集统计，为论文撰写提供可引用的数据基础。

---

## Part A：Schellenberger 数据集预处理

### 预处理流程概述

```
原始 .mat (2000Hz)
   ├── radar_i / radar_q  [step1 → dataset/GDNXXXX/scenario/]
   │     → 椭圆校正 → 反正切解调 → 带通滤波(0.5-10Hz) → 降采样至200Hz
   │     → radar_raw.npy        (全段，未滤波相位，shape (L,))
   │     → radar_phase.npy      (全段，滤波相位，shape (L,))
   │     → radar_spec_input.npy (全段细粒度STFT, shape (1,33,T_full))
   │     → radar_spec_loss.npy  (全段多分辨率STFT, shape (3,F,T_full))
   └── tfm_ecg2 (Lead II)  [step2 → dataset/GDNXXXX/scenario/]
         → NeuroKit2清洗 → R峰检测 → 降采样至200Hz
         → ecg_clean.npy      (全段清洗ECG，shape (L,)，未归一化)
         → rpeak_indices.npy  (全段R峰索引，200Hz坐标)

step3 QC：读取上述全段文件，输出 qc_report.json

step4 分段  [→ dataset/GDNXXXX/scenario/segments/]  ← 独立子目录，不覆盖上述文件
   → radar_raw.npy        (N, 1, 1600)
   → radar_phase.npy      (N, 1, 1600)
   → radar_spec_input.npy (N, 1, 33, T_seg)
   → radar_spec_loss.npy  (N, 3, F, T_seg)
   → ecg.npy              (N, 1, 1600)，per-segment min-max归一化到[0,1]
   → rpeak.npy            (N, 1, 1600)，高斯软标签 σ=5点

分段参数：window=1600点(8s)，stride=800点(50%重叠)
数据划分：5-Fold CV（按受试者，KFold seed=42）
```

---

### Run #1 — Schellenberger

**日期**：2026-04-02
**执行人**：Claude Code (自动化运行)
**数据集路径**：`/home/qhh2237/Datasets/Med_Radar`
**输出路径**：`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset`

#### 1.1 环境

| 项目 | 版本 |
|------|------|
| Python | 3.10 (conda: cyberbrain) |
| PyTorch | 1.13.1 |
| NeuroKit2 | — |
| SciPy | — |
| 硬件 | RTX 4080 SUPER 16GB, Ubuntu |

#### 1.2 预处理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 原始采样率（雷达/ECG） | 2000 Hz | 实测确认，两路均为2000Hz且等长 |
| 目标采样率 | **200 Hz** | 降采样因子=10（分步：×2 × ×5） |
| 降采样方式 | `scipy.signal.decimate`，zero-phase | 分步执行避免数值不稳定 |
| 带通滤波（雷达相位） | 0.5–10 Hz，4阶Butterworth，sosfiltfilt | 在2000Hz下滤波后再降采样 |
| ECG导联 | `tfm_ecg2`（导联II） | R波最明显，便于特征提取 |
| ECG清洗方法 | NeuroKit2 `neurokit` | 含基线漂移校正+高频噪声去除 |
| R峰检测方法 | NeuroKit2 `neurokit`（失败时回退`pantompkins`） | — |
| 高斯Mask σ | **5 采样点 = 25 ms @ 200Hz** | R峰软标签 |
| 分段窗口 | **1600 点 = 8 s** | 覆盖约6–8个心跳周期（@75bpm） |
| 分段步长 | **800 点 = 4 s（50%重叠）** | 扩充训练样本量 |
| radar_spec_input | nperseg=64, noverlap=56, stride=8 @200Hz | → shape (1, 33, ~193) per segment |
| radar_spec_loss | nperseg=[128,256,512], noverlap=[64,128,256] @200Hz | → shape (3, 65, 5) per segment |
| 场景范围 | Resting / Valsalva / Apnea | 忽略TiltUp / TiltDown |
| ECG归一化 | per-segment min-max → [0, 1] | 在step4分段后执行 |

#### 1.3 QC 阈值（初始默认值，待数据驱动调整）

| 指标 | 阈值 | 触发动作 |
|------|------|---------|
| 雷达相位跳变率 | > 1% | 剔除整个受试者 |
| ECG基线漂移比（<0.5Hz能量占比） | > 30% | 剔除整个受试者 |
| R峰检测失效窗口比例 | > 20% | 剔除整个受试者 |

#### 1.4 数据集构成（预处理前）

| 场景 | 受试者数 | 缺失受试者 |
|------|---------|-----------|
| Resting | 30 | — |
| Valsalva | 27 | GDN0015, GDN0024, GDN0026 |
| Apnea | 24 | GDN0001, GDN0002, GDN0003, GDN0015, GDN0024, GDN0026 |

总 .mat 文件数（3场景）：30 + 27 + 24 = **81 个**（TiltUp/TiltDown的54个文件忽略）

#### 1.5 QC 结果

**剔除受试者**：无（全部30人通过）
**通过QC受试者数**：30 / 30

**各受试者质量指标**（雷达相位跳变率 / ECG基线漂移比 / R峰失效率）：

| 受试者 | Resting | Valsalva | Apnea | 结论 |
|--------|---------|----------|-------|------|
| GDN0001 | 0.0000 / 0.0066 / 0.000 | 0.0000 / 0.0071 / 0.000 | — | ✓ |
| GDN0002 | 0.0000 / 0.0260 / 0.000 | 0.0000 / 0.0261 / 0.000 | — | ✓ |
| GDN0003 | 0.0000 / 0.0322 / 0.000 | 0.0000 / 0.0269 / 0.000 | — | ✓ |
| GDN0004 | 0.0000 / 0.0155 / 0.000 | 0.0001 / 0.0151 / 0.000 | 0.0000 / 0.0167 / 0.000 | ✓ |
| GDN0005 | 0.0000 / 0.0258 / 0.000 | 0.0001 / 0.0262 / 0.000 | 0.0000 / 0.0229 / 0.000 | ✓ |
| GDN0006 | 0.0005 / 0.0334 / 0.000 | 0.0003 / 0.0380 / 0.000 | 0.0000 / 0.0362 / 0.000 | ✓ |
| GDN0007 | 0.0000 / 0.0405 / 0.000 | 0.0000 / 0.0390 / 0.000 | 0.0000 / 0.0438 / 0.000 | ✓ |
| GDN0008 | 0.0002 / 0.0240 / 0.000 | 0.0001 / 0.0219 / 0.000 | 0.0000 / 0.0110 / 0.000 | ✓ |
| GDN0009 | 0.0000 / 0.0288 / 0.000 | 0.0000 / 0.0311 / 0.000 | 0.0000 / 0.0252 / 0.000 | ✓ |
| GDN0010 | 0.0000 / 0.0592 / 0.000 | 0.0000 / 0.0598 / 0.000 | 0.0000 / 0.0657 / 0.000 | ✓ |
| GDN0011 | 0.0000 / 0.0252 / 0.000 | 0.0000 / 0.0239 / 0.000 | 0.0000 / 0.0166 / 0.000 | ✓ |
| GDN0012 | 0.0000 / 0.0371 / 0.000 | 0.0000 / 0.0328 / 0.000 | 0.0000 / 0.0257 / 0.000 | ✓ |
| GDN0013 | 0.0000 / 0.0158 / 0.000 | 0.0000 / 0.0146 / 0.000 | 0.0000 / 0.0111 / 0.000 | ✓ |
| GDN0014 | 0.0000 / 0.0076 / 0.000 | 0.0001 / 0.0085 / 0.000 | 0.0000 / 0.0064 / 0.000 | ✓ |
| GDN0015 | 0.0000 / 0.0261 / 0.000 | — | — | ✓ |
| GDN0016 | 0.0000 / 0.0471 / 0.000 | 0.0000 / 0.0424 / 0.000 | 0.0000 / 0.0411 / 0.000 | ✓ |
| GDN0017 | 0.0000 / 0.0289 / 0.000 | 0.0000 / 0.0227 / 0.000 | 0.0000 / 0.0158 / 0.000 | ✓ |
| GDN0018 | 0.0000 / 0.0421 / 0.000 | 0.0000 / 0.0391 / 0.000 | 0.0000 / 0.0271 / 0.000 | ✓ |
| GDN0019 | 0.0000 / 0.0363 / 0.000 | 0.0000 / 0.0343 / 0.000 | 0.0000 / 0.0372 / 0.000 | ✓ |
| GDN0020 | 0.0000 / 0.0199 / 0.000 | 0.0000 / 0.0177 / 0.000 | 0.0000 / 0.0188 / 0.000 | ✓ |
| GDN0021 | 0.0000 / 0.0423 / 0.000 | 0.0000 / 0.0388 / 0.000 | 0.0000 / 0.0364 / 0.000 | ✓ |
| GDN0022 | 0.0000 / 0.0385 / 0.000 | 0.0000 / 0.0371 / 0.000 | 0.0000 / 0.0321 / 0.000 | ✓ |
| GDN0023 | 0.0000 / 0.0787 / 0.000 | 0.0000 / 0.0769 / 0.000 | 0.0000 / 0.0764 / 0.000 | ✓ |
| GDN0024 | 0.0000 / 0.0131 / 0.000 | — | — | ✓ |
| GDN0025 | 0.0000 / 0.0280 / 0.000 | 0.0000 / 0.0205 / 0.000 | 0.0000 / 0.0150 / 0.000 | ✓ |
| GDN0026 | 0.0001 / 0.0209 / 0.000 | — | — | ✓ |
| GDN0027 | 0.0000 / 0.0245 / 0.000 | 0.0000 / 0.0258 / 0.000 | 0.0000 / 0.0331 / 0.000 | ✓ |
| GDN0028 | 0.0000 / 0.0230 / 0.000 | 0.0000 / 0.0218 / 0.000 | 0.0000 / 0.0127 / 0.000 | ✓ |
| GDN0029 | 0.0000 / 0.0221 / 0.000 | 0.0000 / 0.0265 / 0.000 | 0.0000 / 0.0157 / 0.000 | ✓ |
| GDN0030 | 0.0000 / 0.0330 / 0.000 | 0.0000 / 0.0316 / 0.000 | 0.0000 / 0.0336 / 0.000 | ✓ |

**指标观察（论文可引用）**：
- 雷达相位跳变率极低（最大值 0.0005），说明该批次数据无显著运动伪影或传感器掉落
- ECG基线漂移比普遍 < 8%，最高为 GDN0023（~7.8%），远低于30%剔除阈值；NeuroKit2基线校正效果良好
- R峰检测失效率全部为 0.000，说明 Lead II ECG 信号质量优异，Pan-Tompkins 算法检测稳定
- **结论**：该数据集整体信号质量高，无需基于当前阈值剔除任何受试者

#### 1.6 分段统计

| 场景 | 参与受试者 | 总分段数 | 平均每人分段数 |
|------|----------|---------|--------------|
| Resting | 30 | 4,714 | 157.1 |
| Valsalva | 27 | 6,953 | 257.5 |
| Apnea | 24 | 1,140 | 47.5 |
| **合计** | 30 | **12,807** | 427.0 |

**受试者级分段数（verify_dataset.py 输出）**：

| 受试者 | Resting | Valsalva | Apnea | 合计 |
|--------|---------|----------|-------|------|
| GDN0001 | 150 | 249 | 0 | 399 |
| GDN0002 | 154 | 247 | 0 | 401 |
| GDN0003 | 149 | 263 | 0 | 412 |
| GDN0004 | 149 | 271 | 35 | 455 |
| GDN0005 | 151 | 269 | 36 | 456 |
| GDN0006 | 151 | 326 | 34 | 511 |
| GDN0007 | 157 | 246 | 45 | 448 |
| GDN0008 | 153 | 245 | 25 | 423 |
| GDN0009 | 161 | 243 | 55 | 459 |
| GDN0010 | 158 | 246 | 65 | 469 |
| GDN0011 | 161 | 242 | 50 | 453 |
| GDN0012 | 161 | 244 | 99 | 504 |
| GDN0013 | 180 | 251 | 26 | 457 |
| GDN0014 | 149 | 244 | 97 | 490 |
| GDN0015 | 161 | 0 | 0 | 161 |
| GDN0016 | 151 | 246 | 27 | 424 |
| GDN0017 | 149 | 264 | 37 | 450 |
| GDN0018 | 157 | 288 | 22 | 467 |
| GDN0019 | 149 | 275 | 30 | 454 |
| GDN0020 | 159 | 245 | 60 | 464 |
| GDN0021 | 152 | 265 | 25 | 442 |
| GDN0022 | 163 | 272 | 37 | 472 |
| GDN0023 | 168 | 247 | 71 | 486 |
| GDN0024 | 151 | 0 | 0 | 151 |
| GDN0025 | 153 | 256 | 84 | 493 |
| GDN0026 | 204 | 0 | 0 | 204 |
| GDN0027 | 155 | 258 | 40 | 453 |
| GDN0028 | 151 | 247 | 59 | 457 |
| GDN0029 | 152 | 248 | 40 | 440 |
| GDN0030 | 155 | 256 | 41 | 452 |

#### 1.7 5-Fold CV 划分

随机种子 seed=42，`sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)`，按受试者划分（每折6人）：

| Fold | 测试集受试者 | 测试集分段数 |
|------|------------|------------|
| fold_0 | GDN0009, GDN0010, GDN0016, GDN0018, GDN0024, GDN0028 | 2,427 |
| fold_1 | GDN0001, GDN0005, GDN0013, GDN0017, GDN0025, GDN0029 | 2,695 |
| fold_2 | GDN0002, GDN0003, GDN0006, GDN0012, GDN0014, GDN0023 | 2,804 |
| fold_3 | GDN0004, GDN0019, GDN0022, GDN0026, GDN0027, GDN0030 | 2,490 |
| fold_4 | GDN0007, GDN0008, GDN0011, GDN0015, GDN0020, GDN0021 | 2,391 |

各折分段数范围：2,391–2,804（最大差 < 18%，分布较均衡）

#### 1.8 运行时间

step1 与 step2 并行执行：

| 步骤 | 耗时 | 备注 |
|------|------|------|
| step1（雷达处理，30受试者 × 3场景 = 81个文件） | ~2 min | 与 step2 并行 |
| step2（ECG处理，81个文件） | ~2 min | 与 step1 并行 |
| step3（QC，30受试者） | < 1 min | — |
| step4（分段保存，12807个分段） | ~40 sec | — |
| verify_dataset | < 5 sec | — |
| **合计（并行后）** | **~3 min** | — |

#### 1.9 异常与处理记录

| 时间 | 受试者/场景 | 问题描述 | 处理方式 |
|------|-----------|---------|---------|
| 2026-04-02 | GDN0006（全场景） | `measurement_info` 字段中受试者ID拼写错误：`GND0006`（缺字母N），导致输出目录错误命名为 `GND0006/` | 修复 `mat_loader.py`：受试者ID改为始终从文件名解析；删除错误目录后重新处理 |
| 2026-04-02 | GDN0025（全场景） | `measurement_info` 字段中受试者ID为 `GDN0025_no02`（非标准命名），导致输出目录错误命名为 `GDN0025_no02/` | 同上，文件名解析修复后重新处理 |
| 2026-04-02 | 全部受试者 | step4 将分段文件保存到与 step1/step2 相同的目录，覆盖了全段中间文件（`radar_raw.npy`、`radar_phase.npy`、`radar_spec_*.npy`）。若 step3 在 step4 后重跑，QC 会读取分段数据而非全段录音，导致指标错误 | 修复 step4：输出改至 `segments/` 子目录。删除被覆盖文件，重新运行 step1 + step4 |

**数据集原始文件质量说明（论文可引用）**：
> 该数据集部分受试者的 MATLAB .mat 文件中 `measurement_info` 字段存在受试者ID不一致问题（如 GDN0006 误写为 GND0006，GDN0025 附有 `_no02` 后缀）。本研究已通过解析文件名（格式：`GDNXXXX_N_Scenario.mat`）统一提取受试者ID，规避了此问题。

#### 1.10 输出文件校验结果

```
校验状态：全部通过 ✓
受试者数：30 / 30
总分段数：12,807
  Resting : 4,714 段（30人）
  Valsalva: 6,953 段（27人）
  Apnea   : 1,140 段（24人）
Shape校验：radar_raw/phase (N,1,1600) ✓ | ecg/rpeak (N,1,1600) ✓
           radar_spec_input (N,1,33,T) ✓ | radar_spec_loss (N,3,F,T) ✓
值域校验：ecg.npy ∈ [0,1] ✓ | rpeak.npy ∈ [0,1] ✓
```

---

### 复现命令（Schellenberger）

```bash
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet
conda activate cyberbrain

# Step1 和 Step2 可并行运行
python -m data_preprocessing.step1_radar_processing \
  --raw_dir /home/qhh2237/Datasets/Med_Radar \
  --out_dir dataset

python -m data_preprocessing.step2_ecg_processing \
  --raw_dir /home/qhh2237/Datasets/Med_Radar \
  --out_dir dataset

# Step3/4/verify 顺序执行
python -m data_preprocessing.step3_qc \
  --dataset_dir dataset \
  --out_json dataset/qc_report.json

python -m data_preprocessing.step4_segment_save \
  --dataset_dir dataset \
  --qc_report dataset/qc_report.json

python -m data_preprocessing.verify_dataset \
  --dataset_dir dataset
```

---

## Part B：MMECG 数据集预处理

### 预处理流程概述

```
MMECG.h5（/home/qhh2237/Datasets/MMECG/MMECG.h5）
   ├── 读取所有录音 group（11受试者 × ~33录音/人 = ~364条）
   │   每条录音：RCG (35505, 50)，ECG (35505, 1)，attrs: subject_id, state
   │
   ├── RCG 处理：
   │     转置 → (50, T)
   │     → 0.5-40 Hz 宽带带通（4阶Butterworth，sosfilt）
   │     → 逐通道 z-score 归一化（消除 range bin 间 ~5× 幅度差异）
   │
   ├── ECG 处理：
   │     → 0.5-40 Hz 带通
   │     → 滑窗切片后 per-window min-max → [0,1]
   │     → NeuroKit2 Pan-Tompkins R峰检测 → Gaussian软标签 σ=5点
   │
   ├── 滑窗切片：window=1600, stride=800（50%重叠）
   │     → rcg: (N, 50, 1600)
   │     → ecg: (N,  1, 1600)
   │     → rpeak: (N, 1, 1600)
   │     → meta: (N, 2) = [subject_id, state_code]
   │
   ├── 按受试者聚合（跨同一受试者的所有录音合并）
   └── 保存 NPY + metadata_mmecg.json（含 LOSO fold 分配）
```

**state_code 映射**：`{"NB": 0, "IB": 1, "SP": 2, "PE": 3}`

---

### Run #1 — MMECG

**日期**：2026-04-25
**执行脚本**：`scripts/preprocess_mmecg.py`
**数据集路径**：`/home/qhh2237/Datasets/MMECG/MMECG.h5`
**输出路径**：`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset_mmecg/`

#### B.1 预处理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 原始采样率 | 200 Hz | MMECG 数据集原生 200Hz，无需降采样 |
| 雷达带通滤波 | **0.5–40 Hz**，4阶Butterworth，sosfilt | 宽带保留 QRS 高频形态，不去除 ECG 重建所需的高频信息 |
| 雷达归一化 | 逐通道 z-score | 消除各 range bin 间幅度差异，使 SE 注意力按信号质量选 bin |
| ECG 带通滤波 | 0.5–40 Hz | 与 RCG 一致 |
| ECG 归一化 | per-window min-max → [0,1] | 每个分段独立归一化 |
| R峰检测 | NeuroKit2 Pan-Tompkins 1985 | 失败时返回空数组，对应分段 rpeak 全零 |
| 高斯软标签 σ | **5 采样点 = 25ms** | 与 Schellenberger 一致 |
| 分段窗口 | **1600 点 = 8s** | 与 Schellenberger 一致 |
| 分段步长 | **800 点（50%重叠）** | 与 Schellenberger 一致 |
| 数据集划分 | **LOSO 11折** | fold_i 留出 subject_ids[i]；按 subject_id 排序后分配 |

#### B.2 滤波策略设计决策记录

本实验中探索了两种滤波策略，最终选定宽带：

| 策略 | 滤波范围 | 结论 |
|------|---------|------|
| 窄带（0.8-3.5Hz） | 仅保留心跳频带 | **放弃**。0.626 PCC 是窄带RCG vs 窄带ECG的结果，并非vs全ECG。窄带滤波去除了QRS高频形态，模型无法重建ECG形态。 |
| **宽带（0.5-40Hz）**（采用） | 保留完整心脏波形频段 | 包含静态杂波(13-16Hz)但保留QRS形态信息。模型通过z-score+SE注意力自适应抑制噪声range bin。 |

**关键洞察**：FMCW range bin 与 ECG 的宽带 PCC 只有 0.064（单个最优 bin），远低于 SST 等频域方法。但这个低 PCC 包含足够的时序信息（R峰位置），配合 SE 注意力聚合 50 通道，模型仍可学习合理的映射。

#### B.3 数据集统计

| 受试者 | 总窗口数 | 状态构成 |
|--------|---------|---------|
| subject_1 | 344 | NB only（fold 0 测试集，难折）|
| subject_2~11 | 待统计（mmecg_G 完成后更新）| — |
| **合计** | TBD | — |

**受试者数量**：11
**录音总数**：~364条（从 H5 实际读取后确认）

#### B.4 LOSO fold 分配

```
fold_0: test=subject_1,  train=subject_2~11
fold_1: test=subject_2,  train=subject_1, subject_3~11
...
fold_10: test=subject_11, train=subject_1~10
```

详见 `dataset_mmecg/metadata_mmecg.json` 中的 `loso_folds` 字段。

#### B.5 运行时间

| 步骤 | 耗时 | 备注 |
|------|------|------|
| 读取 H5 + 处理所有录音 | ~5-10 min | 含 R峰检测（NeuroKit2 per-window）|
| 保存 NPY + metadata | < 1 min | — |
| **合计** | **~10 min** | — |

#### B.6 复现命令

```bash
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet
conda activate cyberbrain
python scripts/preprocess_mmecg.py
# 如需静默模式
python scripts/preprocess_mmecg.py --quiet
```

---

## 关键设计决策记录

### Schellenberger 相关

| 决策 | 选择 | 备选方案 | 理由 |
|------|------|---------|------|
| ECG导联 | Lead II (`tfm_ecg2`) | Lead I | R波幅度最大，便于特征提取和R峰检测 |
| 目标采样率 | 200 Hz | 250 Hz / 500 Hz | 与前代文献一致；8s=1600点，计算量适中 |
| 高斯软标签 | σ=5点 (25ms) | 二值标签 | 平滑损失面，对R峰定位误差更鲁棒 |
| 分段长度 | 8s (1600点) | 10s | 覆盖6–8个心跳，计算量与上下文平衡 |
| 数据划分 | 5-Fold CV（按受试者） | 按片段随机划分 | 严格防止数据泄露；反映跨受试者泛化能力 |

### MMECG 相关

| 决策 | 选择 | 备选方案 | 理由 |
|------|------|---------|------|
| RCG 滤波范围 | 0.5-40Hz 宽带 | 0.8-3.5Hz 窄带 | 窄带去除 QRS 高频形态，ECG 重建质量极差（实验 mmecg_F 验证）|
| RCG 归一化 | 逐通道 z-score | 全局归一化 | 消除 range bin 间幅度差异；使 SE 注意力基于信号质量而非幅度工作 |
| 分段策略 | 同 Schellenberger | — | 统一超参，便于跨数据集比较 |
| 数据集划分 | LOSO（严格跨受试者）| 随机分割 | 11人数据集用 LOSO 是领域惯例；每折严格留出 1 受试者 |

---

*本文档随每次预处理运行自动/手动更新，每次 Run 追加新章节。*
