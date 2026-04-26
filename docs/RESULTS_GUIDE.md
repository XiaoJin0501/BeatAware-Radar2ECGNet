# 实验结果解读指南

本文件说明 `experiments/` 和 `experiments_mmecg/` 目录的结构、每个输出文件的含义、指标的解读方式，以及消融实验结果的横向比较方法。

---

## 一、目录结构

### 1.1 Schellenberger 实验（`experiments/`）

每次运行 `scripts/train.py` + `scripts/test.py` 后，会在 `experiments/<EXP_TAG>/` 下生成：

```
experiments/
└── <EXP_TAG>/                         ← 由 --exp_tag 指定，如 ExpB_phase
    ├── config.json                    ← 本次实验的完整超参配置（自动保存）
    ├── test_summary.csv               ← 5 fold 汇总 + 均值行（test.py 生成）
    ├── test_summary.json              ← 同上，JSON 格式（方便程序读取）
    └── fold_0/                        ← 每个 fold 独立目录
        ├── checkpoints/
        │   └── best.pt                ← 在 val_pcc 最高的 epoch 保存
        ├── logs/
        │   ├── train.log              ← 文字日志（每 epoch 一行）
        │   └── events.out.tfevents.*  ← TensorBoard 事件文件
        └── results/
            ├── train_history.json     ← 每 epoch 的 train/val 指标序列
            ├── test_metrics.csv              ← 该 fold 的全局测试指标（test.py 生成）
            ├── test_metrics_by_scenario.csv  ← 按场景分组的指标（resting/valsalva/apnea）
            ├── test_metrics_by_subject.csv   ← 按受试者×场景的指标（用于30-subject柱状图）
            └── sample_predictions.png        ← 前 8 个样本 GT vs Pred + 功率谱对比图
```

### 1.2 MMECG 实验（`experiments_mmecg/`）

每次运行 `scripts/train_mmecg.py` + `scripts/test_mmecg.py` 后，会在 `experiments_mmecg/<EXP_TAG>/` 下生成：

```
experiments_mmecg/
└── <EXP_TAG>/                         ← 如 mmecg_G
    ├── summary_loso.json              ← 11折 LOSO 汇总（test_mmecg.py 生成）
    └── fold_0/                        ← 每折（0~10）独立目录
        ├── config.json                ← 本折的完整超参配置
        ├── checkpoints/
        │   └── best.pt                ← 在 val_pcc 最高的 epoch 保存
        ├── logs/
        │   └── train.log              ← 文字日志（每 epoch 一行）
        └── results/
            ├── train_history.json     ← 每 epoch 的 train/val 指标序列
            └── test_metrics.json      ← 该折测试指标（含 per-state 细分）
```

**`summary_loso.json` 结构**：
```json
{
  "pcc_mean": 0.72, "pcc_std": 0.08,
  "mae_mean": 0.08, "mae_std": 0.01,
  "f1_mean":  0.85, "f1_std":  0.05,
  "per_fold": [...],
  "per_state_summary": {
    "NB": {"pcc_mean": 0.75, "pcc_std": 0.07, ...},
    "IB": {...}, "SP": {...}, "PE": {...}
  }
}
```

---

## 二、指标含义与解读

### 2.1 波形重建指标（主要关注）

| 指标 | 公式 | 解读 | 目标方向 |
|------|------|------|---------|
| **PCC** | Pearson 相关系数 | 波形整体形态相似度，0=无关，1=完美，通常 >0.85 视为良好 | 越大越好，目标 >0.85 |
| **MAE** | `mean(|pred - gt|)` | ECG 归一化到 [0,1]，MAE=0.05 意味着时域逐点平均误差 5% | 越小越好 |
| **RMSE** | `sqrt(mean((pred-gt)²))` | 对大误差（如 QRS 峰偏差）惩罚更重 | 越小越好 |
| **PRD** | `sqrt(Σ(p-g)²/Σg²)×100%` | ECG 领域标准，<10% 被认为重建质量良好，>20% 明显失真 | 越小越好，目标 <10% |

> **MMECG 早期训练参考值（fold 0，epoch 5 附近）**：
> - PCC ≈ 0.49（mmecg_D，有 ReLU bug）
> - PCC 预期提升到 0.65+ （GELU 修复后，mmecg_G）
>
> SOTA（radarODE，SST + Neural ODE）在 MMECG 上 PCC ≈ 0.89。

### 2.2 峰值检测指标

| 指标 | 含义 | 解读 | 目标方向 |
|------|------|------|---------|
| **R-peak F1** | 在重建 ECG 上用 NeuroKit2 重检 R 峰，与 GT 比对，容忍 ±25ms（±5 samples @200Hz）| 0=完全检测失败，1=完美；F1>0.9 视为临床可用 | 越大越好，目标 >0.90 |

### 2.3 高级指标（仅 test.py 最终评估时计算，不在训练 val 中计算）

| 指标列名 | 含义 | 单位 | 目标方向 |
|---------|------|------|---------|
| **dtw** | Dynamic Time Warping（Sakoe-Chiba 窗口 25 samples = 125ms，归一化为路径长度） | 无量纲 | 越小越好 |
| **rr_interval_mae** | 相邻 R 峰间隔（RR 间期）的平均绝对误差，反映心率节律精度 | ms | 越小越好 |
| **qrs_width_mae** | QRS 波群时限（QRS onset → J-point）的平均绝对误差，反映心室除极持续时间 | ms | 越小越好 |
| **qt_interval_mae** | QT 间期（QRS onset → T-wave end）的平均绝对误差，与心室复极和心律失常风险相关 | ms | 越小越好 |
| **pr_interval_mae** | PR 间期（P-wave onset → R-peak）的平均绝对误差，反映房室传导时间 | ms | 越小越好 |

**临床参考值范围**（正常成人）：
| 间期 | 正常范围 |
|------|---------|
| RR interval | 600–1000 ms（心率 60–100 bpm）|
| QRS width | 80–120 ms |
| QT interval | 350–450 ms |
| PR interval | 120–200 ms |

---

## 三、各输出文件解读

### 3.1 `train.log`

每 epoch 一行，格式：
```
2026-04-26 02:05:45 [INFO] Epoch  40/150 | train_loss=0.2312 | val_loss=0.3041 |
  val_pcc=0.5218 | val_mae=0.0724 | val_f1=0.8123 | lr=8.5e-05 | 28.3s
  -> Best checkpoint saved (val_pcc=0.5218)
```

- `train_loss`：训练集上 `L_total = Σ [ 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i ]` 的均值
- `val_pcc`：验证集 Pearson 相关系数（**MMECG best checkpoint 的选择依据**）
- `val_mae`：验证集时域 MAE
- `val_f1`：验证集 R峰 F1（NeuroKit2 重检）
- `lr`：当前学习率（CosineAnnealingLR 衰减）

### 3.2 `train_history.json`

包含每 epoch 的完整指标序列，可用于绘制 loss 曲线：
```json
[
  {
    "epoch": 1,
    "train_loss": 0.390,
    "val_pcc":  0.48,
    "val_mae":  0.094,
    "val_f1":   0.71,
    "val_loss": 0.338,
    "lr": 1.0e-4
  },
  ...
]
```

### 3.3 MMECG `test_metrics.json`（per-fold）

```json
{
  "pcc": 0.72,
  "mae": 0.078,
  "rmse": 0.103,
  "prd": 9.8,
  "f1": 0.87,
  "test_subject": 1,
  "n_samples": 344,
  "per_state": {
    "NB": {"pcc": 0.75, "mae": 0.072, "f1": 0.88, "n_samples": 344},
    "IB": {"pcc": 0.68, "mae": 0.085, "f1": 0.82, "n_samples": 0}
  }
}
```

**注意**：fold 0 的测试集是 subject_1，该受试者只有 344 个窗口且只有 NB 状态。折间性能差异可能很大，需看 11 折均值。

### 3.4 MMECG `summary_loso.json`（11折汇总）

论文中上报的数字 = `pcc_mean`、`f1_mean` 等均值 ± 标准差。

---

## 四、MMECG LOSO 评估注意事项

### 4.1 折间差异来源

LOSO 设计中每折测试集只有 1 个受试者，导致折间性能差异天然很大：

| 来源 | 影响 |
|------|------|
| 受试者个体差异 | 不同受试者胸壁厚度/体位不同，RCG-ECG 相关性不同 |
| 窗口数差异 | 各受试者录音数量不同，窗口数从 ~344 到 ~1000+ 不等 |
| 状态分布 | 某些受试者只有 NB 状态，某些有 4 种状态 |

**结论**：单折 PCC 不代表模型整体性能，必须看 11 折均值 ± 标准差。

### 4.2 当前训练诊断历史

| 实验 | 配置 | fold 0 PCC | 问题 |
|------|------|-----------|------|
| mmecg_D | 0.5-40Hz + ReLU bug | 0.49 (epoch5) | ReLU 截断负半周期 |
| mmecg_E | STFT + 2D Conv | 0.21-0.35 | STFT幅度谱是能量包络，丢失时间精细结构 |
| mmecg_F | 0.8-3.5Hz 窄带 | ~0 | 窄带去除 QRS 高频形态，模型无法重建 |
| **mmecg_G** | **0.5-40Hz + GELU 修复** | **TBD（训练中）** | **当前最优配置** |

### 4.3 per-state 指标解读

| 状态 | 预期难度 | 原因 |
|------|---------|------|
| NB（正常呼吸）| 最易 | 呼吸规律，RCG 信噪比最高 |
| IB（不规则呼吸）| 中等 | 呼吸变化引入时域扰动 |
| SP（坐位）| 中等 | 体位改变影响 range bin 选择 |
| PE（运动后）| 最难 | 心率加快，运动伪影 |

---

## 五、各输出文件解读（Schellenberger）

### 5.1 `test_summary.csv`

`test.py` 评估全部 5 folds 后生成，最后一行是 5-fold 均值：

```
fold, mae,   rmse,  pcc,   prd,    rpeak_f1, loss
0,    0.082, 0.107, 0.71,  9.8,    0.83,     0.29
1,    0.079, 0.103, 0.73,  9.4,    0.85,     0.27
2,    0.085, 0.111, 0.69,  10.2,   0.81,     0.31
3,    0.081, 0.106, 0.72,  9.9,    0.84,     0.30
4,    0.080, 0.104, 0.71,  9.6,    0.82,     0.28
mean, 0.081, 0.106, 0.71,  9.8,    0.83,     0.29
```

**论文中上报的数字 = `mean` 行的各指标。**

### 5.2 `sample_predictions.png`

前 8 个验证样本的 GT（蓝色）vs 模型预测（红色）波形对比图，横轴为时间（秒），纵轴为归一化幅度 [0,1]。

**肉眼检查要点**：
- QRS 峰位置是否与 GT 对齐（时间偏移 <25ms 为合格）
- QRS 峰高度是否接近 GT（高度误差反映在 MAE/RMSE 上）
- P 波和 T 波形态是否有隐约轮廓（早期训练可能不明显）
- 是否存在异常振荡或基线漂移

---

## 六、消融实验横向比较

### 6.1 MMECG 消融（计划）

| Exp | 配置 | PCC (11-fold 均值) | 备注 |
|-----|------|------------------|------|
| mmecg_G | 完整模型（fmcw+pam+emd）| TBD | 当前训练中 |
| mmecg_A | fmcw only，无pam，无emd | TBD | 待执行 |
| mmecg_B | fmcw+pam，无emd | TBD | 待执行 |
| mmecg_C | fmcw，无pam，emd | TBD | 待执行 |

### 6.2 Schellenberger 消融（计划）

| Exp | KI | PA | CP | MAE↓ | PCC↑ | F1↑ |
|-----|:--:|:--:|:--:|------|------|-----|
| Model A（基线）| ✓ | | | TBD | TBD | TBD |
| Model B（+PA）| ✓ | ✓ | | TBD | TBD | TBD |
| Model C（完整）| ✓ | ✓ | ✓ | TBD | TBD | TBD |

---

## 七、TensorBoard 可视化

```bash
conda activate cyberbrain
# Schellenberger
tensorboard --logdir experiments/ --port 6006
# MMECG
tensorboard --logdir experiments_mmecg/ --port 6007
# 浏览器打开 http://localhost:6006 或 6007
```

可查看：
- 每 step 的 train loss（total / time / freq / peak 分量）
- 每 epoch 的 val_pcc、val_mae、val_f1、val_loss
- 学习率曲线

---

## 八、快速结果核查 Checklist

训练结束后按以下顺序确认结果可信：

- [ ] `train.log`：loss 总体呈下降趋势，无突然暴涨
- [ ] `train_history.json`：`val_pcc` 总体上升趋势，无 NaN
- [ ] **MMECG**：`summary_loso.json` 存在，11折均值 PCC 合理
- [ ] **MMECG**：`per_state` 细分存在，NB 通常最高，PE 通常最低
- [ ] **Schellenberger**：`test_summary.csv` 的5折 MAE 方差不超过 0.01
- [ ] `sample_predictions.png`（Schellenberger）：QRS 峰与 GT 大致对齐，无全零预测
- [ ] `best.pt` 存在：checkpoint 已保存
