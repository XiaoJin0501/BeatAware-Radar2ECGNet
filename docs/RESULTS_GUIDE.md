# 实验结果解读指南

本文件说明 `experiments/` 目录的结构、每个输出文件的含义、指标的解读方式，以及消融实验结果的横向比较方法。

---

## 一、目录结构

每次运行 `scripts/train.py` + `scripts/test.py` 后，会在 `experiments/<EXP_TAG>/` 下生成：

```
experiments/
└── <EXP_TAG>/                         ← 由 --exp_tag 指定，如 ExpB_phase
    ├── config.json                    ← 本次实验的完整超参配置（自动保存）
    ├── test_summary.csv               ← 5 fold 汇总 + 均值行（test.py 生成）
    ├── test_summary.json              ← 同上，JSON 格式（方便程序读取）
    └── fold_0/                        ← 每个 fold 独立目录
        ├── checkpoints/
        │   └── best.pt                ← 在 val_mae 最低的 epoch 保存
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

---

## 二、指标含义与解读

### 2.1 波形重建指标（主要关注）

| 指标 | 公式 | 解读 | 目标方向 |
|------|------|------|---------|
| **MAE** | `mean(|pred - gt|)` | ECG 归一化到 [0,1]，MAE=0.05 意味着时域逐点平均误差 5% | 越小越好 |
| **RMSE** | `sqrt(mean((pred-gt)²))` | 对大误差（如 QRS 峰偏差）惩罚更重 | 越小越好 |
| **PCC** | Pearson 相关系数 | 波形整体形态相似度，0=无关，1=完美，通常 >0.85 视为良好 | 越大越好，目标 >0.85 |
| **PRD** | `sqrt(Σ(p-g)²/Σg²)×100%` | ECG 领域标准，<10% 被认为重建质量良好，>20% 明显失真 | 越小越好，目标 <10% |

> **smoke_test 参考值**（2 epochs，未充分训练）：
> - Epoch 1: MAE=0.099, PCC=0.641, PRD=60%
> - Epoch 2: MAE=0.086, PCC=0.685, PRD=54%
>
> PRD 在早期偏高是正常的，充分训练后预期 PRD 降至 10–20% 区间。

### 2.2 峰值检测指标

| 指标 | 含义 | 解读 | 目标方向 |
|------|------|------|---------|
| **R-peak F1** | 在重建 ECG 上用 NeuroKit2 重检 R 峰，与 GT 比对，容忍 ±25ms（±5 samples @200Hz）| 0=完全检测失败，1=完美；F1>0.9 视为临床可用 | 越大越好，目标 >0.90 |

> **R-peak F1 的前提**：重建波形的 QRS 峰形态须足够清晰，NeuroKit2 才能准确检测。早期训练 F1 偏低是正常的。smoke_test epoch 2 达到 0.779，说明波形已有基本 QRS 形态。

### 2.3 高级指标（仅 test.py 最终评估时计算，不在训练 val 中计算）

| 指标列名 | 含义 | 单位 | 目标方向 |
|---------|------|------|---------|
| **dtw** | Dynamic Time Warping（Sakoe-Chiba 窗口 25 samples = 125ms，归一化为路径长度） | 无量纲 | 越小越好 |
| **rr_interval_mae** | 相邻 R 峰间隔（RR 间期）的平均绝对误差，反映心率节律精度 | ms | 越小越好 |
| **qrs_width_mae** | QRS 波群时限（QRS onset → J-point）的平均绝对误差，反映心室除极持续时间 | ms | 越小越好 |
| **qt_interval_mae** | QT 间期（QRS onset → T-wave end）的平均绝对误差，与心室复极和心律失常风险相关 | ms | 越小越好 |
| **pr_interval_mae** | PR 间期（P-wave onset → R-peak）的平均绝对误差，反映房室传导时间 | ms | 越小越好 |

**实现方式**：
- DTW：`compute_dtw_metric(max_samples=500)` 随机采样最多 500 条，防止全量计算过久
- RR interval：NeuroKit2 `ecg_peaks` 检测，取 `mean(diff(peaks)) / fs * 1000`
- QRS / QT / PR：NeuroKit2 DWT delineation（`ecg_delineate(method="dwt")`），一次调用提取所有波界点：
  - `ECG_R_Onsets` = QRS 起始点（Q 波前）
  - `ECG_R_Offsets` = QRS 终止点（J 点）
  - `ECG_T_Offsets` = T 波终点
  - `ECG_P_Onsets`  = P 波起始点
- 所有间期误差单位均为 **ms**（对应论文表格中的临床参考值范围）

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
Epoch   1/150 | train_loss=0.3902 | val_mae=0.0994 | val_pcc=0.6409 | lr=5.05e-05 | 806.9s
  -> Best checkpoint saved (val_mae=0.0994)
```

- `train_loss`：训练集上 `L_total = L_time + α·L_freq + β·L_peak` 的均值
- `val_mae`：验证集时域 MAE（**best checkpoint 的选择依据**）
- `val_pcc`：验证集 Pearson 相关系数
- `lr`：当前学习率（CosineAnnealingLR 衰减）
- `806.9s`：本 epoch 耗时（约 13 分钟/epoch @RTX 4080 Super，150 epochs ≈ 32 小时）

### 3.2 `train_history.json`

包含每 epoch 的完整指标序列，可用于绘制 loss 曲线：
```json
[
  {
    "epoch": 1,
    "total": 0.390,   // 总 loss
    "time":  0.114,   // L_time（MAE）分量
    "freq":  0.558,   // L_freq（STFT）分量
    "peak":  0.248,   // L_peak（BCE）分量
    "val_mae":  0.099,
    "val_rmse": 0.140,
    "val_pcc":  0.641,
    "val_prd":  60.0,
    "val_rpeak_f1": 0.779,  // 仅在 --f1_every 指定的 epoch 出现
    "val_loss": 0.338
  },
  ...
]
```

### 3.3 `test_summary.csv`

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

### 3.4 `sample_predictions.png`

前 8 个验证样本的 GT（蓝色）vs 模型预测（红色）波形对比图，横轴为时间（秒），纵轴为归一化幅度 [0,1]。

**肉眼检查要点**：
- QRS 峰位置是否与 GT 对齐（时间偏移 <25ms 为合格）
- QRS 峰高度是否接近 GT（高度误差反映在 MAE/RMSE 上）
- P 波和 T 波形态是否有隐约轮廓（早期训练可能不明显）
- 是否存在异常振荡或基线漂移

---

## 四、当前结果粒度的局限性

**当前 `test.py` 的指标粒度：fold 级别（混合所有受试者和场景）**

```
Val fold 0 = {GDN0005, GDN0012, GDN0019, GDN0027, GDN0030}
             × {resting, valsalva, apnea}
             的所有 segments 混合 → 一个均值
```

`__getitem__` 虽然返回 `subject` 和 `scenario` 字段，但目前 `test.py` **未使用这两个字段**，全部混合计算。

**这意味着目前看不到**：
- 某个受试者特别难/容易（个体差异）
- resting vs valsalva vs apnea 的性能差异（场景泛化性）
- D5 实验（仅 resting 训练 → 跨场景泛化）需要 per-scenario 指标

**何时需要补充 per-scenario 粒度**：在完成 Exp B（基础性能验证）之后，进入 D5 实验时需要改 `test.py` 加入按 `scenario` 分组的细分指标。

---

## 五、消融实验横向比较（当前缺失）

### 5.1 当前状态

每个 `exp_tag` 只有自己的 `test_summary.csv`，**没有自动跨实验汇总脚本**。

消融实验完成后，各 exp 的 `mean` 行需要手动（或用脚本）拼成论文 Table 2：

| Exp | MAE↓ | RMSE↓ | PCC↑ | PRD↓ | F1↑ |
|-----|------|-------|------|------|-----|
| A（基线，无PAM）| - | - | - | - | - |
| B1（raw）       | - | - | - | - | - |
| B2（phase）     | - | - | - | - | - |
| B3（spec）      | - | - | - | - | - |
| D4（无Conformer）| - | - | - | - | - |

### 5.2 待实现：`scripts/summarize_ablation.py`

计划实现一个汇总脚本，自动读取所有 exp 的 `test_summary.csv` 并输出对比表 + bar chart。实现后更新本文档。

---

## 六、TensorBoard 可视化

```bash
conda activate cyberbrain
tensorboard --logdir experiments/ --port 6006
# 浏览器打开 http://localhost:6006
```

可查看：
- 每 step 的 train loss（total / time / freq / peak 分量）
- 每 epoch 的 val_mae、val_pcc、val_prd、val_rpeak_f1
- 学习率曲线

---

## 七、快速结果核查 Checklist

训练结束后按以下顺序确认结果可信：

- [ ] `train.log`：loss 总体呈下降趋势，无突然暴涨
- [ ] `train_history.json`：`val_mae` 逐 epoch 下降，无 NaN
- [ ] `test_summary.csv`：5 fold 之间 MAE 方差不超过 0.01（否则某个 fold 可能有数据问题）
- [ ] `sample_predictions.png`：QRS 峰与 GT 大致对齐，无全零预测
- [ ] `best.pt` 存在：checkpoint 已保存
