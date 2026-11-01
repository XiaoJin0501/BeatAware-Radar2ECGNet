# PAPER_FACTS — REM-Net phase_pd 论文事实清单

导出日期：2026-11-01。只读汇总，不训练、不重选 checkpoint，不补造实验结果。
本清单的 verified 表示能由所列代码/结果核验；不是对临床有效性或原论文复现等价性的背书。missing 表示该具体问题尚无核验依据，不代表数值为零。结果文件以路径引用，代码以行号引用。A–I 全部需求均列出。
**写作边界：**“seed 42”为实验基种子标签，实际初始化通常为 42+物理 fold；phase_pd 为 Shape-200 输出，连续结果来自无参数逐拍组装，不是连续回归模型。旧模型图表不能移植为新模型证据。单折复杂度不能表示全记录逐拍组装耗时。

## A. 数据集 / Experimental Setup

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| MMECG A1 | 11 名被试；性别标签计数 {'girl': 3, 'boy': 8}（3/11=27.27%女性，8/11=72.73%男性）；年龄 30.7273±13.5358 岁（样本SD），范围 [22,65]；均按被试去重。 | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| MMECG A2 总量 | 91 条记录；元数据时长求和 16154.775 s（原始总时长，不是接受窗口长度之和）。 | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| MMECG A2 IB | 7 被试，18 记录，3195.450 s | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| MMECG A2 NB | 11 被试，43 记录，7633.575 s | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| MMECG A2 PE | 6 被试，12 记录，2130.300 s | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| MMECG A2 SP | 3 被试，18 记录，3195.450 s | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| CR_RVS A1 | 30 名被试；性别标签计数 {'F': 16, 'M': 14}（53.33%女性，46.67%男性）；年龄 30.7333±10.0890 岁（样本SD），范围 [21,61]；均按被试去重。 | `/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv` | verified |
| CR_RVS A2 总量 | 81 条记录；元数据时长求和 51722.075 s（原始总时长，不是接受窗口长度之和）。 | `/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv` | verified |
| CR_RVS A2 Apnea | 24 被试，24 记录，4705.225 s | `/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv` | verified |
| CR_RVS A2 Resting | 30 被试，30 记录，19048.575 s | `/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv` | verified |
| CR_RVS A2 Valsalva | 27 被试，27 记录，27968.275 s | `/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv` | verified |
| MMECG A1 健康状况原措辞 | missing — 本轮未建立原文健康描述与本地11名子集的对应证据，不能从年龄/性别推断健康状况。 | `reference_papers/MMECG_baselines model/Chen et al. - 2024 - Contactless Electrocardiogram Monitoring With Millimeter Wave Radar.pdf` | missing |
| CR_RVS A1 原论文 | 原文短语 “30 healthy subjects”；14 male /16 female，30.7±9.9 岁（原论文汇总，与本地重算需区分）。 | `/home/qhh2237/Datasets/CR_RVS/CR_RVS Dataset.pdf, pp.1–2` | verified |
| MMECG A3 已核实 | 77 GHz FMCW；本地200 Hz、50个RCG通道/空间位置，并非50根天线。 | `/home/qhh2237/Datasets/MMECG/metadata/MMECG_Dataset.md:5–55` | verified |
| MMECG A3 原论文设备 | TI AWR1843 + DCA1000；3 TX /4 RX，12虚拟通道；Table1帧周期5ms，§4.2.1报告总带宽3.32GHz（§2另有4GHz概述，不能不加区分）；仰卧、quasi-static、胸前0.4–0.5m，主瓣约对准胸骨。这是原论文设备描述，不证明公开子集每条记录有独立设备日志。 | `reference_papers/MMECG_baselines model/Chen et al. - 2024 - Contactless Electrocardiogram Monitoring With Millimeter Wave Radar.pdf, pp.278–279, §4.1–4.2.1, Table1` | verified |
| CR_RVS A3 | 24 GHz ISM Six-Port CW，bistatic；I/Q两路同步ADC 2000 Hz；约40 cm；受试者躺在tilt table，上身面向雷达。CW无FMCW扫频带宽。 | `/home/qhh2237/Datasets/CR_RVS/CR_RVS Dataset.pdf, pp.3–4` | verified |
| CR_RVS A4 | Task Force Monitor（CNSystems），tfm_ecg2（Lead II），原始2000 Hz；本研究200 Hz。 | `/tmp/rem_dual_head_formal/scripts/build_crrvs_wsst_shape_h5.py:1–18,39–49; /home/qhh2237/Datasets/CR_RVS/CR_RVS Dataset.pdf, pp.3–4` | verified |
| MMECG A4 设备/导联 | 原论文ECG为TI ADS1292 evaluation board；公开本地ECG单通道200 Hz；具体导联名称missing，不能从单通道推测Lead II。 | `reference_papers/MMECG_baselines model/Chen et al. - 2024 - Contactless Electrocardiogram Monitoring With Millimeter Wave Radar.pdf, p.279 §4.1; /home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| A5 bib keys | MMECG: chenmmwave；CR_RVS: schellenberger2020radar；后者 DOI 10.1038/s41597-020-00629-5。 | `docs/paper/sections/references.bib:74–83,660–671` | verified |
| CR_RVS A5 获取/伦理 | 原论文说明figshare公开.mat；伦理批准FAU Erlangen-Nürnberg No.85_15B，参与者书面知情同意。论文自身CC BY4.0；不可未经核实直接等同数据集许可。 | `/home/qhh2237/Datasets/CR_RVS/CR_RVS Dataset.pdf, pp.3,7–9` | verified |
| A5 数据集许可/MMECG获取与伦理 | missing — 数据集许可文本、MMECG正式下载链接及伦理编号尚未逐项核验。 | `docs/paper/sections/references.bib` | missing |
| MMECG A6 排除规则 | 非有限雷达/ECG、ECG std<0.01、峰峰值<0.02、R数<4或>18；注意判定使用已逐窗z-score的ECG。 | `/home/qhh2237/Datasets/MMECG/05B_build_windowed_h5_loso_v2.py:76–85,112–125` | verified |
| A6 两数据集排除数量 | missing — 接受窗口数可核实（B9）；尚无可核验的完整拒绝日志/排除原因计数，不用原始数减窗口数估计。 | `/home/qhh2237/Datasets/MMECG/processed/loso_v2; /home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold` | missing |


### A6 实际测试记录覆盖核对

- **verified MMECG**：元数据91记录，所有test folds中实际出现91记录；无test窗口的记录=[]。来源：`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_XX/test.h5:record_id`与`/home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv`。没有test窗口不等于模型评价时排除；若有，具体拒绝原因/窗口数仍missing。
- **verified CR_RVS**：元数据81记录，所有test folds中实际出现80记录；无test窗口的记录=['CRRVS_0024']（metadata: GDN0009 /Apnea /227.4s，不能归因为原始记录不足8s）。来源：`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_XX/test.h5:record_id`与`/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv`。没有test窗口不等于模型评价时排除；若有，具体拒绝原因/窗口数仍missing。

## B. 预处理与输入

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| B1 MMECG | 使用公开50通道RCG和clean archive；未从原始ADC重建相位。原始RCG相位提取/空间选择算法不由本地输入维度证明。 | `experiments/development/structured_decoder_v0_20261017_ab/rawio.py:59–68` | verified |
| B1 MMECG 处理 | clean archive处理：逐通道detrend→逐通道z-score→四阶Butterworth零相位filtfilt 0.5–20 Hz；ECG0.5–40 Hz；NaN/Inf线性修补，坏值比例>5%标记bad_record；xcorr lag仅审计，不强制时移。原论文3D beamforming后K-means输出N=50空间簇；公开RCG已完成上游处理，本研究没有重做raw ADC beamforming。 | `/home/qhh2237/Datasets/MMECG/04B_preprocess_records.py:23–37,76–104; reference_papers/MMECG_baselines model/Chen et al. - 2024 - Contactless Electrocardiogram Monitoring With Millimeter Wave Radar.pdf, p.279 §4.2.2` | verified |
| B1 CR_RVS | unwrap(angle(I+jQ))→线性detrend→resample_poly至200 Hz→四阶Butterworth零相位filtfilt 0.6–12 Hz；每窗robust-z=(x−median)/(1.4826 MAD+1e−6)，clip至±8。 | `/tmp/rem_dual_head_formal/scripts/build_crrvs_wsst_shape_h5.py:57–78,129–165,274–280` | verified |
| B2 WSST | ssqueezepy ssq_cwt，Morlet，nv=16；取变换幅值中部起连续71频率行（不是已指定Hz带通）；Fourier resample时间至240；非负clip后每通道每窗min–max；float16缓存；50=RCG通道，71=所选频率行，240=时间位置。 | `/tmp/rem_dual_head_formal/scripts/precompute_wsst8s_mmecg.py:40–76` | verified |
| B2 WSST 精确Hz范围 | missing — 代码按频率行索引截取而不保留Hz轴，不能凭71行编造频率上下限。 | `/tmp/rem_dual_head_formal/scripts/precompute_wsst8s_mmecg.py:58–64` | missing |
| B3 STFT | Hann窗128点(640 ms)，noverlap112，hop16点=80 ms；nfft默认128；one-sided 0–100 Hz共65频点，boundary=None,padded=False；1600点得93帧；log1p(abs(Z))后全图均值/SD标准化；输入1×65×93。 | `/tmp/rem_dual_head_formal/scripts/build_crrvs_wsst_shape_h5.py:88–101,279–280` | verified |
| B4 ECG | MM原始200 Hz，四阶Butterworth 0.5–40 Hz；CR从2000 resample_poly到200，带通0.5–40 Hz。Shape目标最终逐拍min–max。旧MM窗口缓存另有逐窗z-score，不能称为原生绝对mV。 | `experiments/development/structured_decoder_v0_20261017_ab/rawio.py:59–77; /tmp/rem_dual_head_formal/scripts/build_crrvs_wsst_shape_h5.py:129–143; /home/qhh2237/Datasets/MMECG/05B_build_windowed_h5_loso_v2.py:118` | verified |
| B4 scaler caveat | 旧连续Fold.fit_scaler在训练H5缓存的ECG上估计median/IQR，并非直接原始记录；新的临床图参考由raw_reference重建，不将其与Shape缓存混称同一幅度空间。 | `/tmp/rem_dual_head_formal/scripts/train_continuous_ecg_pilot.py:195–214,228–235; docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/REPORT.md` | verified |
| B5 窗口 | 8 s=1600点；MM val/test步长2 s，训练按场景记录量clip到2–20 s，并非所有训练窗固定2 s；CR默认4 s。中心[288:1312)共1024点=5.12 s仅用于连续组装临床评分/示例，不是Shape训练目标长度。 | `/home/qhh2237/Datasets/MMECG/05B_build_windowed_h5_loso_v2.py:31–40,88–107; /tmp/rem_dual_head_formal/scripts/build_crrvs_wsst_shape_h5.py:274; docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/REPORT.md` | verified |
| B6 特征点 | NeuroKit2 0.2.12（临床运行manifest）；ecg_clean→ecg_peaks→ecg_delineate(method="prominence")；端点数组长度须与R数相等，坐标0≤a<len(x)才保留，否则NaN。不是人工标注。 | `docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/manifest.json; /tmp/rem_dual_head_formal/src/utils/clinical_endpoints.py:15–44` | verified |
| B7 形态目标 | N=窗口数，每窗1代表拍。排序去重R，候选相邻R间隔>10点；优先首个包含中心800（a≤800≤b）的R–R，若无则选区间中点距800最近者；无可用pair则[700,900]。切片ecg[a:b]右端exclusive，linspace(0,b−a−1,200)插值，随后min–max；范围<1e−8则全0；短切片<2用ecg[700:900]。训练目标取窗缓存ECG，不是连续原始mV。 | `/tmp/rem_dual_head_formal/src/data/mmecg_dataset.py:82–97; /tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py:58–66` | verified |
| B8 事件目标 | Q/R/S/T四通道1600点高斯mask，sigma4样本=20 ms；有限位置转int；单尺度，不是历史R-only双尺度。 | `experiments/phase_resampling_20261027/train_stage1.py:61–67` | verified |

### B9 / C1 各折真实H5库存（只读计数）
状态 verified；表内source为实际划分文件；subject集合互不相交已检查。
|dataset|fold|split|windows|subjects|subject IDs|source|状态|
|---|---|---|---|---|---|---|---|
|MMECG|1|train|1665|9|[5, 9, 10, 13, 14, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_01/train.h5`|verified|
|MMECG|1|val|170|1|[2]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_01/val.h5`|verified|
|MMECG|1|test|170|1|[1]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_01/test.h5`|verified|
|MMECG|2|train|1649|9|[1, 9, 10, 13, 14, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_02/train.h5`|verified|
|MMECG|2|val|1020|1|[5]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_02/val.h5`|verified|
|MMECG|2|test|170|1|[2]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_02/test.h5`|verified|
|MMECG|3|train|1648|9|[1, 2, 10, 13, 14, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_03/train.h5`|verified|
|MMECG|3|val|510|1|[9]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_03/val.h5`|verified|
|MMECG|3|test|1020|1|[5]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_03/test.h5`|verified|
|MMECG|4|train|1644|9|[1, 2, 5, 13, 14, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_04/train.h5`|verified|
|MMECG|4|val|255|1|[10]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_04/val.h5`|verified|
|MMECG|4|test|510|1|[9]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_04/test.h5`|verified|
|MMECG|5|train|1644|9|[1, 2, 5, 9, 14, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_05/train.h5`|verified|
|MMECG|5|val|510|1|[13]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_05/val.h5`|verified|
|MMECG|5|test|255|1|[10]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_05/test.h5`|verified|
|MMECG|6|train|1664|9|[1, 2, 5, 9, 10, 16, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_06/train.h5`|verified|
|MMECG|6|val|765|1|[14]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_06/val.h5`|verified|
|MMECG|6|test|510|1|[13]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_06/test.h5`|verified|
|MMECG|7|train|1653|9|[1, 2, 5, 9, 10, 13, 17, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_07/train.h5`|verified|
|MMECG|7|val|595|1|[16]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_07/val.h5`|verified|
|MMECG|7|test|765|1|[14]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_07/test.h5`|verified|
|MMECG|8|train|1627|9|[1, 2, 5, 9, 10, 13, 14, 29, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_08/train.h5`|verified|
|MMECG|8|val|765|1|[17]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_08/val.h5`|verified|
|MMECG|8|test|595|1|[16]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_08/test.h5`|verified|
|MMECG|9|train|1309|9|[1, 2, 5, 9, 10, 13, 14, 16, 30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_09/train.h5`|verified|
|MMECG|9|val|1955|1|[29]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_09/val.h5`|verified|
|MMECG|9|test|765|1|[17]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_09/test.h5`|verified|
|MMECG|10|train|1627|9|[1, 2, 5, 9, 10, 13, 14, 16, 17]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_10/train.h5`|verified|
|MMECG|10|val|1020|1|[30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_10/val.h5`|verified|
|MMECG|10|test|1955|1|[29]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_10/test.h5`|verified|
|MMECG|11|train|1657|9|[2, 5, 9, 10, 13, 14, 16, 17, 29]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_11/train.h5`|verified|
|MMECG|11|val|170|1|[1]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_11/val.h5`|verified|
|MMECG|11|test|1020|1|[30]|`/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_11/test.h5`|verified|
|CR_RVS|1|train|4164|22|[1, 2, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_01/train.h5`|verified|
|CR_RVS|1|val|195|1|[4]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_01/val.h5`|verified|
|CR_RVS|1|test|1246|7|[3, 7, 11, 15, 16, 21, 28]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_01/test.h5`|verified|
|CR_RVS|2|train|4173|22|[2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 25, 27, 28, 29, 30]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_02/train.h5`|verified|
|CR_RVS|2|val|194|1|[6]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_02/val.h5`|verified|
|CR_RVS|2|test|1238|7|[1, 4, 5, 19, 20, 22, 26]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_02/test.h5`|verified|
|CR_RVS|3|train|4155|22|[1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 26, 28]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_03/train.h5`|verified|
|CR_RVS|3|val|196|1|[5]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_03/val.h5`|verified|
|CR_RVS|3|test|1254|7|[2, 18, 23, 24, 27, 29, 30]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_03/test.h5`|verified|
|CR_RVS|4|train|4304|24|[1, 2, 3, 4, 5, 8, 9, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_04/train.h5`|verified|
|CR_RVS|4|val|205|1|[7]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_04/val.h5`|verified|
|CR_RVS|4|test|1096|5|[6, 10, 12, 17, 25]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_04/test.h5`|verified|
|CR_RVS|5|train|4609|25|[1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_05/train.h5`|verified|
|CR_RVS|5|val|225|1|[10]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_05/val.h5`|verified|
|CR_RVS|5|test|771|4|[8, 9, 13, 14]|`/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold/fold_05/test.h5`|verified|

## C. 划分与种子

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| C1 | MMECG 11折strict LOSO；CR_RVS 5折subject-independent；每折train/val/test精确名单见B9。验证来自独立被试，不从同一被试窗口随机切分。 | `/home/qhh2237/Datasets/MMECG/metadata/mmecg_loso_v2_splits.csv; /home/qhh2237/Datasets/CR_RVS/metadata/crrvs_kfold_splits.csv` | verified |
| C2 | phase_pd正式结果基种子42，实际seed=42+physical_fold；MM43–53、CR43–47。不能写所有模型初始化均为42。 | `experiments/phase_resampling_20261027/formal_spec.json; experiments/phase_resampling_20261027/formal_ablation.py:110–123` | verified |
| C2 phase_pd 多seed均值±SD | missing — 本清单未找到phase_pd完整两数据集多基种子正式结果；旧模型多seed不能替代。 | `experiments/phase_resampling_20261027/formal_spec.json` | missing |

### C1 CR_RVS 分组生成规则补充

状态 verified。按每被试场景记录数分组、组内随机打乱并循环分配五test folds；验证从非test且此前未作为val的被试中选记录数最接近中位数者，平手选字典序最小ID。来源：`/home/qhh2237/Datasets/CR_RVS/03B_create_kfold_splits.py:37–75,110–119`。本次实际名单以B9读取的H5为准，不用旧MM `03B_create_loso_splits.py`的不同规则替代v2固定CSV。

## D. 网络结构

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| D1 | 2个Conv2D：50→128→128，k3×3,s1,p1,biasFalse，各接BN/GELU；第一层后MaxPool(2,1)；第二层后Dropout2D0.1。 | `/tmp/rem_dual_head_formal/src/models/continuous/ce_net.py:21–42` | verified |
| D2 | AdaptiveAvgPool2d((1,None))→squeeze频率→linear interpolate至400，align_corners=False；MM71→35→1，CR65→32→1。 | `/tmp/rem_dual_head_formal/src/models/continuous/ce_net.py:32–42` | verified |
| D3 | 5块、128通道、k7、d=1/2/4/8/16，每块Conv-BN-GELU-Dropout0.1-Conv-BN-GELU，block residual x+b(x)，无全局skip；RF=1+2×6×31=373 latent positions；名义50Hz下7.46s（首尾位置跨度7.44s）；非原始200Hz采样数；这是卷积路径在eval下的名义RF，训练BN统计可跨时序/批次聚合。 | `/tmp/rem_dual_head_formal/src/models/continuous/ce_net.py:81–98` | verified |
| D4 | H400→1600 linear：事件路径align_corners=False，相位路径True；上下文H400→200 False。 | `experiments/phase_resampling_20261027/architecture.py:96–113` | verified |
| D5 | 事件Conv1d128→64,k5,p2,biasFalse→BN→GELU→Conv64→4,k5,p2,biasTrue；返回4×1600 logits；下游Sigmoid。 | `/tmp/rem_dual_head_formal/src/models/continuous/dual_head_rem.py:23–47; experiments/phase_resampling_20261027/architecture.py:112–113` | verified |
| D6 有序解码阈值 | R find_peaks height0.5,distance40样本(200ms)；Q候选[R−20,R)；S[R+1,R+21)；T[S+8,min(1600,S+91,nextR))，各argmax；空区间跳过整组。不对Q/S/T另施0.5阈值。 | `/tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py:117–134` | verified |
| D7 | R logits局部softmax±10样本(±50ms，inclusive)，tau=1；局部期望位置；detach。 | `experiments/phase_resampling_20261027/architecture.py:32–37` | verified |
| D8 | t_i=a+(i/199)(b−a−1)，i=0…199；归一化g=2t/1599−1；bilinear grid_sample,border padding,align_corners=True；grid和P stop-gradient。R–R Phase Alignment只输出t_i。 | `experiments/phase_resampling_20261027/architecture.py:10–30` | verified |
| D9 | phase_pd concat Hbeat128+Pbeat4+Hinterp128=260；Conv260→64,k5,p2,biasFalse→BN→GELU→Conv64→1,k5,p2,biasTrue→外部Sigmoid；无额外输入Interpolate。interp128，window_p132，phase128，phase_d256，phase_p132，phase_pd260。 | `experiments/phase_resampling_20261027/architecture.py:76–105; /tmp/rem_dual_head_formal/src/models/continuous/dual_head_rem.py:29–35` | verified |
| D10 | 选距800最近的decoded R及其下一个；soft refine后有限且60≤RR≤400样本才valid；无效时first仍为所选R/若无R则800，duration=decoded RR median/若无差分160；second=first+duration；越界grid border填充；test绝不用GT替换；matched仅诊断标签（两个边界均≤30样本）。 | `experiments/phase_resampling_20261027/architecture.py:39–58,65–73` | verified |
| D11 | MM total1,481,029；CR total1,424,581。模块精确trainable counts见下表，CPU实例化计数，不加载/修改权重。 | `docs/paper/tab_layers.tex; experiments/phase_resampling_20261027/architecture.py:76–83` | verified |

|dataset|stem|TCN|event head|shape head|total|source|状态|
|---|---|---|---|---|---|---|---|
|MMECG|205568|1149440|42372|83649|1481029|`experiments/phase_resampling_20261027/architecture.py`|verified|
|CR_RVS|149120|1149440|42372|83649|1424581|`experiments/phase_resampling_20261027/architecture.py`|verified|

### D10 正式fallback / unmatched（subject macro，比率0–1）

来源：`experiments/phase_resampling_20261027/formal_ablation_20261027/supplement_S1.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | arm | n_subjects | fallback_rate | valid_unmatched_rate | r_coverage | t_coverage |
|---|---|---|---|---|---|---|
| mmecg | A0 | 11 | NA | NA | 0.8353503263847412 | 0.8468376077986871 |
| mmecg | A1 | 11 | NA | NA | 0.8379685670715393 | 0.8496619022663375 |
| mmecg | A2 | 11 | 0.08671460679133315 | 0.5149763989917442 | 0.8165833617536918 | 0.8287178987622074 |
| mmecg | A3 | 11 | 0.20277584024387094 | 0.49779600904920596 | 0.6768404261955926 | 0.6885660648643236 |
| mmecg | A4 | 11 | 0.06152555884013684 | 0.5176492731479944 | 0.848656893876053 | 0.8596605804811296 |
| mmecg | S1 | 11 | NA | NA | NA | NA |
| mmecg | S2 | 11 | NA | NA | 0.8827950796117059 | 0.8942243632983223 |
| crrvs | A0 | 30 | NA | NA | 0.847147989474269 | 0.8710063213609132 |
| crrvs | A1 | 30 | NA | NA | 0.848445714751141 | 0.8737178943143256 |
| crrvs | A2 | 30 | 0.08350939316236264 | 0.4915090672773729 | 0.8414846729271375 | 0.8683651881475247 |
| crrvs | A3 | 30 | 0.16408707617427687 | 0.4644482184883149 | 0.7593562661911366 | 0.7879468356979114 |
| crrvs | A4 | 30 | 0.0986356661307605 | 0.4904075874212857 | 0.8243532947965834 | 0.8516082608467663 |
| crrvs | S1 | 30 | NA | NA | NA | NA |
| crrvs | S2 | 30 | NA | NA | 0.8867011412721636 | 0.905242023889208 |

## E. 训练

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| E1 | AdamW lr1e−4,weight_decay1e−4,betas默认(0.9,0.999),eps默认1e−8；无scheduler；最多100epochs,batch16,patience15；MM drop_lastFalse，CRTrue。 | `experiments/phase_resampling_20261027/formal_ablation.py:110–123,132–166; experiments/phase_resampling_20261027/train_stage1.py:120–141` | verified |
| E2 | L=Lshape+0.3Lanchor；Lshape=L1+0.5(1−PCC)；BCEWithLogits默认mean，无pos_weight/类别权重。 | `/tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py:178–187` | verified |
| E3 | alpha=clip(((epoch−1)/100−0.15)/0.30,0,1)：epochs1–16 alpha0；17–45线性混合；46起alpha1。训练无效pred pair用GT bounds；evaluation始终pred，包括fallback。 | `experiments/phase_resampling_20261027/architecture.py:60–73; experiments/phase_resampling_20261027/formal_ablation.py:139–143` | verified |
| E4 | 所核实phase_pd loop无数据增强、autocast/GradScaler、梯度裁剪调用；FP32常规backward/optimizer.step。 | `experiments/phase_resampling_20261027/formal_ablation.py:121–146; experiments/phase_resampling_20261027/train_stage1.py:132–156` | verified |
| E5 多任务/五baseline | validation harmonic(shape arithmeticPCC, independent event macroF1@100ms)，并列按shapePCC，再−RMSE；不是按formal Fisher testPCC选。 | `/tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_mmecg.py:83–86; experiments/phase_resampling_20261027/formal_ablation.py:62–67` | verified |
| E5 S1/S2 | S1按val shapePCC,−RMSE,R2；S2按val independent macroF1@100ms,−anchorBCE；所有方法test不参与选择。 | `experiments/phase_resampling_20261027/formal_ablation.py:62–67; experiments/phase_resampling_20261027/formal_spec.json` | verified |
| E6 历史训练硬件/软件环境 | missing — 该文件是复杂度测量环境，不足以证明所有历史训练均用同一GPU/Python/Torch/CUDA。下文只列测量环境，不冒充训练环境。 | `docs/paper/sections/complexity_20261030/environment.json` | missing |
| E6 已核实测量环境 | Python3.12.13 / Torch2.6.0+cu124 / CUDA12.4 / NVIDIA RTX4080SUPER；这是复杂度运行环境。 | `docs/paper/sections/complexity_20261030/environment.json` | verified |

### E6 phase_pd 各折历史epoch时间与所选epoch
每epoch时间含该runner的验证等开销，不称纯GPU训练时间；复用运行按selected.json追溯。
|dataset|fold|selected epoch|logged epochs|median seconds/epoch|sum logged seconds|source|状态|
|---|---|---|---|---|---|---|---|
|mmecg|1|67|82|9.236|753.438|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/stage1/runs/mmecg/fold_01/phase_pd/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_01/A4/selected.json`|verified|
|mmecg|2|60|75|13.929|1054.546|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/stage1/runs/mmecg/fold_02/phase_pd/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_02/A4/selected.json`|verified|
|mmecg|3|71|86|10.758|921.218|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_03/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_03/A4/selected.json`|verified|
|mmecg|4|52|67|9.468|629.630|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_04/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_04/A4/selected.json`|verified|
|mmecg|5|88|100|10.670|1062.607|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_05/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_05/A4/selected.json`|verified|
|mmecg|6|67|82|12.062|980.315|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_06/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_06/A4/selected.json`|verified|
|mmecg|7|98|100|11.201|1113.443|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_07/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_07/A4/selected.json`|verified|
|mmecg|8|70|85|11.894|1003.868|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_08/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_08/A4/selected.json`|verified|
|mmecg|9|56|71|16.445|1142.257|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_09/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_09/A4/selected.json`|verified|
|mmecg|10|63|78|13.232|1022.897|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_10/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_10/A4/selected.json`|verified|
|mmecg|11|49|64|9.163|584.494|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_11/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/mmecg/fold_11/A4/selected.json`|verified|
|crrvs|1|73|88|6.160|535.303|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/stage1/runs/crrvs/fold_01/phase_pd/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_01/A4/selected.json`|verified|
|crrvs|2|54|69|6.127|418.301|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_02/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_02/A4/selected.json`|verified|
|crrvs|3|99|100|6.100|604.676|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_03/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_03/A4/selected.json`|verified|
|crrvs|4|86|100|6.371|631.567|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_04/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_04/A4/selected.json`|verified|
|crrvs|5|27|42|6.750|278.480|`/home/qhh2237/Projects/BeatAware-Radar2ECGNet/experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_05/A4/history.json`; `experiments/phase_resampling_20261027/formal_ablation_20261027/runs/crrvs/fold_05/A4/selected.json`|verified|


### E6 已记录训练时长合计

- **verified mmecg**：890已记录epochs，所有折日志seconds合计10268.713 s（2.8524 h）；epoch时间中位数11.145 s。来源为上表逐折history.json。包括runner记录的训练/验证开销，不含未记录的数据预处理、标签缓存或停机时间，不是端到端项目耗时。
- **verified crrvs**：399已记录epochs，所有折日志seconds合计2468.328 s（0.6856 h）；epoch时间中位数6.157 s。来源为上表逐折history.json。包括runner记录的训练/验证开销，不含未记录的数据预处理、标签缓存或停机时间，不是端到端项目耗时。

## F. 五个 adapted baselines

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| Chen-TCN | 输入time；spatio-temporal causal TCN；原论文μ-law256类自回归，改为共享Shape-200/anchor readouts；latent channels=64；原论文bib候选`chenmmwave`（具体引用归属见下方missing）。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:10–67; /tmp/rem_dual_head_formal/src/models/baselines/continuous/mmecg_tcn.py` | verified |
| Chen-TCN 原论文/官方代码精确对应 | missing — 本地实现可核实；具体上游URL、版本/commit及论文与该本地别名的一一对应尚未全部核验，不将bib候选视作已核实论文来源。 | `docs/paper/sections/references.bib; /tmp/rem_dual_head_formal/src/models/baselines/continuous/mmecg_tcn.py` | missing |
| radarODE-MTL | 输入TF；官方DCNResNet/squeeze representation经本地适配；替换原任务readout，不声称保留原论文全部多任务优化；latent channels=1024；原论文bib候选`radarodemt`（具体引用归属见下方missing）。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:10–67; /tmp/rem_dual_head_formal/src/models/baselines/continuous/radarode_mtl.py` | verified |
| radarODE-MTL 原论文/官方代码精确对应 | missing — 本地实现可核实；具体上游URL、版本/commit及论文与该本地别名的一一对应尚未全部核验，不将bib候选视作已核实论文来源。 | `docs/paper/sections/references.bib; /tmp/rem_dual_head_formal/src/models/baselines/continuous/radarode_mtl.py` | missing |
| Conformer | 输入time；轻量conv embed/Conformer representation；不是原论文完整cross-modal AE预训练协议；latent channels=64；原论文bib候选`croos-modalAutoencoder`（具体引用归属见下方missing）。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:10–67; /tmp/rem_dual_head_formal/src/models/baselines/continuous/cross_modal_conformer.py` | verified |
| Conformer 原论文/官方代码精确对应 | missing — 本地实现可核实；具体上游URL、版本/commit及论文与该本地别名的一一对应尚未全部核验，不将bib候选视作已核实论文来源。 | `docs/paper/sections/references.bib; /tmp/rem_dual_head_formal/src/models/baselines/continuous/cross_modal_conformer.py` | missing |
| WaveGRU-Net | 输入time；CNN/BiGRU representation，改为同一双头；latent channels=128；原论文bib候选`xu2025wavegrunet`（具体引用归属见下方missing）。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:10–67; /tmp/rem_dual_head_formal/src/models/baselines/continuous/wavegru_net.py` | verified |
| WaveGRU-Net 原论文/官方代码精确对应 | missing — 本地实现可核实；具体上游URL、版本/commit及论文与该本地别名的一一对应尚未全部核验，不将bib候选视作已核实论文来源。 | `docs/paper/sections/references.bib; /tmp/rem_dual_head_formal/src/models/baselines/continuous/wavegru_net.py` | missing |
| Attention-DCGAN | 输入time；attention CNN生成器representation；改为同一双头；不可称原GAN训练损失复现；latent channels=64；原论文bib候选`ecglikegan`（具体引用归属见下方missing）。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:10–67; /tmp/rem_dual_head_formal/src/models/baselines/continuous/attention_dcgan.py` | verified |
| Attention-DCGAN 原论文/官方代码精确对应 | missing — 本地实现可核实；具体上游URL、版本/commit及论文与该本地别名的一一对应尚未全部核验，不将bib候选视作已核实论文来源。 | `docs/paper/sections/references.bib; /tmp/rem_dual_head_formal/src/models/baselines/continuous/attention_dcgan.py` | missing |
| F 双头训练 | 全部随机初始化的adapted训练，common shape sigmoid200 + event logits4×1600，Conv k5/BN/GELU/Conv k5；共享现有shape+0.3BCE损失与validation harmonic规则。原生continuous/softmax/GAN输出不是当前主表输出。 | `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py:19–67; /tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_mmecg.py:83–110` | verified |
| F 已核实训练流程差异 | baseline MM runner: AdamW lr1e−4,wd1e−4,batch16,max100,patience15,seed42+fold，drop_last=True；CR runner复用该train并替换CR数据/模型，所以同样True；phase_pd MM drop_last=False。共同loss/selector不表示全部训练细节完全一致。 | `/tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_mmecg.py:28–29,95–129; /tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_crrvs.py:79–80; experiments/phase_resampling_20261027/formal_ablation.py:121–123` | verified |
| F 逐方法超参搜索记录 | missing — 未找到完整逐方法search trial/预算/选参记录；不能声称系统调参或确定没有任何调参。 | `/tmp/rem_dual_head_formal/experiments/specs/q1_dual_endpoint_baselines_mmecg.json; /tmp/rem_dual_head_formal/experiments/specs/q1_dual_endpoint_baselines_crrvs.json` | missing |

### F 引用身份核对

状态 verified：本地baseline registry将cross_modal_conformer对应Liu等2024 Cross-Modal Autoencoder（`/tmp/rem_dual_head_formal/src/models/baselines/continuous/__init__.py:34–46`），对应本地bib key为`croos-modalAutoencoder`（拼写保留原文件）；`conformer` key实际上是Gulati等语音Conformer，不能作为该雷达baseline的原论文引用。来源：`docs/paper/sections/references.bib:469–478,674–684`。其余registry论文名称映射可核验，但官方代码版本/原论文逐层等价性仍missing。

### F 补充输入适配

状态 verified：radarODE-MTL在CR_RVS把log-STFT `[B,1,65,93]` 用 `F.interpolate(size=(71,240),mode="bilinear",align_corners=False)` 调整，另四baseline仍使用时域RCG。来源：`/tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_crrvs.py:31–37`。五baseline保留representation但输出头/监督已适配，不等同原论文原生性能或原生连续解码。radarODE官方源码本地入口为`/home/qhh2237/Projects/radarODE-MTL`，来源：`/tmp/rem_dual_head_formal/src/models/baselines/continuous/radarode_mtl.py:39`；具体上游commit与许可证仍missing。

## G. 评估方法

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| G1 | 每窗Shape200：PCC=中心化向量内积/范数乘积（分母floor1e−8）；RMSE=sqrt(mean((pred−ref)^2))。主表：subject内Fisher-z均PCC（clip±(1−1e−7)）→tanh，再subject等权均值；RMSE窗均→subject等权均；SD为population(ddof0)。 | `experiments/phase_resampling_20261027/formal_ablation.py:184–194; /tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py:98–111; docs/paper/sections/results_verified_20261101/REPORT.md` | verified |
| G2 | R匹配order-preserving最大匹配数、同数量最小总误差，±30样本=150ms；端点计入需有限且命中容差。coverage=matched/eligible；R-MDR=1−matchedGT/eligibleGT；R-FDR=1−matchedPred/predCount；分母0不擅填0。 | `/tmp/rem_dual_head_formal/src/utils/clinical_endpoints.py:47–76; /tmp/rem_dual_head_formal/scripts/evaluate_formal_anchor_intervals.py; docs/paper/sections/results_verified_20261101/REPORT.md` | verified |
| G2 Cov合并 | 每subject合计Q/R/S/T matched ÷ eligible，再subject等权平均；不是4种coverage直接平均，也不是全部beats pooling。 | `docs/paper/sections/ablation_merged_20261101/spec.json` | verified |
| G3 | Wilcoxon双侧signed-rank,zero_method=wilcox,method=auto；architecture每dataset四对比Holm；factorial每dataset四效应单独Holm；S1 posthoc两dataset为独立Holm2；bootstrap paired subject10000次，seed42，percentile2.5/97.5。 | `experiments/phase_resampling_20261027/formal_spec.json; docs/paper/sections/ablation_merged_20261101/spec.json` | verified |
| G4 | 每8s窗口对相邻预测R逐拍phase_pd解码，200点按[a,b−1]逆映射，invalid/uncovered固定0；每窗仅中心1024评分；供给predicted R到NK prominence；同参考delineator；两端均匹配。不是原生连续回归。 | `docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/REPORT.md; docs/paper/sections/figures/plot_phase_pd_bland_altman.py` | verified |
| G5 | Bland & Altman2007 J Biopharm Stat17:571–582，DOI10.1080/10543400701329422；重复测量changing true value，一元不等样本ANOVA估计within+between差值方差；LoA=bias±1.96sqrt(sum)；subject bootstrap10000。只两端匹配心拍；参考自动端点，不是临床人工真值。 | `docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/REPORT.md` | verified |
| G6 场景线性混合模型公式 | missing — 未找到phase_pd对应已执行模型/设计矩阵/随机效应及拟合输出；不构造公式或p值。 | `scripts/figures/plot_scenario_results.py:1–400` | missing |

### G4 临床baseline可比性限制

状态 verified。五baseline补算临床结果已完成，来源`docs/paper/sections/figures/source_data/baseline_clinical_20261101/manifest.json`和`REPORT.md`。每个预测R–R用各方法原生shape head在独立逐拍居中8s雷达context推理，禁止复制代表拍；REM phase_pd则重用父窗口特征逐拍相位采样。参考/检测/匹配/去重/评分相同，但输入雷达上下文支持和总推理计算量不相同，不能声称完整生成协议/输入预算完全相同。临床均值必须连同coverage/可评估被试数引用。

## H. Results / Abstract 的结果可引用性

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| H1 主表 | 完整108个单元格见下方；REM是phase_pd A4，五baseline为adapted。所有统计单位/覆盖率按各列明确，QRS/QT为连续组装端点，RR为直接event间期。 | `docs/paper/sections/results_verified_20261101/baseline_compare_verified.csv; docs/paper/sections/results_verified_20261101/REPORT.md` | verified |
| H1 REM vs5 baselines Wilcoxon-Holm | missing — 未发现phase_pd vs五baseline正式预定义统计输出；本次导出不新增检验。不能用旧模型p或消融p代替。 | `docs/paper/sections/results_verified_20261101/per_subject_verified.csv` | missing |
| H2 | 架构ΔPCC、CI、raw/adjusted p见table1；析因四效应见table2；非容量匹配因果实验。 | `experiments/phase_resampling_20261027/formal_ablation_20261027/table1_modules.csv; experiments/phase_resampling_20261027/formal_ablation_20261027/table2_factorial.csv; experiments/phase_resampling_20261027/formal_spec.json` | verified |
| H3 | 多任务描述结果见table3；S1与A4事后PCC检验见merged source；S2事件显著性没有注册结果。 | `docs/paper/sections/ablation_merged_20261101/shape_only_posthoc_statistics.csv; experiments/phase_resampling_20261027/formal_ablation_20261027/table3_multitask.csv` | verified |
| H3 event-only vs联合显著性 | missing — 该表为描述性，不存在可引用事件任务显著性检验。 | `experiments/phase_resampling_20261027/formal_ablation_20261027/table3_multitask.csv` | missing |
| H4 phase_pd场景median[IQR]/LMM场景p | missing — 当前场景图来自原双头模型；未找到phase_pd同口径subject×scenario median[IQR]及LMM输出，不迁移旧值。 | `docs/paper/sections/figures/source_data/fig_scenario_wise_results.csv` | missing |
| H5 phase_pd定位CDF中位误差 | missing — 该图REM来自旧dual_head_rem而非phase_pd；旧图40/35等不能作为新模型结果；现主表phase_pd事件MAE可引用但不是CDF median。 | `docs/paper/sections/figures/source_data/localization_cdf_main_20261029/manifest.json` | missing |
| H6 | 临床Bland–Altman与roundtrip见下方CSV摘录；主表MAE为subject-macro，BA为paired pooled observations，不互换。 | `docs/paper/sections/figures/bland_altman_stats_v2.csv; docs/paper/sections/figures/roundtrip_check.csv` | verified |
| H7 | HR均由predicted R计算，中心5.12秒，60/meanRR；不可用少数有限窗口替代全部失败统计。MAE/coverage见BA原stats；|误差|>5的R计数关联见下表，非因果归因。 | `docs/paper/sections/figures/bland_altman_stats.csv; docs/paper/sections/figures/source_data/phase_pd_bland_altman_v2_20261029/hr_attribution_summary.csv` | verified |
| H8 | oracle0、jitter5/10/20/40/80/128ms，5repeats种子0–4；paired subject ΔvsA0，10k bootstrap；预测star横坐标仅matched pair、纵坐标所有窗口，二者分母不同必须披露。 | `docs/paper/sections/figures/source_data/anchor_robustness_delta_20261028/REPORT.md` | verified |
| H9 | 复杂度12行见下表：batch1,FP32,50 warmup/200同步GPU测量，单次forward含ordered decode/phase sampling，排除TF preprocessing；fvcore按2FLOP/MAC+自定义算子处理。不是全记录逐拍生成代价。 | `docs/paper/sections/complexity_20261030/REPORT.md` | verified |
| H10 GDN0022 | QT eligible828,matched167,coverage20.169%；bias−95.898ms,MAE101.946ms；signed median−100ms,IQR[−115,−85]；97.01%为负。三场景均存在，不能断定是被试病理或单纯模型原因。 | `docs/paper/sections/figures/source_data/phase_pd_bland_altman_v2_20261029/worst_qt_subject.json; docs/paper/sections/figures/source_data/phase_pd_bland_altman_v2_20261029/REPORT.md` | verified |
| H11 phase_pd 示例图ID/PCC | missing — 当前v4/advantage示例来自旧模型；下方保留旧面板ID/PCC审计供辨认，但不得以phase_pd名义引用。 | `docs/paper/sections/figures/source_data/examples_v4_apnea_revision_20261030/manifest.json` | missing |

## I. 可复现性

| 项目 | 数值 / 经核实事实 | 来源 | 状态 |
|---|---|---|---|
| I1 仓库 | git remote origin = git@github.com:XiaoJin0501/BeatAware-Radar2ECGNet.git；对应URL https://github.com/XiaoJin0501/BeatAware-Radar2ECGNet 。此处只核验remote，不证明匿名公众可访问。 | `.git/config` | verified |
| I1 对外公开/模型权重许可 | missing — 未核实仓库公开可见性、模型下载入口及许可证；不能写代码/权重已公开或开放源代码许可。 | `仓库根目录（未找到LICENSE*）` | missing |
| I2 数据获取 | CR原论文figshare .mat；本地MM/CR元数据路径见A，原始数据不得随repo发布。 | `/home/qhh2237/Datasets/CR_RVS/CR_RVS Dataset.pdf, p.7; /home/qhh2237/Datasets/MMECG/metadata/mmecg_metadata.csv` | verified |
| I2 完整可复现分发包 | missing — 代码依赖/tmp/rem_dual_head_formal和绝对本地Dataset路径；需保留/发布受许可的冻结代码、配置、hash和环境才可称开箱复现。 | `experiments/phase_resampling_20261027/frozen_bridge.py:17–44` | missing |

### H1 全方法主表原始数值（PCC为0–1，论文百分比×100）

来源：`docs/paper/sections/results_verified_20261101/baseline_compare_verified.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | model | metric | mean | std_population | n_subjects | n_subjects_total | coverage_subject_mean | source |
|---|---|---|---|---|---|---|---|---|
| mmecg | mmecg_tcn | pcc | 0.8653790788217024 | 0.056782565522714504 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | mmecg_tcn | rmse | 0.1370436122471636 | 0.052792716662319245 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | mmecg_tcn | q | 42.20520317078789 | 15.268613730931621 | 11 | 11 | 0.8164225333788515 | frozen structured event-head matching, seed42 |
| mmecg | mmecg_tcn | r | 38.97522723247718 | 15.077770099061599 | 11 | 11 | 0.8164225333788515 | frozen structured event-head matching, seed42 |
| mmecg | mmecg_tcn | s | 41.27376075094946 | 18.487858566337383 | 11 | 11 | 0.8164225333788515 | frozen structured event-head matching, seed42 |
| mmecg | mmecg_tcn | t | 39.38547790129139 | 17.58618482256086 | 11 | 11 | 0.826771088885653 | frozen structured event-head matching, seed42 |
| mmecg | mmecg_tcn | rr | 19.940253488491223 | 2.877453046966148 | 11 | 11 | 0.7329204213411387 | frozen structured event-head matching, seed42 |
| mmecg | mmecg_tcn | qrs | 7.747805939519032 | 2.847466372942212 | 11 | 11 | 0.8149696105833698 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | mmecg_tcn | qt | 25.665889155543496 | 6.825651952612144 | 11 | 11 | 0.7399624766166117 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | radarode_mtl | pcc | 0.8844061927361921 | 0.06386431140932934 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | radarode_mtl | rmse | 0.10875790972601283 | 0.03614416103077898 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | radarode_mtl | q | 46.00925744213946 | 12.861480614769675 | 11 | 11 | 0.825925748260229 | frozen structured event-head matching, seed42 |
| mmecg | radarode_mtl | r | 43.24359102166541 | 13.203447576586868 | 11 | 11 | 0.825925748260229 | frozen structured event-head matching, seed42 |
| mmecg | radarode_mtl | s | 43.88572217138931 | 15.455654789593828 | 11 | 11 | 0.825925748260229 | frozen structured event-head matching, seed42 |
| mmecg | radarode_mtl | t | 42.92219580348922 | 12.791443467605799 | 11 | 11 | 0.8362834559993789 | frozen structured event-head matching, seed42 |
| mmecg | radarode_mtl | rr | 23.294473704632647 | 3.3968856800038885 | 11 | 11 | 0.739264127851595 | frozen structured event-head matching, seed42 |
| mmecg | radarode_mtl | qrs | 7.262538436729631 | 2.5340688822459394 | 11 | 11 | 0.8427453050188528 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | radarode_mtl | qt | 24.699744561247943 | 6.461928377309445 | 11 | 11 | 0.7688651792628902 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | cross_modal_conformer | pcc | 0.7278525341640819 | 0.13263664887538992 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | cross_modal_conformer | rmse | 0.15499459477988156 | 0.04355613058738681 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | cross_modal_conformer | q | 46.48693271474941 | 13.264534410110516 | 10 | 11 | 0.41404389478162157 | frozen structured event-head matching, seed42 |
| mmecg | cross_modal_conformer | r | 44.83274182543067 | 12.56893580047375 | 10 | 11 | 0.41404389478162157 | frozen structured event-head matching, seed42 |
| mmecg | cross_modal_conformer | s | 45.85674157896406 | 14.401080550572884 | 10 | 11 | 0.41404389478162157 | frozen structured event-head matching, seed42 |
| mmecg | cross_modal_conformer | t | 43.28580443944146 | 14.011631580693852 | 10 | 11 | 0.4219965491749002 | frozen structured event-head matching, seed42 |
| mmecg | cross_modal_conformer | rr | 20.98281401717208 | 4.698437553052975 | 10 | 11 | 0.2606585590353778 | frozen structured event-head matching, seed42 |
| mmecg | cross_modal_conformer | qrs | 12.011197072394356 | 2.6371472699684593 | 10 | 11 | 0.40610600780490674 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | cross_modal_conformer | qt | 47.69788149000562 | 3.956004609320395 | 10 | 11 | 0.29060981327800284 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | wavegru_net | pcc | 0.7037481394681063 | 0.13070375907520013 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | wavegru_net | rmse | 0.15526746890761636 | 0.0521738154960697 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | wavegru_net | q | 52.62270199363014 | 12.274914204236772 | 11 | 11 | 0.6366412858583895 | frozen structured event-head matching, seed42 |
| mmecg | wavegru_net | r | 51.69310203108309 | 11.30567178859218 | 11 | 11 | 0.6366412858583895 | frozen structured event-head matching, seed42 |
| mmecg | wavegru_net | s | 52.544150400206306 | 12.625571795123188 | 11 | 11 | 0.6366412858583895 | frozen structured event-head matching, seed42 |
| mmecg | wavegru_net | t | 67.56962126431644 | 15.350832144591351 | 11 | 11 | 0.6414597130303948 | frozen structured event-head matching, seed42 |
| mmecg | wavegru_net | rr | 49.229295325849314 | 11.21029083500212 | 11 | 11 | 0.3737267237221371 | frozen structured event-head matching, seed42 |
| mmecg | wavegru_net | qrs | 11.479744785863698 | 1.9059686852323043 | 11 | 11 | 0.628342382429831 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | wavegru_net | qt | 52.303093812105175 | 3.608859613222134 | 11 | 11 | 0.45436902754838043 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | attention_dcgan | pcc | 0.689964698119597 | 0.1356621412129857 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | attention_dcgan | rmse | 0.15111720291050998 | 0.04000206792138219 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | attention_dcgan | q | 46.54215350972711 | 14.424669398337935 | 11 | 11 | 0.4172542670760518 | frozen structured event-head matching, seed42 |
| mmecg | attention_dcgan | r | 44.8043043481264 | 12.821739164463061 | 11 | 11 | 0.4172542670760518 | frozen structured event-head matching, seed42 |
| mmecg | attention_dcgan | s | 45.44986344606679 | 15.368474009228287 | 11 | 11 | 0.4172542670760518 | frozen structured event-head matching, seed42 |
| mmecg | attention_dcgan | t | 79.14206475521884 | 10.04025672799798 | 11 | 11 | 0.4212298409970307 | frozen structured event-head matching, seed42 |
| mmecg | attention_dcgan | rr | 36.25696921969449 | 11.024987268956986 | 11 | 11 | 0.21219372360488195 | frozen structured event-head matching, seed42 |
| mmecg | attention_dcgan | qrs | 14.451183929130037 | 3.2844306430852632 | 11 | 11 | 0.3935765586885939 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | attention_dcgan | qt | 51.65885865415385 | 5.926139170227097 | 11 | 11 | 0.24587160740974684 | native baseline independently decoded per-predicted-beat clinical endpoints |
| mmecg | rem_net | pcc | 0.9139722368933938 | 0.034921713024241526 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | rem_net | rmse | 0.11079175634817644 | 0.031107630687365675 | 11 | 11 | NA | common frozen Shape200 targets / formal test predictions |
| mmecg | rem_net | q | 42.597587160190926 | 10.88464043772865 | 11 | 11 | 0.848656893876053 | frozen structured event-head matching, seed42 |
| mmecg | rem_net | r | 39.0179782511286 | 11.11768411339619 | 11 | 11 | 0.848656893876053 | frozen structured event-head matching, seed42 |
| mmecg | rem_net | s | 43.1182889592617 | 14.641935140766845 | 11 | 11 | 0.848656893876053 | frozen structured event-head matching, seed42 |
| mmecg | rem_net | t | 39.846587750866725 | 13.910235625259602 | 11 | 11 | 0.8596605804811296 | frozen structured event-head matching, seed42 |
| mmecg | rem_net | rr | 17.611883802325615 | 3.197717639884752 | 11 | 11 | 0.786610041679172 | frozen structured event-head matching, seed42 |
| mmecg | rem_net | qrs | 6.8871144587805055 | 2.2097992709707612 | 11 | 11 | 0.8729996624216274 | phase_pd continuous onset-offset pairs; roundtrip PASS |
| mmecg | rem_net | qt | 17.53239796796987 | 6.139088667788741 | 11 | 11 | 0.8278327557907729 | phase_pd continuous onset-offset pairs; roundtrip PASS |
| crrvs | mmecg_tcn | pcc | 0.8816836436589559 | 0.07505609141115602 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | mmecg_tcn | rmse | 0.12198751966158548 | 0.053966943726847225 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | mmecg_tcn | q | 28.717647984097727 | 14.837911793743247 | 30 | 30 | 0.7576341955888837 | frozen structured event-head matching, seed42 |
| crrvs | mmecg_tcn | r | 21.529442428194805 | 10.442996433827748 | 30 | 30 | 0.7576341955888837 | frozen structured event-head matching, seed42 |
| crrvs | mmecg_tcn | s | 26.354286084312548 | 12.439717825199136 | 30 | 30 | 0.7576341955888837 | frozen structured event-head matching, seed42 |
| crrvs | mmecg_tcn | t | 26.589106625263636 | 15.288760223777931 | 30 | 30 | 0.7747975870056 | frozen structured event-head matching, seed42 |
| crrvs | mmecg_tcn | rr | 19.505467210858892 | 8.336530377894196 | 30 | 30 | 0.6612067389672216 | frozen structured event-head matching, seed42 |
| crrvs | mmecg_tcn | qrs | 11.952743818180265 | 5.2509012441665766 | 30 | 30 | 0.7699609432649605 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | mmecg_tcn | qt | 30.69127305444816 | 10.971011433275951 | 30 | 30 | 0.6910742352778626 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | radarode_mtl | pcc | 0.8916431844234467 | 0.08376230699107208 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | radarode_mtl | rmse | 0.10524443791558345 | 0.05576207812770371 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | radarode_mtl | q | 29.710816846119606 | 13.729813529013253 | 30 | 30 | 0.8967441380767089 | frozen structured event-head matching, seed42 |
| crrvs | radarode_mtl | r | 23.002199623867387 | 8.45003419909885 | 30 | 30 | 0.8967441380767089 | frozen structured event-head matching, seed42 |
| crrvs | radarode_mtl | s | 25.64066287822486 | 9.450994028694994 | 30 | 30 | 0.8967441380767089 | frozen structured event-head matching, seed42 |
| crrvs | radarode_mtl | t | 26.652149811273137 | 13.580674877540462 | 30 | 30 | 0.9133589870952117 | frozen structured event-head matching, seed42 |
| crrvs | radarode_mtl | rr | 19.125079779085027 | 8.227041565013819 | 30 | 30 | 0.8528057894347031 | frozen structured event-head matching, seed42 |
| crrvs | radarode_mtl | qrs | 10.776079477889063 | 5.140452319399967 | 30 | 30 | 0.9177709149237769 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | radarode_mtl | qt | 23.62512880632829 | 14.57435098206256 | 30 | 30 | 0.8815041886009016 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | cross_modal_conformer | pcc | 0.778571355342865 | 0.11527187726189637 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | cross_modal_conformer | rmse | 0.12357942784825961 | 0.05659137989557643 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | cross_modal_conformer | q | 25.920942859381668 | 15.35863775142015 | 30 | 30 | 0.7953337962300886 | frozen structured event-head matching, seed42 |
| crrvs | cross_modal_conformer | r | 17.396188240533437 | 12.090272520472274 | 30 | 30 | 0.7953337962300886 | frozen structured event-head matching, seed42 |
| crrvs | cross_modal_conformer | s | 24.379148934354145 | 14.571794357552617 | 30 | 30 | 0.7953337962300886 | frozen structured event-head matching, seed42 |
| crrvs | cross_modal_conformer | t | 24.604924758297088 | 12.54515113112142 | 30 | 30 | 0.8128676230774474 | frozen structured event-head matching, seed42 |
| crrvs | cross_modal_conformer | rr | 13.108972912668888 | 12.056065195481871 | 30 | 30 | 0.7212899125635808 | frozen structured event-head matching, seed42 |
| crrvs | cross_modal_conformer | qrs | 11.642096190172605 | 5.3932938615403945 | 30 | 30 | 0.8129072950207342 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | cross_modal_conformer | qt | 62.7190043907717 | 9.80089224855267 | 30 | 30 | 0.7132038429496781 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | wavegru_net | pcc | 0.7753328422705332 | 0.12383973048811128 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | wavegru_net | rmse | 0.12623089974125226 | 0.059277749999749094 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | wavegru_net | q | 29.324731981720404 | 15.842475419820111 | 30 | 30 | 0.7746166150465243 | frozen structured event-head matching, seed42 |
| crrvs | wavegru_net | r | 21.128476421688 | 10.646084310865294 | 30 | 30 | 0.7746166150465243 | frozen structured event-head matching, seed42 |
| crrvs | wavegru_net | s | 26.870216647026314 | 12.640936753386418 | 30 | 30 | 0.7746166150465243 | frozen structured event-head matching, seed42 |
| crrvs | wavegru_net | t | 26.202113112264893 | 15.614628208870316 | 30 | 30 | 0.7976506479190992 | frozen structured event-head matching, seed42 |
| crrvs | wavegru_net | rr | 17.34799839454077 | 10.61037490087786 | 30 | 30 | 0.6781995244923108 | frozen structured event-head matching, seed42 |
| crrvs | wavegru_net | qrs | 10.895952124597972 | 4.742479222502704 | 30 | 30 | 0.8002580743054625 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | wavegru_net | qt | 58.66795000429212 | 10.99674511768186 | 30 | 30 | 0.6990070071479744 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | attention_dcgan | pcc | 0.7697629580895106 | 0.12286930057662664 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | attention_dcgan | rmse | 0.1242876158406337 | 0.056725021183452226 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | attention_dcgan | q | 27.865320065751515 | 14.373082439256128 | 30 | 30 | 0.8349620059574802 | frozen structured event-head matching, seed42 |
| crrvs | attention_dcgan | r | 19.515939969286478 | 9.411438847923382 | 30 | 30 | 0.8349620059574802 | frozen structured event-head matching, seed42 |
| crrvs | attention_dcgan | s | 25.149612908606354 | 13.023402770559132 | 30 | 30 | 0.8349620059574802 | frozen structured event-head matching, seed42 |
| crrvs | attention_dcgan | t | 26.585434313593044 | 14.259567083504894 | 30 | 30 | 0.8484099621015454 | frozen structured event-head matching, seed42 |
| crrvs | attention_dcgan | rr | 16.1725512706844 | 9.423355461870168 | 30 | 30 | 0.7392252478376474 | frozen structured event-head matching, seed42 |
| crrvs | attention_dcgan | qrs | 11.597589252924637 | 6.205920485190449 | 30 | 30 | 0.8450675506185632 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | attention_dcgan | qt | 63.4270223108144 | 7.26729122919201 | 30 | 30 | 0.7239679630661532 | native baseline independently decoded per-predicted-beat clinical endpoints |
| crrvs | rem_net | pcc | 0.9044548749923706 | 0.07594037670813109 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | rem_net | rmse | 0.10080457963049412 | 0.051297203237519344 | 30 | 30 | NA | common frozen Shape200 targets / formal test predictions |
| crrvs | rem_net | q | 28.06831056189062 | 15.131789939181994 | 30 | 30 | 0.8243532947965834 | frozen structured event-head matching, seed42 |
| crrvs | rem_net | r | 21.247763779247762 | 9.982927127583174 | 30 | 30 | 0.8243532947965834 | frozen structured event-head matching, seed42 |
| crrvs | rem_net | s | 26.38175467148474 | 9.978544744016316 | 30 | 30 | 0.8243532947965834 | frozen structured event-head matching, seed42 |
| crrvs | rem_net | t | 23.954586095641492 | 13.043055301566747 | 30 | 30 | 0.8516082608467663 | frozen structured event-head matching, seed42 |
| crrvs | rem_net | rr | 16.87228120162281 | 6.431843488080346 | 30 | 30 | 0.760067855794672 | frozen structured event-head matching, seed42 |
| crrvs | rem_net | qrs | 11.596872349142634 | 4.796606683662281 | 30 | 30 | 0.8609302603640218 | phase_pd continuous onset-offset pairs; roundtrip PASS |
| crrvs | rem_net | qt | 24.599850890608096 | 15.32347003892851 | 30 | 30 | 0.8085783574727422 | phase_pd continuous onset-offset pairs; roundtrip PASS |

### H1 临床主表coverage/可评估被试

来源：`docs/paper/sections/results_verified_20261101/clinical_coverage.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | model | metric | n_assessable | n_total | coverage |
|---|---|---|---|---|---|
| mmecg | mmecg_tcn | qrs | 11 | 11 | 0.8149696105833698 |
| mmecg | mmecg_tcn | qt | 11 | 11 | 0.7399624766166117 |
| mmecg | radarode_mtl | qrs | 11 | 11 | 0.8427453050188528 |
| mmecg | radarode_mtl | qt | 11 | 11 | 0.7688651792628902 |
| mmecg | cross_modal_conformer | qrs | 10 | 11 | 0.40610600780490674 |
| mmecg | cross_modal_conformer | qt | 10 | 11 | 0.29060981327800284 |
| mmecg | wavegru_net | qrs | 11 | 11 | 0.628342382429831 |
| mmecg | wavegru_net | qt | 11 | 11 | 0.45436902754838043 |
| mmecg | attention_dcgan | qrs | 11 | 11 | 0.3935765586885939 |
| mmecg | attention_dcgan | qt | 11 | 11 | 0.24587160740974684 |
| mmecg | rem_net | qrs | 11 | 11 | 0.8729996624216274 |
| mmecg | rem_net | qt | 11 | 11 | 0.8278327557907729 |
| crrvs | mmecg_tcn | qrs | 30 | 30 | 0.7699609432649605 |
| crrvs | mmecg_tcn | qt | 30 | 30 | 0.6910742352778626 |
| crrvs | radarode_mtl | qrs | 30 | 30 | 0.9177709149237769 |
| crrvs | radarode_mtl | qt | 30 | 30 | 0.8815041886009016 |
| crrvs | cross_modal_conformer | qrs | 30 | 30 | 0.8129072950207342 |
| crrvs | cross_modal_conformer | qt | 30 | 30 | 0.7132038429496781 |
| crrvs | wavegru_net | qrs | 30 | 30 | 0.8002580743054625 |
| crrvs | wavegru_net | qt | 30 | 30 | 0.6990070071479744 |
| crrvs | attention_dcgan | qrs | 30 | 30 | 0.8450675506185632 |
| crrvs | attention_dcgan | qt | 30 | 30 | 0.7239679630661532 |
| crrvs | rem_net | qrs | 30 | 30 | 0.8609302603640218 |
| crrvs | rem_net | qt | 30 | 30 | 0.8085783574727422 |

### H2 架构对比ΔPCC / CI / p

来源：`experiments/phase_resampling_20261027/formal_ablation_20261027/table1_modules.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | arm | pcc_percent | rmse | r_mae_ms | r_subjects_assessable | delta_pp | ci_low_pp | ci_high_pp | p_raw | p_holm | star |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | A0 | 90.31058116392656 | 0.11404400522058661 | 41.712188458710784 | 11 | -1.0866425254128196 | -1.7293411357836288 | -0.4498987035317859 | 0.013671875 | 0.041015625 | * |
| mmecg | A1 | 91.13249616189437 | 0.11171362684531645 | 40.32172531965666 | 11 | -0.2647275274450129 | -0.7850841094147075 | 0.22967297922481145 | 0.51953125 | 1.0 | NA |
| mmecg | A2 | 91.50277863849293 | 0.10836701907894829 | 40.707133164516726 | 11 | 0.10555494915355336 | -0.3630804202773354 | 0.6522221998734907 | 0.9658203125 | 1.0 | NA |
| mmecg | A3 | 81.29057071425699 | 0.13532193343747745 | 44.302898054770345 | 11 | -10.106652975082397 | -13.741990436207164 | -7.037704194133933 | 0.0009765625 | 0.00390625 | * |
| mmecg | A4 | 91.39722368933938 | 0.11079175634817644 | 39.0179782511286 | 11 | NA | NA | NA | NA | NA | NA |
| crrvs | A0 | 89.82642809549968 | 0.10048910876115164 | 21.4843823620299 | 30 | -0.6190594037373861 | -1.005108172694842 | -0.28089651962121337 | 0.0023674946278333664 | 0.007102483883500099 | * |
| crrvs | A1 | 89.78895326455434 | 0.10337339205046495 | 21.67020046113154 | 30 | -0.6565342346827189 | -1.1812009910742443 | -0.021466483672460027 | 0.0023674946278333664 | 0.007102483883500099 | * |
| crrvs | A2 | 90.44561942418416 | 0.10179383878906569 | 21.09544003950359 | 30 | 0.0001319249471028646 | -0.3593282947937647 | 0.37079910437266034 | 0.7303419969975948 | 0.7303419969975948 | NA |
| crrvs | A3 | 88.1446756919225 | 0.10589602515101433 | 23.278886119232663 | 30 | -2.3008118073145547 | -3.747468108932177 | -0.9384186367193869 | 0.001718590036034584 | 0.006874360144138336 | * |
| crrvs | A4 | 90.44548749923706 | 0.10080457963049412 | 21.247763779247762 | 30 | NA | NA | NA | NA | NA | NA |


### H2 析因效应ΔPCC / CI / p

来源：`experiments/phase_resampling_20261027/formal_ablation_20261027/table2_factorial.csv`；状态 verified，逐行摘录。

| dataset | effect | delta_pp | ci_low_pp | ci_high_pp | p_raw | p_holm | star | delta_r_mae_ms | r_complete_subjects | pcc_subjects |
|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | phase | 0.7284625010056929 | 0.29878713867881085 | 1.1481598967855624 | 0.013671875 | 0.041015625 | * | -1.154401181361056 | 11 | 11 |
| mmecg | event_probability | 0.3581800244071267 | 0.00835325907577169 | 0.7105639983307231 | 0.1015625 | 0.166015625 | NA | -1.539809026221125 | 11 | 11 |
| mmecg | interaction | -0.9274699471213601 | -1.9004872305826708 | -0.009231445464221003 | 0.0830078125 | 0.166015625 | NA | -0.29869177433400684 | 11 | 11 |
| mmecg | window_context | 10.106652975082397 | 7.037704194133932 | 13.741990436207164 | 0.0009765625 | 0.00390625 | * | -5.284919803641746 | 11 | 11 |
| crrvs | phase | 0.6378627816836039 | 0.21876091013352078 | 1.042688009639581 | 0.0003800932317972183 | 0.0015203729271888733 | * | -0.405689502205047 | 30 | 30 |
| crrvs | event_probability | -0.018803377946217854 | -0.3006039634346962 | 0.2725697383284565 | 0.9515264742076397 | 0.9515264742076397 | NA | 0.16907091942290683 | 30 | 30 |
| crrvs | interaction | 0.03734290599822998 | -0.6435394535462061 | 0.6374146789312362 | 0.24494642950594425 | 0.4898928590118885 | NA | -0.0334943593574654 | 30 | 30 |
| crrvs | window_context | 2.3008118073145547 | 0.9384186367193859 | 3.7474681089321757 | 0.001718590036034584 | 0.005155770108103752 | * | -2.0311223399848988 | 30 | 30 |

### H3 多任务比较

来源：`experiments/phase_resampling_20261027/formal_ablation_20261027/table3_multitask.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | arm | pcc_percent | rmse | q_mae_ms | r_mae_ms | s_mae_ms | t_mae_ms | q_coverage | r_coverage | s_coverage | t_coverage | overall_coverage_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | S1 | 89.61773352189498 | 0.11356591501019218 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| mmecg | S2 | NA | NA | 43.5417932260891 | 40.38270361474678 | 43.06438830276047 | 41.521255892359314 | 0.8827950796117059 | 0.8827950796117059 | 0.8827950796117059 | 0.8942243632983223 | 88.55525714306215 |
| mmecg | A0 | 90.31058116392656 | 0.11404400522058661 | 44.79911888112219 | 41.712188458710784 | 44.03997218589502 | 41.89953689354131 | 0.8353503263847412 | 0.8353503263847412 | 0.8353503263847412 | 0.8468376077986871 | 83.81250876614271 |
| mmecg | A4 | 91.39722368933938 | 0.11079175634817644 | 42.597587160190926 | 39.0179782511286 | 43.1182889592617 | 39.846587750866725 | 0.848656893876053 | 0.848656893876053 | 0.848656893876053 | 0.8596605804811296 | 85.13142512549602 |
| crrvs | S1 | 89.40199077129364 | 0.10771635634203752 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| crrvs | S2 | NA | NA | 27.02537213717371 | 20.254012753735466 | 25.82313484706024 | 24.396413465063457 | 0.8867011412721636 | 0.8867011412721636 | 0.8867011412721636 | 0.905242023889208 | 89.11949079132062 |
| crrvs | A0 | 89.82642809549968 | 0.10048910876115164 | 28.274965066677705 | 21.4843823620299 | 26.257107108667544 | 25.86908806797941 | 0.847147989474269 | 0.847147989474269 | 0.847147989474269 | 0.8710063213609132 | 85.29298635016195 |
| crrvs | A4 | 90.44548749923706 | 0.10080457963049412 | 28.06831056189062 | 21.247763779247762 | 26.38175467148474 | 23.954586095641492 | 0.8243532947965834 | 0.8243532947965834 | 0.8243532947965834 | 0.8516082608467663 | 83.09591048564229 |

### H3 S1新增事后统计

来源：`docs/paper/sections/ablation_merged_20261101/shape_only_posthoc_statistics.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | arm | delta_pp | ci_low_pp | ci_high_pp | p_raw | p_holm | star | family |
|---|---|---|---|---|---|---|---|---|
| mmecg | S1 | -1.7794901674444026 | -2.546719068830664 | -0.943600034171885 | 0.0048828125 | 0.007488865405321121 | * | post-hoc S1 across2datasets |
| crrvs | S1 | -1.0434967279434204 | -2.1165744562943773 | 0.1669233093659069 | 0.0037444327026605606 | 0.007488865405321121 | * | post-hoc S1 across2datasets |

### G4 往返检验（当前重采样流程，不称误差下限）

来源：`docs/paper/sections/figures/roundtrip_check.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| n_eligible | n_pairs | n_subjects | coverage | detection_failure_rate | mae | bias | bias_ci_low | bias_ci_high | loa_low | loa_high | naive_loa_low | naive_loa_high | within_variance | between_variance | slope | slope_ci_low | slope_ci_high | icc_a1 | dataset | metric | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 19266 | 19266 | 11 | 1.0 | 0.0 | 0.7780546039655352 | 0.5574587355963874 | 0.4079543381085623 | 0.7477069375006365 | -3.2943746790528223 | 4.409292150245597 | -3.289444356759655 | 4.40436182795243 | 3.7922728773554515 | 0.06982121214064144 | -0.006675588525754605 | -0.044086322702141786 | 0.02158003569786336 | 0.9220459052183025 | mmecg | qrs | True |
| 19075 | 19070 | 11 | 0.999737876802097 | 0.0002621231979029748 | 1.0175668589407447 | 0.49423177766124804 | 0.30856726862231604 | 0.6231620382856169 | -9.924588443464886 | 10.913051998787383 | -9.923481444543391 | 10.911944999865888 | 28.21454882397478 | 0.042379226875604636 | 0.0003504902209422124 | -0.007690810769145596 | 0.006570373832849417 | 0.9720208247578707 | mmecg | qt | True |
| 26195 | 26195 | 30 | 1.0 | 0.0 | 1.5079213590379843 | 1.3212445123115097 | 1.0232433726279693 | 1.6595457791742323 | -5.094603188738993 | 7.737092213362012 | -5.086031966776212 | 7.728520991399232 | 9.906753227204401 | 0.8083398910471125 | -0.05067576297273717 | -0.0635726400030462 | -0.03371942192259259 | 0.9374279678916528 | crrvs | qrs | True |
| 26114 | 26105 | 30 | 0.9996553572796202 | 0.00034464272037981747 | 1.2886420226010342 | 0.6048649683968589 | 0.45599726461714 | 0.7515195323319328 | -6.555000413774666 | 7.764730350568383 | -6.553542480284382 | 7.7632724170781 | 13.190877430883114 | 0.15347708047106065 | 0.003857908862379416 | -0.0008602144520505331 | 0.010965467771162895 | 0.991212530155962 | crrvs | qt | True |

### H6 重复测量BA完整统计

来源：`docs/paper/sections/figures/bland_altman_stats_v2.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | metric | n_eligible | n_pairs | n_subjects | coverage | detection_failure_rate | mae | bias | bias_ci_low | bias_ci_high | loa_low | loa_high | naive_loa_low | naive_loa_high | within_variance | between_variance | slope | slope_ci_low | slope_ci_high | icc_a1 | beat_slope | beat_slope_ci_low | beat_slope_ci_high | subject_mean_slope | subject_mean_intercept | subject_mean_slope_ci_low | subject_mean_slope_ci_high | subject_mean_bootstrap_valid | subject_mean_bootstrap_total | regression_line_drawn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | rr | 19266 | 15825 | 11 | 0.8213952039862971 | 0.17860479601370294 | 15.682751011660125 | 0.2190625134990867 | -0.27161675633368737 | 0.5915588010498359 | -41.74476756185137 | 42.18289258884955 | -41.74252605359117 | 42.18065108058935 | 458.0745851923766 | 0.31854126349789164 | -0.0038707726916011385 | -0.01899814078792866 | 0.00731574220879662 | 0.9801043854658245 | -0.0038707726916011385 | -0.01899814078792866 | 0.00731574220879662 | -0.007391443328671162 | 6.3589292738217775 | -0.015326732013186833 | 0.0018752646089232306 | 10000 | 10000 | False |
| mmecg | qrs | 19266 | 16824 | 11 | 0.873248209280598 | 0.126751790719402 | 7.541607227769853 | 0.6401569186875892 | -5.3651238751358905 | 4.942650343942937 | -19.241329027887865 | 20.52164286526304 | -18.420302591893854 | 19.70061642926903 | 47.119318869144664 | 55.77361205647935 | 0.8334291173469479 | 0.6089545968781442 | 1.212699428509886 | -0.25504209036027586 | 0.8334291173469479 | 0.6089545968781442 | 1.212699428509886 | -1.8375329395054474 | 168.84446726776395 | -4.001054853153725 | -0.24567626659368277 | 10000 | 10000 | True |
| mmecg | qt | 19075 | 15875 | 11 | 0.8322411533420708 | 0.16775884665792917 | 18.976692913385826 | 4.463937007874016 | -4.841838288434836 | 11.717499366250044 | -47.612590192781276 | 56.54046420852931 | -46.707105871241694 | 55.634979886989726 | 546.3852814749114 | 159.56137754231847 | -0.010450824247947443 | -0.37347566484158734 | 0.706325576935881 | 0.3261183248616677 | -0.010450824247947443 | -0.37347566484158734 | 0.706325576935881 | -0.09403584074114177 | 35.87809212045808 | -0.8082243583338692 | 0.9298016413648199 | 10000 | 10000 | False |
| crrvs | rr | 26193 | 20793 | 30 | 0.7938380483335242 | 0.20616195166647577 | 15.19277680965349 | -0.9176279975297802 | -1.6574546591572892 | -0.1949021380911779 | -43.2078252606214 | 41.37256926556184 | -43.198544295246606 | 41.36328830018705 | 459.92854582141 | 5.622470565304361 | -0.008121305437269322 | -0.017081928854519896 | -0.0007507969347422987 | 0.9894631986738979 | -0.008121305437269322 | -0.017081928854519896 | -0.0007507969347422987 | -0.008191160022225768 | 6.571860434859189 | -0.02051265572824169 | -0.0007498640664754594 | 10000 | 10000 | True |
| crrvs | qrs | 26195 | 22368 | 30 | 0.853903416682573 | 0.14609658331742703 | 11.608100858369099 | -2.166934907010014 | -6.214995480323679 | 1.8383331682145219 | -31.02261781759469 | 26.688748003574663 | -30.741397813932554 | 26.407527999912528 | 99.52547864546442 | 117.22026173255742 | -0.03860899565601624 | -0.4258506207220143 | 0.4936931974224502 | -0.0547457473720516 | -0.03860899565601624 | -0.4258506207220143 | 0.4936931974224502 | -1.4939221998528915 | 115.9747692491103 | -2.184372042159486 | -0.8314499004681987 | 10000 | 10000 | True |
| crrvs | qt | 26114 | 20891 | 30 | 0.7999923412728804 | 0.20000765872711956 | 22.4421999904265 | -1.5001675362596334 | -8.065854129573221 | 4.740642898690921 | -59.51761753024523 | 56.51728245772597 | -59.077242972285305 | 56.07690789976604 | 511.5598610736772 | 364.6439352103243 | 0.1984085989985325 | -0.04976317085471043 | 0.5508756428849848 | 0.24895189588111039 | 0.1984085989985325 | -0.04976317085471043 | 0.5508756428849848 | -1.0020287027656085 | 414.6192712845043 | -1.698921723554256 | -0.41559876342182783 | 10000 | 10000 | True |

### H7 HR计数关联（分母为有限HR且|误差|>5的窗口）

来源：`docs/paper/sections/figures/source_data/phase_pd_bland_altman_v2_20261029/hr_attribution_summary.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | total_windows | finite_hr_windows | unavailable_hr_windows | bad_windows | bad_fraction_finite | any_missed_n | any_missed_fraction | any_extra_n | any_extra_fraction | neither_n | neither_fraction | missed_only_n | extra_only_n | both_n | missed_only_fraction | extra_only_fraction | both_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | 7735 | 7644 | 91 | 1367 | 0.17883307169021453 | 1366 | 0.9992684711046086 | 292 | 0.21360643745427946 | 0 | 0.0 | 1075 | 1 | 291 | 0.7863935625457206 | 0.000731528895391368 | 0.21287490855888808 |
| crrvs | 5605 | 5373 | 232 | 1052 | 0.19579378373348222 | 1052 | 1.0 | 177 | 0.1682509505703422 | 0 | 0.0 | 875 | 0 | 177 | 0.8317490494296578 | 0.0 | 0.1682509505703422 |

### H8 预测锚点

来源：`docs/paper/sections/figures/source_data/anchor_robustness_delta_20261028/predicted_points.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | realised_err_ms | delta_context_pp | ci_low_pp | ci_high_pp | pcc_absolute | context_pcc_absolute | n_subjects | n_subjects_with_anchor_matches |
|---|---|---|---|---|---|---|---|---|
| mmecg | 37.071448864909904 | 1.0866425254128196 | 0.44989870353178546 | 1.7293411357836286 | 0.9139722368933938 | 0.9031058116392656 | 11 | 11 |
| crrvs | 19.760310907328552 | 0.6190594037373861 | 0.2808965196212133 | 1.0051081726948419 | 0.9044548749923706 | 0.8982642809549968 | 30 | 30 |

### H8 各sigma paired ΔPCC

来源：`docs/paper/sections/figures/anchor_robustness_delta.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | sigma_ms | realised_err_ms | delta_context_pp | ci_low_pp | ci_high_pp | pcc_absolute | fallback_rate |
|---|---|---|---|---|---|---|---|
| mmecg | 0 | 0.0 | 1.0132789611816406 | 0.16477136449380358 | 1.9188948923891236 | 0.913238601251082 | 0.0 |
| mmecg | 5 | 3.9894826231743568 | 1.0056367787447844 | 0.15967245264486837 | 1.9060697203332713 | 0.9131621794267134 | 0.0 |
| mmecg | 10 | 7.978967000904195 | 0.9892028028314762 | 0.14491275765679107 | 1.8805182603272514 | 0.9129978396675803 | 0.0 |
| mmecg | 20 | 15.957931769705853 | 0.9316218983043326 | 0.1074569089846182 | 1.8026607686823075 | 0.912422030622309 | 0.0 |
| mmecg | 40 | 31.91586504193191 | 0.7380560311404146 | -0.026869987899607088 | 1.5406758785247805 | 0.9104863719506698 | 0.0 |
| mmecg | 80 | 63.81132017338144 | 0.2354894984852196 | -0.45202665979211815 | 1.0021344612945218 | 0.9054607066241178 | 0.0 |
| mmecg | 128 | 101.46326069621746 | -0.3386392376639634 | -1.1292177520015034 | 0.5302012400193628 | 0.899719419262626 | 0.0 |
| crrvs | 0 | 0.0 | 0.8135811487833658 | 0.40425271292527515 | 1.210026388367017 | 0.9064000924428304 | 0.0 |
| crrvs | 5 | 4.001729947706968 | 0.8098417917887368 | 0.40161267320315047 | 1.2052084575096764 | 0.9063626988728841 | 0.0 |
| crrvs | 10 | 8.003461667821098 | 0.8012845516204844 | 0.3949022958676024 | 1.1946255048116046 | 0.9062771264712015 | 0.0 |
| crrvs | 20 | 16.006920185575968 | 0.7710405588150016 | 0.3739154587189356 | 1.1574938764174765 | 0.9059746865431467 | 0.0 |
| crrvs | 40 | 32.01315568522678 | 0.6712815364201856 | 0.29597943822542727 | 1.0434675345818198 | 0.9049770963191985 | 0.0 |
| crrvs | 80 | 64.01182160048441 | 0.4150157769521075 | 0.061792711416880176 | 0.7620211154222489 | 0.9024144387245178 | 0.0 |
| crrvs | 128 | 102.29697140606565 | 0.15316669146219875 | -0.232984468340874 | 0.5346870660781848 | 0.8997959478696188 | 0.0 |

### H9 复杂度

来源：`docs/paper/sections/complexity_20261030/complexity.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| method | dataset | Params_M | FLOPs_G | GPU_latency_ms | CPU_latency_ms | GPU_model |
|---|---|---|---|---|---|---|
| Chen-TCN | MMECG | 0.650786 | 2.929018001 | 2.0532759372144938 | 27.539956383407116 | NVIDIA GeForce RTX 4080 SUPER |
| radarODE-MTL | MMECG | 42.275527 | 23.694319666 | 7.688074605539441 | 618.1956971995533 | NVIDIA GeForce RTX 4080 SUPER |
| Conformer | MMECG | 0.465558 | 0.1878784 | 1.0485888924449682 | 2.8814675752073526 | NVIDIA GeForce RTX 4080 SUPER |
| WaveGRU-Net | MMECG | 0.789334 | 1.63246594 | 1.8029676284641027 | 40.374118369072676 | NVIDIA GeForce RTX 4080 SUPER |
| Attention-DCGAN | MMECG | 4.387991 | 1.9056128 | 0.6975366268306971 | 16.972143901512027 | NVIDIA GeForce RTX 4080 SUPER |
| REM-Net | MMECG | 1.481029 | 5.56823541 | 1.2959153391420841 | 48.601173562929034 | NVIDIA GeForce RTX 4080 SUPER |
| Chen-TCN | CR_RVS | 0.636674 | 1.764067109 | 1.9968280103057623 | 13.588929316028953 | NVIDIA GeForce RTX 4080 SUPER |
| radarODE-MTL | CR_RVS | 42.232456 | 22.168057906 | 6.236654706299305 | 226.86747158877552 | NVIDIA GeForce RTX 4080 SUPER |
| Conformer | CR_RVS | 0.463206 | 0.1841152 | 1.0834787972271442 | 2.869319636374712 | NVIDIA GeForce RTX 4080 SUPER |
| WaveGRU-Net | CR_RVS | 0.779926 | 1.58981634 | 1.77370710298419 | 39.09212118014693 | NVIDIA GeForce RTX 4080 SUPER |
| Attention-DCGAN | CR_RVS | 4.356631 | 1.8554368 | 0.7120240479707718 | 18.310598330572248 | NVIDIA GeForce RTX 4080 SUPER |
| REM-Net | CR_RVS | 1.424581 | 1.99784181 | 1.2020180001854897 | 17.842826433479786 | NVIDIA GeForce RTX 4080 SUPER |

### H10 GDN0022各场景

来源：`docs/paper/sections/figures/source_data/phase_pd_bland_altman_v2_20261029/worst_qt_subject_scenes.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| subject | official_subject_id | record | scene | test_windows | eligible_qt_beats | matched_qt_beats | coverage | bias_ms | mae_ms | p25_ms | median_ms | p75_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | GDN0022 | CRRVS_0059 | Resting | 80 | 339 | 47 | 0.13864306784660768 | -101.06382978723404 | 105.1063829787234 | -115.0 | -105.0 | -95.0 |
| 22 | GDN0022 | CRRVS_0060 | Valsalva | 80 | 342 | 69 | 0.20175438596491227 | -99.05797101449275 | 103.40579710144928 | -115.0 | -100.0 | -85.0 |
| 22 | GDN0022 | CRRVS_0061 | Apnea | 37 | 147 | 51 | 0.3469387755102041 | -86.86274509803921 | 97.05882352941177 | -102.5 | -95.0 | -80.0 |

### H8 完整稳健性报告（含crossovers与matched/valid/all误差口径）

来源：`docs/paper/sections/figures/source_data/anchor_robustness_delta_20261028/REPORT.md`；状态：verified，仅针对该文件所标明的模型与口径。

# Paired ΔPCC anchor robustness revision

Inference only, all16 frozen A4 checkpoints. Original sigma0–80 observations are reused unchanged. Y is a paired subject-level difference, in percentage points, with10000 paired-subject bootstrap CI.
Shared y-limits: False (predefined CI-span ratio threshold2). Original absolute figure archived under previous_absolute_figure/.

## MMECG

| Sigma ms | Realised anchor error ms | ΔPCC vs Context [95% CI] pp | Fallback % |
|---:|---:|---|---:|
| 0 | 0.00 | +1.013 [+0.165, +1.919] | 0.000 |
| 5 | 3.99 | +1.006 [+0.160, +1.906] | 0.000 |
| 10 | 7.98 | +0.989 [+0.145, +1.881] | 0.000 |
| 20 | 15.96 | +0.932 [+0.107, +1.803] | 0.000 |
| 40 | 31.92 | +0.738 [-0.027, +1.541] | 0.000 |
| 80 | 63.81 | +0.235 [-0.452, +1.002] | 0.000 |
| 128 | 101.46 | -0.339 [-1.129, +0.530] | 0.000 |

Predicted-anchor star: error=37.07ms on matched central pairs; ALL-window ΔPCC=+1.087 [+0.450, +1.729]pp.
Statistical crossover, interpolating paired lowerCI to0: 28.72ms realised. Mean crossing: 79.25ms realised.

| Predicted-window subset | N windows | Anchor MAE ms | RR-span MAE ms [95% CI] | Common MAE ms | Differential MAE ms | Same-sign % | Common-energy % |
|---|---:|---:|---|---:|---:|---:|---:|
| matched | 3201 | 37.07 | 16.60 [14.64, 18.69] | 35.75 | 8.30 | 84.5 | 91.0 |
| valid | 7285 | 488.97 | 68.08 [50.09, 86.03] | 487.24 | 34.04 | 91.6 | 97.9 |
| all | 7735 | 508.87 | 95.68 [63.61, 131.91] | 503.59 | 47.84 | 91.0 | 95.7 |

### Median absolute predicted-anchor errors

R error means the FIRST R boundary error e0, not the mean error of both boundaries. RR error is e1−e0. Primary medians use an equal-subject empirical mixture CDF; pooled-window medians are supplied separately.

| Subset | Median absolute R error (ms) | Median absolute RR error (ms) | Pooled median absolute R error (ms) | Pooled median absolute RR error (ms) |
|---|---:|---:|---:|---:|
| matched | 31.84 | 12.03 | 33.27 | 11.68 |
| valid | 583.77 | 18.33 | 590.61 | 17.10 |
| all | 600.29 | 19.39 | 599.89 | 18.24 |

Matched-pair diagnostics support common-mode dominance within paired endpoints.
## CRRVS

| Sigma ms | Realised anchor error ms | ΔPCC vs Context [95% CI] pp | Fallback % |
|---:|---:|---|---:|
| 0 | 0.00 | +0.814 [+0.404, +1.210] | 0.000 |
| 5 | 4.00 | +0.810 [+0.402, +1.205] | 0.000 |
| 10 | 8.00 | +0.801 [+0.395, +1.195] | 0.000 |
| 20 | 16.01 | +0.771 [+0.374, +1.157] | 0.000 |
| 40 | 32.01 | +0.671 [+0.296, +1.043] | 0.000 |
| 80 | 64.01 | +0.415 [+0.062, +0.762] | 0.000 |
| 128 | 102.30 | +0.153 [-0.233, +0.535] | 0.000 |

Predicted-anchor star: error=19.76ms on matched central pairs; ALL-window ΔPCC=+0.619 [+0.281, +1.005]pp.
Statistical crossover, interpolating paired lowerCI to0: 72.04ms realised. Mean crossing: not observed through sigma128ms.

| Predicted-window subset | N windows | Anchor MAE ms | RR-span MAE ms [95% CI] | Common MAE ms | Differential MAE ms | Same-sign % | Common-energy % |
|---|---:|---:|---|---:|---:|---:|---:|
| matched | 2282 | 19.76 | 15.83 [14.46, 17.20] | 17.93 | 7.91 | 72.7 | 75.5 |
| valid | 5016 | 584.76 | 66.48 [55.24, 78.52] | 583.67 | 33.24 | 86.5 | 98.7 |
| all | 5605 | 617.64 | 126.56 [86.43, 184.74] | 609.68 | 63.28 | 85.5 | 96.0 |

### Median absolute predicted-anchor errors

R error means the FIRST R boundary error e0, not the mean error of both boundaries. RR error is e1−e0. Primary medians use an equal-subject empirical mixture CDF; pooled-window medians are supplied separately.

| Subset | Median absolute R error (ms) | Median absolute RR error (ms) | Pooled median absolute R error (ms) | Pooled median absolute RR error (ms) |
|---|---:|---:|---:|---:|
| matched | 15.67 | 11.55 | 14.96 | 11.27 |
| valid | 775.65 | 19.59 | 757.65 | 19.27 |
| all | 774.30 | 21.93 | 765.77 | 22.35 |

Matched-pair diagnostics support common-mode dominance within paired endpoints.

## Interpretation safeguards

c=(e0+e1)/2, d=(e1−e0)/2, RR-span error=2|d|. Common-energy share is per-subject mean(c²)/(mean(c²)+mean(d²)), then subject macro.
Primary star error and span diagnosis use the SAME matched subset; report valid/all subsets separately. Star y/CI always retains ALL windows.
Do not replace ALL-window errors by matched-subset errors. Valid/all rows include unmatched cycle choices and fallback; common-mode energy alone does not demonstrate correct beat correspondence or a small physiological latency.
These are errors of the SPECIFIC central phase-grid pair against the target central pair, without rematching to another cycle. They are not the usual per-beat event-head R-MAE or all-beat RR-interval MAE.
Same-sign rate and larger common component can support common-mode dominance WITHIN PAIRS, not a constant per-record offset or acquisition synchronization explanation.
Independent Gaussian perturbations differ from the empirical correlated predicted-error distribution; the star need not lie on the jitter curve.
Crossover is loss of paired statistical support above Context, not necessarily a mean intersection. No new significance-based model or checkpoint selection.
CI bands are pointwise, not simultaneous; the interpolated crossover is a grid-dependent diagnostic, not a95% confidence interval for the threshold itself.
Shortened legend labels use one shared row; orange predicted-anchor stars and whiskers, white-edged 3-point markers, no annotation or horizontal grid, and a left-panel-only y label. Axis labels are8pt and ticks7pt; data coordinates and curve statistics are unchanged.
All old data and reports remain unchanged. Only the three requested figure files are replaced, after archival. The new delta source is anchor_robustness_delta.csv.



### H7 窗口HR完整统计

来源：`docs/paper/sections/figures/bland_altman_stats.csv`；状态verified。单位bpm；±5准确率在文末有限配对结果中另列。

| dataset | metric | n_eligible | n_pairs | n_subjects | coverage | mae | bias | bias_ci_low | bias_ci_high | loa_low | loa_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | hr | 7735 | 7644 | 11 | 0.9882352941176471 | 4.596305190359576 | -4.145206074578447 | -5.970447657868808 | -2.745117141980683 | -25.57578185941966 | 17.285369710262767 |
| crrvs | hr | 5605 | 5373 | 30 | 0.9586083853702052 | 4.485050896862112 | -3.9769755536749307 | -5.10062971102537 | -2.945348317841198 | -23.371380725729317 | 15.417429618379456 |

## 不可误引为phase_pd的旧图审计
以下数值来源 verified，但作为新模型结果的状态为 **missing**。保留是为了论文逐图核对，不能把数值转换成新模型声明。

### 旧定位CDF各端点中位数（不是phase_pd）

来源：`docs/paper/sections/figures/source_data/localization_cdf_main_20261029/median_annotations.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | metric | remnet_median_ms | baseline | baseline_median_ms |
|---|---|---|---|---|
| mmecg | q | 40.0 | mmecg_tcn | 35.0 |
| mmecg | r | 35.0 | mmecg_tcn | 32.5 |
| mmecg | s | 37.5 | mmecg_tcn | 35.0 |
| mmecg | t | 35.0 | mmecg_tcn | 30.0 |
| crrvs | q | 20.0 | cross_modal_conformer | 20.0 |
| crrvs | r | 15.0 | cross_modal_conformer | 10.0 |
| crrvs | s | 20.0 | cross_modal_conformer | 15.0 |
| crrvs | t | 20.0 | cross_modal_conformer | 20.0 |

### 旧v4当前展示面板 IDs / PCC（不是phase_pd）

来源：`docs/paper/sections/figures/source_data/examples_v4_apnea_revision_20261030/per_window_results.csv`；状态：**verified**（逐行摘录，未改变原统计口径）。

| dataset | scene | subject | fold | window_index | record_id | start | central_start | central_stop | selection_representative_pcc | rem_net_pcc | mmecg_tcn_pcc | radarode_mtl_pcc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmecg | NB | 9 | 4 | 353 | MMECG_0021 | 5200 | 5488 | 6512 | 0.9156167865216782 | 0.9068865180015564 | 0.8356436491012573 | 0.8946477174758911 |
| mmecg | IB | 16 | 8 | 59 | MMECG_0041 | 23600 | 23888 | 24912 | 0.9192544952044244 | 0.8661132454872131 | 0.5134637951850891 | 0.8510346412658691 |
| mmecg | SP | 29 | 10 | 1072 | MMECG_0069 | 20800 | 21088 | 22112 | 0.894946904592367 | 0.8465814590454102 | 0.8658306002616882 | 0.9069433212280273 |
| mmecg | PE | 14 | 7 | 617 | MMECG_0039 | 8800 | 9088 | 10112 | 0.9084562132824212 | 0.879526674747467 | 0.8286213278770447 | 0.8785006403923035 |
| crrvs | NB | 2 | 3 | 49 | CRRVS_0003 | 73600 | 73888 | 74912 | 0.9357019152549435 | 0.9387128353118896 | 0.9466899633407593 | 0.9417563676834106 |
| crrvs | IB | 13 | 5 | 444 | CRRVS_0035 | 51200 | 51488 | 52512 | 0.9181857062282571 | 0.9178445935249329 | 0.7847528457641602 | 0.8868279457092285 |
| crrvs | SP | 7 | 1 | 322 | CRRVS_0018 | 1600 | 1888 | 2912 | 0.874133776232427 | 0.8449724316596985 | 0.8452491760253906 | 0.8470168113708496 |

### H6参考分布 / H7 HR ±5 bpm — 从冻结配对CSV只读汇总

来源：`docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/paired_values.csv`；状态 verified；以下SD为样本SD，参考分布仅限成功匹配配对，不代表全部GT；HR不可用窗口不进入有限配对准确率，另见失败数。
字段：`dataset, fold, subject, record, start, ref_r, metric, ref, pred, centre_distance`。
- mmecg/hr: reference mean±sampleSD = 73.211456 ± 9.729556; N=7644; HR MAE=4.596305 bpm; within±5bpm=82.116693%.
- mmecg/rr: reference mean±sampleSD = 826.906793 ± 107.532956; N=15825; 单位ms。
- mmecg/qrs: reference mean±sampleSD = 92.211424 ± 5.096401; N=16824; 单位ms。
- mmecg/qt: reference mean±sampleSD = 376.563150 ± 22.673790; N=15875; 单位ms。
- crrvs/hr: reference mean±sampleSD = 61.770743 ± 8.822321; N=5373; HR MAE=4.485051 bpm; within±5bpm=80.420622%.
- crrvs/rr: reference mean±sampleSD = 983.612514 ± 149.329030; N=20793; 单位ms。
- crrvs/qrs: reference mean±sampleSD = 80.228451 ± 10.122606; N=22368; 单位ms。
- crrvs/qt: reference mean±sampleSD = 416.770140 ± 22.441286; N=20891; 单位ms。

## 导出检查
- 仅读取元数据、H5索引、已落盘结果和代码；没有训练、推理或新增显著性检验。
- 所有缺失项明确标为missing；不同模型、样本口径与聚合器不混用。
- 复杂度环境不是历史训练环境；元数据年龄SD不是原论文SD。
- CR临床低/负ICC、GDN0022低coverage、大误差保留，不为论文叙事删除。
- 后续补齐missing需新增可核验来源，再更新本文件；不能用估计填空。

## 关键来源版本指纹

状态 verified；导出时仓库HEAD：`cb3af51ec378fb72d863a4dd235a1e90c409b24a`。以下SHA256锁定本次读取的文件版本。

|source|SHA256|
|---|---|
|`experiments/phase_resampling_20261027/architecture.py`|`75b9503ee97c0acc48be0b233da6a698e035292a72b35fdc3bad5d319be1dfc8`|
|`experiments/phase_resampling_20261027/formal_spec.json`|`0feaae999e6db61b5038c8b8c69119f3f71a639ba088731cb1dbc9c40f675c02`|
|`docs/paper/sections/results_verified_20261101/baseline_compare_verified.csv`|`dde7204e10dabdf8a19692a926157ef4e6ab04f803d6a315dc4cc0e1fa143615`|
|`experiments/phase_resampling_20261027/formal_ablation_20261027/table1_modules.csv`|`ba82890a19fdb2c8c7d8bacaa1c422c5d1e83c57af65106efe72e8baa0b2687b`|
|`experiments/phase_resampling_20261027/formal_ablation_20261027/table2_factorial.csv`|`703d06f5b7ee0daef6e4ddf1e442847eddbc6124c2e6af4dcfad5dfcbc2d8741`|
|`docs/paper/sections/figures/bland_altman_stats_v2.csv`|`22dedb1a4e611096ad4f22fa387e38e7963bfea7cd454b4bbc0731cb926e1452`|
|`docs/paper/sections/figures/roundtrip_check.csv`|`1c1c4fa90359c89ee7864eeb2598feeabfa616c58403efbe980805ba49c71824`|
|`docs/paper/sections/figures/anchor_robustness_delta.csv`|`92846503a1a46ea513f0f9dbf5168db8849735288bad91c74450871ebc37f5f9`|
|`docs/paper/sections/complexity_20261030/complexity.csv`|`1b77e083c3475ef89d8aad93d057363bd7f099e4b7cc12bc871f18273e335da5`|
