# 数据集与下载说明

从公开数据开始执行以下数值任务，不需要 149 GB 历史环境。大小为原始下载量，解压、处理、依赖和运行还需要额外空间。

| `--dataset` | 来源及下载量 | 任务和主要限制 |
|---|---|---|
| `mfpt` | [MathWorks MFPT](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)，20 文件约 42.6 MB | 原生台架振动分类；来源为 CC BY-NC-SA 4.0 |
| `cmapss-fd001`、`cmapss-fd002`、`cmapss-fd003`、`cmapss-fd004` | [NASA C-MAPSS](https://zenodo.org/records/15346912)，共同 ZIP 约 12.4 MB | 四个仿真发动机子集；单位为剩余循环数，非现场测量 |
| `femto-rul` | [NASA FEMTO](https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip) 约 1.16 GB，另下载固定版[挑战说明](https://github.com/aref9069/IEEE-PHM-2012-Prognostic-Challenge-/blob/9c476de6405b8c31b53bba8d6b0955ea57172206/IEEEPHM2012-Challenge-Details.pdf)约 2.28 MB | 台架轴承剩余秒数；一条挑战标注有来源歧义，另报去除此条的敏感性指标 |
| `xjtu-rul` | [XJTU-SY 作者数据](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)，六个与作者来源哈希一致的[固定镜像分卷](https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/tree/274226336036cd8deb2c40b4ba323af2d11a7290/Data)，共约 4.45 GB | 15 只台架轴承全寿命；目标为距末次采集的名义分钟，非精确物理失效时刻 |
| `xjtu-terminal` | 与 `xjtu-rul` 相同的六分卷 | 只读取每只轴承末次记录，预测拆检损伤位置；仅 15 只轴承，无早晚期严重度标签 |
| `engine-qa` | [EngineMT-QA 固定版本](https://huggingface.co/datasets/pandalin98/EngineMT-QA/tree/ea2051a15512212976c22c1a075ec549c7d48eef)，QA 约 57.5 MB，选定 HDF5 数组约 28 MB | 28 个历史发动机方向的数值问答标注；仿真、不代表物理风险或因果真值 |
| `ldm-rul` | [Paderborn LDM 作者特征](https://github.com/alireza-javanmardi/bearing-RUL/tree/d7167c927aad5b7f197b1d735b21339e3e17844e/data/LDM)，约 158 MB | 14 次有明确试验终点的轴承运行；预测距作者特征末行的采集次数，不是小时；使用派生特征而非重算原波形 |
| `cwru` | [PDMBench CWRU](https://huggingface.co/datasets/submission096/CWRU)，约 7.1 MB | 原特征表分类；录波组根据连续段推断，不能称物理轴承独立 |
| `electric-motor-vibrations` | [PDMBench 电机振动](https://huggingface.co/datasets/submission096/ElectricMotorVibrations)，约 67.8 MB | 三轴振动分类；原题要求的声学输入未包含在此子任务中 |
| `rotor-broken-bar` | [PDMBench 转子](https://huggingface.co/datasets/submission096/RotorBrokenBar)，约 327.4 MB | 按来源记录的类别分类，窗口按原记录分组 |
| `hust-bearing` | [PDMBench HUST](https://huggingface.co/datasets/submission096/HUST)，约 1.03 GB | 按来源文件分组的轴承分类，不代表不同物理轴承独立 |
| `paderborn` | [PDMBench Padeborn](https://huggingface.co/datasets/submission096/Padeborn)，约 3.90 GB | 数字轴承编号跨类别复用，按编号整体分组；同组可含多类，不能报告单一组投票准确率 |
| `mendeley-bearing` | [PDMBench Mendeley](https://huggingface.co/datasets/submission096/Mendeley)，约 4.24 GB | 长录波二分类，每条均匀抽取最多 64 个不重叠窗口；录波可共用物理硬件 |

各数据源须分别遵守其使用条款。以上均是**数值研究适配**，不是完整原题或官方成绩。FEMTO 原始 ZIP 的 SHA-256 是本研究固定的全文件哈希，不声称来自发布者；其他来源版本和哈希见 [数据清单](../src/phmforge_bench/resources/pdmb_fault_sources.json) 及相应准备代码。

如果需要全部来源，按上述入口下载的文件合计约 **15.4 GB**；只运行部分任务时按需下载即可。这个估计不含软件依赖、准备过程中的解压副本或训练输出。XJTU 的原始压缩包约 4.45 GB，解压过程还需要约 12.3 GB，计算完毕后程序删除自己的解压副本；FEMTO 读取嵌套压缩包时也需要额外磁盘空间。

尚未完整接入的原题和来源包括 UoC、PlanetaryPdM，以及需要独立维修收益、合规或物理安全真值的完整题目。EngineMT-QA 已支持发布者问答标注的数值适配，但它缺少可认证的完整物理健康合同。下载数据不会自动使原题全部要求可评分；不要把分类或 RUL 子任务计为完整原题完成。

## 从公开源准备和运行

先按 [README](../README.md#安装) 安装本仓库。任选需要的示例执行；`--data-root` 须指向未完成准备的新目录：

```bash
export PHM_RUN_ROOT="$HOME/phmforge-runs"
mkdir -p "$PHM_RUN_ROOT/tmp" "$PHM_RUN_ROOT/mpl" "$PHM_RUN_ROOT/xdg"
export TMPDIR="$PHM_RUN_ROOT/tmp" TMP="$PHM_RUN_ROOT/tmp" TEMP="$PHM_RUN_ROOT/tmp"
export MPLCONFIGDIR="$PHM_RUN_ROOT/mpl" XDG_CACHE_HOME="$PHM_RUN_ROOT/xdg"

phmforge-bench list
phmforge-bench prepare --dataset cwru --data-root "$PHM_RUN_ROOT/data/cwru"
phmforge-bench run-reference --data-root "$PHM_RUN_ROOT/data/cwru" \
  --run-dir "$PHM_RUN_ROOT/runs/cwru" --epochs 1
```

将 `cwru` 换为上表的任一名称即可运行相应数值任务，输出目录必须是新的。参考训练和冷评分的结果位于各自运行目录的 `evaluation/score.json`；一轮只用于检查链路，不能证明模型性能。大于 32 MiB 的文件采用分段下载并可恢复中断，全部下载完成后仍要通过固定 SHA-256 才会被使用；若代理或源服务器不支持范围请求，可使用下方的离线来源。

MFPT 从 [MathWorks 固定版本](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/tree/d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d) 下载 [清单](../src/phmforge_bench/resources/mfpt_manifest.json) 中的 20 个文件，逐一检查大小和 SHA-256。来源数据的许可为 CC BY-NC-SA 4.0；使用时须遵守来源条款。

C-MAPSS 从 [NASA 数据的 Zenodo 镜像](https://zenodo.org/records/15346912)下载 `CMAPSSData.zip`，检查压缩包 SHA-256 为 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`，并校验所选子集的三个原始文件。FD001 使用 80 台训练、20 台开发和 100 台测试发动机；其余子集按发动机固定划分训练／开发，使用各自原始测试发动机和标签。

FEMTO 只从原始嵌套压缩包读取 6 条学习记录及 11 条挑战观测前缀，完整后续、挑战文档和测试标签不会放入公开开发目录。时间使用名义 10 秒采集间隔，因为源日志有时钟异常；各工况只有两条学习记录。挑选轮数后最终参考模型使用两条学习记录重训，不能将重训后的学习集分数称为独立验证。

XJTU 使用 7-Zip 26.05 成功解压六分卷；系统需提供兼容的 `7zz` 或 `7z`，若不在 PATH 则传 `--extractor /path/to/7zz`。逐一校验发布者分卷 SHA-256 后，程序核对 15 只轴承的 9,216 条 32,768×2 原始振动记录并重算特征。每个工况各用 3/1/1 只轴承训练、开发及测试；每只测试轴承随机取 5 个前缀，结果按查询及轴承计分。名义时间按一次采集一分钟计算；终态损伤位置不是逐时故障标签。预处理较耗时，公开来源可能被记忆，匿名化不能杜绝该风险。

`xjtu-terminal` 只从相同分卷提取 15 个末次采集文件，损伤位置来自数据作者的拆检信息。同样按 9/3/3 只轴承分组；由于类别极少，某些类别可能没有训练样本，不可将其作为可靠现场诊断模型。批量运行会复用另一 XJTU 入口已下载的分卷。

`engine-qa` 固定数据集版本，下载并校验两份 QA 文件；通过 HTTPS 范围请求按需读取 146 条完整记录与 280 条前 60 行记录（另有一条前缀与完整记录重叠），逐条验证冻结的数组 SHA-256。该方式不下载或验证完整约 18.8 GB HDF5 的全文件哈希，远程服务器必须支持精确范围请求；如果提供完整 HDF5，则额外检查完整文件大小和发布者 SHA-256。训练／开发与原题记录不重合，但不能证明不同物理发动机独立，不同原题也可能共用记录。十周期子任务仅有 40 条训练和 48 条开发问答组合，其中一个开发联合类别没有训练样本。QA 文件和原题源身份公开，不能当成完全保密的盲测。命名传感器与匿名通道的对应关系、物理因果、故障概率校准均未得到证实。

LDM 从固定作者仓库取 14 次运行的 FFT、运行工况及温度文件，逐文件校验 SHA-256，按 7/2/5 个完整试验轴承划分。保留实际特征与采集索引，分别预测 5 个测试轴承、每只 5 个前缀，单位为剩余采集次数。B01 无预设终点、B05 主动中断、B16 的终点栏缺失，因此排除；采集间隔不规则，不能换算成未经核实的运行小时。这是作者公开的实测派生特征，未重新下载或重算 [152 GB 原波形](https://zenodo.org/records/10868257)。它与 Paderborn KAt 分类数据不同，也不能冒充原题可能指定的 Mendeley 数据。

XJTU 和 LDM 每个测试前缀都在全新的独立进程中预测；该进程只能读取当前前缀，不能读取同一轴承的其他时间点或先前查询输出。最终评分汇总后报告每只轴承及总体误差。逐条冷启动比一次批量预测慢，默认外评超时为 300 秒；低配置机器可使用 `run-reference --timeout 900` 等方式延长总限时。

准备完成后，数据根目录下的 `READY.json` 记录来源及划分；`public/` 包含训练和开发资料，`holdout/` 与 `private/` 包含外评输入或标签。不要把整个数据根目录交给正在接受评测的 Agent。参考模型训练和独立评分命令见 [README](../README.md#运行真实训练和独立评分)。

## 手动下载或离线提供

MFPT 的 `--source` 指向包含 `train_data/` 和 `test_data/` 两个子目录的仓库根目录，不是其中单个 `.mat` 文件。下面获取与自动下载一致的固定版本，并把结果准备到另一个目录：

```bash
mkdir -p "$PHM_RUN_ROOT/sources"
git clone https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data.git \
  "$PHM_RUN_ROOT/sources/mfpt_native"
git -C "$PHM_RUN_ROOT/sources/mfpt_native" checkout d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d
phmforge-bench prepare --dataset mfpt \
  --source "$PHM_RUN_ROOT/sources/mfpt_native" \
  --data-root "$PHM_RUN_ROOT/data/mfpt-offline"
```

C-MAPSS 的 `--source` 指向**原始 ZIP 文件**，不是解压后的目录：

```bash
mkdir -p "$PHM_RUN_ROOT/sources"
curl --fail --location --retry 3 \
  'https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1' \
  --output "$PHM_RUN_ROOT/sources/CMAPSSData.zip"
phmforge-bench prepare --dataset cmapss-fd001 \
  --source "$PHM_RUN_ROOT/sources/CMAPSSData.zip" \
  --data-root "$PHM_RUN_ROOT/data/cmapss-offline"
```

其他分类数据的 `--source` 指向对应固定版本的 `PdM.csv`。FEMTO 的 `--source` 指向完整 NASA 外层 ZIP，`--challenge-pdf` 指向固定版本挑战说明 PDF：

```bash
phmforge-bench prepare --dataset femto-rul --source /path/to/NASA_FEMTO.zip \
  --challenge-pdf /path/to/IEEEPHM2012-Challenge-Details.pdf \
  --data-root "$PHM_RUN_ROOT/data/femto"
```

XJTU 的 `--source` 指向包含全部六个原始 `.rar` 分卷的目录；LDM 的 `--source` 指向含 `fft/`、`op/`、`temp/` 的作者特征目录：

```bash
phmforge-bench prepare --dataset xjtu-rul --source /path/to/xjtu-volumes \
  --extractor /path/to/7zz --data-root "$PHM_RUN_ROOT/data/xjtu"
phmforge-bench prepare --dataset xjtu-terminal --source /path/to/xjtu-volumes \
  --extractor /path/to/7zz --data-root "$PHM_RUN_ROOT/data/xjtu-terminal"
phmforge-bench prepare --dataset ldm-rul --source /path/to/LDM \
  --data-root "$PHM_RUN_ROOT/data/ldm"
```

EngineMT-QA 的 `--source` 指向包含固定版 `train_qa.jsonl`、`test_qa.jsonl` 的目录；如该目录还含 `time_series_data.h5`，将校验完整文件后离线读取，否则继续在线获取并校验所需数组。批量运行可以用 `--source-map` 指定本地来源：JSON 键为 `scenario list` 中的 `prepare_dataset`，值为对应文件或目录路径；FEMTO 挑战说明另用键 `femto-challenge-pdf`。来源仍须通过固定哈希检查。

已有来源文件也可直接传入其实际路径；程序仍会校验固定哈希，文件缺失或不一致时会失败，不会自动替换为合成数据。重新准备时测试身份会随机化，因此新输出文件哈希不必与以前的准备结果相同。

v0.4 于 2026-11-02 在本服务器实际自动下载并校验 CWRU、C-MAPSS FD002 与电机振动；其他 v0.4 新增来源在相同服务器使用逐文件哈希匹配的本地原始文件完成准备。电机的在线和离线观测一致，测试身份按设计随机化。自动下载在其他网络或机器上的可用性尚需实测。

v0.5 的 LDM 另在本服务器完整自动下载了 42 个固定作者文件，并核对在线和本地来源的训练／开发数值一致；XJTU 的全部六分卷使用本地哈希匹配原始文件完成准备，公开镜像另实际验证了第一分卷连续两个 16 MiB 分段，尚未重新完整在线下载六卷。每次准备时测试身份和截断点会随机化。

## 重放历史实验需要什么

当前服务器的历史环境制品在：

```text
/srv/yyf-zyq/PHM_20261010/_shared/phmforge_research_environment_20261027_v030
```

它由 `READY.json`、`environment.json.gz`、`objects/` 和 `provenance/` 组成，固定清单哈希与字节数见 [environment.lock.json](../environment.lock.json)。仓库根目录的 `environment_store` 是仅在这台服务器上配置的本地链接，GitHub 克隆不包含该链接或制品。其他机器须先取得**整份制品**，再把路径传给 `--store`；当前没有公共下载地址。

制品不只是原始数据集，还包括各实验当时的预处理、划分、模型和程序。单独下载下面的上游数据不能自动重建这些材料，也不能保证历史任务立即可运行。完整性校验及历史重放命令见 [完整实验环境](COMPLETE_ENVIRONMENT.md)。

## 历史实验涉及的数据来源

下表给出主要来源及历史目录中的名称，方便核对研究材料；历史标签不一定等于独立数据集数量。链接用于查找来源，不表示 `prepare` 已支持该来源，也不保证从该链接单独下载后能复现封存实验。

| 历史标签或来源 | 数据及来源 | 说明 |
|---|---|---|
| `CMAPSS_FD001` | [NASA C-MAPSS](https://zenodo.org/records/15346912) | 发动机仿真；独立示例支持自动准备 |
| `MFPT` | [MFPT 原始文件](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/tree/d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d) / [PDMBench 文件](https://huggingface.co/datasets/submission096/MFPT) | 独立示例使用前者的固定文件；历史实验也使用过处理版本 |
| `CWRU` | [PDMBench CWRU](https://huggingface.co/datasets/submission096/CWRU) | 轴承诊断；原源缺少部分寿命任务所需的失效终点 |
| `ELECTRICMOTORVIBRATIONS` | [PDMBench ElectricMotorVibrations](https://huggingface.co/datasets/submission096/ElectricMotorVibrations) | 电机振动故障分类 |
| `ROTORBROKENBAR` | [PDMBench RotorBrokenBar](https://huggingface.co/datasets/submission096/RotorBrokenBar) / [IEEE DataPort 原始数据入口](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) | 电机断条数据；原始完整包的下载可能需要登录 |
| `FEMTO` | [PDMBench FEMTO](https://huggingface.co/datasets/submission096/FEMTO) | 轴承退化与寿命预测 |
| `HUST` | [PDMBench HUST](https://huggingface.co/datasets/submission096/HUST) / [轴承原数据](https://data.mendeley.com/datasets/cbv7jyx4p9/1) | 河内理工大学轴承数据；与下方 HUST 电池数据不同 |
| `PADERBORN` | [PDMBench Padeborn](https://huggingface.co/datasets/submission096/Padeborn) | 来源仓库名实际拼写为 `Padeborn` |
| `MENDELEY` | [PDMBench Mendeley](https://huggingface.co/datasets/submission096/Mendeley) | 此名称对应固定的轴承诊断来源；Mendeley 也是多个不同数据集的发布平台 |
| `XJTU`、`XJTU_TERMINAL` | [PDMBench XJTU](https://huggingface.co/datasets/submission096/XJTU) / [XJTU-SY 原始入口](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) | 轴承全寿命数据及其历史终点代理任务 |
| `UOC` | [PDMBench UoC](https://huggingface.co/datasets/submission096/UoC) | 齿轮数据；部件标签与部分原题要求不完全一致 |
| `PlanetaryG1`、`PlanetaryG2` | [PDMBench PlanetaryPdM](https://huggingface.co/datasets/submission096/PlanetaryPdM) | 同来源的不同历史派生任务，不计作两个独立数据源 |
| `ENGINE_QA_HEALTH`、`ENGINE_QA_TREND` | [EngineMT-QA](https://huggingface.co/datasets/pandalin98/EngineMT-QA) | 基于发动机仿真的问答／代理标注，不是现场物理健康真值 |
| `IMS_ENDPOINT` | [NASA PCoE 的 IMS 入口](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | 历史任务使用记录终点代理，不能直接称精确失效寿命 |
| `LDM_RTF` | [LDM 作者数据](https://zenodo.org/records/10868257) | 轴承退化数据；用于明确更换来源的历史适配任务 |
| `HUST_BATTERY` | [HUST-LFP 电池数据 v2](https://data.mendeley.com/datasets/nsc7hnsg4s/2) | 华中科技大学电池数据，与上方 HUST 轴承数据不是同一来源 |
| `COMFAULDA` | [COMFAULDA 作者数据](https://zenodo.org/records/8208454) | 实测旋转台架数据，用于后续程序复核实验 |

历史来源调查曾记录 14 个 PDMBench 文件的仓库、固定版本、文件大小和来源哈希，见 [historical_source_pins.json](historical_source_pins.json)。其中还包括只调查过的 Azure、MAFAULDA；不能据此把它们计为已完成的独立 PHMForge 任务。历史数据的来源许可、标签语义、实测或仿真性质及任务权限应逐项核查。
