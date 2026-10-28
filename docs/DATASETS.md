# 数据集与下载说明

先选择运行方式，所需数据不同：

| 目标 | 必需数据 | 如何取得 |
|---|---|---|
| 独立轴承故障分类示例 `pdm_fault_008` | MFPT 轴承振动的 20 个原始 `.mat` 文件，合计约 42.6 MB | `prepare --dataset mfpt` 自动下载；也可手动提供指定版本 |
| 独立发动机寿命示例 `pdm_rul_001` | NASA C-MAPSS `CMAPSSData.zip`，约 12.4 MB；只使用 FD001 的三个文件 | `prepare --dataset cmapss-fd001` 自动下载；也可手动提供压缩包 |
| 重放或复核我们以前的实际实验 | 独立的历史环境制品，约 149 GB，含数据、划分、程序、权重、依赖和记录 | 当前仅在本服务器保存，**没有公开下载地址**；须取得整份制品 |

MFPT 是实验台实测振动；C-MAPSS 是 NASA 发动机仿真数据。两个示例均为原题的**数值子任务研究适配**，不是完整原题评分。`prepare` 目前只支持上述两个数据集；历史来源表中的其他数据并不能直接传给它使用。

## 自动下载两个独立示例

先按 [README](../README.md#安装) 安装本仓库。任选需要的示例执行；`--data-root` 须指向未完成准备的新目录：

```bash
export PHM_RUN_ROOT="$HOME/phmforge-runs"
mkdir -p "$PHM_RUN_ROOT/tmp" "$PHM_RUN_ROOT/mpl" "$PHM_RUN_ROOT/xdg"
export TMPDIR="$PHM_RUN_ROOT/tmp" TMP="$PHM_RUN_ROOT/tmp" TEMP="$PHM_RUN_ROOT/tmp"
export MPLCONFIGDIR="$PHM_RUN_ROOT/mpl" XDG_CACHE_HOME="$PHM_RUN_ROOT/xdg"

phmforge-bench prepare --dataset mfpt --data-root "$PHM_RUN_ROOT/data/mfpt"
phmforge-bench prepare --dataset cmapss-fd001 --data-root "$PHM_RUN_ROOT/data/cmapss"
```

MFPT 从 [MathWorks 固定版本](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/tree/d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d) 下载 [清单](../src/phmforge_bench/resources/mfpt_manifest.json) 中的 20 个文件，逐一检查大小和 SHA-256。来源数据的许可为 CC BY-NC-SA 4.0；使用时须遵守来源条款。

C-MAPSS 从 [NASA 数据的 Zenodo 镜像](https://zenodo.org/records/15346912)下载 `CMAPSSData.zip`，检查压缩包 SHA-256 为 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`，并分别校验 `train_FD001.txt`、`test_FD001.txt` 和 `RUL_FD001.txt`。它使用 80 台训练、20 台开发和 100 台测试发动机。

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

已有来源文件也可直接传入其实际路径；程序仍会校验固定哈希，文件缺失或不一致时会失败，不会自动替换为合成数据。重新准备时测试身份会随机化，因此新输出文件哈希不必与以前的准备结果相同。

2026-10-28 在本服务器分别执行两个数据集的自动下载和已有文件导入，四条准备流程均通过；两种方式得到的观测和标签一致，测试身份按设计各自随机化。这是数据准备验证，不是新的模型训练结果。

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
