# PHMForge 科研适配版

本仓库把 PHMForge 风格的工具服务、公开数据准备、真实模型训练、Agent 工作区和独立评分整合为一个可运行的科研基准。它是研究适配版，**不是 PHMForge 官方完整实现，也不是官方成绩**。

v0.7 保留现有 55 个公开数据数值适配入口，并为其余 20 个题号增加明确的可用性说明和部分新入口。数据、划分和模型可以共用：**数值流程跑通不等于完整回答原题，也不是 55 次独立模型实验**。新增 IMS 和 UoC 后共有 19 个可单独准备的数值任务类型，不需要 149 GB 历史快照。数据范围、下载大小与验证情况见 [数据集说明](docs/DATASETS.md) 和 [验证记录](docs/VALIDATION.md)。

| 原题号的当前数值适配 | 数量 | 含义 |
|---|---:|---|
| 原有入口 | 55 | 数值子任务；共享 13 类模型，原题的费用、安全等部分可能缺失 |
| 新增相关来源 | 7 | 可做部分数值任务，其中两题使用 Paderborn LDM 而非原先的 Paderborn 分类数据 |
| 显式替代数据 | 9 | 使用与原题不同的数据源，仅可作为替代研究；须主动选择 |
| 暂不可运行 | 4 | 两个行星齿轮题缺少原题所需部件标签，两个发动机题是无对应观测和真值的合成模板 |

上述 7 和 9 都是部分数值适配，不是另外 16 道原题的完整复现；合计 75 是原题目录，不是 75 道全部解决的题。通过 `scenario list --all` 可查看全部题号与状态，逐题对应关系见 [75 个题号的适配范围](docs/SOURCE75_SCOPE.md)。

v0.3 的**历史实验环境快照**另外保存原输入、划分、拟合权重、程序和日志，使用 `environment` 命令恢复或重放；它不属于从数据开始运行新任务的前提。该约 149 GB 的快照目前没有公开下载地址。见 [完整实验环境](docs/COMPLETE_ENVIRONMENT.md)。

**给其他机器的使用者：** 克隆后安装依赖，按需下载来源数据；数据和依赖没有放在 Git 仓库中。模型只保证实际训练与执行，不保证准确。跨机器安装、网络和内核兼容性仍需在使用者机器上验证；Agent 运行另需兼容的模型接口。

| 使用方式 | 内容 |
|---|---|
| `environment list / restore` | 查看实际历史任务、恢复冻结的代码、数据和运行环境 |
| `environment replay` | 重新执行历史训练或预测命令；预测与历史输出逐项核对 |
| `environment agent` | 在历史任务的训练／开发资料上启动新的 coding Agent 复核，提交后再外评 |
| `prepare / run-reference` | 从公开源准备数据并执行数值训练及评分；划分不等同于全部历史实验 |

完整环境保留原 75 题的覆盖台账：**55 个数值方向实际尝试、51 个正常交付外评、4 个预算未交付；另 16 个原目标不具备条件、4 个明确合成题排除**。后续程序复核、改目标实验和失败记录另列。完整保存实验环境，不等于原版 75 题全部科学有效，也不抹去旧实验的权限和标签限制。

从公开下载开始可独立运行的**数值子任务**：

| 任务 | 数据来源 | 任务与外部评分 | 数据性质 |
|---|---|---|---|
| `pdm_fault_008` | MFPT | 根据振动特征识别正常、内圈故障和外圈故障；按样本及采集记录评分 | 台架实测 |
| `pdm_rul_001` | NASA C-MAPSS FD001 | 在测试发动机的末次观测时预测剩余循环数；计算 MAE、RMSE | 仿真，不是现场实测 |
| `research_fd002_rul`–`research_fd004_rul` | NASA C-MAPSS FD002–FD004 | 分别预测各子集发动机的剩余循环数 | 仿真 |
| `research_femto_bearing_rul` | NASA PRONOSTIA/FEMTO | 预测 11 条挑战轴承前缀的剩余秒数；另报已知有歧义的标注 | 台架实测 |
| `research_xjtu_bearing_rul` | XJTU-SY | 用当前信号特征预测至末次采集的名义分钟数；按整轴承分组 | 台架实测 |
| `pdm_fault_002_terminal_adaptation` | XJTU-SY | 根据末次振动记录预测拆检损伤位置；没有早期／晚期故障严重度标签 | 台架实测，只有 15 只轴承 |
| `research_engine_qa_component_health`、`research_engine_qa_ten_cycle_trend` | EngineMT-QA | 覆盖 28 条发动机方向的发布者问答标注；不等于物理健康真值 | 仿真及问答标注 |
| `research_ldm_bearing_rul` | Paderborn LDM 作者特征 | 预测到记录末端的剩余采集次数，非小时；更换来源的研究适配 | 作者公开的台架实测派生特征 |
| `research_ims_recording_endpoint` | NASA IMS | 预测剩余采集次数至记录结束；不是经核实的物理失效时间 | 三次台架实验，测试四个传感器通道共用一次实验终点 |
| `research_uoc_pinion_state_classification` | UoC | 九种输入小齿轮状态分类；原始采集及硬件独立性未核实 | 台架数据，样本级探索划分 |
| `research_cwru_fault_classification` | CWRU | 原始特征表类别预测；分组由连续段推断 | 台架数据 |
| `research_electric_motor_fault_classification`、`research_rotor_fault_classification` | ElectricMotorVibrations、RotorBrokenBar | 电机振动和转子数据源标签分类 | 台架数据 |
| `research_hust_bearing_fault_classification`、`research_paderborn_bearing_fault_classification`、`research_mendeley_bearing_fault_classification` | HUST、Padeborn、Mendeley | 按来源标签开展轴承故障分类 | 台架数据 |

每个任务需要的数据、自动和手动下载方法、固定来源版本，以及历史实验涉及的其他数据见 [数据集与下载说明](docs/DATASETS.md)。

这些数值流程不等于 75 道原题全部自然语言分析、维修或安全决策。`phmforge-bench list --all` 分别展示原题目录和可独立运行的数值任务；不要把多种数据划分、阶段或重复会话相加成独立原题。

## 运行现有 55 个方向

原题包括费用、设备安全、法规合规和多模态比较等要求；公开数据没有相应可靠真值的部分会标为未知。发动机方向的“失效概率区间”只是数据集的问答类别，不是已校准的实际失效概率。每个方向的原题、数据集、可计算部分和限制可直接查询：

```bash
phmforge-bench scenario list
phmforge-bench scenario list --all
phmforge-bench scenario list --task pdm_fault_002
```

以下命令会按需下载并准备全部 55 个入口依赖的数据，并逐项执行；它们实际共享 13 类数值数据及相应模型。默认只使用 CPU。全部原始下载约 15 GB，解压及运行需要更多空间，网络请求和 XJTU 解压可能耗时较长。XJTU 需要 `7zz`/`7z`：

```bash
export PHM_RUN_ROOT="$HOME/phmforge-runs"
mkdir -p "$PHM_RUN_ROOT/tmp" "$PHM_RUN_ROOT/mpl" "$PHM_RUN_ROOT/xdg"
export TMPDIR="$PHM_RUN_ROOT/tmp" TMP="$PHM_RUN_ROOT/tmp" TEMP="$PHM_RUN_ROOT/tmp"
export MPLCONFIGDIR="$PHM_RUN_ROOT/mpl" XDG_CACHE_HOME="$PHM_RUN_ROOT/xdg"

phmforge-bench scenario run-all --prepare-missing \
  --data-home "$PHM_RUN_ROOT/data-55" --run-dir "$PHM_RUN_ROOT/runs/all-55" \
  --epochs 10 --timeout 900
```

进度及失败原因写入 `runs/all-55/BATCH.json`，各题的 `SCENARIO.json` 和 `evaluation/score.json` 保存在其子目录；有失败时命令返回非零状态。输出目录必须不存在，避免覆盖。网络失败不会用虚构数据替换；修复数据源后可指定新运行目录重试，已准备好的数据会复用。只运行指定方向时，可重复添加 `--task`：

```bash
phmforge-bench scenario run-all --prepare-missing \
  --data-home "$PHM_RUN_ROOT/data-55" --run-dir "$PHM_RUN_ROOT/runs/two-cases" \
  --task engine_health_combined_001 --task pdm_fault_008 --epochs 2
```

也可先单独准备、再运行一个方向；发动机 `engine-qa` 一次准备 28 个原题记录，其他方向的数据准备名称见 `scenario list`：

```bash
phmforge-bench prepare --dataset engine-qa --data-root "$PHM_RUN_ROOT/data-55/engine-qa"
phmforge-bench scenario run --task engine_health_combined_001 \
  --data-root "$PHM_RUN_ROOT/data-55/engine-qa" \
  --run-dir "$PHM_RUN_ROOT/runs/engine-case" --epochs 10
```

这些数据、划分和参考模型不等同于此前 4B/9B 历史实验的代码及成绩。相同数据集的多个入口可能得到同一组预测，不能据此声称完成了相互独立的 55 次实验。

## 其余 20 个题号

`scenario list --all` 逐题显示数据源关系和限制；`scenario list --task pdm_rul_002` 可看完整原题及为什么更换数据。运行原有 55 个加 7 个相关来源入口，显式添加 `--include-additional`；如果也接受 9 个不同来源的替代研究，再改用 `--include-substitutions`。默认批处理仍只选择原来的 55 项。新增相关来源要求约 1.08 GB IMS 压缩包、约 74 MB UoC 文件以及约 158 MB LDM 特征文件，还需额外解压空间。IMS 和 XJTU 都需要可用的 7-Zip：

```bash
phmforge-bench scenario run-all --include-additional --prepare-missing \
  --data-home "$PHM_RUN_ROOT/data-75" --run-dir "$PHM_RUN_ROOT/runs/related-62" \
  --epochs 2 --timeout 900

# 显式同意原题所称来源被替换后，再运行可用的 71 个数值入口
phmforge-bench scenario run-all --include-substitutions --prepare-missing \
  --data-home "$PHM_RUN_ROOT/data-75" --run-dir "$PHM_RUN_ROOT/runs/with-substitutions-71" \
  --epochs 2 --timeout 900
```

也可以通过 `--task <题号>` 明确单独选择一个可用的替代来源入口。结果的 `source_relation`、限制和缺失字段会明确写入 `SCENARIO.json`。IMS 的目标是距公开记录末端的采集次数，不是已核实的失效时刻；其最终实验只有四个轴承通道，不足以输出五台设备的排名。预测的低 RUL 也不能直接当成经验证的物理安全风险。

具体安装、真实训练、MCP 和隔离检查见 [验证记录](docs/VALIDATION.md)。

## 安装

需要 **Linux、Python 3.12、bubblewrap (`bwrap`)**、网络访问数据源和软件包源，以及 CPU 运行所需的磁盘空间。XJTU 另需可解压多卷 RAR 的 `7zz` 或 `7z`，本次使用 7-Zip 26.05。安装系统包的方法依发行版而异，例如 Debian/Ubuntu 可以运行 `sudo apt install bubblewrap python3.12-venv`。容器或服务器需要允许普通用户创建 bubblewrap 使用的命名空间；不可用时评分会明确失败。

```bash
git clone https://github.com/zyqzyq007/phmforge-research-adapter.git
cd phmforge-research-adapter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
phmforge-bench list
```

`requirements.txt` 固定主要依赖及 CPU 版 PyTorch；传递依赖仍由安装时的软件源解析，并非包含全部哈希的锁文件。Git 仓库**不捆绑**原始数据、大型依赖、模型权重、实验结果或密钥；下面的准备命令会从固定来源下载并校验数据。离线使用时，可以先另行取得对应来源文件，再传 `--source`。

如果使用搬迁过的独立 Python，创建虚拟环境时出现 `encodings` 或解释器前缀错误，应使用 Python 的实际安装路径重新创建环境。下载依赖出现哈希不匹配时，保留校验并从官方 PyPI 源重新取得失败的依赖，再继续安装；不要关闭校验或更换版本来掩盖下载错误。

例如，如果 `python3.12` 是指向搬迁后解释器的链接，可以用其实际路径在新目录创建环境：

```bash
PHM_PYTHON_312="$(readlink -f "$(command -v python3.12)")"
"$PHM_PYTHON_312" -m venv .venv-corrected
source .venv-corrected/bin/activate
```

如果下载时出现代理超时或 HTTPS 证书主机名不匹配，请先检查当前网络和代理；在支持直接访问官方源的网络中，可针对安装命令暂时取消代理环境变量。仍须保持 TLS 和文件哈希校验，或者按[数据说明](docs/DATASETS.md#手动下载或离线提供)提供已校验的本地源文件。

## 运行真实训练和独立评分

所有目录都可以改成自己的路径。准备数据、参考模型输出和评分结果应使用不同目录。首次运行会访问数据源。

```bash
export PHM_RUN_ROOT="$HOME/phmforge-runs"
mkdir -p "$PHM_RUN_ROOT/tmp" "$PHM_RUN_ROOT/mpl" "$PHM_RUN_ROOT/xdg"
export TMPDIR="$PHM_RUN_ROOT/tmp" TMP="$PHM_RUN_ROOT/tmp" TEMP="$PHM_RUN_ROOT/tmp"
export MPLCONFIGDIR="$PHM_RUN_ROOT/mpl" XDG_CACHE_HOME="$PHM_RUN_ROOT/xdg"

# 实测 MFPT：下载并校验 20 条原始记录，制作训练、开发和测试划分
phmforge-bench prepare --dataset mfpt --data-root "$PHM_RUN_ROOT/data/mfpt"
phmforge-bench run-reference --data-root "$PHM_RUN_ROOT/data/mfpt" \
  --run-dir "$PHM_RUN_ROOT/runs/mfpt-reference" --epochs 10

# NASA 仿真 FD001：下载、校验原始压缩包和三个 FD001 文件
phmforge-bench prepare --dataset cmapss-fd001 --data-root "$PHM_RUN_ROOT/data/cmapss"
phmforge-bench run-reference --data-root "$PHM_RUN_ROOT/data/cmapss" \
  --run-dir "$PHM_RUN_ROOT/runs/cmapss-reference" --epochs 10
```

输出目录必须尚不存在，防止意外覆盖。`run-reference` 实际拟合神经网络，记录优化器、非零梯度和参数变化，再调用隔离进程生成预测并独立评分；冷评分时不挂载训练或开发数据。结果位于 `evaluation/score.json`，执行状态和提交文件哈希分别位于 `RUN.json`、`evaluation/STATUS.json` 与 `evaluation/submission_manifest.json`。分数随数据划分、训练轮数和配置变化，不应当作为已验证的通用模型表现。

更多任务只需将 `--dataset` 替换为数据清单中的名称；也可使用 `--source` 指定对应的本地原始文件。准备过程会检查来源哈希，错误或不完整时失败，不会用合成数据替代。较大文件支持断点续传并在完成后校验 SHA-256。

## 运行 Agent

可使用内置流程对一个 coding Agent 做端到端实验。需要原生 Codex CLI、支持 **Responses API** 的模型接口和 `deepseek-flash` 模型。设置密钥时不要把密钥写进仓库、命令记录或实验文件：

在 Linux x86-64 上，可以使用以下固定版本的原生 CLI；本仓库已实测 Codex CLI 0.165.0。若已安装兼容的原生 CLI，也可将其绝对路径传给 `--cli`。

```bash
export PHM_CODEX_DIR="$PHM_RUN_ROOT/tools/codex-0.165.0"
mkdir -p "$PHM_CODEX_DIR"
curl --fail --location --retry 3 \
  'https://github.com/openai/codex/releases/download/rust-v0.165.0/codex-x86_64-unknown-linux-musl.tar.gz' \
  --output "$PHM_CODEX_DIR/codex.tar.gz"
printf '%s  %s\n' 'be635d3486a5bd060d0fff582ae6d30160280c31344dd85719707fa920e187f0' \
  "$PHM_CODEX_DIR/codex.tar.gz" | sha256sum --check
tar -xzf "$PHM_CODEX_DIR/codex.tar.gz" -C "$PHM_CODEX_DIR"
"$PHM_CODEX_DIR/codex-x86_64-unknown-linux-musl" --version
```

然后使用自己的接口密钥运行：

```bash
# 在安全的交互式终端中读取，不显示输入；使用自己的模型接口凭据
read -rs -p 'DEEPSEEK_API_KEY: ' DEEPSEEK_API_KEY; printf '\n'
export DEEPSEEK_API_KEY

phmforge-bench agent --data-root "$PHM_RUN_ROOT/data/mfpt" \
  --run-dir "$PHM_RUN_ROOT/runs/mfpt-agent" \
  --cli "$PHM_CODEX_DIR/codex-x86_64-unknown-linux-musl" \
  --model-base-url https://api.deepseek.com/
```

针对 55 个方向中的具体原题，可在准备好对应数据后使用 `scenario agent`。该命令将原题与适配限制放入 Agent 工作区，仍然只对数值子任务做外部评分，使用接口会产生费用：

```bash
phmforge-bench scenario agent --task engine_health_combined_001 \
  --data-root "$PHM_RUN_ROOT/data-55/engine-qa" \
  --run-dir "$PHM_RUN_ROOT/runs/engine-agent" \
  --cli "$PHM_CODEX_DIR/codex-x86_64-unknown-linux-musl"
```

也可以用 `phmforge-bench scenario workspace --task <任务编号> --data-root <对应数据目录> --out <新工作区>` 为其他 Agent 生成题目及训练／开发数据。调用方须自行隔离该 Agent。

Agent 可访问训练/开发资料、22 个 PHMForge 风格的 MCP 工具及一份实际训练的初始程序，可以保留或替换该程序。它在隔离环境中开发；指定接口的凭据只在外部网关，测试观测和标签在提交前不挂载。Agent 正常结束并确认收到真实模型响应后，系统冻结 `/submission`，在独立无网络进程中对测试样本推理评分。`PROTOCOL.json` 固定 CLI 版本、模型和预算；`model_gateway/agent/` 记录实际返回模型及用量。可通过 `--max-requests`、`--max-input-tokens`、`--max-output-tokens`、`--max-output-per-request` 和 `--timeout` 设置预算；输入 token 阈值是已完成请求的准入门槛，最后一个已在途请求可能跨过阈值。调用模型会产生接口费用。

该接口需要所选服务确实支持 Responses API 和指定模型，不同供应商的兼容性需要自行验证。日志和模型请求可能含训练数据及实验内容，请保存在 Git 仓库外并审查后分享。隔离依赖 Linux 内核和 bubblewrap；它降低常见的文件和网络泄露风险，不是针对恶意内核或公开数据记忆的完整安全保证。

只需检查连接和评分流程时，可在 `agent` 命令末尾加 `--smoke --timeout 600 --max-requests 40`。该模式要求 Agent 保留初始程序并避免大规模搜索，得到的结果只用于软件接通检查，不作为 Agent 能力对比。一次 10 请求的接通检查曾因预算耗尽而未提交；实际所需预算会随模型和任务变化。

也可自行接入其他 Agent：

```bash
phmforge-bench workspace --data-root "$PHM_RUN_ROOT/data/mfpt" \
  --out "$PHM_RUN_ROOT/workspaces/mfpt"
# 按 TASK.md 在安全隔离环境中生成独立目录 /path/to/submission，包含 solution.py 和模型文件
phmforge-bench evaluate --data-root "$PHM_RUN_ROOT/data/mfpt" \
  --submission /path/to/submission --out "$PHM_RUN_ROOT/runs/my-evaluation"
```

`workspace` 只复制训练和开发材料，**自行启动的 Agent 仍需要用户自行隔离**；单靠目录分离无法阻止在宿主机运行的程序读取其他文件。提交程序必须输出与测试身份一一对应的预测，缺失、重复、未知身份、非有限数值等均会被拒绝。

## 工具服务

```bash
phmforge-bench serve --data-root "$PHM_RUN_ROOT/data/mfpt" \
  --out "$PHM_RUN_ROOT/runs/mcp-artifacts" --transport mcp
```

支持 MCP stdio 和 `--transport jsonl`。这是独立进程服务，本身不提供文件系统隔离，使用内置 `agent` 流程时才会被放入 Agent 隔离环境。服务可发现 22 个工具名称；能否执行取决于数据和任务。缺少可靠物理依据的健康评估、时序诊断、维护优化、合规、安全建议和网页搜索明确返回 `unavailable`，不生成伪造结论。`load_ground_truth` 只提供明确选择的训练或开发标签；最终测试标签由外部评分器读取。

## 数据、划分和局限

- [MathWorks MFPT 原始数据](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)为 **CC BY-NC-SA 4.0**：仅限非商业用途，商业用途需要另行获得许可。下载清单固定来源版本和各文件 SHA-256。划分按采集记录进行：11 条训练、3 条开发、6 条测试；同一记录的窗口不会跨集合，但不能据此声称设备独立。
- [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) 是仿真数据，使用 [NASA progpy 指向的数据镜像](https://github.com/nasa/progpy/blob/master/src/progpy/datasets/nasa_cmapss.py)。FD001 按 80 个训练、20 个开发和 100 个测试发动机划分。参考模型仅用训练发动机拟合和归一化、用开发发动机选择；训练 RUL 封顶 125，测试真值不封顶。公开训练文件包含完整训练及开发轨迹，开发集可用于调参，不是盲测集。
- 每次准备数据时会为测试窗口／发动机生成随机身份并打乱测试样本或整台发动机的顺序，映射保留在私有目录，以减少直接通过公开原始编号和顺序查标签。同一份准备好的数据可重复评分；重新准备时身份、顺序及文件哈希会变化，特征、划分和真值不变。测试资料虽在本流程中隔离，这些都是公开数据，预训练模型或使用者仍可能已经见过原数据或标签，也可能根据数据特征识别样本；这些限制需要在研究报告中说明。
- 所有运行材料均为科研评估，不提供已验证的维护、合规或安全决策。

## 上游和许可证

原始 PHMForge 固定提交的 [LICENSE](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/LICENSE) 写为 MIT，同一提交的 [README](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/README.md) 写为 Apache 2.0。原始声明保存在 `third_party/`；上游差异尚未厘清，本仓库新增代码尚未指定许可证。各数据集须分别遵守其来源条款。
