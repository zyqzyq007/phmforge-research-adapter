# PHMForge 科研适配版

本仓库把 PHMForge 风格的工具服务、公开数据准备、真实模型训练、Agent 工作区和独立评分整合为一个可运行的科研基准。它是研究适配版，**不是 PHMForge 官方完整实现，也不是官方成绩**。

v0.3 增加了**实际实验环境快照**：历史任务输入、数据划分、拟合权重、Agent 程序、工具版本、训练／推理命令和结果按原文件 SHA-256 保存。v0.3.1 修复了从只读环境制品恢复文件时的权限问题，不改变该制品的内容或哈希。通过 `environment` 命令恢复和执行；大型环境制品与 Git 源码分开存储。具体内容、取得方式和运行记录见 [完整实验环境](docs/COMPLETE_ENVIRONMENT.md)。

本机已配置 `environment_store` 链接；搬迁时按 `environment.lock.json` 复制配套制品，再做完整校验。GitHub 单独克隆只包含源码与清单，尚无公开的大数据下载地址。

**给其他机器的使用者：** 仅克隆 GitHub，可以从公开数据下载并独立运行下文的 MFPT 和 C-MAPSS FD001 两个数值子任务。重放历史 51 个正常交付方向还需要单独取得约 149 GB 的环境制品；该制品目前没有公开下载地址，也不能仅靠下载原始数据自动还原。复制前请核对数据来源的许可，并避免把包含测试标签的制品交给待评测 Agent。

| 使用方式 | 内容 |
|---|---|
| `environment list / restore` | 查看实际历史任务、恢复冻结的代码、数据和运行环境 |
| `environment replay` | 重新执行历史训练或预测命令；预测与历史输出逐项核对 |
| `environment agent` | 在历史任务的训练／开发资料上启动新的 coding Agent 复核，提交后再外评 |
| `prepare / run-reference` | 下列两个从公开下载开始的独立示例；其划分不等同于全部历史实验 |

完整环境保留原 75 题的覆盖台账：**55 个数值方向实际尝试、51 个正常交付外评、4 个预算未交付；另 16 个原目标不具备条件、4 个明确合成题排除**。后续程序复核、改目标实验和失败记录另列。完整保存实验环境，不等于原版 75 题全部科学有效，也不抹去旧实验的权限和标签限制。

从公开下载开始可独立运行的两个**数值子任务**：

| 任务 | 数据来源 | 任务与外部评分 | 数据性质 |
|---|---|---|---|
| `pdm_fault_008` | MFPT | 根据振动特征识别正常、内圈故障和外圈故障；按样本及采集记录评分 | 台架实测 |
| `pdm_rul_001` | NASA C-MAPSS FD001 | 在测试发动机的末次观测时预测剩余循环数；计算 MAE、RMSE | 仿真，不是现场实测 |

每个示例需要的数据、自动和手动下载方法、固定来源版本，以及历史实验涉及的其他数据见 [数据集与下载说明](docs/DATASETS.md)。这两个示例可以分别下载数据运行；完整历史重放仍需单独的环境制品。

这两个流程不涵盖相应原题中的全部自然语言分析和维修决策。`phmforge-bench list --all` 查看原题与独立示例清单；`phmforge-bench environment list --store /path/to/environment` 查看完整历史环境。不要把不同划分、阶段或重复会话相加成独立任务数。

具体安装、真实训练、MCP 和隔离检查见 [验证记录](docs/VALIDATION.md)。

## 安装

需要 **Linux、Python 3.12、bubblewrap (`bwrap`)**、网络访问数据源和软件包源，以及 CPU 运行所需的磁盘空间。安装系统包的方法依发行版而异，例如 Debian/Ubuntu 可以运行 `sudo apt install bubblewrap python3.12-venv`。容器或服务器需要允许普通用户创建 bubblewrap 使用的命名空间；不可用时评分会明确失败。

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

下载外可指定已有数据：MFPT 使用 `--source /path/to/mfpt_native` 指向含固定 20 个原始 `.mat` 文件的目录；FD001 使用 `--source /path/to/CMAPSSData.zip`。准备过程会检查来源哈希，错误或不完整时失败，不会用合成数据替代。

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
- 每次准备数据时会为测试窗口／发动机生成随机身份，映射保留在私有目录，防止直接通过公开原始编号查标签。同一份准备好的数据可重复评分；重新准备时身份及文件哈希会变化，特征、划分和真值不变。测试资料虽在本流程中隔离，这些都是公开数据，预训练模型或使用者仍可能已经见过原数据或标签，也可能根据数据特征识别样本；这些限制需要在研究报告中说明。
- 所有运行材料均为科研评估，不提供已验证的维护、合规或安全决策。

## 上游和许可证

原始 PHMForge 固定提交的 [LICENSE](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/LICENSE) 写为 MIT，同一提交的 [README](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/README.md) 写为 Apache 2.0。原始声明保存在 `third_party/`；上游差异尚未厘清，本仓库新增代码尚未指定许可证。各数据集须分别遵守其来源条款。
