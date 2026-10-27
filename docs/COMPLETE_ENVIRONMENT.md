# 实际实验环境副本

此环境保存 A238–A467 的实际研究材料，包括原数据与派生数据、训练／开发／外部划分、拟合权重、程序、工具、提示词、执行命令、模型交互记录、预测、评分与反证。每份历史归档仍独立标识，字节内容按 SHA-256 去重。追加的迁移入口不修改历史程序、标签或成绩。

## 组成与边界

- Git 仓库：安装包、可迁移执行入口、测试、场景台账、环境摘要，以及 `research_snapshot/` 中逐字节保存的实际科研主脚本。
- 环境制品：`READY.json`、`environment.json.gz`、`objects/`、`provenance/`。需要整体取得，不能只复制 README 或索引。大小、固定 SHA 和本机位置记录在仓库根目录的 `environment.lock.json`。
- 保存实际 CPython 3.12 与当时安装的数值依赖；主机 Linux 内核、glibc 和 bubblewrap 仍由目标机器提供。不是整个操作系统磁盘镜像。
- 不含 API 凭据。模型服务需要自己的可用接口；历史模型响应不会被替代成新的 Agent 调用。
- 历史数据、日志和第三方模型分别受原有条款约束。Git 源码不自动授予这些材料的公开再分发权；环境制品当前按本地科研复现材料提供。

原始 75 题有完整状态台账。真实数值方向、改目标适配、代理标注任务、预算失败、环境问题、后验诊断、方法失败分别保留。Engine 的作者 QA 一致性不是物理风险真值；NASA 仿真与台架实测仍分别标记。旧实验已知的信息暴露不会因打包而消失。

## 校验与列出任务

按 README 安装后，把路径换成环境制品实际位置。所有新运行目录必须与只读环境分开。

也可直接使用快照中保存的 Python 与依赖，无需重新解析 pip 版本。需要系统 `python3`（3.10 或更高版本）先恢复运行时；缓存放在环境制品之外：

```bash
python3 scripts/frozen_runtime.py --store /path/to/environment \
  --cache /path/to/runtime-cache -- environment list --store /path/to/environment
```

后续把 `phmforge-bench` 替换为以上命令中 `--` 之前的部分即可。此方式与重新从 PyPI 安装有不同的用途：它保留实际使用过的已安装依赖文件，不把当前依赖解析结果称为历史环境。

```bash
export PHM_ENV_STORE=/path/to/environment
export PHM_REPLAY_ROOT="$HOME/phmforge-replay"
mkdir -p "$PHM_REPLAY_ROOT/tmp"
export TMPDIR="$PHM_REPLAY_ROOT/tmp" TMP="$PHM_REPLAY_ROOT/tmp" TEMP="$PHM_REPLAY_ROOT/tmp"

phmforge-bench environment verify --store "$PHM_ENV_STORE" --full
phmforge-bench environment list --store "$PHM_ENV_STORE" --records > "$PHM_REPLAY_ROOT/catalog.json"
```

`--full` 重新计算全部对象哈希，数据量大时耗时较长；不带此参数只检查清单哈希、对象存在性和大小。`list` 列出历史 coding 会话与原生程序复核会话，不能把会话数当独立数据集数。

## 真实预测重放

下面的 HUST 轴承案例使用当时的真实权重、原测试观测及原提交程序，重新启动无网络隔离进程；只把该条历史命令所需文件挂进去。

```bash
phmforge-bench environment replay --store "$PHM_ENV_STORE" \
  --record runs/20261023T105942Z_coverage306/h/evaluation/detail306_h_initial/final_command.json \
  --out "$PHM_REPLAY_ROOT/hust-bearing"
```

新结果写到 `output/predictions.json`、`stdout.log`、`stderr.log` 和 `RESULT.json`。默认逐项对比历史预测，浮点相对／绝对容差为 `1e-6`；程序失败、缺少依赖、缺少历史输出或预测不一致会报错。**重放成功不是预测正确或方法胜出。** 程序的原评分与限制仍保留在对应归档中。

`--no-compare` 只关闭历史预测等价判断，不会生成预测占位值。训练命令通过各会话的 `full_fit/PROCESS.json` 执行；如果记录要求参数拟合，入口检查新产生的训练审计，不能用保存的训练日志代替新拟合。

逐条查询隔离的 XJTU 等任务应使用 `environment replay-case --case <目录中的案例ID> --out <新目录> --store "$PHM_ENV_STORE"`，一次重放该案例的全部最终查询。部分旧评测会删除临时程序副本；本制品按原复制规则从封存提交恢复这些挂载，保留源文件哈希，修复记录随验证材料保存。

## 新的 coding Agent 复核

有明确训练／开发隔离声明、唯一外部执行命令的旧 coding 案例可恢复为新工作区。入口拒绝把带有最终 `test.npz` 的历史开发挂载用于新复核。

```bash
phmforge-bench environment workspace --store "$PHM_ENV_STORE" \
  --case runs/20261023T105942Z_coverage306/h/detail306_h_initial \
  --out "$PHM_REPLAY_ROOT/hust-inputs"

# DEEPSEEK_API_KEY 仅由外部进程环境提供，不能写进文件。
phmforge-bench environment agent --store "$PHM_ENV_STORE" \
  --case runs/20261023T105942Z_coverage306/h/detail306_h_initial \
  --out "$PHM_REPLAY_ROOT/hust-review" \
  --cli /absolute/path/to/codex \
  --model-base-url https://your-responses-provider.example/v1 \
  --timeout 1200 --max-requests 40
```

Agent 从历史提交程序开始，可保留、修改或替换，使用同版本公共数据与工具。最终观测和标签只由提交后的外评进程读取。分类／RUL 子任务重新计算数值指标；完整自然语言分析和维护建议仍需原任务专用评分器，不会自动被计为通过。

这是一轮**新的历史程序复核**，不声称随机 Agent 会重现原对话或从零开发行为。`--smoke` 仅检查 MCP、模型接口、提交与外评接通。后期的原生 broker 方法会话提供冻结源码、配置、候选程序和各次 fit/predict 重放记录；没有伪装成同一套旧 MCP 工作流。

## 恢复源代码或搬迁

```bash
phmforge-bench environment restore --store "$PHM_ENV_STORE" \
  --path project/scripts --out "$PHM_REPLAY_ROOT/research-scripts"
phmforge-bench environment restore --store "$PHM_ENV_STORE" \
  --path runs/20261027T104923Z_rebuild462 --out "$PHM_REPLAY_ROOT/rebuild462"
```

可将整个环境制品目录复制到另一台 Linux x86-64 机器，再执行 `verify --full`。重放入口通过清单转换历史宿主路径，不依赖原服务器同名目录。恢复的原始科研脚本保留原来的绝对路径和版本；直接调用它们需要相应配置迁移，优先用已捕获命令的 `replay` 入口。

环境制品含外部标签和已公开结果，**不要整体挂载到被评测 Agent**。`workspace` 只准备资料；自行启动 Agent 时仍须设置隔离。内置 `environment agent` 才负责开发与外评的进程边界。

本轮实际验证结果由 `docs/VALIDATION.md` 和 `environment.lock.json` 对应的验证报告给出；未经执行的历史命令只标为保存，不标为已复现。
