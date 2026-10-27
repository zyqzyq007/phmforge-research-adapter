# PHMForge 真实计算研究适配版（研究预览）

这是 PHMForge 风格工具的研究适配代码，不是 PHMForge 官方仓库的完整 fork，也不是 75 道原题的完整可复现版本。本仓库作为研究预览公开，许可证、数据准备和复现仍存在下文所列限制。

## 包含的内容

- `src/verified_phmforge.py`：经研究修改的工具实现，包含真实训练、预测、开发集验证和部分数据访问限制。
- `src/real_*_model.py`：故障分类、发动机和轴承寿命等数值后端。部分后端需要分别准备对应的研究适配数据与配置。
- `src/verified_benchmark_worker.py`：MCP / JSONL 工具服务入口。
- `src/verified_benchmark_scoring.py`：数值指标和身份对齐工具。
- `metadata/`：A246 运行包使用的工具目录和数据注册表快照，供检查数据契约；运行时会从所指定的数据目录读取这两个文件。
- `third_party/`：固定上游提交的许可证原文和声明差异说明。
- `requirements-observed.txt`：本机已验环境中的主要包版本，不是经过干净环境安装验证的依赖锁文件。

此处的 `src/` 复制自研究项目 `scripts/` 中较晚的源码快照；它比 A246/v4 共享运行包内的适配器包含更多数据格式和实验权限控制。它们不能视为完全同一个版本。

## 可用范围与限制

2026-10-27 本机对本目录中的代码进行了实际 MCP 冒烟测试：发现 22 个工具；使用 MFPT 训练集运行 1 轮 MLP，共 17 次优化器更新和 17 次非零梯度更新；570 个开发样本的推理和逐样本准确率重算一致。开发准确率为 0.5，它用于训练期选择检查，不是盲测成绩，也不能代表模型效果或其他数据集的可用性。

另一个独立测试在现有 A246/v4 隔离运行包中完成同样链路。旧实验还验证过 CWRU、MFPT、电机振动和断条转子的真实数值后端。C-MAPSS 是 NASA 仿真数据，不能描述为现场实测；不同历史实验分别使用了其他准备器和数据，不能把它们统称为本目录开箱即用的全部能力。

`assess_component_health`、`diagnose_timing_issues`、`optimize_maintenance_schedule`、`check_compliance`、`generate_safety_recommendations`、`web_search` 在此配置中明确返回 `unavailable`。最终测试标签不从此适配器提供；训练/开发标签的处理属于本研究的权限设计，不等同于官方 benchmark 的统一访问规则。

FEMTO、XJTU、LDM/IMS 等后续格式的代码被保留，但本候选目录没有提供它们完整的数据准备、注册、端到端复现流程；2026-10-27 的候选目录测试没有重新验证这些格式。Engine 输入仅支持有限统计，物理通道语义及健康决策合同仍有缺口。

## 本机运行

准备 Python 3.12、PyTorch、NumPy、MCP 等依赖，并提供符合注册表格式的数据目录；当前只验证了原机器上已有的运行环境，未验证全新环境安装。数据、权重、API 密钥和第三方依赖不包含在本目录内。

数据目录须包含 `dataset_registry.json`、`tool_catalog.json` 和相应数据文件。以下示例使用 JSONL；须将两个路径替换为实际位置，而不能仅复制注册表：

```sh
PHM_DATA_ROOT=/path/to/verified-data/public
PHM_RUN_ROOT=/path/to/new-run
mkdir -p "$PHM_RUN_ROOT/tmp" "$PHM_RUN_ROOT/mpl" "$PHM_RUN_ROOT/xdg"
printf '%s\n' '{"tool":"load_dataset","args":{"dataset_name":"MFPT","split":"train"}}' |
  TMPDIR="$PHM_RUN_ROOT/tmp" TMP="$PHM_RUN_ROOT/tmp" TEMP="$PHM_RUN_ROOT/tmp" \
  MPLCONFIGDIR="$PHM_RUN_ROOT/mpl" XDG_CACHE_HOME="$PHM_RUN_ROOT/xdg" \
  python src/verified_benchmark_worker.py --family forge --data "$PHM_DATA_ROOT" \
    --output "$PHM_RUN_ROOT" --transport jsonl
```

将 `--transport jsonl` 改为 `--transport mcp` 可用于 MCP 客户端。此命令直接在本机进程执行，**没有网络或文件系统隔离**；涉及最终测试标签隔离时，应使用研究项目的 `scripts/launch_verified_environment.py` 与已资格化的 v4 运行包，或另行实现并验证隔离。

本机历史启动说明、接口限制和真实训练证据见研究项目的 `docs/phmforge_repairs246.md` 与 `docs/verified_benchmark_environments240.md`。运行日志或结果可能包含数据路径及实验内容，应单独审查后再分享。

## 尚需处理的问题

1. **许可证：** 上游固定提交 [LICENSE 文件](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/LICENSE) 写 MIT，其 [README](https://github.com/DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance/blob/28430199f7b8ccba924d49e06c2d59b761846038/README.md) 写 Apache 2.0；需核清上游口径、保留适用声明，并为本项目新增代码确定许可证。本目录尚未选择 LICENSE。
2. **数据：** 各数据源的许可、来源、下载方式和预处理需逐项核查；部分研究数据可能不允许直接再分发。本候选目录不含数据和模型权重。
3. **可复现性：** 补充数据准备脚本、固定依赖及干净环境安装，验证跨机器训练与评分。现有 22 个可发现工具不代表 22 个都能给出有依据的物理结论。
4. **安全检查：** 提交前复核所选文件及 Git 历史，禁止提交密钥、私人数据、实验日志、测试标签或本机依赖目录。本次候选目录的简单模式扫描未发现常见密钥形式，但不能替代完整审核。
