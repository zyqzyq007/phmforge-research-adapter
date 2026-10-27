#!/usr/bin/env python3
"""Build the offline Chinese briefing from frozen result files, without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/fullbench187"


def read(path: Path):
    return json.loads(path.read_text())


def build(run: Path):
    ledger_path = RESULTS / "final_source_ledger214.jsonl"
    ledger = [json.loads(s) for s in ledger_path.read_text().splitlines() if s.strip()]
    scope = read(RESULTS / "final_scope214.json")
    checkpoint = read(RESULTS / "current_execution_checkpoint.json")
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == scope["final_ledger_sha256"]
    assert len(ledger) == 774
    inventory = read(run / "selected_sources/TASK_INVENTORY187.json")
    questions = {}
    for key, contract in [("phmforge", "PHMForge_public"), ("assetops_official_HF", "AssetOps_HF/scenarios")]:
        for row in inventory[key]["rows"]:
            questions[(contract, str(row["task_id"]))] = row.get("question", "")
    scenarios = read(run / "selected_sources/phm_scenarios.json")["pdm_scenarios"]
    forge_tools = {x["task_id"]: x.get("required_tools", []) for x in scenarios}
    native = {}
    cohorts = [
        ("A187_native_results.json", "20261018T170243Z_fullbench187"),
        ("A190_native_results.json", "20261018T185541Z_assethf190"),
        ("A202_native_results.json", "20261019T032531Z_assetextra202"),
    ]
    for file, cohort in cohorts:
        for row in read(RESULTS / file)["rows"]:
            native[(cohort, row["job"])] = row
    recovery = read(RESULTS / "recovery_native_results223.json")
    quarantined = {x["original_task_id"] for x in recovery["rows"]}
    rows = []
    for raw in ledger:
        ep = raw.get("episode") or {}
        score = raw.get("original_scoring") or {}
        n = native.get((ep.get("run"), ep.get("job")), {})
        ns = n.get("native_score") or {}
        ev = raw.get("evidence") or {}
        rows.append({
            "contract": raw["contract"], "id": str(raw["original_task_id"]),
            "category": raw.get("category", ""), "revision": raw.get("revision", ""),
            "question": questions.get((raw["contract"], str(raw["original_task_id"])), ""),
            "status": raw.get("execution_status", "unknown"),
            "available": bool(score.get("available")), "passed": score.get("passed"),
            "method": score.get("method", ""), "reason": score.get("reason", ""),
            "judge_rationale": ns.get("rationale", ""), "judge_score": ns.get("score"),
            "judge_dimensions": ns.get("details", {}),
            "elapsed": ep.get("elapsed_seconds"), "start": ep.get("started_utc"),
            "end": ep.get("ended_utc"), "run": ep.get("run"), "job": ep.get("job"),
            "not_run_reason": raw.get("non_execution_reason", raw.get("input_qualification", "")),
            "missing": raw.get("missing_source_items", []),
            "question_sha": raw.get("question_sha256"), "answer_sha": ev.get("answer_sha256"),
            "calls_sha": ev.get("calls_sha256"), "tool_calls": ev.get("tool_call_records"),
            "error_returns": ev.get("error_returns"), "same_text_count": raw.get("exact_text_group_size", 1),
            "quarantined": raw["contract"] == "PHMForge_public" and str(raw["original_task_id"]) in quarantined,
            "required_tools": forge_tools.get(str(raw["original_task_id"]), []) if raw["contract"] == "PHMForge_public" else [],
        })
    finished = [x for x in rows if x["status"] == "finished"]
    assert len(finished) == 333
    assert sum(x["available"] for x in finished) == 332
    assert sum(x["available"] and x["passed"] is True for x in finished) == 220
    assert len(quarantined) == 3
    elapsed = sorted(x["elapsed"] for x in finished if isinstance(x["elapsed"], (int, float)))
    sources = [
        ("阶段汇报", "docs/briefing20261020.md", "当前结论、运行范围与主要反证"),
        ("来源版本核查", "docs/dataset_check_current20261019.md", "官方版本、原题哈希、配置与缺件"),
        ("774项冻结台账", "results/fullbench187/final_source_ledger214.jsonl", "完整分母；SHA256固定，不覆盖原分"),
        ("最终范围", "results/fullbench187/final_scope214.json", "333终态、332原分、441未运行"),
        ("当前执行检查点", "results/fullbench187/current_execution_checkpoint.json", "运行状态及各轮审计入口"),
        ("Forge工具与评分资格", "docs/forge_contract_qualification193.md", "75题、22工具、固定否认控制与模型简化"),
        ("HF评分和覆盖", "docs/hf152_scoring_and_coverage191_192.md", "裁判量表与轨迹截断"),
        ("底层数据来源", "docs/assetops_underlying_data_audit200.md", "压缩机与液压循环的原始来源和映射限制"),
        ("通信污染审计", "docs/legacy_rpc_validity221.md", "3题7次错配，旧尝试隔离"),
        ("恢复补跑结果", "docs/benchmark_recovery_results223.md", "同3原题补跑；1958关联、1668算术返回核验"),
        ("工单证据范围", "docs/workorder_evidence_scope224.md", "47题、160处ID、时序与状态表述"),
        ("工单分组核验", "docs/native_bundle_audit220.md", "411/424满足跨度和覆盖，保留评分反证"),
        ("文件型任务核验", "docs/file_task_evidence_audit229.md", "8题文件不可用；3拒绝、5披露代理"),
        ("引擎健康核验", "docs/engine_diagnostic_evidence230.md", "30题、13信号摘要、1806规则返回"),
        ("时间边界反证", "docs/time_and_score_counterevidence215.md", "未来观测访问不自动等于训练泄漏"),
        ("20道原题断言", "docs/native_claim_audit219.md", "替代设备、日期与事件计数"),
        ("导波正常完成结果", "docs/scoped_problem_addendum178_result.md", "有效B被替换；原7例0/7、82.484mm"),
        ("导波干净普通复核", "docs/scoped_problem_addendum179.md", "正常结束仍0/7；不同起点，非严格配对"),
        ("导波损伤算子对照", "docs/damage_operator_validation225.md", "10组合成几何、60次计算与机制反证"),
        ("历史16会话", "docs/validation_scope_results185.md", "4来源、8场景、成功与失败完整保留"),
        ("11张候选卡", "results/fullbench187/candidate_cards217.json", "问题边界、反证与迭代决定"),
        ("普通复核计划", "docs/ordinary_scope_review_plan226.md", "4会话方案已资格核查，尚未运行"),
        ("近期方法适用边界", "docs/differential_validation_boundary228.md", "不能把差分验证一概视作依赖隐藏oracle"),
        ("归档索引", "results/archive_index.json", "逐文件核验的原任务、轨迹、代码和评分归档"),
    ]
    if (run / "SUCCESS_AUDIT237.json").is_file():
        sources.insert(-1, ("成功案例核查", "docs/success_case_examples237.md", "Fan主故障判定与31张预防性工单的成功范围、原评分和逐项核验"))
    provenance = []
    for title, rel, purpose in sources:
        p = ROOT / rel
        assert p.is_file(), rel
        provenance.append({"title": title, "path": rel, "purpose": purpose, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_utc": checkpoint.get("updated_utc", checkpoint["utc"]),
        "official_version_check": "2026-10-19 10:33 UTC / 北京时间18:33",
        "scope": scope, "rows": rows,
        "native_joined": len(native),
        "duration": {"count": len(elapsed), "median_seconds": statistics.median(elapsed), "max_seconds": max(elapsed)},
        "recovery": recovery,
        "candidates": read(RESULTS / "candidate_cards217.json"),
        "forge_tools": read(run / "forge_tools.json"),
        "asset_tools": read(run / "selected_sources/NATIVE_TOOL_NAMES.json"),
        "asset_tool_provenance": read(run / "asset_tools_provenance.json"),
        "examples": read(run / "EXAMPLES232.json") if (run / "EXAMPLES232.json").is_file() else None,
        "source_counts": dict(Counter(x["contract"] for x in rows)),
        "sources": provenance,
        "research_confirmed": 0,
        "new_model_calls_for_report": 0,
        "main_data_versions": {
            "PHMForge": "28430199f7b8ccba924d49e06c2d59b761846038",
            "AssetOps_main": "e844f6468c6c68b328022e7ca8e31da140842cc7",
            "AssetOps_compat": "c6b96755b8a182aa090ee028852cbc83279b2982",
            "AssetOps_HF": "5e25bb7f2cd37fb68b9a9e1f99d170ca5be7ce17",
        },
    }
    template_path = ROOT / ("reports/phm_briefing_visual.template.html" if report["examples"] else "reports/phm_agent_report.template.html")
    template = template_path.read_text()
    data = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    assert template.count("__REPORT_DATA__") == 1
    out = ROOT / "reports/phm_agent_report_20261020.html"
    out.write_text(template.replace("__REPORT_DATA__", data))
    (run / "report_data.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary = {"output": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "ledger_rows": len(rows), "finished": len(finished), "native_grade_available": 332, "native_passed": 220, "new_model_calls": 0, "all_evidence_files_exist": True, "native_rows_joined": len(native)}
    (run / "BUILD_RESULT231.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    build(parser.parse_args().run)
