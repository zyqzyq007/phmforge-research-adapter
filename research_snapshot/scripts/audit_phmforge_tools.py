"""Inventory the frozen PHMForge tools and replay targeted validity controls.

These are constructed software controls, not PHM performance experiments.
Original function bodies are unchanged; framework argument wrappers are omitted.
Previously qualified model training is not repeated.
"""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd


def discover(source, output):
    scope = dict(json=json, np=np, pd=pd, Path=Path, os=os, re=re)
    fixtures = output / "control_data"
    fixtures.mkdir(parents=True)
    scope.update(DATA_DIR=fixtures)
    inventory, functions = [], {}
    for file in sorted(source.glob("*.py")):
        tree = ast.parse(file.read_text())
        helpers = [copy.deepcopy(n) for n in tree.body
                   if isinstance(n, ast.FunctionDef)]
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(b, ast.Name) and b.id == "BaseTool" for b in node.bases):
                continue
            values = {n.target.id: n.value for n in node.body
                      if isinstance(n, ast.AnnAssign)}
            name = ast.literal_eval(values["name"])
            method = next(n for n in node.body
                          if isinstance(n, ast.FunctionDef) and n.name == "_run")
            inventory.append(dict(
                tool=name, file=file.name, class_name=node.name,
                line_start=method.lineno, line_end=method.end_lineno,
                source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                description=ast.literal_eval(values["description"]),
            ))
            function = copy.deepcopy(method)
            function.name = "replay_" + name
            helpers.append(function)
            functions[name] = function.name
        exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])),
                     str(file), "exec"), scope)
    return inventory, {k: scope[v] for k, v in functions.items()}, fixtures


def audit(source, output):
    output.mkdir(parents=True, exist_ok=True)
    inventory, functions, fixtures = discover(source, output)
    assert len(inventory) == 22
    records = []

    def call(tool, **arguments):
        value = functions[tool](SimpleNamespace(), **arguments)
        records.append(dict(tool=tool, arguments=arguments, output=value))
        return value

    def check(name, observed, explanation):
        assert observed, name
        checks.append(dict(control=name, reproduced=True, meaning=explanation))

    checks = []
    health = call("assess_component_health", component="Fan",
                  efficiency=1.2, flow_modifier=1.2)
    check("out_of_declared_range_health", "Healthy" in health,
          "Values above the declared 0–1 range are accepted as Healthy.")
    timing = call("diagnose_timing_issues", efficiency_deviation=0., flow_deviation=0.)
    check("zero_deviations_fault", "Combined efficiency-flow fault" in timing,
          "Zero baseline deviations still produce a fault diagnosis.")
    one = call("optimize_maintenance_schedule", rul_estimate=100.,
               preventive_cost=1., failure_cost=10000., safety_margin=.2)
    two = call("optimize_maintenance_schedule", rul_estimate=100.,
               preventive_cost=100000., failure_cost=10000., safety_margin=.2)
    check("optimal_threshold_independent_of_costs",
          "threshold: 80 cycles" in one and "threshold: 80 cycles" in two,
          "Threshold is RUL*(1-margin); no cost objective is optimized.")
    compliance = call("check_compliance", standard="NOT_A_REAL_STANDARD",
                      safety_integrity_level=99, current_pfd=0.)
    check("arbitrary_standard_compliant", "COMPLIANT" in compliance,
          "Unknown standard and unsupported SIL still return COMPLIANT.")
    one = call("generate_safety_recommendations", risk_level="high",
               failure_mode="bearing", current_controls="none")
    two = call("generate_safety_recommendations", risk_level="high",
               failure_mode="bearing", current_controls="redundant validated protection")
    check("safety_controls_do_not_change_actions",
          one.split("Current controls:")[0] == two.split("Current controls:")[0],
          "Existing controls are echoed, not used to choose actions.")
    invalid = call("generate_safety_recommendations", risk_level="undefined",
                   failure_mode="bearing")
    check("unknown_risk_defaults_to_medium", "within 1 week" in invalid,
          "Unknown risk is silently mapped to a fixed medium-risk template.")

    assert not os.environ.get("BRAVE_API_KEY")
    search = call("web_search", query="audit-control-no-network", count=3)
    check("search_placeholder_without_service", "would return 3 results" in search,
          "This is an explicit placeholder, not an actual search result.")
    (fixtures / "RUL_FD001.txt").write_text("10\n20\n")
    loaded = call("load_ground_truth", dataset_name="CMAPSS_FD001")
    check("ground_truth_loader_is_real_io", "Loaded 2 ground truth" in loaded,
          "Counterevidence: original loader reads a file and returns a count, not its values.")
    rul_verify = call("verify_ground_truth", ground_truth="[10,20,30]", predictions="[10]")
    class_verify = call("verify_classification", ground_truth='["a","b","c"]',
                        predictions='["a"]')
    check("verification_ignores_omitted_targets",
          "1/1" in rul_verify and "1/1" in class_verify,
          "zip truncates; supplied subset can get a perfect ratio without full coverage.")
    mae = call("calculate_mae", ground_truth="[1,1,1]", predictions="[1]")
    rmse = call("calculate_rmse", ground_truth="[1,1,1]", predictions="[1]")
    accuracy = call("calculate_accuracy", ground_truth="[1,1,1]", predictions="[1]")
    check("metric_broadcast_accepts_wrong_length",
          "0.00" in mae and "0.00" in rmse and "100.00%" in accuracy,
          "NumPy broadcasting accepts unequal coverage; arithmetic is real but validation is missing.")
    (fixtures / "CWRU_train.csv").write_text("feature,label\n1,0\n")
    dataset = call("load_dataset", dataset_name="CWRU", split="test")
    check("train_file_reported_as_test", "Loaded CWRU test: 1 rows" in dataset,
          "Only CWRU_train.csv exists, yet missing test split reports a successful test load.")
    first = call("analyze_engine_signals", sensor_data='{"speed_krpm":3}')
    second = call("analyze_engine_signals", sensor_data='{"speed_rpm":3000}')
    check("sensor_threshold_ignores_units", "NORMAL" in first and "WARNING" in second,
          "All sensor names and quantities share a fixed absolute threshold of 1000.")
    trend = call("detect_degradation_trend",
                 cycle_data='[{"cycle":1,"value":1},{"cycle":2,"value":100},{"cycle":3,"value":1}]')
    check("trend_uses_endpoints", "Stable" in trend,
          "An endpoint trend summary does not establish absence of internal anomalies.")
    maintenance = call("calculate_maintenance_cost", preventive_cost=-100.,
                       frequency_per_year=1, downtime_hours=0., hourly_rate=0.)
    failure = call("calculate_failure_cost", failure_probability=2., repair_cost=100.,
                   downtime_hours=0., hourly_rate=0.)
    check("costs_accept_invalid_inputs", "$-100.00" in maintenance and "200.00%" in failure,
          "Actual arithmetic has no nonnegative-cost/probability-range validation.")
    risk = call("assess_safety_risk", failure_mode="control", severity=100,
                probability=100, detectability=100)
    check("risk_accepts_out_of_range_ratings", "RPN=1000000" in risk,
          "Declared 1–10 ordinal ratings are not enforced; risk classes are fixed thresholds.")
    report = dict(
        scope="22 active original tool function bodies; no framework wrapper replay",
        original_tool_count=len(inventory), tools_inventory=inventory,
        probes=records, checks=checks,
        classification="constructed software validity controls; not empirical PHM Agent failures",
        model_tools="Reuse hash-matched A239 parameter/weight/input controls and A240 real backend qualification",
        new_Agent_sessions=0, new_LLM_calls=0, new_training_jobs=0,
    )
    (output / "TOOL_REPLAY242.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(tools=len(inventory), calls=len(records),
                          reproduced_controls=len(checks))))


async def audit_native(source, output):
    """Exercise actual FastMCP dispatch, retaining original function bodies.

    Extract helpers and response models to avoid the irrelevant LangChain
    dependency imported by the full server module. No stdio transport claimed.
    """
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "not_a_trained_model.pkl"
    checkpoint.write_bytes(pickle.dumps({"control": "plain dictionary; no model"}))
    data_tree = ast.parse((source / "tools/data_tools.py").read_text())
    helpers = [copy.deepcopy(n) for n in data_tree.body if isinstance(n, ast.FunctionDef)]
    calls, catalogs = [], {}
    for server in ["prognostics", "maintenance"]:
        path = source / "mcp_servers" / (server + "_server.py")
        tree = ast.parse(path.read_text())
        nodes = helpers + [copy.deepcopy(n) for n in tree.body
                           if isinstance(n, ast.ClassDef)
                           or isinstance(n, ast.FunctionDef) and n.name != "main"]
        mcp = FastMCP("audit-original-" + server)
        scope = dict(__name__="__main__", mcp=mcp, BaseModel=BaseModel,
                     json=json, np=np, os=os, Path=Path, re=re, DATA_DIR=output)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(path), "exec"), scope)
        catalogs[server] = [x.name for x in await mcp.list_tools()]
        requests = ([
            ("train_rul_model", dict(dataset="ABSENT_DATA", model_type="transformer", epochs=1)),
            ("train_fault_classifier", dict(dataset="ABSENT_DATA", model_type="transformer", epochs=200)),
            ("predict_rul", dict(model_path=str(checkpoint), test_data="absent_a.csv", unit_id=1)),
            ("predict_rul", dict(model_path=str(checkpoint), test_data="absent_b.csv", unit_id=2)),
            ("classify_faults", dict(model_path=str(checkpoint), test_data="absent_a.csv", unit_id=1)),
            ("diagnose_timing_issues", dict(efficiency_deviation=0., flow_deviation=0.)),
        ] if server == "prognostics" else [
            ("check_compliance", dict(standard="NOT_A_REAL_STANDARD",
                                      safety_integrity_level=99, current_pfd=0.)),
        ])
        def serialize(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, (tuple, list)):
                return [serialize(x) for x in value]
            if isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            return value
        for tool, arguments in requests:
            item = dict(server=server, tool=tool, arguments=arguments)
            try:
                item["result"] = serialize(await mcp.call_tool(tool, arguments))
            except Exception as exc:
                item.update(error_type=type(exc).__name__, error=str(exc))
            calls.append(item)
    report = dict(
        scope="Actual FastMCP list_tools/call_tool dispatch; original function bodies and response models; no stdio transport",
        catalogs=catalogs, calls=calls,
        new_LLM_calls=0, actual_model_training_jobs=0,
        original_source_functions_modified=False,
    )
    (output / "NATIVE_MCP_REPLAY242.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(tool_counts={k: len(v) for k, v in catalogs.items()},
                          calls=len(calls), errors=sum("error" in x for x in calls))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.native:
        asyncio.run(audit_native(args.source, args.output))
    else:
        audit(args.source, args.output)
