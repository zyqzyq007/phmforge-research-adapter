#!/usr/bin/env python3
"""Audit PHMForge test access, scoring, and feedback rules without model calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET


PIN = "28430199f7b8ccba924d49e06c2d59b761846038"
REPO = "DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance"
RAW = f"https://raw.githubusercontent.com/{REPO}/{PIN}"

FILES = {
    "README.md": "README.md",
    "benchmark_pass1.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/benchmark_pass1.py",
    "run_pass3.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/run_pass3.py",
    "agents.md": "ReActXen/src/reactxen/demo/intent_implementation_demo/agents.md",
    "agents.py": "ReActXen/src/reactxen/agents/react/agents.py",
    "create_reactxen_agent.py": "ReActXen/src/reactxen/prebuilt/create_reactxen_agent.py",
    "data_tools.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/tools/data_tools.py",
    "model_tools.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/tools/model_tools.py",
    "metric_tools.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/tools/metric_tools.py",
    "web_search_tool.py": "ReActXen/src/reactxen/demo/intent_implementation_demo/tools/web_search_tool.py",
    "phm_scenarios.json": "ReActXen/src/reactxen/demo/intent_implementation_demo/scenarios/phm_scenarios.json",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def call_keywords(node: ast.Call) -> list[str]:
    return [x.arg for x in node.keywords if x.arg]


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def self_attribute_names(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
        ):
            names.append(child.attr)
    return names


def extract_evaluate_pass(source: str):
    tree = ast.parse(source)
    fn = function(tree, "evaluate_pass")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"re": re}
    exec(compile(module, "<evaluate_pass>", "exec"), namespace)
    return namespace["evaluate_pass"]


def source_url(relative: str) -> str:
    return f"{RAW}/{quote(relative, safe='/')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--a244-bundle", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    source = run / "source"
    official = source / "official"
    audit_dir = run / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    tree_doc = load_json(source / "github_tree.json")
    tree_hash = {x["path"]: x["sha"] for x in tree_doc.get("tree", []) if x["type"] == "blob"}
    commits = load_json(source / "github_commits.json")
    branches = load_json(source / "github_branches.json")
    tags = load_json(source / "github_tags.json")
    releases = load_json(source / "github_releases.json")
    scenario = load_json(official / "phm_scenarios.json")
    scenarios = scenario["pdm_scenarios"]

    file_checks = []
    for local_name, repo_path in FILES.items():
        data = (official / local_name).read_bytes()
        observed = git_blob_sha1(data)
        expected = tree_hash.get(repo_path)
        file_checks.append(
            {
                "local": str(official / local_name),
                "repository_path": repo_path,
                "url": source_url(repo_path),
                "bytes": len(data),
                "sha256": sha256(data),
                "git_blob_sha1": observed,
                "github_tree_git_blob_sha1": expected,
                "matches_pinned_github_tree": observed == expected,
            }
        )
    if not all(x["matches_pinned_github_tree"] for x in file_checks):
        raise RuntimeError("A downloaded official file does not match the pinned GitHub tree")

    latest_commit = commits[0]
    main_branch = next(x for x in branches if x["name"] == "main")
    other_branches = [
        {"name": x["name"], "sha": x["commit"]["sha"]}
        for x in branches
        if x["name"] != "main"
    ]

    benchmark_source = (official / "benchmark_pass1.py").read_text()
    agent_source = (official / "agents.py").read_text()
    create_source = (official / "create_reactxen_agent.py").read_text()
    model_source = (official / "model_tools.py").read_text()
    data_source = (official / "data_tools.py").read_text()
    metric_source = (official / "metric_tools.py").read_text()
    web_source = (official / "web_search_tool.py").read_text()
    pass3_source = (official / "run_pass3.py").read_text()

    benchmark_tree = ast.parse(benchmark_source)
    agent_tree = ast.parse(agent_source)
    run_one = function(benchmark_tree, "run_one_scenario")
    create_calls = [
        node
        for node in ast.walk(run_one)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_reactxen_agent"
    ]
    if len(create_calls) != 1:
        raise RuntimeError("Expected exactly one create_reactxen_agent call")
    create_call = create_calls[0]
    question_keyword = next(x for x in create_call.keywords if x.arg == "question")
    key_keyword = next(x for x in create_call.keywords if x.arg == "key")

    prompt_methods = {}
    for name in ("_build_agent_prompt", "_build_reflection_prompt"):
        method = function(agent_tree, name)
        fmt_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
        ]
        prompt_methods[name] = {
            "format_keywords": sorted({x.arg for call in fmt_calls for x in call.keywords if x.arg}),
            "contains_self_key": "key" in self_attribute_names(method),
        }

    evaluate_pass = extract_evaluate_pass(benchmark_source)
    classification_stub = evaluate_pass(
        {"classification_type": "Fault Classification", "ground_truth": {}},
        "classification attempted",
    )
    rul_self_report = evaluate_pass(
        {
            "classification_type": "RUL Prediction",
            "ground_truth": {"expected_mae_range": [7, 18], "expected_rmse_range": [11, 26]},
        },
        "MAE=12.0 cycles; RMSE=16.0 cycles",
    )

    scenario_counts = {}
    for item in scenarios:
        ctype = item["classification_type"]
        scenario_counts[ctype] = scenario_counts.get(ctype, 0) + 1
    rul_scenarios = [x for x in scenarios if x["classification_type"] == "RUL Prediction"]
    rul_prompts_request_all = sum(
        "all test" in x.get("input_question", "").lower()
        or "every unit" in x.get("input_question", "").lower()
        for x in rul_scenarios
    )

    paper_html = (source / "arxiv_2604.01532v3.html").read_text()
    paper_text = re.sub(r"<[^>]+>", " ", paper_html)
    paper_text = re.sub(r"\s+", " ", paper_text)
    arxiv_root = ET.parse(source / "arxiv_api_2604.01532.xml").getroot()
    atom = {"atom": "http://www.w3.org/2005/Atom"}
    entry = arxiv_root.find("atom:entry", atom)
    paper_updated = entry.findtext("atom:updated", namespaces=atom)
    paper_id = entry.findtext("atom:id", namespaces=atom)

    explicit_policy_terms = [
        term
        for term in (
            "filesystem permissions",
            "filesystem access policy",
            "network access policy",
            "test label visibility",
            "score query limit",
            "retry limit",
        )
        if term in (benchmark_source + agent_source + create_source).lower()
    ]

    checks = {
        "pinned_commit_is_latest_seen_main": latest_commit["sha"] == PIN,
        "main_branch_matches_pinned_commit": main_branch["commit"]["sha"] == PIN,
        "official_files_match_github_tree": all(
            x["matches_pinned_github_tree"] for x in file_checks
        ),
        "repository_has_75_scenarios": len(scenarios) == 75,
        "repository_has_5_categories": len(scenario_counts) == 5,
        "paper_v3_states_99_scenarios": "99 SME-authored scenarios" in paper_text,
        "paper_v3_states_39_mcp_tools": "39 MCP-native tools" in paper_text,
        "paper_states_claude_code_evaluated_manually": (
            "Claude Code is an interactive CLI agent" in paper_text
            and "evaluated manually" in paper_text
        ),
        "paper_states_deterministic_evaluator_25_subset": (
            "automated ReAct/ReActXen harness on the 25-scenario" in paper_text
        ),
        "question_passed_to_agent": (
            isinstance(question_keyword.value, ast.Name)
            and question_keyword.value.id == "question"
            and 'question = scenario.get("input_question", "")' in benchmark_source
        ),
        "ground_truth_passed_as_key": isinstance(key_keyword.value, ast.Call)
        and isinstance(key_keyword.value.func, ast.Name)
        and key_keyword.value.func.id == "str",
        "ground_truth_key_absent_from_agent_prompt": (
            not prompt_methods["_build_agent_prompt"]["contains_self_key"]
            and not prompt_methods["_build_reflection_prompt"]["contains_self_key"]
        ),
        "key_stored_and_used_for_exact_match": (
            "self.key = key" in agent_source
            and "return EM(self.answer, self.key)" in agent_source
        ),
        "agent_factory_prints_agent_args_to_stdout": (
            "print (agent_args)" in create_source or "print(agent_args)" in create_source
        ),
        "predict_rul_loads_final_ground_truth_file": (
            "RUL_{fd_num}.txt" in model_source
            and "_load_rul_ground_truth(fd)" in model_source
        ),
        "predictions_are_ground_truth_plus_noise": (
            "preds = [max(0.0, round(g + s, 2))" in model_source
        ),
        "single_unit_predict_rul_returns_actual_rul": (
            "actual RUL = {gt[unit_id - 1]}" in model_source
        ),
        "load_ground_truth_tool_returns_count_only": (
            "Loaded {len(values)} ground truth RUL values" in data_source
        ),
        "verify_tools_zip_truncate_arrays": (
            "zip(pred, gt)" in metric_source
            and "zip(pred, gt)" in metric_source
        ),
        "evaluator_runs_after_agent_run": (
            benchmark_source.index("agent.run()")
            < benchmark_source.index("correct, reason = evaluate_pass")
        ),
        "classification_self_report_alone_passes": classification_stub[0],
        "rul_self_reported_metrics_alone_pass": rul_self_report[0],
        "pass3_reuses_same_scenario_set": (
            "SAME 25 scenarios" in pass3_source
            and "run_one_scenario(sc, framework, model_id)" in pass3_source
        ),
        "resume_is_default": "resume: bool = True" in benchmark_source,
        "web_search_is_conditional_on_brave_key": (
            'os.environ.get("BRAVE_API_KEY"' in web_source
            and "API key not configured" in web_source
        ),
        "no_explicit_unified_permission_terms_in_audited_code": not explicit_policy_terms,
    }

    permission_rows = [
        {
            "surface": "scenario question",
            "official_automated": "visible to the agent",
            "coding_agent_manual": "paper says manual, but no file/network prompt or permission table is released",
        },
        {
            "surface": "scenario ground_truth object",
            "official_automated": "loaded by the runner and passed as the React agent's exact-answer key; not inserted into prompt construction, but printed in agent_args stdout",
            "coding_agent_manual": "not specified",
        },
        {
            "surface": "final RUL labels",
            "official_automated": "predict_rul reads RUL_FDxxx.txt and the single-unit branch returns actual RUL directly",
            "coding_agent_manual": "not specified",
        },
        {
            "surface": "final classification labels",
            "official_automated": "classify_faults creates deterministic synthetic labels rather than reading a test-label artifact",
            "coding_agent_manual": "not specified",
        },
        {
            "surface": "test observations",
            "official_automated": "load_dataset reads the configured train/test files; non-CMAPSS missing test split may fall back to the train file",
            "coding_agent_manual": "not specified",
        },
        {
            "surface": "filesystem",
            "official_automated": "agent receives domain tools rather than an arbitrary shell; tools read PHMFORGE_DATA_DIR",
            "coding_agent_manual": "no released unified filesystem sandbox or access policy",
        },
        {
            "surface": "network",
            "official_automated": "web_search can call Brave when BRAVE_API_KEY is configured; otherwise returns a placeholder",
            "coding_agent_manual": "no released network policy; our adaptation should disable network",
        },
        {
            "surface": "score feedback during one run",
            "official_automated": "evaluate_pass is called after agent.run; it is not passed into the agent loop",
            "coding_agent_manual": "not specified, so clean feedback isolation cannot be assumed",
        },
        {
            "surface": "post-score modification / reruns",
            "official_automated": "resume is enabled by default and there is no documented retry cap or prohibition on changing code between score observations",
            "coding_agent_manual": "not specified",
        },
        {
            "surface": "pass-all-3",
            "official_automated": "reruns the same selected scenarios three times; no separate held-out confirmation set is enforced",
            "coding_agent_manual": "the paper calls Claude Code evaluation manual; no per-attempt interaction log policy is released",
        },
        {
            "surface": "scenario/tool release identity",
            "official_automated": f"latest public main pin has {len(scenarios)} scenarios and the checked code constructs 22 tools",
            "coding_agent_manual": "paper v3 claims 99 scenarios and 39 MCP tools, with 24 BESS cases and 17 battery tools not present at the pinned public main",
        },
    ]

    report = {
        "audit_id": "A251",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "zero-model official PHMForge access/scoring audit; no official score claimed",
        "official_source": {
            "repository": REPO,
            "requested_pin": PIN,
            "latest_commit_seen_on_main": {
                "sha": latest_commit["sha"],
                "committed_utc": latest_commit["commit"]["committer"]["date"],
                "message": latest_commit["commit"]["message"].splitlines()[0],
            },
            "main_branch_sha": main_branch["commit"]["sha"],
            "other_branch_heads": other_branches,
            "public_tags": tags,
            "public_releases": releases,
            "file_checks": file_checks,
            "transport_note": (
                "The environment's HTTPS chain failed default CA verification. Read-only fetches used "
                "curl -k, then every repository file was authenticated by matching its Git blob SHA-1 "
                "to the independent recursive GitHub tree response for the pinned commit."
            ),
        },
        "paper": {
            "url": "https://arxiv.org/abs/2604.01532v3",
            "id_seen": paper_id,
            "updated_utc": paper_updated,
            "claims_99_scenarios": checks["paper_v3_states_99_scenarios"],
            "claims_39_tools": checks["paper_v3_states_39_mcp_tools"],
            "automated_25_subset": checks["paper_states_deterministic_evaluator_25_subset"],
            "manual_claude_code_full_set": checks["paper_states_claude_code_evaluated_manually"],
        },
        "public_repository": {
            "scenario_count": len(scenarios),
            "scenario_counts_by_category": scenario_counts,
            "rul_scenarios": len(rul_scenarios),
            "rul_scenarios_requesting_all_units": rul_prompts_request_all,
            "source_scenario_sha256": sha256((official / "phm_scenarios.json").read_bytes()),
        },
        "prompt_data_flow": {
            "ground_truth_passed_as_key": checks["ground_truth_passed_as_key"],
            "ground_truth_key_absent_from_prompt_methods": checks[
                "ground_truth_key_absent_from_agent_prompt"
            ],
            "prompt_method_details": prompt_methods,
            "key_used_by_generic_exact_match": checks["key_stored_and_used_for_exact_match"],
            "agent_factory_prints_args": checks["agent_factory_prints_agent_args_to_stdout"],
        },
        "scoring": {
            "evaluator_runs_after_agent": checks["evaluator_runs_after_agent_run"],
            "evaluator_checks_text_not_artifacts": True,
            "classification_stub_result": list(classification_stub),
            "rul_self_report_result": list(rul_self_report),
            "resume_default": checks["resume_is_default"],
            "pass3_reuses_same_scenario_set": checks["pass3_reuses_same_scenario_set"],
        },
        "permission_matrix": permission_rows,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "conclusions": {
            "official_automated_test_labels_are_hidden": False,
            "official_automated_final_rul_is_agent_visible_via_tool": True,
            "official_automated_key_directly_prompt_injected": False,
            "official_automated_score_feedback_is_agent_internal": False,
            "official_automated_protocol_has_documented_retry_cap": False,
            "official_external_coding_agent_protocol_reproducible_from_release": False,
            "paper_and_public_main_release_identity_match": False,
            "official_75_task_score_claim_allowed": False,
            "official_99_task_score_claim_allowed": False,
            "research_adaptation_must_hide_final_labels": True,
            "research_adaptation_must_use_external_scorer": True,
            "research_adaptation_must_disable_network_and_hidden_paths": True,
            "research_adaptation_score_is_official_phmforge_score": False,
        },
        "blocking_findings": [
            "The latest public main is internally a 75-scenario/22-tool release, while paper v3 describes a 99-scenario/39-tool benchmark with a 17-tool BESS server absent from that main pin.",
            "No reproducible external coding-agent harness is released for the paper's 99-task Claude Code evaluation; it is described as manual.",
            "The automated RUL prediction tool reads final RUL files and exposes actual RUL for single-unit calls, so the official automated RUL score is not a blind final-label evaluation.",
            "Pass@1 evaluates answer text rather than model artifacts, and the repository documents resumable reruns without a score-feedback or modification cap.",
        ],
        "recommended_research_protocol": {
            "official_score_claim": False,
            "visible_to_agent": [
                "scenario question",
                "training observations",
                "development observations",
                "development labels through an explicit review tool",
            ],
            "hidden_from_agent": [
                "final test labels",
                "hidden test paths",
                "official scorer implementation during generation",
                "prior task scores",
            ],
            "tool_and_file_policy": (
                "read-only task runtime, private writable output, no host home, no generic network, "
                "and no arbitrary path traversal"
            ),
            "evaluation_order": (
                "freeze candidate code and predictions, terminate writers, then read hidden labels "
                "with a separate scorer"
            ),
            "feedback_policy": (
                "no task score is returned during generation; any later revision is a new predeclared "
                "session and must not reuse a completed authorization"
            ),
            "allowed_scope": [
                "pdm_fault_001",
                "pdm_fault_013",
                "pdm_fault_015",
            ],
        },
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
        "new_training_jobs": 0,
        "new_scientific_scores": 0,
        "new_confirmed_PHM_problems": 0,
        "fully_qualified_original_tasks": 0,
    }

    (audit_dir / "ACCESS_AUDIT251.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (run / "PROTOCOL251.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "scope": "official PHMForge access and scoring audit with no model calls",
                "official_pin": PIN,
                "paper": "https://arxiv.org/abs/2604.01532v3",
                "new_LLM_calls": 0,
                "new_training_jobs": 0,
                "stop_after_audit": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "checks_passed": report["checks_passed"],
                "checks_total": report["checks_total"],
                "repo_scenarios": len(scenarios),
                "paper_updated_utc": paper_updated,
                "official_hidden_labels": False,
                "new_LLM_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
