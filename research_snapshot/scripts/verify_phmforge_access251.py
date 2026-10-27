#!/usr/bin/env python3
"""Independent checks for A251 without importing the primary audit implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


PIN = "28430199f7b8ccba924d49e06c2d59b761846038"


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--a244-bundle", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    source = run / "source"
    official = source / "official"
    report = json.loads((run / "audit/ACCESS_AUDIT251.json").read_text())
    tree = json.loads((source / "github_tree.json").read_text())
    tree_hash = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    commits = json.loads((source / "github_commits.json").read_text())
    tags = json.loads((source / "github_tags.json").read_text())
    releases = json.loads((source / "github_releases.json").read_text())
    scenarios = json.loads((official / "phm_scenarios.json").read_text())["pdm_scenarios"]
    model_tools = (official / "model_tools.py").read_text()
    pass1 = (official / "benchmark_pass1.py").read_text()
    agents = (official / "agents.py").read_text()
    creator = (official / "create_reactxen_agent.py").read_text()
    paper = (source / "arxiv_2604.01532v3.html").read_text()
    paper_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", paper))

    member = (
        "source/official/ReActXen/src/reactxen/demo/intent_implementation_demo/"
        "scenarios/phm_scenarios.json"
    )
    with tarfile.open(args.a244_bundle, "r:gz") as handle:
        archived_scenarios = handle.extractfile(member).read()
    scenario_bytes = (official / "phm_scenarios.json").read_bytes()

    expected_files = {
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
    match_rows = []
    for local, remote in expected_files.items():
        observed = blob_sha1((official / local).read_bytes())
        match_rows.append(
            dict(local=local, observed=observed, expected=tree_hash.get(remote),
                 match=observed == tree_hash.get(remote))
        )

    checks = {
        "latest_commit_still_matches_pin": commits[0]["sha"] == PIN,
        "all_11_source_files_match_tree": all(x["match"] for x in match_rows),
        "downloaded_scenarios_match_A244_archived_bytes": scenario_bytes == archived_scenarios,
        "scenario_sha256_matches_A244": (
            hashlib.sha256(scenario_bytes).hexdigest()
            == "3d55c7acc802084deb4c3e93a648755e1d4e01322e215872720db0123860036d"
        ),
        "scenario_count_is_75": len(scenarios) == 75,
        "public_tags_empty": tags == [],
        "public_releases_empty": releases == [],
        "paper_reports_99_tasks": "99 SME-authored scenarios" in paper_text,
        "paper_reports_39_tools": "39 MCP-native tools" in paper_text,
        "paper_reports_manual_coding_agent_evaluation": (
            "Claude Code is an interactive CLI agent" in paper_text
            and "evaluated manually" in paper_text
        ),
        "predict_rul_reads_ground_truth_file": "RUL_{fd_num}.txt" in model_tools,
        "predict_rul_derives_prediction_from_truth": (
            "preds = [max(0.0, round(g + s, 2))" in model_tools
        ),
        "single_unit_returns_actual_rul": "actual RUL = {gt[unit_id - 1]}" in model_tools,
        "factory_prints_key_bearing_agent_args": (
            "print (agent_args)" in creator or "print(agent_args)" in creator
        ),
        "key_not_in_prompt_builder": (
            'question=self.question' in agents
            and "self.key" not in agents[
                agents.index("def _build_agent_prompt") : agents.index("def is_finished", agents.index("def _build_agent_prompt"))
            ]
        ),
        "pass1_scores_after_agent": pass1.index("agent.run()") < pass1.index(
            "correct, reason = evaluate_pass"
        ),
        "report_has_26_checks": report["checks_total"] == 26 and report["checks_passed"] == 26,
        "report_makes_no_official_score_claim": (
            report["conclusions"]["research_adaptation_score_is_official_phmforge_score"]
            is False
            and report["fully_qualified_original_tasks"] == 0
        ),
        "report_has_zero_new_model_work": all(
            report[key] == 0
            for key in (
                "new_LLM_calls",
                "new_Agent_sessions",
                "new_training_jobs",
                "new_scientific_scores",
            )
        ),
    }

    result = {
        "audit_id": "A251-independent",
        "source_file_checks": match_rows,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
    }
    output = run / "audit/INDEPENDENT_VERIFY251.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"],
                      "all_passed": result["all_passed"]}))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
