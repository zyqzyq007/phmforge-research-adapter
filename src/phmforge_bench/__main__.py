"""Run source-bound PHM research tasks and isolated external evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .data import prepare_cmapss, prepare_mfpt, resource_json, write_json
from .runner import evaluate, train_reference, workspace


def main():
    pdmb_sources = resource_json("pdmb_fault_sources.json")
    parser = argparse.ArgumentParser(prog="phmforge-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="List integrated tasks and the original scenario denominator")
    ls.add_argument("--all", action="store_true", help="Show the historical 75-scenario inventory")
    prep = sub.add_parser("prepare", help="Download/verify a source and create a reproducible split")
    prep.add_argument("--dataset", choices=["mfpt", "femto-rul", "xjtu-rul", "xjtu-terminal", "ldm-rul", "ims-endpoint", "engine-qa", *(f"cmapss-fd00{i}" for i in range(1, 5)),
                                           *pdmb_sources], required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--source", type=Path, help="Pinned offline source file or source directory (see dataset documentation)")
    prep.add_argument("--challenge-pdf", type=Path, help="Existing pinned FEMTO challenge document (femto-rul only)")
    prep.add_argument("--extractor", type=Path, help="Path to a compatible 7zz/7z executable (XJTU and IMS)")
    prep.add_argument("--workers", type=int, default=4)
    ws = sub.add_parser("workspace", help="Create a training/development-only agent workspace")
    ws.add_argument("--data-root", type=Path, required=True)
    ws.add_argument("--out", type=Path, required=True)
    ref = sub.add_parser("reference", help="Train a genuine MLP reference submission")
    ref.add_argument("--data-root", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--epochs", type=int, default=10)
    score = sub.add_parser("evaluate", help="Cold-run a frozen submission and independently score it")
    score.add_argument("--data-root", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--timeout", type=int, default=300)
    both = sub.add_parser("run-reference", help="Train and externally evaluate a real reference")
    both.add_argument("--data-root", type=Path, required=True)
    both.add_argument("--run-dir", type=Path, required=True)
    both.add_argument("--epochs", type=int, default=10)
    both.add_argument("--timeout", type=int, default=300)
    agent = sub.add_parser("agent", help="Run Codex/deepseek-flash in an isolated training workspace and score its submission")
    agent.add_argument("--data-root", type=Path, required=True)
    agent.add_argument("--run-dir", type=Path, required=True)
    agent.add_argument("--cli", type=Path, required=True, help="Path to a native Codex CLI executable")
    agent.add_argument("--model-base-url", default="https://api.deepseek.com/", help="HTTPS Responses-compatible provider endpoint")
    agent.add_argument("--epochs", type=int, default=1, help="Epochs for the initial reference submission")
    agent.add_argument("--timeout", type=int, default=1200)
    agent.add_argument("--max-requests", type=int, default=40)
    agent.add_argument("--max-input-tokens", type=int, default=4000000)
    agent.add_argument("--max-output-tokens", type=int, default=65536)
    agent.add_argument("--max-output-per-request", type=int, default=8192)
    agent.add_argument("--smoke", action="store_true", help="Check plumbing with the fitted starter; not a scientific Agent comparison")
    serve = sub.add_parser("serve", help="Run the PHMForge-compatible MCP/JSONL server")
    serve.add_argument("--data-root", type=Path, required=True)
    serve.add_argument("--out", type=Path, required=True)
    serve.add_argument("--transport", choices=["mcp", "jsonl"], default="mcp")
    scenarios = sub.add_parser("scenario", help="Inspect and run explicitly scoped historical task adaptations")
    scenarios_sub = scenarios.add_subparsers(dest="scenario_command", required=True)
    scenarios_list = scenarios_sub.add_parser("list", help="List the default 55 cases, additional cases, or inspect one case")
    scenarios_list.add_argument("--task")
    scenarios_list.add_argument("--include-additional", action="store_true", help="Include seven related-source partial adaptations")
    scenarios_list.add_argument("--include-substitutions", action="store_true", help="Also include nine explicitly substituted datasets")
    scenarios_list.add_argument("--all", action="store_true", help="Include unavailable cases and show all 75 identities")
    scenarios_workspace = scenarios_sub.add_parser("workspace", help="Create an Agent workspace for one case")
    scenarios_workspace.add_argument("--task", required=True)
    scenarios_workspace.add_argument("--data-root", required=True, type=Path)
    scenarios_workspace.add_argument("--out", required=True, type=Path)
    scenarios_run = scenarios_sub.add_parser("run", help="Fit and evaluate the numeric part of one case")
    scenarios_run.add_argument("--task", required=True)
    scenarios_run.add_argument("--data-root", required=True, type=Path)
    scenarios_run.add_argument("--run-dir", required=True, type=Path)
    scenarios_run.add_argument("--epochs", type=int, default=10)
    scenarios_run.add_argument("--timeout", type=int, default=300)
    scenarios_run.add_argument("--submission", type=Path, help="Reuse a separately fitted submission for the same source dataset")
    scenarios_agent = scenarios_sub.add_parser("agent", help="Run a coding agent on one original question's numeric adaptation")
    scenarios_agent.add_argument("--task", required=True)
    scenarios_agent.add_argument("--data-root", required=True, type=Path)
    scenarios_agent.add_argument("--run-dir", required=True, type=Path)
    scenarios_agent.add_argument("--cli", required=True, type=Path)
    scenarios_agent.add_argument("--model-base-url", default="https://api.deepseek.com/")
    scenarios_agent.add_argument("--epochs", type=int, default=1)
    scenarios_agent.add_argument("--timeout", type=int, default=1200)
    scenarios_agent.add_argument("--max-requests", type=int, default=40)
    scenarios_agent.add_argument("--max-input-tokens", type=int, default=4000000)
    scenarios_agent.add_argument("--max-output-tokens", type=int, default=65536)
    scenarios_agent.add_argument("--max-output-per-request", type=int, default=8192)
    scenarios_agent.add_argument("--smoke", action="store_true")
    scenarios_all = scenarios_sub.add_parser("run-all", help="Run the default 55 entries; additional sources require an explicit flag")
    scenarios_all.add_argument("--data-home", required=True, type=Path)
    scenarios_all.add_argument("--run-dir", required=True, type=Path)
    scenarios_all.add_argument("--epochs", type=int, default=10)
    scenarios_all.add_argument("--timeout", type=int, default=600)
    scenarios_all.add_argument("--prepare-missing", action="store_true", help="Download and prepare missing public sources; may require many GB")
    scenarios_all.add_argument("--source-map", type=Path, help="JSON map from dataset names to offline source paths")
    scenarios_all.add_argument("--extractor", type=Path, help="Path to 7zz/7z for XJTU and IMS")
    scenarios_all.add_argument("--workers", type=int, default=4)
    scenarios_all.add_argument("--task", action="append", help="Limit to the specified task(s); may be repeated")
    scenarios_all.add_argument("--include-additional", action="store_true", help="Include seven related-source partial adaptations")
    scenarios_all.add_argument("--include-substitutions", action="store_true", help="Also run nine explicitly substituted datasets")
    environment = sub.add_parser("environment", help="Restore and execute the full archived research environment")
    envsub = environment.add_subparsers(dest="environment_command", required=True)
    for name, help_text in [("verify", "Verify the complete object store"), ("list", "List actual archived task cases"),
                            ("restore", "Restore an original source/data/program tree"), ("replay", "Cold-run an actual recorded training or inference command"),
                            ("replay-case", "Replay every recorded final query of a task case"),
                            ("workspace", "Restore a recorded task's training-only workspace"),
                            ("agent", "Run a new coding-agent review of an archived submission")]:
        item = envsub.add_parser(name, help=help_text)
        item.add_argument("--store", required=True, type=Path)
        if name == "verify":
            item.add_argument("--full", action="store_true")
            item.add_argument("--workers", type=int, default=4)
        elif name == "list":
            item.add_argument("--dataset")
            item.add_argument("--records", action="store_true")
        elif name == "restore":
            item.add_argument("--path", required=True)
            item.add_argument("--out", required=True, type=Path)
        elif name == "replay":
            item.add_argument("--record", required=True)
            item.add_argument("--out", required=True, type=Path)
            item.add_argument("--timeout", type=int, default=300)
            item.add_argument("--no-compare", action="store_true", help="Execute without claiming historical prediction reproduction")
        else:
            item.add_argument("--case", required=True)
            item.add_argument("--out", required=True, type=Path)
            if name == "agent":
                item.add_argument("--cli", required=True, type=Path)
                item.add_argument("--model-base-url", required=True)
                item.add_argument("--timeout", type=int, default=1200)
                item.add_argument("--max-requests", type=int, default=40)
                item.add_argument("--max-input-tokens", type=int, default=4000000)
                item.add_argument("--max-output-tokens", type=int, default=65536)
                item.add_argument("--max-output-per-request", type=int, default=8192)
                item.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "environment":
        from .environment import main as environment_main
        result = environment_main(args)
    elif args.command == "scenario":
        from .scenario import create_workspace, inventory, run, run_all
        if args.scenario_command == 'list':
            result = inventory(args.task, args.include_additional, args.include_substitutions, args.all)
        elif args.scenario_command == 'workspace':
            result = create_workspace(args.task, args.data_root, args.out)
        elif args.scenario_command == 'run':
            result = run(args.task, args.data_root, args.run_dir, args.epochs, args.timeout, args.submission)
        elif args.scenario_command == 'agent':
            from .agent import run_agent
            result = run_agent(args.data_root, args.run_dir, args.cli, args.model_base_url,
                args.epochs, args.timeout, args.max_requests, args.max_input_tokens,
                args.max_output_tokens, args.max_output_per_request, args.smoke, scenario_task_id=args.task)
        else:
            source_map = json.loads(args.source_map.read_text()) if args.source_map else None
            if source_map is not None and not isinstance(source_map, dict):
                parser.error('--source-map must be a JSON object of source names to paths')
            result = run_all(args.data_home, args.run_dir, args.epochs, args.timeout,
                             args.prepare_missing, source_map, args.workers, args.extractor, args.task,
                             args.include_additional, args.include_substitutions)
    elif args.command == "list":
        inv = resource_json("scenario_inventory.json")
        runnable = [x for x in inv["tasks"] if x["release_status"] == "integrated_research_adaptation"]
        runnable.extend({"id": f"research_fd00{i}_rul", "dataset": f"CMAPSS_FD00{i}",
                         "release_status": "integrated_numeric_subtask"} for i in range(2, 5))
        runnable.append({"id": "research_femto_bearing_rul", "dataset": "FEMTO",
                         "release_status": "integrated_numeric_subtask"})
        runnable.append({"id": "research_xjtu_bearing_rul", "dataset": "XJTU",
                         "release_status": "integrated_numeric_subtask"})
        runnable.append({"id": "pdm_fault_002_terminal_adaptation", "dataset": "XJTU_TERMINAL",
                         "release_status": "integrated_numeric_subtask"})
        runnable.extend({"id": task, "dataset": name,
                         "release_status": "integrated_numeric_subtask"}
                        for task, name in [("research_engine_qa_component_health", "ENGINE_QA_HEALTH"),
                                           ("research_engine_qa_ten_cycle_trend", "ENGINE_QA_TREND")])
        runnable.append({"id": "research_ldm_bearing_rul", "dataset": "LDM_RTF",
                         "release_status": "integrated_numeric_subtask"})
        runnable.append({"id": "research_ims_recording_endpoint", "dataset": "IMS_ENDPOINT",
                         "release_status": "integrated_numeric_subtask"})
        runnable.extend({"id": item["task_id"], "dataset": item["dataset"],
                         "release_status": "integrated_numeric_subtask"} for item in pdmb_sources.values())
        result = {"original_scenarios": len(inv["tasks"]), "standalone_research_adaptations": len(runnable),
                  "runnable": runnable, "historical_counts_not_release_coverage": inv["historical_counts"]}
        if args.all:
            from .scenario import inventory as scenario_inventory
            current = {row['task_id']: row for row in scenario_inventory(include_unavailable=True)['tasks']}
            result['all_scenarios_note'] = 'Historical snapshot fields are not current release coverage; see current_source_status and scenario list --all.'
            result['all_scenarios'] = [{**{key: value for key, value in row.items() if key != 'release_status'},
                'historical_release_status': row['release_status'],
                'current_source_status': current[row['id']]['status'],
                'current_source_dataset': current[row['id']]['dataset'],
                'current_source_relation': current[row['id']]['source_relation']} for row in inv['tasks']]
    elif args.command == "prepare":
        if args.challenge_pdf and args.dataset != "femto-rul":
            parser.error("--challenge-pdf applies only to --dataset femto-rul")
        if args.extractor and args.dataset not in {"xjtu-rul", "xjtu-terminal", "ims-endpoint"}:
            parser.error("--extractor applies only to XJTU and IMS datasets")
        if args.dataset == "mfpt":
            result = prepare_mfpt(args.data_root, args.source, args.workers)
        elif args.dataset == "femto-rul":
            from .femto import prepare_femto
            result = prepare_femto(args.data_root, args.source, args.challenge_pdf)
        elif args.dataset == "xjtu-rul":
            from .xjtu import prepare_xjtu
            result = prepare_xjtu(args.data_root, args.source, args.workers, args.extractor)
        elif args.dataset == "xjtu-terminal":
            from .xjtu import prepare_xjtu_terminal
            result = prepare_xjtu_terminal(args.data_root, args.source, args.workers, args.extractor)
        elif args.dataset == "engine-qa":
            from .engine import prepare_engine
            result = prepare_engine(args.data_root, args.source, args.workers)
        elif args.dataset == "ldm-rul":
            from .ldm import prepare_ldm
            result = prepare_ldm(args.data_root, args.source, args.workers)
        elif args.dataset == "ims-endpoint":
            from .ims import prepare_ims
            result = prepare_ims(args.data_root, args.source, args.extractor)
        elif args.dataset.startswith("cmapss-"):
            result = prepare_cmapss(args.data_root, args.source, args.dataset.split("-", 1)[1])
        else:
            from .pdmb_fault import prepare_pdmb_fault
            result = prepare_pdmb_fault(args.data_root, args.dataset, args.source)
    elif args.command == "workspace":
        result = workspace(args.data_root, args.out)
    elif args.command == "reference":
        result = train_reference(args.data_root, args.out, args.epochs)
    elif args.command == "evaluate":
        result = evaluate(args.data_root, args.submission, args.out, args.timeout)
    elif args.command == "agent":
        from .agent import run_agent
        result = run_agent(args.data_root, args.run_dir, args.cli, args.model_base_url,
            args.epochs, args.timeout, args.max_requests, args.max_input_tokens,
            args.max_output_tokens, args.max_output_per_request, args.smoke)
    elif args.command == "run-reference":
        run = args.run_dir.resolve()
        if run.exists():
            raise FileExistsError(run)
        run.mkdir(parents=True)
        try:
            fit = train_reference(args.data_root, run / "submission", args.epochs)
            scored = evaluate(args.data_root, run / "submission", run / "evaluation", args.timeout)
            result = {"status": "completed", "training": fit, "evaluation": scored}
            write_json(run / "RUN.json", result)
        except Exception as exc:
            write_json(run / "RUN.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise
    else:
        worker = Path(__file__).resolve().parents[1] / "verified_benchmark_worker.py"
        command = [sys.executable, "-B", str(worker), "--family", "forge", "--data", str(args.data_root.resolve() / "public"),
                   "--output", str(args.out.resolve()), "--transport", args.transport]
        return subprocess.call(command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if args.command == 'scenario' and args.scenario_command == 'run-all' and result['status'] == 'completed_with_failures' else 0


if __name__ == "__main__":
    raise SystemExit(main())
