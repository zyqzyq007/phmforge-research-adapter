"""Run source-bound PHM research tasks and isolated external evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .data import prepare_cmapss, prepare_mfpt, resource_json, write_json
from .runner import evaluate, train_reference, workspace


def main():
    parser = argparse.ArgumentParser(prog="phmforge-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="List integrated tasks and the original scenario denominator")
    ls.add_argument("--all", action="store_true", help="Show the historical 75-scenario inventory")
    prep = sub.add_parser("prepare", help="Download/verify a source and create a reproducible split")
    prep.add_argument("--dataset", choices=["mfpt", "cmapss-fd001"], required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--source", type=Path, help="Existing verified MFPT source directory or CMAPSSData.zip")
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
    score.add_argument("--timeout", type=int, default=120)
    both = sub.add_parser("run-reference", help="Train and externally evaluate a real reference")
    both.add_argument("--data-root", type=Path, required=True)
    both.add_argument("--run-dir", type=Path, required=True)
    both.add_argument("--epochs", type=int, default=10)
    both.add_argument("--timeout", type=int, default=120)
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
    elif args.command == "list":
        inv = resource_json("scenario_inventory.json")
        runnable = [x for x in inv["tasks"] if x["release_status"] == "integrated_research_adaptation"]
        result = {"original_scenarios": len(inv["tasks"]), "standalone_research_adaptations": len(runnable),
                  "runnable": runnable, "historical_counts_not_release_coverage": inv["historical_counts"]}
        if args.all:
            result["all_scenarios"] = inv["tasks"]
    elif args.command == "prepare":
        result = (prepare_mfpt(args.data_root, args.source, args.workers) if args.dataset == "mfpt"
                  else prepare_cmapss(args.data_root, args.source))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
