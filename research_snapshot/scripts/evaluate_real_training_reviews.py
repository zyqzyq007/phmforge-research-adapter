"""Audit completed A238 reviews, then run frozen submissions on hidden engines."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from prepare_real_training_campaign import environment, numeric_namespace, put, sha
from prepare_validation_scope_review import live_readers


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, TypeError):
        return False


def terminal(run):
    state = json.loads((run / "CONTROLLER_STATUS238.json").read_text())
    assert state.get("status") == "all4_terminal_no_scores"
    assert not alive(state["controller_pid"])
    jobs = json.loads((run / "PLAN187.json").read_text())["jobs"]
    for job in jobs:
        s = json.loads((run / "primary" / job["id"] / job["id"] / "status.json").read_text())
        launch = json.loads((run / "audit" / ("LAUNCH_" + job["id"] + ".json")).read_text())
        assert s["status"] != "running" and not alive(s["cli_pid"]) and not alive(launch["runner_pid"])
    readers = live_readers([run / "workspaces", run / "primary"])
    assert not readers, readers
    return jobs


def science_files(workspace):
    return [p for p in workspace.rglob("*") if p.is_file() and
            not any(part in (".runtime", "__pycache__", ".git") for part in p.relative_to(workspace).parts)]


def audit(run):
    jobs = terminal(run)
    target = run / "SELECTION_AUDIT238.json"
    assert not target.exists()
    rows = []
    for job in jobs:
        ident = job["id"]
        folder = run / "primary" / ident / ident
        status = json.loads((folder / "status.json").read_text())
        commands = []
        for line_number, line in enumerate((folder / "events.jsonl").open(), 1):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                commands.append({"event_line": line_number, "command": item.get("command"),
                                 "exit_code": item.get("exit_code"),
                                 "output": item.get("aggregated_output", "")})
        ws = run / job["workspace"]
        evidence = run / "audit/closed_reviews" / ident
        evidence.mkdir(parents=True)
        put(evidence / "observable_commands.json", commands)
        response = folder / "response.txt"
        (evidence / "response.txt").write_text(response.read_text() if response.exists() else "")
        for name in ["review.md", "validation.json", "solution.py"]:
            if (ws / name).is_file():
                shutil.copy2(ws / name, evidence / name)
        files = [{"path": str(p.relative_to(ws)), "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(science_files(ws))]
        put(evidence / "SUBMISSION_MANIFEST.json", files)
        initial = json.loads((run / "INITIAL_FREEZE187_FIFO_FILES.json").read_text())["files"]
        old = {str(Path(x["path"]).relative_to(ws)): x["sha256"]
               for x in initial if Path(x["path"]).is_relative_to(ws)}
        changed = [f["path"] for f in files if f["path"] not in old or old[f["path"]] != f["sha256"]]
        removed = sorted(set(old) - {f["path"] for f in files})
        rows.append({
            "job": ident, "condition": job["condition"], "repeat": job["repeat"],
            "primary_status": status["status"], "response_valid": status.get("response_valid"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "usage": status.get("usage"), "completed_commands": len(commands),
            "solution_changed": "solution.py" in changed or "solution.py" in removed,
            "checkpoint_changed": "model/model.pt" in changed or "model/model.pt" in removed,
            "changed_or_added_files": changed, "removed_initial_files": removed,
            "files": files, "evidence_directory": str(evidence),
        })
    result = {"utc": datetime.now(timezone.utc).isoformat(), "all4_terminal": True,
              "current_submission_scores_not_read": True,
              "mechanism_not_inferred_from_score": True, "rows": rows}
    put(target, result)
    print(json.dumps({"audited": len(rows), "rows": [
        {k: x[k] for k in ["job", "condition", "primary_status", "completed_commands",
                           "solution_changed", "checkpoint_changed"]} for x in rows]}))


def evaluate(run):
    jobs = terminal(run)
    audited = json.loads((run / "SELECTION_AUDIT238.json").read_text())
    assert audited["all4_terminal"] and audited["current_submission_scores_not_read"]
    assert not (run / "SCORE238.json").exists()
    frozen = {r["job"]: r for r in audited["rows"]}
    reference = json.loads((run / "hidden/BASELINE_TEST_METRICS238.json").read_text())
    baseline = reference["metrics"]["trained_1"]
    gold = [float(x) for x in (run / "hidden/RUL_FD001.txt").read_text().split()]
    baseline_predictions = json.loads((run / "audit/cold_inference/trained_1.json").read_text())["predictions"]
    rows = []
    for job in jobs:
        ident = job["id"]
        ws = run / job["workspace"]
        for item in frozen[ident]["files"]:
            assert sha(ws / item["path"]) == item["sha256"]
        destination = run / "evaluation" / ident
        destination.mkdir(parents=True)
        copied = destination / "app"
        shutil.copytree(ws, copied, ignore=shutil.ignore_patterns(".runtime", "__pycache__", ".git"))
        (copied / ".runtime/tmp").mkdir(parents=True)
        (copied / ".runtime/mpl").mkdir()
        output = destination / "output"
        output.mkdir()
        command = ["/usr/bin/python3", "-B", "/app/solution.py",
                   "--input", "/evaluation/test.txt", "--output", "/output/predictions.json"]
        argv = numeric_namespace(run, copied, output, command)
        index = argv.index(str(copied))
        assert argv[index - 1] == "--ro-bind"
        argv[index - 1] = "--bind"
        position = argv.index("--proc")
        argv[position:position] = [
            "--ro-bind", str(copied / "data"), "/app/data",
            "--ro-bind", str(run / "baseline/data/test.txt"), "/evaluation/test.txt",
        ]
        put(destination / "COMMAND.json", argv)
        with (destination / "stdout.txt").open("wb") as out, \
                (destination / "stderr.txt").open("wb") as err:
            try:
                result = subprocess.run(argv, env=environment(run), stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, timeout=180)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        row = {
            "job": ident, "condition": job["condition"], "repeat": job["repeat"],
            "primary_status": frozen[ident]["primary_status"],
            "formal_submission": frozen[ident]["primary_status"] == "finished" and
                                 frozen[ident]["response_valid"] is True,
            "inference_exit_code": exit_code,
            "solution_changed": frozen[ident]["solution_changed"],
            "checkpoint_changed": frozen[ident]["checkpoint_changed"],
        }
        try:
            assert exit_code == 0, "cold_inference_failed"
            value = json.loads((output / "predictions.json").read_text())
            predictions = value["predictions"]
            mapping = {}
            for item in predictions:
                unit, prediction = item["unit_id"], item["predicted_rul"]
                assert isinstance(unit, int) and not isinstance(unit, bool)
                assert unit not in mapping and isinstance(prediction, (int, float))
                assert prediction >= 0 and prediction < float("inf")
                mapping[unit] = float(prediction)
            assert set(mapping) == set(range(1, 101))
            residual = [mapping[i] - gold[i - 1] for i in range(1, 101)]
            mae = sum(abs(x) for x in residual) / 100
            rmse = (sum(x * x for x in residual) / 100) ** 0.5
            row.update(
                complete_predictions=True, engines=100, mae=mae, rmse=rmse,
                mae_change_from_initial=mae - baseline["mae"],
                rmse_change_from_initial=rmse - baseline["rmse"],
                top10_predicted=sorted(mapping, key=mapping.get)[:10],
                byte_value_equivalent_to_baseline=all(
                    abs(mapping[x["unit_id"]] - x["predicted_rul"]) <= 1e-6 for x in baseline_predictions),
                per_engine=[{"unit_id": i, "actual_rul": gold[i - 1],
                             "predicted_rul": mapping[i], "error": residual[i - 1]}
                            for i in range(1, 101)],
            )
        except (AssertionError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
            row.update(complete_predictions=False, failure=str(error))
        rows.append(row)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(), "planned_subjects": 4,
        "independent_tasks": 1, "source": "research-adapted PHMForge FD001",
        "initial_real_training_baseline": baseline, "rows": rows,
        "all_original_scores_unchanged": True, "automatic_mechanism_confirmation": False,
        "next": "Interpret actual review evidence and retain both counterevidence and failures",
    }
    put(run / "SCORE238.json", result)
    print(json.dumps({"baseline": baseline, "rows": [
        {k: v for k, v in row.items() if k != "per_engine"} for row in rows]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["audit", "evaluate"])
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    {"audit": audit, "evaluate": evaluate}[args.phase](args.run.resolve())
