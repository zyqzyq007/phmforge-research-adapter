#!/usr/bin/env python3
"""Prepare, run, and externally score the first PHMForge coding-Agent batch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env, launch, status_path


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4")
DEPS = Path("/srv/yyf-zyq/PHM_20261010/_shared/forge_A187_f6edf732_runtime_recovery/forge_deps")
CLI = Path("/data/yyf-zyq/PHM_20261010/20261015T215024Z_independent95/runtime/codex0163/package_files/package/vendor/x86_64-unknown-linux-musl/bin/codex")
CLI_SHA256 = "399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215"
MODEL_CATALOG = Path("/data/yyf-zyq/PHM_20261010/20261017T030500Z_multibench128/runtime/deepseek_cli_catalog128.json")
A251 = Path("/data/yyf-zyq/PHM_20261010/20261020T172244Z_phmforgeaccess251/RUN_CONTENTS.tar.gz")

TASKS = [
    {
        "job": "forgefault252_001",
        "task_id": "pdm_fault_001",
        "dataset": "CWRU",
        "archive": "/data/yyf-zyq/PHM_20261010/20261020T153929Z_cwru248/RUN_CONTENTS.tar.gz",
        "data_prefix": "dataset",
        "model_member": "models/mlp30/model.pt",
        "audit_member": "models/mlp30/training_audit.json",
        "labels_member": "hidden/test_labels248.json",
        "baseline": {"accuracy": 0.7821556496255292, "macro_f1": 0.5432516854469674},
    },
    {
        "job": "forgefault252_013",
        "task_id": "pdm_fault_013",
        "dataset": "MFPT",
        "archive": "/data/yyf-zyq/PHM_20261010/20261020T165033Z_mfpt013_250/RUN_CONTENTS.tar.gz",
        "data_prefix": "replay/datasets/MFPT",
        "model_member": "replay/outputs/MFPT/MFPT_mlp_6787354e15ff/model.pt",
        "audit_member": "replay/outputs/MFPT/MFPT_mlp_6787354e15ff/training_audit.json",
        "labels_member": "replay/external_labels/MFPT/labels.json",
        "baseline": {"accuracy": 0.9438596491228071, "macro_f1": 0.9571830985915493},
    },
    {
        "job": "forgefault252_015",
        "task_id": "pdm_fault_015",
        "dataset": "ELECTRICMOTORVIBRATIONS",
        "archive": "/data/yyf-zyq/PHM_20261010/20261020T161706Z_fault015_249/RUN_CONTENTS.tar.gz",
        "data_prefix": "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS",
        "model_member": "replay/outputs/ELECTRICMOTORVIBRATIONS/ELECTRICMOTORVIBRATIONS_mlp_4f83024fde03/model.pt",
        "audit_member": "replay/outputs/ELECTRICMOTORVIBRATIONS/ELECTRICMOTORVIBRATIONS_mlp_4f83024fde03/training_audit.json",
        "labels_member": "replay/external_labels/ELECTRICMOTORVIBRATIONS/labels.json",
        "baseline": {"accuracy": 0.6687306501547987, "macro_f1": 0.6294689343977786},
    },
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_bytes(bundle: Path, member: str) -> bytes:
    with tarfile.open(bundle, "r:gz") as handle:
        source = handle.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"{bundle}::{member}")
        return source.read()


def member_to(bundle: Path, member: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(member_bytes(bundle, member))


def scenario_question(task_id: str) -> tuple[str, str]:
    data = member_bytes(A251, "source/official/phm_scenarios.json")
    scenarios = json.loads(data)["pdm_scenarios"]
    item = next(x for x in scenarios if x["task_id"] == task_id)
    return item["input_question"], hashlib.sha256(item["input_question"].encode()).hexdigest()


def observations_without_labels(source: Path, destination: Path):
    data = np.load(source, allow_pickle=False)
    payload = {k: data[k] for k in data.files if k not in ("labels", "groups")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def labels_by_id(path: Path) -> dict[str, object]:
    rows = json.loads(path.read_text())
    result = {}
    for row in rows:
        ident = str(row["sample_id"])
        if ident in result:
            raise ValueError("Duplicate hidden label identity: " + ident)
        result[ident] = row.get("label", row.get("published_label"))
    return result


def npz_labels_by_id(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    ids = data["sample_ids"].tolist()
    labels = data["labels"].tolist()
    if len(ids) != len(labels) or len(set(map(str, ids))) != len(ids):
        raise ValueError("Invalid development labels")
    return {str(ident): label for ident, label in zip(ids, labels)}


def baseline_solution() -> str:
    return '''"""Frozen A246/A248 source-bound baseline inference."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "baseline"))
from real_fault_model import infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    infer(HERE / "baseline/model.pt", args.input, args.output)
'''


def agent_instructions() -> str:
    return """Use only this task workspace. Never use `/tmp`; use `.runtime/` for temporary files.
Do not access the internet, other experiment archives, hidden labels, a scorer, or prior task scores.
The `data/` and `baseline/` directories are read-only. You may read them and create new files elsewhere.
CPU PyTorch, NumPy, SciPy and scikit-learn are available. Do not train an LLM.
"""


def prompt_for(spec: dict, question: str, class_names: list) -> str:
    return f"""You are reviewing a real source-bound PHM fault-classification program for a research adaptation of PHMForge task `{spec['task_id']}`.

Original scenario:
{question}

The fixed public source is {spec['dataset']}. The target classes are represented by the source's reversible labels {class_names}. Do not invent physical fault names beyond the source metadata.

Available inputs:
- `data/train.npz`: fitting observations with `sample_ids`, `groups`, `labels`, and the model input features.
- `data/val_observations.npz`: development observations only; no labels or groups.
- `data/val_labels.json`: explicit development labels joined by `sample_id`.
- `data/dataset.json`: source, preprocessing, class and group metadata.
- `baseline/`: the frozen real MLP candidate, its training audit, and inference code.

The final test observations and labels are absent. You receive no test score or score feedback. Do not attempt to locate them.

Review the supplied candidate and its evidence. It is an existing working program, not something that must be replaced. Keep it if your checks support it; make justified changes if they improve the defensible task result. Use only fitting and development information. You may train a new CPU model, engineer features, perform validation, and create controls. Record what you actually ran. Group-aware validation is preferred where the metadata supports it because overlapping windows are not independent devices.

Required final interface:

`python3 solution.py --input PATH_TO_TEST_OBSERVATIONS_NPZ --output PATH_TO_PREDICTIONS_JSON`

The input will contain `sample_ids` and the same observation fields as development, but no labels or groups. The output JSON must contain a `predictions` list with exactly one object per input row:

`{{"sample_id": "...", "predicted_label": ...}}`

Labels must use the checkpoint/source label vocabulary. Every sample ID must occur exactly once. The process must work when this directory is mounted at `/app`. Save `solution.py`, all required code/checkpoints outside `.runtime`, `review.md` describing tests and decisions, and `validation.json` with development results. Keep every temporary or cache file under `.runtime`. You have 1200 seconds, two numerical threads, and CPU-only execution. Test metrics will be computed externally after your workspace is frozen; do not invent them.
"""


def copy_runtime_code(workspace: Path):
    baseline = workspace / "baseline"
    baseline.mkdir()
    for name in ("real_fault_model.py", "real_rul_model.py"):
        shutil.copy2(RUNTIME / "public/code" / name, baseline / name)


def prepare(run: Path):
    if (run / "PROTOCOL252.json").exists():
        raise FileExistsError("Frozen protocol already exists")
    for directory in ("audit", "hidden", "logs", "plans", "primary", "runtime/tmp", "runtime/mpl", "workspaces"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    if sha(CLI) != CLI_SHA256:
        raise RuntimeError("CLI binary hash changed")
    shutil.copy2(CLI, run / "audit/codex_binary")
    shutil.copy2(MODEL_CATALOG, run / "model_catalog252.json")
    shutil.copy2(MODEL_CATALOG, run / "model_catalog187.json")
    cli_config = """default_permissions = "iot-offline"
model_catalog_json = "/home/jupyter-yyf-zyq/.codex/models.json"
model = "deepseek-flash"
model_provider = "deepseek"
model_reasoning_effort = "max"
web_search = "disabled"
approval_policy = "never"

[model_providers.deepseek]
name = "DeepSeek"
base_url = "https://api.deepseek.com/"
env_key = "DEEPSEEK_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

[shell_environment_policy]
exclude = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]

[features]
apps = false
multi_agent = false

[permissions.iot-offline]

[permissions.iot-offline.filesystem]
":root" = "write"
":tmpdir" = "write"

[permissions.iot-offline.filesystem.":workspace_roots"]
"." = "write"

[permissions.iot-offline.network]
enabled = false
"""
    (run / "cli_config252.toml").write_text(cli_config)
    # replay_diagnostic_cli's quota-aware launcher uses these historical names.
    # Keep byte-identical aliases so the frozen 252 profile is what actually runs.
    (run / "cli_config187.toml").write_text(cli_config)

    jobs, frozen = [], []
    for spec in TASKS:
        bundle = Path(spec["archive"])
        ws = run / "workspaces" / spec["job"]
        data, hidden, baseline = ws / "data", run / "hidden" / spec["job"], ws / "baseline"
        data.mkdir(parents=True)
        hidden.mkdir(parents=True)
        copy_runtime_code(ws)
        prefix = spec["data_prefix"]
        for name in ("dataset.json", "train.npz", "val.npz", "test.npz"):
            member_to(bundle, f"{prefix}/{name}", run / "audit/source" / spec["job"] / name)
        shutil.copy2(run / "audit/source" / spec["job"] / "dataset.json", data / "dataset.json")
        shutil.copy2(run / "audit/source" / spec["job"] / "train.npz", data / "train.npz")
        observations_without_labels(run / "audit/source" / spec["job"] / "val.npz", data / "val_observations.npz")
        labels = npz_labels_by_id(run / "audit/source" / spec["job"] / "val.npz")
        write_json(data / "val_labels.json", labels)
        observations_without_labels(run / "audit/source" / spec["job"] / "test.npz", hidden / "test.npz")
        member_to(bundle, spec["labels_member"], hidden / "labels.json")
        member_to(bundle, spec["model_member"], baseline / "model.pt")
        member_to(bundle, spec["audit_member"], baseline / "training_audit.json")
        shutil.copy2(run / "audit/source" / spec["job"] / "test.npz", run / "audit/source" / spec["job"] / "test_labels_absent_check.npz")
        test_keys = set(np.load(hidden / "test.npz", allow_pickle=False).files)
        if "labels" in test_keys or "groups" in test_keys:
            raise ValueError("Test observations contain labels or groups")
        (ws / "solution.py").write_text(baseline_solution())
        (ws / "AGENTS.md").write_text(agent_instructions())
        question, question_sha = scenario_question(spec["task_id"])
        (ws / "task_question.md").write_text(question + "\n")
        metadata = json.loads((data / "dataset.json").read_text())
        text = prompt_for(spec, question, metadata["class_names"])
        prompt_path = run / "plans" / (spec["job"] + ".txt")
        prompt_path.write_text(text)
        write_json(ws / "readonly_inputs.json", ["data", "baseline", "task_question.md"])
        for sub in ("tmp", "mpl", "xdg"):
            (ws / ".runtime" / sub).mkdir(parents=True)
        native = {
            "id": spec["job"], "family": "phmforge_fault_review",
            "task_id": spec["task_id"], "condition": "ordinary_review", "repeat": 1,
            "workspace": str(ws.relative_to(run)), "prompt_file": str(prompt_path),
            "answer_mode": "text",
        }
        plan_path = run / "plans" / (spec["job"] + ".json")
        write_json(plan_path, {"jobs": [native]})
        mounts = run / "plans" / (spec["job"] + "_mounts.json")
        write_json(mounts, {"mounts": [{"source": str(DEPS), "destination": "/opt/phm_python/extra"}]})
        baseline_copy = run / "audit/baseline" / spec["job"]
        baseline_copy.mkdir(parents=True)
        shutil.copytree(ws, baseline_copy / "workspace")
        jobs.append(dict(native, plan=str(plan_path), mounts=str(mounts),
                         dataset=spec["dataset"], baseline_metrics=spec["baseline"],
                         question_sha256=question_sha))
        for path in ws.rglob("*"):
            if path.is_file() and ".runtime" not in path.parts:
                frozen.append({"path": str(path), "sha256": sha(path)})
        for path in (plan_path, mounts, prompt_path):
            frozen.append({"path": str(path), "sha256": sha(path)})

    write_json(run / "PLAN252.json", {"jobs": jobs})
    write_json(run / "INITIAL_FREEZE187_FIFO_FILES.json", {"files": frozen})
    protocol = {
        "run_id": run.name,
        "frozen_utc": utc(),
        "classification": "source-bound PHMForge classification coding-Agent research adaptation; official score not claimed",
        "model": "deepseek-flash",
        "cli": str(CLI),
        "cli_sha256": CLI_SHA256,
        "cli_version_online": "0.163.0",
        "cli_version_online_checked_utc": utc(),
        "subject_timeout_seconds": 1200,
        "max_new_model_sessions": 3,
        "max_concurrency": 8,
        "local_max_concurrency": 3,
        "max_new_primary_rolling24h": 120,
        "tasks": [x["task_id"] for x in TASKS],
        "authorization": "Current active goal instructs deepseek-flash coding-Agent runs on qualified PHMForge scenes",
        "visible": ["question", "train observations/labels", "development observations/labels", "frozen baseline", "dataset metadata"],
        "hidden": ["final test labels", "prior scores", "scorer", "other task workspaces"],
        "network": "disabled in CLI code-tool profile; outer process keeps only model API capability",
        "evaluation": "freeze all writers, invoke solution on label-free test observations, then join hidden labels",
        "automatic_retry": False,
        "new_Agent_sessions": 0,
        "new_training_jobs_by_researcher": 0,
        "official_original_benchmark_score_qualified": False,
        "fully_qualified_original_tasks": 0,
    }
    write_json(run / "PROTOCOL252.json", protocol)
    write_json(run / "OWNERSHIP.json", {"owner": "root", "controller_pid": None,
                                       "writers_stopped": True,
                                       "status": "prepared_not_launched"})
    print(json.dumps({"status": "prepared", "jobs": len(jobs), "new_model_sessions": 0}))


def preflight(run: Path, jobs: list[dict]):
    results = []
    env = clean_env(run)
    env.update(PYTHONPATH="/opt/phm_python/extra", TMPDIR="/app/.runtime/tmp",
               TMP="/app/.runtime/tmp", TEMP="/app/.runtime/tmp",
               MPLCONFIGDIR="/app/.runtime/mpl", XDG_CACHE_HOME="/app/.runtime/xdg",
               PYTHONDONTWRITEBYTECODE="1")
    for job in jobs:
        ws = run / job["workspace"]
        folder = run / "audit" / (job["id"] + "_preflight")
        folder.mkdir(parents=True, exist_ok=False)
        blocked = [str(run / "hidden"), str(run / "audit/source"),
                   str(run / "plans"), str(PROJECT),
                   "/home/jupyter-yyf-zyq/.bashrc"]
        probe = (
            "from pathlib import Path;import json,numpy as np,torch;"
            "print(json.dumps({'torch':str(torch.__version__),"
            "'val_has_no_labels':'labels' not in np.load('data/val_observations.npz',allow_pickle=False).files,"
            "'test_absent':not Path('data/test.npz').exists(),"
            "'hidden_absent':not any(Path(p).exists() for p in " + repr(blocked) + ")}))"
        )
        inner = [str(CLI), "sandbox", "-P", "iot-offline", "-C", "/app",
                 "--", "/usr/bin/python3", "-B", "-c", probe]
        args = SimpleNamespace(
            cli=CLI, runtime_mounts_file=Path(job["mounts"]),
            credential_env_file=None, isolation_config=run / "cli_config252.toml",
            model_catalog=run / "model_catalog252.json", workspace_alias="/app",
            permission_profile="iot-offline",
        )
        argv = isolate_filesystem(inner, args, ws, folder, None)
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        (folder / "stdout.txt").write_text(result.stdout)
        (folder / "stderr.txt").write_text(result.stderr)
        write_json(folder / "COMMAND.json", argv)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-3000:])
        output = json.loads(result.stdout.strip().splitlines()[-1])
        results.append({"job": job["id"], **output})
    write_json(run / "PREFLIGHT252.json", {"passed": True, "checks": results,
                                          "new_model_sessions": 0})
    print(json.dumps({"preflight": "passed", "jobs": len(results)}))


def run_jobs(run: Path):
    for name in ("cli_config187.toml", "model_catalog187.json"):
        if not (run / name).is_file():
            raise FileNotFoundError("Quota launcher requires frozen input: " + name)
    protocol = json.loads((run / "PROTOCOL252.json").read_text())
    jobs = json.loads((run / "PLAN252.json").read_text())["jobs"]
    preflight(run, jobs)
    if any(status_path(run, job).exists() for job in jobs):
        raise FileExistsError("A reserved or terminal session already exists")
    env = clean_env(run)
    pending = list(jobs)
    active = {}
    progress = {"controller_pid": os.getpid(), "started_utc": utc(),
                "planned": len(jobs), "launched": [], "completed": []}
    serial = 0
    while pending or active:
        for ident, (process, job) in list(active.items()):
            if process.poll() is None:
                continue
            with (run / "logs" / (ident + ".process_exit.json")).open("w") as handle:
                json.dump({"job": ident, "returncode": process.returncode,
                           "ended_utc": utc()}, handle, indent=2)
            path = status_path(run, job)
            row = json.loads(path.read_text()) if path.exists() else {"status": "runner_error_no_status"}
            progress["completed"].append({"job": ident, "status": row["status"],
                                          "runner_exit_code": process.returncode})
            del active[ident]
        while pending and len(active) < protocol["local_max_concurrency"]:
            serial += 1
            process, admission = launch(run, PROJECT, protocol, pending[0], serial, env)
            if process is None:
                progress["waiting"] = admission
                break
            job = pending.pop(0)
            active[job["id"]] = (process, job)
            progress["launched"].append(admission)
        progress.update(updated_utc=utc(), pending=len(pending), active_jobs=list(active))
        write_json(run / "CONTROLLER_STATUS252.json", progress)
        owner = json.loads((run / "OWNERSHIP.json").read_text())
        owner.update(controller_pid=os.getpid(), writers_stopped=False,
                     status="coding_agent_sessions_running_or_capacity_wait", heartbeat_utc=utc())
        write_json(run / "OWNERSHIP.json", owner)
        if pending or active:
            time.sleep(5)
    progress.update(status="all_selected_sessions_terminal_no_scoring", ended_utc=utc(),
                    terminal_counts=dict(Counter(x["status"] for x in progress["completed"])))
    write_json(run / "CONTROLLER_STATUS252.json", progress)
    owner.update(writers_stopped=True, status="terminal_waiting_freeze_then_external_score")
    write_json(run / "OWNERSHIP.json", owner)
    print(json.dumps({"status": progress["status"], "completed": progress["completed"]}))


def classification_score(actual, predicted):
    labels = sorted(set(actual) | set(predicted), key=lambda x: (str(type(x)), x))
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    matrix = []
    per_class = {}
    for label in labels:
        matrix.append([sum(a == label and p == target for a, p in zip(actual, predicted))
                       for target in labels])
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"support": tp + fn, "precision": precision,
                                 "recall": recall, "f1": f1}
    return {"n": len(actual), "accuracy": accuracy,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels),
            "balanced_accuracy": sum(x["recall"] for x in per_class.values()) / len(labels),
            "confusion_matrix": matrix, "class_order": labels, "per_class": per_class}


def execute_solution(run: Path, job: dict, workspace: Path, test: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    public = destination / "public"
    shutil.copytree(workspace, public, ignore=shutil.ignore_patterns(".runtime"))
    (public / "evaluation").mkdir()
    shutil.copy2(test, public / "evaluation/test_observations.npz")
    output = destination / "output"
    output.mkdir()
    for name in ("tmp", "mpl"):
        (output / ".runtime" / name).mkdir(parents=True)
    argv = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-net", "--die-with-parent", "--new-session"]
    for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--ro-bind", str(DEPS), "/opt/phm_python/extra",
             "--ro-bind", str(public), "/app", "--bind", str(output), "/output",
             "--proc", "/proc", "--dev", "/dev", "--chdir", "/app",
             "--remount-ro", "/",
             "--setenv", "PYTHONPATH", "/opt/phm_python/extra",
             "--setenv", "TMPDIR", "/output/.runtime/tmp",
             "--setenv", "TMP", "/output/.runtime/tmp",
             "--setenv", "TEMP", "/output/.runtime/tmp",
             "--setenv", "MPLCONFIGDIR", "/output/.runtime/mpl",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
             "/usr/bin/python3", "-B", "/app/solution.py",
             "--input", "/app/evaluation/test_observations.npz",
             "--output", "/output/predictions.json"]
    write_json(destination / "COMMAND.json", argv)
    start = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=240)
    write_json(destination / "STATUS.json", {"returncode": result.returncode,
                                             "elapsed_seconds": time.monotonic() - start})
    (destination / "stdout.txt").write_text(result.stdout)
    (destination / "stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        return {"status": "execution_error", "stderr_tail": result.stderr[-2000:]}
    prediction_path = output / "predictions.json"
    if not prediction_path.is_file():
        return {"status": "missing_predictions"}
    value = json.loads(prediction_path.read_text())
    return {"status": "finished", "predictions": value.get("predictions", value),
            "artifact_sha256": sha(prediction_path)}


def score(run: Path):
    if (run / "EVALUATION252.json").exists():
        raise FileExistsError("External evaluation already frozen")
    protocol = json.loads((run / "PROTOCOL252.json").read_text())
    jobs = json.loads((run / "PLAN252.json").read_text())["jobs"]
    controller = json.loads((run / "CONTROLLER_STATUS252.json").read_text())
    if controller.get("status") != "all_selected_sessions_terminal_no_scoring":
        raise RuntimeError("Agent sessions are not all terminal")
    rows = []
    for job in jobs:
        workspace = run / job["workspace"]
        session = status_path(run, job)
        state = json.loads(session.read_text())
        test = run / "hidden" / job["id"] / "test.npz"
        gold_rows = json.loads((run / "hidden" / job["id"] / "labels.json").read_text())
        gold = {str(x["sample_id"]): x.get("label", x.get("published_label")) for x in gold_rows}
        candidate = execute_solution(run, job, workspace, test, run / "evaluation" / job["id"] / "candidate")
        baseline_ws = run / "audit/baseline" / job["id"] / "workspace"
        baseline = execute_solution(run, job, baseline_ws, test, run / "evaluation" / job["id"] / "baseline")
        row = {"job": job["id"], "task_id": job["task_id"], "dataset": job["dataset"],
               "session_status": state["status"], "candidate_execution": candidate["status"],
               "baseline_execution": baseline["status"]}
        for name, result in (("candidate", candidate), ("baseline", baseline)):
            if result["status"] != "finished":
                continue
            predictions = result["predictions"]
            by_id = {str(x["sample_id"]): x["predicted_label"] for x in predictions}
            if len(by_id) != len(predictions) or set(by_id) != set(gold):
                raise ValueError(f"{name} prediction identities do not exactly cover hidden labels")
            ids = sorted(gold, key=str)
            metrics = classification_score([gold[i] for i in ids], [by_id[i] for i in ids])
            result["metrics"] = metrics
            row[name + "_metrics"] = metrics
        if "candidate_metrics" in row and "baseline_metrics" in row:
            row["delta"] = {
                "accuracy": row["candidate_metrics"]["accuracy"] - row["baseline_metrics"]["accuracy"],
                "macro_f1": row["candidate_metrics"]["macro_f1"] - row["baseline_metrics"]["macro_f1"],
                "balanced_accuracy": row["candidate_metrics"]["balanced_accuracy"] - row["baseline_metrics"]["balanced_accuracy"],
            }
            row["candidate_worse_than_baseline"] = (
                row["delta"]["accuracy"] < 0 or row["delta"]["macro_f1"] < 0
            )
        rows.append(row)
    result = {
        "evaluation_id": "A252",
        "utc": utc(),
        "classification": "post-freeze external numerical score of source-bound PHMForge research adaptations",
        "official_original_benchmark_score_qualified": False,
        "fully_qualified_original_tasks": 0,
        "sessions": len(rows),
        "finished_sessions": sum(x["session_status"] == "finished" for x in rows),
        "candidate_scoring_complete": sum("candidate_metrics" in x for x in rows),
        "baseline_scoring_complete": sum("baseline_metrics" in x for x in rows),
        "regressions": sum(bool(x.get("candidate_worse_than_baseline")) for x in rows),
        "new_confirmed_PHM_problems": 0,
        "rows": rows,
    }
    write_json(run / "EVALUATION252.json", result)
    print(json.dumps({k: result[k] for k in ("sessions", "finished_sessions",
                                             "candidate_scoring_complete", "baseline_scoring_complete",
                                             "regressions")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run", "score"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    {"prepare": prepare, "run": run_jobs, "score": score}[args.phase](run)


if __name__ == "__main__":
    main()
