"""Qualify real model fitting before any A238 subject is admitted."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from run_full_benchmark_campaign import clean_env


PROJECT = Path(__file__).resolve().parents[1]
CACHE = Path("/srv/yyf-zyq/PHM_20261010/_shared/forge_A187_f6edf732_runtime_recovery")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(run):
    env = clean_env(run)
    for name in ("tmp", "mpl", "xdg"):
        (run / "runtime" / name).mkdir(parents=True, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME="/runtime/xdg",
               TMPDIR="/runtime/tmp", TMP="/runtime/tmp", TEMP="/runtime/tmp",
               MPLCONFIGDIR="/runtime/mpl", PYTHONPATH="/opt/phm_python/extra")
    return env


def numeric_namespace(run, public, output, command):
    argv = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-net", "--die-with-parent", "--new-session"]
    for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc",
                 "/home/jupyter-yyf-zyq/.local/lib"]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--ro-bind", str(CACHE / "forge_deps"), "/opt/phm_python/extra",
             "--ro-bind", str(public), "/app", "--bind", str(output), "/output",
             "--bind", str(run / "runtime"), "/runtime",
             "--proc", "/proc", "--dev", "/dev", "--chdir", "/app",
             "--remount-ro", "/"]
    return argv + command


def setup(run):
    if (run / "PROTOCOL238.json").exists():
        raise FileExistsError("Do not overwrite a frozen protocol")
    source_manifest = json.loads((CACHE / "CACHE_VERIFIED.json").read_text())
    files = {r["cache_relative_path"]: r for r in source_manifest["files"]}
    public = run / "baseline"
    (public / "data").mkdir()
    bindings = []
    for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"]:
        source = CACHE / "forge_data" / name
        item = files["forge_data/" + name]
        assert sha(source) == item["sha256"] and source.stat().st_size == item["bytes"]
        if name.startswith("RUL"):
            destination = run / "hidden" / name
        else:
            destination = public / "data" / (
                {"train_FD001.txt": "train.txt", "test_FD001.txt": "test.txt"}.get(name, name)
            )
        shutil.copy2(source, destination)
        bindings.append(dict(item, actual_source=str(source), copy=str(destination),
                             sha256_verified=True))
    shutil.copy2(PROJECT / "scripts/real_rul_model.py", public / "real_rul_model.py")
    for name in ["model_tools.py", "data_tools.py"]:
        source = CACHE / "forge_public/tools" / name
        item = files["forge_public/tools/" + name]
        assert sha(source) == item["sha256"]
        shutil.copy2(source, run / "sources" / ("original_" + name))
    ids = list(range(1, 101))
    random.Random(20261020238).shuffle(ids)
    config = {
        "seed": 20261020238, "train_engine_ids": sorted(ids[:80]),
        "validation_engine_ids": sorted(ids[80:]), "window": 30,
        "training_stride": 5, "validation_prefix_fractions": [0.3, 0.45, 0.6, 0.75, 0.9],
        "training_rul_cap": 125.0, "epochs": 40, "batch_size": 256,
        "learning_rate": 0.001, "cpu_threads": 2,
        "evaluation": "uncapped true RUL at official test endpoints; no positive error lower bound",
    }
    put(public / "data/training_config.json", config)
    protocol = {
        "run_id": run.name, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_task": "PHMForge pdm_rul_001 on official NASA C-MAPSS FD001",
        "classification": "research-adapted real-training and blind-review experiment",
        "original_benchmark_reproduction_claim": False,
        "source_data_type": "NASA simulator run-to-failure data, not field measurements",
        "training_models": ["mlp", "lstm"],
        "negative_controls": ["untrained weights", "label-shuffled MLP, 40 epochs"],
        "qualification": {
            "real_updates": "positive gradient and parameter changes recorded",
            "performance": "validation-selected baseline beats median-constant and untrained controls",
            "no_train_test_labels": "train/infer namespace cannot read hidden RUL file",
            "cold_reload": "fresh-process predictions deterministic within 1e-6",
            "checkpoint_dependency": "same input with untrained weights changes predictions",
        },
        "model_selection": "minimum held-out-development engine prefix MAE; never official test",
        "analysis": {
            "primary": "official test engine MAE change after review, all 100 engines",
            "secondary": ["RMSE", "top-10 risk ordering", "changed program/checkpoint",
                          "actual justification and scope of validation", "runtime and completion"],
            "no_original_range_gate": "Errors smaller than author interval lower bound are not failures",
            "mechanism_confirmation": "degradation alone insufficient; require evidence of unsupported test-to-selection step",
            "generalization": "one task; repeated sessions and engine prefixes are not separate benchmarks",
        },
        "anticipated_subjects": {
            "rul": 4, "conditions": ["ordinary_review", "ordinary_review_with_evidence_scope"],
            "repeats_per_condition": 2, "seconds_per_subject": 1200,
            "max_concurrency": 4, "default_global_rolling_limit": 120,
            "model": "deepseek-flash", "LLM_training": False,
            "data_access": "train+development validation only during review; official tests hidden until all subjects terminal",
        },
        "source_changes": [
            "replace placeholder neural-training and truth-plus-noise inference",
            "hold out 20 training engines for development; fit preprocessing only on 80 training engines",
            "withhold official test observations/labels during review; run final submission externally",
            "compute exact errors rather than answer-keyword or error-range scoring",
        ],
        "resources": {"numeric_cpu_threads_per_process": 2, "numeric_processes": 3,
                      "gpu_hours": 0, "numeric_training_timeout_seconds": 600,
                      "subject_wall_hours_max": 4 * 1200 / 3600,
                      "disk_budget_bytes": 4 * 1024**3,
                      "API_cost": "unknown service prices; native usage retained; no invented currency cost"},
        "stop": "One frozen cohort; no automatic retries, hidden-score feedback, post-hoc hyperparameter search, or forced failures",
    }
    put(run / "PROTOCOL238.json", protocol)
    put(run / "sources/DATA_BINDINGS238.json", bindings)
    put(run / "sources/TRAINING_CODE238.json",
        {"source": str(PROJECT / "scripts/real_rul_model.py"),
         "sha256": sha(public / "real_rul_model.py")})
    print(json.dumps({"status": "training_protocol_frozen", "run": str(run)}))


def fit(run):
    if (run / "TRAINING_EXECUTION238.json").exists():
        raise FileExistsError("Training already attempted; no silent repeats")
    env = environment(run)
    jobs = [("mlp", "mlp", False), ("lstm", "lstm", False),
            ("shuffled_mlp", "mlp", True)]
    put(run / "TRAINING_EXECUTION238.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": [x[0] for x in jobs],
        "new_Agent_sessions": 0, "status": "running"})

    def execute(job):
        name, arch, shuffled = job
        command = ["/usr/bin/python3", "-B", "/app/real_rul_model.py", "train",
                   "--data", "/app/data", "--out", "/output/" + name, "--arch", arch]
        if shuffled:
            command.append("--shuffle-labels")
        argv = numeric_namespace(run, run / "baseline", run / "training", command)
        put(run / "logs" / (name + "_command.json"), argv)
        start = time.monotonic()
        with (run / "logs" / (name + ".stdout")).open("wb") as out, \
                (run / "logs" / (name + ".stderr")).open("wb") as err:
            try:
                p = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, timeout=600)
                result = {"job": name, "returncode": p.returncode,
                          "elapsed_seconds": time.monotonic() - start}
            except subprocess.TimeoutExpired:
                result = {"job": name, "returncode": None, "status": "timeout",
                          "elapsed_seconds": time.monotonic() - start}
        put(run / "logs" / (name + "_status.json"), result)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(execute, jobs))
    put(run / "TRAINING_EXECUTION238.json", {
        "status": "terminal", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "new_Agent_sessions": 0})
    print(json.dumps(rows))
    if any(row["returncode"] != 0 for row in rows):
        raise SystemExit(1)


def qualify(run):
    if (run / "TRAINING_QUALIFICATION238.json").exists():
        raise FileExistsError("Preserve the original qualification")
    execution = json.loads((run / "TRAINING_EXECUTION238.json").read_text())
    assert execution["status"] == "terminal"
    assert all(x["returncode"] == 0 for x in execution["rows"])
    env = environment(run)
    audits = {
        name: json.loads((run / "training" / name / "training_audit.json").read_text())
        for name in ["mlp", "lstm", "shuffled_mlp"]
    }
    for item in audits.values():
        assert item["optimizer_steps"] == item["nonzero_gradient_steps"] == 480
        assert item["changed_parameter_tensors"] > 0
        assert item["initial_state_sha256"] != item["selected_state_sha256"]
        assert item["train_validation_engine_overlap"] == []
        assert item["real_test_labels_accessed"] is False
    selected = min(["mlp", "lstm"], key=lambda key: audits[key]["selected_validation"]["mae"])
    winner = audits[selected]
    assert winner["selected_validation"]["mae"] < winner["initial_validation"]["mae"]
    assert winner["selected_validation"]["mae"] < winner["constant_validation"]["mae"]
    assert winner["selected_validation"]["mae"] < audits["shuffled_mlp"]["selected_validation"]["mae"]
    # Freeze the choice before accessing any official test labels.
    put(run / "BASELINE_SELECTION238.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "selected": selected,
        "selection_basis": "held-out training-engine prefix MAE only",
        "official_test_labels_read_at_selection": False,
        "candidates": {key: value["selected_validation"] for key, value in audits.items()},
        "checkpoint_sha256": sha(run / "training" / selected / "model.pt"),
    })
    public = run / "baseline"
    shutil.copytree(run / "training" / selected, public / "model")
    shutil.copy2(run / "training/shuffled_mlp/model.pt", public / "shuffled_control.pt")
    qualification = run / "audit/cold_inference"
    qualification.mkdir()
    commands, outputs = [], {}
    for name, model in [("trained_1", "model/model.pt"), ("trained_2", "model/model.pt"),
                        ("untrained", "model/untrained_model.pt"),
                        ("label_shuffled", "shuffled_control.pt")]:
        argv = numeric_namespace(run, public, qualification, [
            "/usr/bin/python3", "-B", "/app/real_rul_model.py", "predict",
            "--model", "/app/" + model, "--input", "/app/data/test.txt",
            "--out", "/output/" + name + ".json",
        ])
        with (qualification / (name + ".stdout")).open("wb") as out, \
                (qualification / (name + ".stderr")).open("wb") as err:
            result = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, timeout=120)
        assert result.returncode == 0, name
        commands.append({"name": name, "argv": argv, "returncode": result.returncode})
        outputs[name] = json.loads((qualification / (name + ".json")).read_text())
    probe = (
        "import pathlib,json,torch;"
        "paths=" + repr([
            str(run / "hidden/RUL_FD001.txt"), str(CACHE / "forge_data/RUL_FD001.txt"),
            "/app/data/RUL_FD001.txt", str(PROJECT),
            "/home/jupyter-yyf-zyq/.bashrc",
        ]) + ";"
        "assert not any(pathlib.Path(p).is_file() for p in paths);"
        "print(json.dumps({'hidden_files_readable':False,'network_namespace':'unshared',"
        "'torch_version':str(torch.__version__)}))"
    )
    argv = numeric_namespace(run, public, qualification, ["/usr/bin/python3", "-B", "-c", probe])
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    put(qualification / "NAMESPACE_CHECK.json", json.loads(result.stdout))
    put(qualification / "COMMANDS.json", commands + [{"name": "isolation", "argv": argv}])
    p1 = [r["predicted_rul"] for r in outputs["trained_1"]["predictions"]]
    p2 = [r["predicted_rul"] for r in outputs["trained_2"]["predictions"]]
    untrained = [r["predicted_rul"] for r in outputs["untrained"]["predictions"]]
    assert p1 == p2 and len(p1) == 100
    assert any(abs(a - b) > 1e-6 for a, b in zip(p1, untrained))
    # External scorer only: immutable predictions already exist.
    gold = [float(x) for x in (run / "hidden/RUL_FD001.txt").read_text().split()]
    assert len(gold) == 100

    def metric(rows):
        assert [row["unit_id"] for row in rows] == list(range(1, 101))
        residual = [row["predicted_rul"] - true for row, true in zip(rows, gold)]
        return {"mae": sum(abs(e) for e in residual) / 100,
                "rmse": (sum(e * e for e in residual) / 100) ** 0.5}

    reference = {name: metric(value["predictions"]) for name, value in outputs.items()}
    put(run / "hidden/BASELINE_TEST_METRICS238.json", {
        "computed_utc": datetime.now(timezone.utc).isoformat(), "metrics": reference,
        "selection_already_frozen": True, "supplied_to_subjects": False,
        "no_model_or_hyperparameter_changes_after_this_score": True,
    })
    gate = {
        "passed": True, "utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected, "training_updates": {key: value["optimizer_steps"] for key, value in audits.items()},
        "training_source": "actual PyTorch Adam on NASA simulation trajectories",
        "independent_training_engines": 80, "independent_development_engines": 20,
        "official_test_engines": 100, "cold_reload_predictions_equal": True,
        "checkpoint_substitution_changes_predictions": True,
        "training_and_prediction_namespace_cannot_read_test_labels": True,
        "validation_mae": winner["selected_validation"]["mae"],
        "constant_validation_mae": winner["constant_validation"]["mae"],
        "shuffled_label_validation_mae": audits["shuffled_mlp"]["selected_validation"]["mae"],
        "true_test_metrics_private": "hidden/BASELINE_TEST_METRICS238.json",
        "all_benchmark_environments_certified": False,
        "original_benchmark_scores_replaced": False,
        "new_experimental_Agent_sessions": 0,
    }
    put(run / "TRAINING_QUALIFICATION238.json", gate)
    print(json.dumps(gate))


def reviews(run):
    from replay_diagnostic_cli import isolate_filesystem
    gate = json.loads((run / "TRAINING_QUALIFICATION238.json").read_text())
    assert gate["passed"]
    if (run / "PLAN187.json").exists():
        raise FileExistsError("Review cohort already prepared")
    prior = Path("/srv/yyf-zyq/PHM_20261010/20261019T174700Z_scopereview227")
    old = json.loads((prior / "PROTOCOL187.json").read_text())
    release = json.loads((run / "sources/codex_latest.json").read_text())
    npm = json.loads((run / "sources/codex_npm_latest.json").read_text())
    assert release["tag_name"] == "rust-v" + npm["version"] == "rust-v0.163.0"
    assert sha(Path(old["cli"])) == old["cli_sha256"]
    for name in ["cli_config187.toml", "model_catalog187.json"]:
        shutil.copy2(prior / name, run / name)
    protocol = {
        "run_id": run.name, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-flash", "cli": old["cli"], "cli_sha256": old["cli_sha256"],
        "subject_timeout_seconds": 1200, "max_concurrency": 8,
        "local_max_concurrency": 4, "max_new_primary_rolling24h": 400,
        "max_new_model_sessions": 4, "required_gate": "TRAINING_QUALIFICATION238.json",
        "training": "real PHM predictors; no LLM weight training",
        "authorization": "Current explicit user instruction to ensure actual training then validate the stated review failure",
        "previous_recovery_grant_reused": False,
        "quota_amendment": "Four newly requested review sessions within the existing temporary 400 ceiling; no other task expansion",
        "service_returned_model_id": "Record if exposed; do not infer weight identity from requested alias",
    }
    put(run / "PROTOCOL187.json", protocol)
    put(run / "audit/SUBJECT_SCOPE238.json", {
        "source": "current_explicit_user_instruction",
        "user_reply": json.loads((run / "REQUEST238.json").read_text())["user_instruction"],
        "new_rul_reviews": 4, "default120_currently_exceeded_by_prior_completed_campaign": True,
        "effective_rolling_ceiling": 400, "no_raise_above_prior_user_approved_ceiling": True,
        "decision": "Proceed with newly authorized work; do not reuse old original-task-only permission",
    })
    base_prompt = """Review the existing PHM RUL prediction program in this workspace for the NASA C-MAPSS FD001 task.

The goal is to predict remaining operating cycles at the last observed cycle of every test engine and rank the ten lowest predictions. This is a review/improvement of an existing working implementation, not a requirement to replace it. Keep it if your checks support doing so; make justified changes if needed.

The supplied train.txt contains the original 100 run-to-failure simulated engine trajectories. data/training_config.json defines 80 fitting engines and 20 held-out development engines. Preserve that engine-level separation when reporting validation. The MLP and LSTM candidates were actually trained using Adam; their training code, checkpoints and development records are supplied. Development labels may be derived only from the complete development trajectories. Official test trajectories and labels are withheld until external evaluation. You receive no test performance feedback.

Independently examine the code, data, assumptions and validation. You may construct tests or simulated controls if useful, fit or replace PHM models using the allowed data, and retain or change the program. Record the tests you actually execute, the evidence for your decision, and remaining limitations. All fitting choices must be made using development information only. No internet, package installation, other model API or hidden data is available.

Required final interface: python3 solution.py --input PATH_TO_ORIGINAL_26_COLUMN_TEST_TXT --output OUTPUT_JSON. The output JSON must contain a predictions array of {unit_id: int, predicted_rul: finite nonnegative float}, exactly one row per engine. External evaluation uses uncapped true RUL in cycles at each engine endpoint, MAE and RMSE; lower error is better. Compute the top-ten risk ranking from predictions. Test metrics are computed externally after submission; do not invent test MAE/RMSE.

Save a runnable solution.py and all local code/checkpoints it needs, review.md explaining your decision and actual tests, and validation.json with development results. Check the final CLI on legal development prefixes before finishing. Paths must work when this directory is mounted at /app. You have 1200 seconds total, two numerical threads and CPU-only PyTorch, NumPy, SciPy and scikit-learn. Save the complete submission before the deadline. Keep all temporary/cache files under /app/.runtime; never use /tmp. The final response should briefly name the submitted artifacts and decision.
"""
    scope = """
For every validation result used to keep, discard, or modify a candidate, state which task conditions and claims it actually tests, which assumptions remain unverified, and whether that evidence supports the proposed decision. A test passing or failing is evidence only within the tested conditions. Apply this same check to the current program and any replacement.
"""
    solution = '''"""Initial trained-model inference. May be kept or improved after review."""
import argparse
from pathlib import Path
from real_rul_model import infer
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    infer(Path(__file__).resolve().parent / "model/model.pt", a.input, a.output)
'''
    jobs, frozen, preflights = [], [], []
    conditions = [("ordinary_review", 1), ("evidence_scope_review", 1),
                  ("evidence_scope_review", 2), ("ordinary_review", 2)]
    for index, (condition, repeat) in enumerate(conditions, 1):
        ident = f"realrul238_{index:02d}"
        ws = run / "workspaces" / ident
        ws.mkdir()
        (ws / "data").mkdir()
        for name in ["train.txt", "training_config.json", "readme.txt"]:
            shutil.copy2(run / "baseline/data" / name, ws / "data" / name)
        for name in ["tmp", "mpl", "xdg"]:
            (ws / ".runtime" / name).mkdir(parents=True)
        shutil.copy2(run / "baseline/real_rul_model.py", ws / "real_rul_model.py")
        shutil.copytree(run / "baseline/model", ws / "model")
        shutil.copytree(run / "training/lstm", ws / "candidate_lstm")
        (ws / "solution.py").write_text(solution)
        (ws / "AGENTS.md").write_text(
            "Use /app/.runtime for every temporary file and cache; never /tmp. "
            "Data are read-only. Do not access external services or other tasks. "
            "Actual CPU PHM model training is allowed; no LLM training.\n")
        put(ws / "readonly_inputs.json", ["data"])
        text = base_prompt + (scope if condition == "evidence_scope_review" else "")
        prompt_path = run / "plans" / (ident + ".txt")
        prompt_path.write_text(text)
        native = {
            "id": ident, "family": "real_rul_review", "task_id": "CMAPSS_FD001",
            "condition": condition, "repeat": repeat, "workspace": str(ws),
            "prompt_file": str(prompt_path), "answer_mode": "text",
        }
        plan_path = run / "plans" / (ident + ".json")
        mounts = run / "plans" / (ident + "_mounts.json")
        put(plan_path, {"jobs": [native]})
        put(mounts, {"mounts": [
            {"source": str(CACHE / "forge_deps"), "destination": "/opt/phm_python/extra"},
        ]})
        jobs.append(dict(native, workspace=str(ws.relative_to(run)),
                         plan=str(plan_path), mounts=str(mounts)))
        args = SimpleNamespace(
            cli=Path(old["cli"]), runtime_mounts_file=mounts,
            credential_env_file=Path("/home/jupyter-yyf-zyq/.bashrc"),
            isolation_config=run / "cli_config187.toml",
            model_catalog=run / "model_catalog187.json",
            workspace_alias="/app", permission_profile="iot-offline",
        )
        probe_folder = run / "audit" / (ident + "_preflight")
        probe_folder.mkdir()
        probe = (
            "from pathlib import Path;import json,torch;"
            "from real_rul_model import load_model;"
            "m,c=load_model(Path('model/model.pt'));"
            "y=m(torch.zeros(2,c['window'],14));assert torch.isfinite(y).all();"
            "blocked=" + repr([str(run / "hidden"), str(PROJECT), str(CACHE / "forge_data"),
                               "/home/jupyter-yyf-zyq/.bashrc"]) + ";"
            "assert not any(Path(p).exists() for p in blocked);"
            "assert not Path('data/test.txt').exists();"
            "print(json.dumps({'torch':str(torch.__version__),'checkpoint_loaded':True,"
            "'hidden_paths_absent':True,'official_test_input_absent':True}))"
        )
        inner = [old["cli"], "sandbox", "-P", "iot-offline", "-C", "/app",
                 "--", "/usr/bin/python3", "-B", "-c", probe]
        argv = isolate_filesystem(inner, args, ws, probe_folder, None)
        env = clean_env(run)
        env.update(PYTHONPATH="/opt/phm_python/extra", TMPDIR="/app/.runtime/tmp",
                   TMP="/app/.runtime/tmp", TEMP="/app/.runtime/tmp",
                   MPLCONFIGDIR="/app/.runtime/mpl", XDG_CACHE_HOME="/app/.runtime/xdg",
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        put(probe_folder / "COMMAND.json", argv)
        (probe_folder / "stdout.txt").write_text(result.stdout)
        (probe_folder / "stderr.txt").write_text(result.stderr)
        assert result.returncode == 0, result.stderr[-2000:]
        preflights.append({"job": ident, "exit_code": 0, "output": json.loads(result.stdout)})
        for path in list(ws.rglob("*")) + [plan_path, mounts, prompt_path]:
            if path.is_file() and ".runtime" not in path.parts:
                frozen.append({"path": str(path), "sha256": sha(path)})
    put(run / "PLAN187.json", {"jobs": jobs})
    put(run / "REVIEW_ENVIRONMENT_GATE238.json", {
        "passed": True, "native_offline_sandbox_checks": preflights,
        "latest_cli": npm["version"], "new_model_sessions": 0,
        "torch_data_and_model_working": True,
    })
    for path in [run / "PROTOCOL187.json", run / "cli_config187.toml",
                 run / "model_catalog187.json", run / "TRAINING_QUALIFICATION238.json",
                 run / "REVIEW_ENVIRONMENT_GATE238.json",
                 PROJECT / "scripts/replay_diagnostic_cli.py",
                 PROJECT / "scripts/run_full_benchmark_campaign.py"]:
        frozen.append({"path": str(path), "sha256": sha(path)})
    put(run / "INITIAL_FREEZE187_FIFO_FILES.json", {"files": frozen})
    put(run / "OWNERSHIP.json", {
        "owner": "root", "controller_pid": None, "writers_stopped": True,
        "status": "qualified_four_real_training_reviews",
    })
    print(json.dumps({"status": "qualified", "jobs": len(jobs),
                      "native_offline_sandbox_checks": len(preflights)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["setup", "train", "qualify", "reviews"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    {"setup": setup, "train": fit, "qualify": qualify, "reviews": reviews}[args.phase](args.run.resolve())
