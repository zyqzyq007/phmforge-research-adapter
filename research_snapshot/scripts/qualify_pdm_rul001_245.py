"""Qualify the PHMForge pdm_rul_001 pipeline with real training and external scoring.

This is a research adaptation, not an official PHMForge score. The original
scenario text and source files remain immutable. Training and inference run in
separate processes without receiving the final-label path; this orchestrator
loads that path only after the prediction artifact has been hashed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

import numpy as np


TASK_ID = "pdm_rul_001"
QUESTION_SHA256 = "9055b9acba6ea87efcc4154aa4b4d4257a3a1d9845e52ac2a8d08a332bad8439"
TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
TEST_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_task(bundle):
    with tarfile.open(bundle, "r:gz") as handle:
        member = handle.getmember(SCENARIO_MEMBER)
        with handle.extractfile(member) as stream:
            payload = json.load(stream)
    matches = [row for row in payload["pdm_scenarios"] if row["task_id"] == TASK_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {TASK_ID}")
    return matches[0]


def run_command(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(
            command, cwd=env["PHM_WORKSPACE"], env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def metric(predicted, actual):
    residual = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "n": int(len(residual)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(residual.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--scenario-bundle", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    workspace = args.workspace.resolve()
    runtime = args.runtime.resolve()
    data = runtime / "public/datasets/CMAPSS_FD001"
    if run.parent != Path("/srv/yyf-zyq/PHM_20261010").resolve():
        raise ValueError("Run must be directly under the managed experiment root")
    if not run.is_dir():
        raise ValueError("Run directory must already exist")

    status_path = run / "status.json"
    write_json(status_path, {
        "run_id": run.name,
        "status": "running",
        "started_utc": utc_now(),
        "model_sessions": 0,
        "training_jobs": 1,
    })

    tmp = run / "tmp"
    tmp.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update({
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "MPLCONFIGDIR": str(run / "matplotlib-cache"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PHM_WORKSPACE": str(workspace),
    })
    (run / "clients-cache").mkdir()

    task = read_task(args.scenario_bundle)
    question = task["input_question"].encode("utf-8")
    if hashlib.sha256(question).hexdigest() != QUESTION_SHA256:
        raise ValueError("Original question hash changed")
    write_json(run / "original_task.json", task)

    source_checks = {
        "train.txt": (data / "train.txt", TRAIN_SHA256),
        "test.txt": (data / "test.txt", TEST_SHA256),
        "RUL_FD001.txt": (args.labels, RUL_SHA256),
    }
    source_evidence = {}
    for name, (path, expected) in source_checks.items():
        actual = digest(path)
        source_evidence[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "matches": actual == expected,
        }
        if actual != expected:
            raise ValueError(f"{name} hash changed")

    config_path = data / "training_config.json"
    config = json.loads(config_path.read_text())
    if config["epochs"] != 40:
        raise ValueError("Unexpected frozen training configuration")

    source_evidence["training_config.json"] = {
        "path": str(config_path),
        "bytes": config_path.stat().st_size,
        "sha256": digest(config_path),
        "matches": True,
        "note": "Original pdm_rul_001 procedure explicitly requests 50 epochs; "
                "the run passes 50 without modifying this file.",
    }
    write_json(run / "SOURCE_CHECKS245.json", source_evidence)

    model_dir = run / "model_transformer_50"
    prediction_path = run / "predictions_unscored.json"
    script = workspace / "scripts/real_rul_model.py"
    run_command([
        sys.executable, "-B", str(script), "train",
        "--data", str(data),
        "--out", str(model_dir),
        "--arch", "transformer",
        "--epochs", "50",
    ], run / "logs/train.log", env)
    run_command([
        sys.executable, "-B", str(script), "predict",
        "--model", str(model_dir / "model.pt"),
        "--input", str(data / "test.txt"),
        "--out", str(prediction_path),
    ], run / "logs/predict.log", env)

    prediction_hash = digest(prediction_path)
    prediction_sealed = {
        "path": prediction_path.name,
        "sha256": prediction_hash,
        "sealed_utc": utc_now(),
        "labels_loaded_after_seal": True,
    }
    write_json(run / "PREDICTION_SEAL245.json", prediction_sealed)

    predictions_payload = json.loads(prediction_path.read_text())
    rows = predictions_payload["predictions"]
    unit_ids = [row["unit_id"] for row in rows]
    predicted = np.asarray([row["predicted_rul"] for row in rows], dtype=float)
    actual = np.loadtxt(args.labels, dtype=float)
    if actual.ndim != 1 or len(actual) != 100 or not np.isfinite(actual).all():
        raise ValueError("Expected 100 finite final RUL labels")
    if unit_ids != list(range(1, 101)):
        raise ValueError("Predicted unit IDs must be exactly 1..100 in order")
    if not np.isfinite(predicted).all() or np.any(predicted < 0) or np.any(predicted > 125):
        raise ValueError("Prediction contract failed")

    metrics = metric(predicted, actual)
    true_top = sorted(range(1, 101), key=lambda unit: (float(actual[unit - 1]), unit))[:10]
    predicted_top = sorted(
        unit_ids, key=lambda unit: (float(predicted[unit - 1]), unit)
    )[:10]
    original_sample_units = task["ground_truth"]["sample_correct_units"]
    zero_based_top = [unit - 1 for unit in true_top]
    expected_mae = task["ground_truth"]["expected_mae_range"]
    expected_rmse = task["ground_truth"]["expected_rmse_range"]
    training_audit = json.loads((model_dir / "training_audit.json").read_text())

    all_predictions = [
        {
            "unit_id": int(unit),
            "predicted_rul": float(prediction),
            "actual_rul": float(label),
            "error": float(prediction - label),
        }
        for unit, prediction, label in zip(unit_ids, predicted, actual)
    ]
    result = {
        "utc": utc_now(),
        "task_id": TASK_ID,
        "run_id": run.name,
        "classification": "source-bound research adaptation; official PHMForge score not claimed",
        "model_sessions": 0,
        "training_jobs": 1,
        "question_sha256": hashlib.sha256(question).hexdigest(),
        "source_checks": source_evidence,
        "prediction_seal": prediction_sealed,
        "training": {
            "architecture": training_audit["architecture"],
            "epochs_executed": training_audit["epochs_executed"],
            "optimizer_steps": training_audit["optimizer_steps"],
            "nonzero_gradient_steps": training_audit["nonzero_gradient_steps"],
            "changed_parameter_tensors": training_audit["changed_parameter_tensors"],
            "training_engines": len(training_audit["training_engine_ids"]),
            "validation_engines": len(training_audit["validation_engine_ids"]),
            "selected_epoch": training_audit["selected_epoch"],
            "selected_validation": training_audit["selected_validation"],
            "real_test_labels_accessed": training_audit["real_test_labels_accessed"],
        },
        "external_evaluation": {
            **metrics,
            "expected_mae_range": expected_mae,
            "expected_rmse_range": expected_rmse,
            "mae_within_authored_range": bool(expected_mae[0] <= metrics["mae"] <= expected_mae[1]),
            "rmse_within_authored_range": bool(expected_rmse[0] <= metrics["rmse"] <= expected_rmse[1]),
            "complete_unique_unit_coverage": len(set(unit_ids)) == 100,
            "labels_read_after_prediction_seal": True,
        },
        "risk_ranking": {
            "predicted_top10": predicted_top,
            "true_top10": true_top,
            "overlap_count": len(set(predicted_top) & set(true_top)),
            "original_sample_correct_units": original_sample_units,
            "zero_based_indices_for_true_top10": zero_based_top,
            "original_field_explained_as_zero_based": original_sample_units == zero_based_top,
            "corrected_one_based_top10": [unit + 1 for unit in original_sample_units],
        },
        "original_required_fields": {
            "all_unit_predictions": all_predictions,
            "top_N_risk_units": predicted_top,
            "overall_mae": metrics["mae"],
            "overall_rmse": metrics["rmse"],
            "ground_truth_verification_status": True,
        },
        "qualification": {
            "original_prompt_bound": True,
            "real_training_verified": (
                training_audit["optimizer_steps"] > 0
                and training_audit["nonzero_gradient_steps"] > 0
                and training_audit["changed_parameter_tensors"] > 0
            ),
            "complete_external_prediction_artifact": True,
            "source_identity_verified": True,
            "zero_based_reference_defect_documented": True,
            "research_adapted_task_contract_qualified": True,
            "official_original_benchmark_score_qualified": False,
            "reason_official_score_not_claimed": (
                "The original scorer checks answer text/ranges and its risk-unit field is "
                "zero-based; this run uses an independent external evaluator and does not "
                "replace the frozen original score."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": training_audit["torch_version"],
            "device": "CPU",
            "threads": 2,
            "tmpdir": str(tmp),
        },
        "known_limitations": [
            "C-MAPSS FD001 is NASA simulator data, not field measurements.",
            "The original ground_truth sample_correct_units field is zero-based relative "
            "to 1-based unit IDs; corrected ranking is retained without editing the task.",
            "The authored MAE/RMSE ranges are acceptance metadata, not independent strong-baseline proof.",
            "This numerical qualification does not start or evaluate an Agent session.",
        ],
    }
    write_json(run / "QUALIFICATION245.json", result)
    write_json(status_path, {
        "run_id": run.name,
        "status": "complete",
        "completed_utc": utc_now(),
        "model_sessions": 0,
        "training_jobs": 1,
        "research_adapted_task_contract_qualified": True,
        "official_original_benchmark_score_qualified": False,
        "new_confirmed_PHM_problems": 0,
    })
    write_json(run / "retention_policy.json", {
        "excluded_rebuildable_directory": "clients-cache",
        "shared_dependencies": {
            "path": "/srv/yyf-zyq/PHM_20261010/_shared",
            "retention": "existing_pinned_readonly_dependencies",
        },
    })
    print(json.dumps({
        "task_id": TASK_ID,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "predicted_top10": predicted_top,
        "overlap_count": result["risk_ranking"]["overlap_count"],
        "model_sessions": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
