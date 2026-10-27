"""Qualify a five-label RotorBrokenBar classifier without changing the source.

This is a source-bound research adaptation, not an official PHMForge score.
The original task asks for broken-bar severity, while the pinned PDMBench
metadata describe a one-channel current binary task and the pinned CSV contains
five vibration channels with labels 0..4. The resulting semantic conflict is
reported explicitly rather than repaired by invention.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np


TASK_ID = "pdm_fault_004"
QUESTION_SHA256 = "284359e2776da2447fdb82c718a1694566f7c8fb1aae2cdaf90e808be8621091"
SOURCE_SHA256 = "a58317c3dca1caed67c491415dcf6653dbd3fe0210da6f2fd13d843dec0e3d37"
PDMB_COMMIT = "a3415dd5f2b7cb45ec7f973c2f330bdafd2dd006"
PDMB_DATASET_INFO_BLOB = "7c30ea21798beaa52d14dbb005b1726280dfdd94"
SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)
CLASSES = [0, 1, 2, 3, 4]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run_command(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=env["PHM_WORKSPACE"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def read_task(bundle):
    with tarfile.open(bundle, "r:gz") as stream:
        member = stream.getmember(SCENARIO_MEMBER)
        with stream.extractfile(member) as payload:
            scenarios = json.load(payload)["pdm_scenarios"]
    matches = [row for row in scenarios if row["task_id"] == TASK_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {TASK_ID}")
    return matches[0]


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
    }


def class_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    lookup = {label: index for index, label in enumerate(CLASSES)}
    for truth, estimate in zip(actual, predicted):
        matrix[lookup[int(truth)], lookup[int(estimate)]] += 1
    per_class = {}
    f1_values = []
    recall_values = []
    for label in CLASSES:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        per_class[str(label)] = {
            "support": int(np.sum(actual == label)),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    return {
        "n": int(len(actual)),
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--external-labels", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--scenario-bundle", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run = args.run.resolve()
    workspace = args.workspace.resolve()
    if run.parent != Path("/srv/yyf-zyq/PHM_20261010").resolve():
        raise ValueError("Run must be directly under the managed experiment root")
    if not run.is_dir() or any(run.iterdir()):
        raise ValueError("Run directory must already exist and be empty")

    status_path = run / "status.json"
    write_json(
        status_path,
        {
            "run_id": run.name,
            "status": "running",
            "started_utc": utc_now(),
            "model_sessions": 0,
            "training_jobs": 1,
        },
    )

    temporary = run / "tmp"
    temporary.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "MPLCONFIGDIR": str(run / "matplotlib-cache"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PHM_WORKSPACE": str(workspace),
        }
    )
    for key in ("TEST_LABELS", "LABELS_PATH", "GROUND_TRUTH_PATH"):
        environment.pop(key, None)
    (run / "clients-cache").mkdir()

    task = read_task(args.scenario_bundle)
    question_hash = hashlib.sha256(task["input_question"].encode("utf-8")).hexdigest()
    if question_hash != QUESTION_SHA256:
        raise ValueError("Original question hash changed")
    write_json(run / "original_task.json", task)

    source_sha = digest(args.source_csv)
    if source_sha != SOURCE_SHA256:
        raise ValueError("RotorBrokenBar source hash changed")
    receipt = json.loads(args.source_receipt.read_text())
    if (
        receipt["actual_sha256"] != SOURCE_SHA256
        or not receipt["pinned_hash_verified"]
        or not receipt["complete_file"]
    ):
        raise ValueError("Source receipt does not authenticate the pinned file")

    data = run / "dataset"
    data.mkdir()
    copied = []
    for name in ("train.npz", "val.npz", "test.npz", "changed_input.npz", "dataset.json"):
        copied.append(copy_file(args.prepared / name, data / name))
    hidden_labels = run / "hidden/test_labels247.json"
    copied.append(copy_file(args.external_labels, hidden_labels))
    copied.append(copy_file(args.source_csv, run / "source/RotorBrokenBar.csv"))
    copied.append(copy_file(args.source_receipt, run / "source/RotorBrokenBar.receipt.json"))

    code = run / "code"
    code.mkdir()
    copied.append(copy_file(workspace / "scripts/real_fault_model.py", code / "real_fault_model.py"))
    copied.append(copy_file(workspace / "scripts/real_rul_model.py", code / "real_rul_model.py"))
    copied.append(copy_file(workspace / "scripts/fault_model_cli_247.py", code / "fault_model_cli.py"))
    copied.append(copy_file(Path(__file__), code / Path(__file__).name))
    write_json(run / "SOURCE_AND_INPUTS247.json", copied)

    info = json.loads((data / "dataset.json").read_text())
    if info["class_names"] != CLASSES:
        raise ValueError("Unexpected published label set")
    test = np.load(data / "test.npz", allow_pickle=False)
    if "labels" in test.files:
        raise ValueError("Test observations exposed labels")
    test_ids = [str(value) for value in test["sample_ids"].tolist()]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate external sample identity")

    model_cli = code / "fault_model_cli.py"
    model_dir = run / "models/mlp30"
    prediction = run / "predictions/mlp30.json"
    repeat = run / "predictions/mlp30_cold_reload.json"
    untrained = run / "predictions/untrained.json"
    changed = run / "predictions/changed_input.json"
    run_command(
        [
            sys.executable,
            "-B",
            str(model_cli),
            "train",
            "--data",
            str(data),
            "--out",
            str(model_dir),
            "--arch",
            "mlp",
            "--epochs",
            str(args.epochs),
        ],
        run / "logs/train.log",
        environment,
    )
    for model, input_path, output_path, log_name in [
        (model_dir / "model.pt", data / "test.npz", prediction, "predict.log"),
        (model_dir / "model.pt", data / "test.npz", repeat, "cold_reload.log"),
        (model_dir / "untrained_model.pt", data / "test.npz", untrained, "untrained.log"),
        (model_dir / "model.pt", data / "changed_input.npz", changed, "changed_input.log"),
    ]:
        run_command(
            [
                sys.executable,
                "-B",
                str(model_cli),
                "predict",
                "--model",
                str(model),
                "--input",
                str(input_path),
                "--out",
                str(output_path),
            ],
            run / f"logs/{log_name}",
            environment,
        )

    seals = {
        path.name: {
            "sha256": digest(path),
            "sealed_utc": utc_now(),
            "labels_loaded_after_all_seals": True,
        }
        for path in (prediction, repeat, untrained, changed)
    }
    write_json(run / "PREDICTION_SEALS247.json", seals)

    label_rows = json.loads(hidden_labels.read_text())
    if len(label_rows) != len(test_ids):
        raise ValueError("External labels and observations differ in size")
    by_id = {row["sample_id"]: row for row in label_rows}
    if len(by_id) != len(label_rows) or set(by_id) != set(test_ids):
        raise ValueError("External labels do not cover the sealed observations")
    ordered = [by_id[sample_id] for sample_id in test_ids]
    actual = np.asarray([row["label"] for row in ordered], dtype=int)
    groups = np.asarray([row["group"] for row in ordered])

    payload = json.loads(prediction.read_text())
    predicted = np.asarray([row["predicted_label"] for row in payload["predictions"]], dtype=int)
    predicted_ids = [row["sample_id"] for row in payload["predictions"]]
    if predicted_ids != test_ids:
        raise ValueError("Prediction identities or order changed")
    metrics = class_metrics(actual, predicted)

    repeat_payload = json.loads(repeat.read_text())
    repeat_predictions = np.asarray(
        [row["predicted_label"] for row in repeat_payload["predictions"]], dtype=int
    )
    untrained_payload = json.loads(untrained.read_text())
    untrained_predictions = np.asarray(
        [row["predicted_label"] for row in untrained_payload["predictions"]], dtype=int
    )
    changed_payload = json.loads(changed.read_text())
    changed_predictions = np.asarray(
        [row["predicted_label"] for row in changed_payload["predictions"]], dtype=int
    )

    group_metrics = {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_metrics[group] = {
            "n": int(selected.sum()),
            "label": int(Counter(actual[selected].tolist()).most_common(1)[0][0]),
            "accuracy": float(np.mean(actual[selected] == predicted[selected])),
        }

    training_audit = json.loads((model_dir / "training_audit.json").read_text())
    expected_range = task["ground_truth"]["expected_accuracy_range"]
    checks = {
        "source_hash_matches_public_pin": True,
        "question_hash_matches_frozen_A187": True,
        "labels_absent_from_test_observations": "labels" not in test.files,
        "labels_read_after_all_prediction_seals": True,
        "complete_unique_external_identity_coverage": set(predicted_ids) == set(test_ids),
        "nonzero_parameter_updates": (
            training_audit["optimizer_steps"] > 0
            and training_audit["nonzero_gradient_steps"] > 0
            and training_audit["changed_parameter_tensors"] > 0
        ),
        "cold_reload_predictions_equal": bool(np.array_equal(predicted, repeat_predictions)),
        "weight_dependency": bool(not np.array_equal(predicted, untrained_predictions)),
        "input_dependency": bool(not np.array_equal(predicted, changed_predictions)),
        "train_validation_group_isolation": not bool(
            set(np.load(data / "train.npz", allow_pickle=False)["groups"].tolist())
            & set(np.load(data / "val.npz", allow_pickle=False)["groups"].tolist())
        ),
        "all_five_published_labels_present": set(actual.tolist()) == set(CLASSES),
        "finite_model_inputs": bool(np.isfinite(test["features"]).all()),
    }

    result = {
        "utc": utc_now(),
        "task_id": TASK_ID,
        "run_id": run.name,
        "classification": "source-bound five-label research adaptation; official PHMForge score not claimed",
        "model_sessions": 0,
        "training_jobs": 1,
        "question_sha256": question_hash,
        "source": {
            "sha256": source_sha,
            "public_pin": receipt["pin"],
            "source_rows": info["source_rows"],
            "channels": info["channels"],
            "labels": CLASSES,
            "group_definition": info["group_definition"],
        },
        "training": {
            "architecture": "mlp",
            "epochs_executed": training_audit["epochs_executed"],
            "optimizer_steps": training_audit["optimizer_steps"],
            "nonzero_gradient_steps": training_audit["nonzero_gradient_steps"],
            "changed_parameter_tensors": training_audit["changed_parameter_tensors"],
            "selected_epoch": training_audit["selected_epoch"],
            "selected_validation_accuracy": training_audit["selected_validation_accuracy"],
            "train_rows": training_audit["train_rows"],
            "validation_rows": training_audit["validation_rows"],
            "test_labels_accessed": training_audit["test_labels_accessed"],
        },
        "external_evaluation": {
            **metrics,
            "authored_accuracy_range": expected_range,
            "accuracy_within_authored_range": bool(
                expected_range[0] <= metrics["accuracy"] <= expected_range[1]
            ),
            "true_label_counts": dict(
                sorted((str(k), v) for k, v in Counter(actual.tolist()).items())
            ),
            "group_metrics": group_metrics,
        },
        "controls": checks,
        "control_summary": {
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
        },
        "contract_audit": {
            "original_question": {
                "asks_severity_1_to_4_broken_bars": True,
                "expected_fault_types": task["ground_truth"].get("expected_fault_types"),
                "expected_accuracy_range": expected_range,
            },
            "pdmb_metadata": {
                "commit": PDMB_COMMIT,
                "dataset_info_git_blob": PDMB_DATASET_INFO_BLOB,
                "declared_name": "Rotor Broken Bar",
                "declared_features": ["current"],
                "declared_channels": 1,
                "declared_fault_types": ["normal", "broken_bar"],
            },
            "observed_source": {
                "columns": ["Vib_acpe", "Vib_acpi", "Vib_axial", "Vib_base", "Vib_carc", "torque", "label"],
                "five_published_labels": True,
                "rows": info["source_rows"],
            },
            "semantic_conflict": (
                "PDMBench metadata says one current channel and two labels; the pinned CSV "
                "contains five vibration channels and five labels. The original prompt asks "
                "for four broken-bar severity levels. No label-to-severity map is supplied."
            ),
        },
        "qualification": {
            "source_bound_five_label_classifier_qualified": all(checks.values()),
            "research_adapted_task_contract_qualified": False,
            "official_original_benchmark_score_qualified": False,
            "fully_qualified_original_tasks": 0,
            "reason_task_contract_not_qualified": (
                "Real training and external scoring passed, but the pinned source metadata, "
                "observed channels and five labels, and the original severity wording cannot "
                "be reconciled without inventing a physical label map or changing the task."
            ),
        },
        "new_model_sessions": 0,
        "new_training_jobs": 1,
        "new_confirmed_PHM_problems": 0,
    }
    write_json(run / "QUALIFICATION247.json", result)
    write_json(
        status_path,
        {
            "run_id": run.name,
            "status": "finished",
            "finished_utc": utc_now(),
            "checks_passed": result["control_summary"]["checks_passed"],
            "checks_total": result["control_summary"]["checks_total"],
            "research_adapted_task_contract_qualified": False,
            "official_original_benchmark_score_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
