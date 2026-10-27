"""Independent verification for the A247 RotorBrokenBar adaptation."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)
QUESTION_SHA256 = "284359e2776da2447fdb82c718a1694566f7c8fb1aae2cdaf90e808be8621091"
SOURCE_SHA256 = "a58317c3dca1caed67c491415dcf6653dbd3fe0210da6f2fd13d843dec0e3d37"
CLASSES = [0, 1, 2, 3, 4]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def close(left, right, tolerance=1e-12):
    return bool(abs(float(left) - float(right)) <= tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scenario-bundle", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    result = json.loads((run / "QUALIFICATION247.json").read_text())
    seals = json.loads((run / "PREDICTION_SEALS247.json").read_text())
    source = json.loads((run / "source/RotorBrokenBar.receipt.json").read_text())
    dataset_info = json.loads((run / "dataset/dataset.json").read_text())
    training = json.loads((run / "models/mlp30/training_audit.json").read_text())

    with tarfile.open(args.scenario_bundle, "r:gz") as stream:
        member = stream.getmember(SCENARIO_MEMBER)
        with stream.extractfile(member) as payload:
            scenarios = json.load(payload)["pdm_scenarios"]
    task = next(row for row in scenarios if row["task_id"] == "pdm_fault_004")
    question_sha = hashlib.sha256(task["input_question"].encode("utf-8")).hexdigest()

    paths = {
        "mlp30.json": run / "predictions/mlp30.json",
        "mlp30_cold_reload.json": run / "predictions/mlp30_cold_reload.json",
        "untrained.json": run / "predictions/untrained.json",
        "changed_input.json": run / "predictions/changed_input.json",
    }
    current_seals = {
        name: digest(path) for name, path in paths.items()
    }
    seal_hashes_equal = all(
        current_seals[name] == seals[name]["sha256"] for name in current_seals
    )

    prediction = json.loads(paths["mlp30.json"].read_text())
    repeat = json.loads(paths["mlp30_cold_reload.json"].read_text())
    untrained = json.loads(paths["untrained.json"].read_text())
    changed = json.loads(paths["changed_input.json"].read_text())
    prediction_ids = [row["sample_id"] for row in prediction["predictions"]]

    label_rows = json.loads((run / "hidden/test_labels247.json").read_text())
    by_id = {row["sample_id"]: row for row in label_rows}
    ordered = [by_id[sample_id] for sample_id in prediction_ids]
    actual = np.asarray([row["label"] for row in ordered], dtype=int)
    groups = np.asarray([row["group"] for row in ordered])
    predicted = np.asarray(
        [row["predicted_label"] for row in prediction["predictions"]], dtype=int
    )
    repeat_predictions = np.asarray(
        [row["predicted_label"] for row in repeat["predictions"]], dtype=int
    )
    untrained_predictions = np.asarray(
        [row["predicted_label"] for row in untrained["predictions"]], dtype=int
    )
    changed_predictions = np.asarray(
        [row["predicted_label"] for row in changed["predictions"]], dtype=int
    )

    accuracy = float(accuracy_score(actual, predicted))
    balanced = float(balanced_accuracy_score(actual, predicted))
    matrix = confusion_matrix(actual, predicted, labels=CLASSES).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    recorded = result["external_evaluation"]
    metric_match = (
        close(accuracy, recorded["accuracy"])
        and close(balanced, recorded["balanced_accuracy"])
        and close(macro_f1, recorded["macro_f1"])
        and matrix == recorded["confusion_matrix"]
    )
    per_class_match = True
    for index, label in enumerate(CLASSES):
        recorded_class = recorded["per_class"][str(label)]
        per_class_match &= (
            int(support[index]) == recorded_class["support"]
            and close(precision[index], recorded_class["precision"])
            and close(recall[index], recorded_class["recall"])
            and close(f1[index], recorded_class["f1_score"])
        )

    group_metrics = {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_metrics[group] = {
            "n": int(selected.sum()),
            "label": int(np.unique(actual[selected]).item()),
            "accuracy": float(accuracy_score(actual[selected], predicted[selected])),
        }
    group_match = all(
        group_metrics[group] == recorded["group_metrics"][group]
        for group in group_metrics
    )

    test = np.load(run / "dataset/test.npz", allow_pickle=False)
    expected_range = task["ground_truth"]["expected_accuracy_range"]
    checks = {
        "question_hash": question_sha == QUESTION_SHA256,
        "source_hash": digest(run / "source/RotorBrokenBar.csv") == SOURCE_SHA256,
        "source_receipt_matches_pin": (
            source["actual_sha256"] == SOURCE_SHA256
            and source["pinned_hash_verified"] is True
            and source["complete_file"] is True
        ),
        "prediction_seals_unchanged": seal_hashes_equal,
        "labels_absent_from_test_observations": "labels" not in test.files,
        "complete_unique_prediction_identity": (
            len(prediction_ids) == len(set(prediction_ids))
            and len(prediction_ids) == len(by_id)
            and set(prediction_ids) == set(by_id)
        ),
        "all_published_labels_present": set(actual.tolist()) == set(CLASSES),
        "class_names_preserved": dataset_info["class_names"] == CLASSES,
        "nonzero_parameter_updates": (
            training["optimizer_steps"] == 1050
            and training["nonzero_gradient_steps"] == 1050
            and training["changed_parameter_tensors"] == 6
        ),
        "cold_reload_byte_identical": (
            paths["mlp30.json"].read_bytes()
            == paths["mlp30_cold_reload.json"].read_bytes()
        ),
        "prediction_outputs_match_training": bool(np.array_equal(predicted, repeat_predictions)),
        "weight_dependency": bool(not np.array_equal(predicted, untrained_predictions)),
        "input_dependency": bool(not np.array_equal(predicted, changed_predictions)),
        "metrics_independently_match": bool(metric_match and per_class_match),
        "groups_independently_match": bool(group_match),
        "authored_range_evaluation": (
            recorded["accuracy_within_authored_range"]
            == bool(expected_range[0] <= accuracy <= expected_range[1])
        ),
    }

    verification = {
        "run_id": result["run_id"],
        "task_id": result["task_id"],
        "independent_implementation": "sklearn.metrics",
        "metrics": {
            "n": int(len(actual)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced,
            "confusion_matrix": matrix,
            "per_class": {
                str(label): {
                    "support": int(support[index]),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1_score": float(f1[index]),
                }
                for index, label in enumerate(CLASSES)
            },
            "group_metrics": group_metrics,
            "authored_accuracy_range": expected_range,
            "within_authored_range": bool(expected_range[0] <= accuracy <= expected_range[1]),
        },
        "prediction_hashes": current_seals,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "all_checks_passed": all(checks.values()),
        "classification": (
            "independent numeric verification of a source-bound five-label adaptation; "
            "not an official PHMForge score"
        ),
        "original_task_contract_qualified": False,
        "official_original_benchmark_score_qualified": False,
    }
    output = run / "INDEPENDENT_VERIFY247.json"
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "checks_passed": verification["checks_passed"],
        "checks_total": verification["checks_total"],
        "output": str(output),
    }))
    if not verification["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
