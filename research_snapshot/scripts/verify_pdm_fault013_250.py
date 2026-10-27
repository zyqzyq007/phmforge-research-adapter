"""Independent sklearn verification for the A250 pdm_fault_013 binding."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
import sklearn
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
QUESTION_SHA256 = "e4e0aa6b13fdbf3239950c97b75f71aa15af1bf4969960dfe5d5db12c8136c9a"
SOURCE_SHA256 = "581c6cb1fb02dcd97a31ae9785061d178101bbc146fb8713d8c510466fb086ee"
CLASSES = [0, 1, 2]


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
    result = json.loads((run / "QUALIFICATION250.json").read_text())
    task_output = json.loads((run / "RESEARCH_OUTPUT250.json").read_text())
    receipt = json.loads(
        (run / "source/inputs/pdmb_attempt2/MFPT.receipt.json").read_text()
    )
    dataset_info = json.loads(
        (run / "replay/datasets/MFPT/dataset.json").read_text()
    )
    training = json.loads(
        (
            run
            / "replay/outputs/MFPT/MFPT_mlp_6787354e15ff/training_audit.json"
        ).read_text()
    )
    labels = json.loads(
        (run / "replay/external_labels/MFPT/labels.json").read_text()
    )
    primary = json.loads(
        (run / "replay/outputs/MFPT/predictions_117397e57214.json").read_text()
    )
    cold = json.loads(
        (run / "replay/outputs/MFPT/predictions_0e425b981347.json").read_text()
    )
    untrained = json.loads(
        (run / "replay/outputs/MFPT/predictions_a6faefdf0a59.json").read_text()
    )
    changed = json.loads(
        (run / "replay/outputs/MFPT/predictions_c982447247e9.json").read_text()
    )

    with tarfile.open(args.scenario_bundle, "r:gz") as stream:
        member = stream.getmember(SCENARIO_MEMBER)
        with stream.extractfile(member) as payload:
            scenarios = json.load(payload)["pdm_scenarios"]
    task = next(row for row in scenarios if row["task_id"] == "pdm_fault_013")
    question_sha = hashlib.sha256(task["input_question"].encode("utf-8")).hexdigest()

    label_by_id = {row["sample_id"]: row for row in labels}
    prediction_ids = [row["sample_id"] for row in primary["predictions"]]
    ordered = [label_by_id[sample_id] for sample_id in prediction_ids]
    actual = np.asarray([row["label"] for row in ordered], dtype=int)
    groups = np.asarray([row["group"] for row in ordered])
    predicted = np.asarray(
        [row["predicted_label"] for row in primary["predictions"]], dtype=int
    )
    cold_predicted = np.asarray(
        [row["predicted_label"] for row in cold["predictions"]], dtype=int
    )
    untrained_predicted = np.asarray(
        [row["predicted_label"] for row in untrained["predictions"]], dtype=int
    )
    changed_predicted = np.asarray(
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
    group_match = group_metrics == recorded["group_metrics"]

    test = np.load(run / "replay/datasets/MFPT/test.npz", allow_pickle=False)
    changed_npz = np.load(
        run / "replay/datasets/MFPT/changed_input.npz", allow_pickle=False
    )
    split_groups = dataset_info["train_validation_groups"]
    expected_range = task["ground_truth"]["expected_accuracy_range"]
    checks = {
        "question_hash": question_sha == QUESTION_SHA256,
        "source_hash": digest(run / "source/inputs/pdmb_attempt2/MFPT.csv")
        == SOURCE_SHA256,
        "source_receipt_matches_pin": (
            receipt["actual_sha256"] == SOURCE_SHA256
            and receipt["pinned_hash_verified"] is True
            and receipt["complete_file"] is True
        ),
        "labels_absent_from_test_observations": (
            "labels" not in test.files and "groups" not in test.files
        ),
        "complete_unique_prediction_identity": (
            len(prediction_ids) == len(set(prediction_ids))
            and len(prediction_ids) == len(label_by_id)
            and set(prediction_ids) == set(label_by_id)
            and set(prediction_ids) == set(test["sample_ids"].tolist())
        ),
        "all_published_labels_present": set(actual.tolist()) == set(CLASSES),
        "class_names_preserved": dataset_info["class_names"] == CLASSES,
        "group_isolation": (
            not set(split_groups["train"]) & set(split_groups["val"])
            and not set(split_groups["train"]) & set(groups.tolist())
            and not set(split_groups["val"]) & set(groups.tolist())
        ),
        "each_group_single_class": all(
            len(set(actual[groups == group].tolist())) == 1
            for group in set(groups.tolist())
        ),
        "nonzero_parameter_updates": (
            training["optimizer_steps"] == 850
            and training["nonzero_gradient_steps"] == 850
            and training["changed_parameter_tensors"] == 6
        ),
        "training_test_labels_isolated": training["test_labels_accessed"] is False,
        "primary_cold_bytes_equal": (
            (run / "replay/outputs/MFPT/predictions_117397e57214.json").read_bytes()
            == (run / "replay/outputs/MFPT/predictions_0e425b981347.json").read_bytes()
        ),
        "primary_cold_labels_equal": bool(
            np.array_equal(predicted, cold_predicted)
        ),
        "weight_dependency": bool(
            not np.array_equal(predicted, untrained_predicted)
        ),
        "input_dependency": bool(
            not np.array_equal(predicted, changed_predicted)
        ),
        "changed_input_identity": (
            changed_npz["sample_ids"].tolist() == test["sample_ids"].tolist()
        ),
        "changed_input_values": (
            not np.array_equal(changed_npz["features"], test["features"])
            or not np.array_equal(changed_npz["signals"], test["signals"])
        ),
        "metrics_independently_match": bool(metric_match and per_class_match),
        "groups_independently_match": bool(group_match),
        "task_output_fields": all(
            field in task_output for field in task["ground_truth"]["required_fields"]
        ),
        "task_output_identity": (
            [row["sample_id"] for row in task_output["classifications"]]
            == prediction_ids
            and [row["unit_id"] for row in task_output["classifications"]]
            == list(range(len(prediction_ids)))
        ),
        "task_output_classes": [
            row["fault_type"] for row in task_output["classifications"]
        ]
        == [f"Fault_{label}" for label in predicted.tolist()],
        "authored_range_evaluation": (
            recorded["accuracy_within_authored_range"]
            == bool(expected_range[0] <= accuracy <= expected_range[1])
            and bool(expected_range[0] <= accuracy <= expected_range[1])
        ),
        "official_score_not_claimed": (
            result["qualification"][
                "official_original_benchmark_score_qualified"
            ]
            is False
            and task_output["official_PHMForge_score"] is False
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
            "within_authored_range": bool(
                expected_range[0] <= accuracy <= expected_range[1]
            ),
        },
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "all_checks_passed": all(checks.values()),
        "classification": (
            "independent numeric verification of the source-bound "
            "pdm_fault_013 research adaptation; not an official score"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "official_original_benchmark_score_qualified": False,
    }
    output = run / "INDEPENDENT_VERIFY250.json"
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "checks_passed": verification["checks_passed"],
                "checks_total": verification["checks_total"],
                "all_checks_passed": verification["all_checks_passed"],
                "output": str(output),
            }
        )
    )
    if not verification["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
