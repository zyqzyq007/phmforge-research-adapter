"""Independent sklearn verification for the A249 pdm_fault_015 binding."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASSES = [0, 1, 2, 3]


def close(left, right, tolerance=1e-12):
    return bool(abs(float(left) - float(right)) <= tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()

    result = json.loads((run / "QUALIFICATION249.json").read_text())
    task_output = json.loads((run / "RESEARCH_OUTPUT249.json").read_text())
    labels = json.loads(
        (run / "replay/external_labels/ELECTRICMOTORVIBRATIONS/labels.json").read_text()
    )
    primary = json.loads(
        (
            run
            / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_dad57d865701.json"
        ).read_text()
    )
    cold = json.loads(
        (
            run
            / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_e7f072795497.json"
        ).read_text()
    )
    untrained = json.loads(
        (
            run
            / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_ae0201e5e3ed.json"
        ).read_text()
    )
    changed = json.loads(
        (
            run
            / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_e404aa3e5c25.json"
        ).read_text()
    )
    dataset_info = json.loads(
        (run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/dataset.json").read_text()
    )
    test = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/test.npz",
        allow_pickle=False,
    )
    changed_npz = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/changed_input.npz",
        allow_pickle=False,
    )

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
            "labels": sorted(set(actual[selected].tolist())),
            "accuracy": float(accuracy_score(actual[selected], predicted[selected])),
        }

    split_groups = dataset_info["train_validation_groups"]
    test_groups = set(groups.tolist())
    expected_range = result["expected_accuracy_range"]
    checks = {
        "metrics_recomputed": bool(metric_match),
        "per_class_metrics_recomputed": bool(per_class_match),
        "unique_prediction_ids": len(prediction_ids) == len(set(prediction_ids)),
        "all_external_labels_covered": set(prediction_ids) == set(label_by_id),
        "all_test_observations_covered": set(prediction_ids)
        == set(test["sample_ids"].tolist()),
        "all_four_classes_present": set(actual.tolist()) == set(CLASSES),
        "task_output_fields": all(
            field in task_output
            for field in ["classifications", "accuracy", "confusion_matrix"]
        ),
        "task_output_identity": [
            row["sample_id"] for row in task_output["classifications"]
        ]
        == prediction_ids,
        "task_output_classes": [
            row["fault_type"] for row in task_output["classifications"]
        ]
        == [f"Fault_{label}" for label in predicted.tolist()],
        "each_group_single_class": all(
            len(values["labels"]) == 1 for values in group_metrics.values()
        ),
        "group_isolation": (
            not set(split_groups["train"]) & set(split_groups["val"])
            and not set(split_groups["train"]) & test_groups
            and not set(split_groups["val"]) & test_groups
        ),
        "test_group_count": len(test_groups)
        == dataset_info["groups_by_split"]["test"],
        "primary_cold_labels_equal": np.array_equal(predicted, cold_predicted),
        "primary_cold_bytes_equal": (
            (
                run
                / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_dad57d865701.json"
            ).read_bytes()
            == (
                run
                / "replay/outputs/ELECTRICMOTORVIBRATIONS/predictions_e7f072795497.json"
            ).read_bytes()
        ),
        "weight_dependency": not np.array_equal(predicted, untrained_predicted),
        "input_dependency": not np.array_equal(predicted, changed_predicted),
        "changed_input_identity": set(changed_npz["sample_ids"].tolist())
        == set(test["sample_ids"].tolist()),
        "changed_input_values": not np.array_equal(
            changed_npz["features"], test["features"]
        ),
        "authored_range_evaluation": (
            recorded["accuracy_within_authored_range"]
            == bool(expected_range[0] <= accuracy <= expected_range[1])
            and not (expected_range[0] <= accuracy <= expected_range[1])
        ),
        "official_score_not_claimed": result["qualification"][
            "official_original_benchmark_score_qualified"
        ]
        is False
        and task_output["official_PHMForge_score"] is False,
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
            "pdm_fault_015 research adaptation; not an official score"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "official_original_benchmark_score_qualified": False,
    }
    output = run / "INDEPENDENT_VERIFY249.json"
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    print(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "checks_passed": verification["checks_passed"],
                "checks_total": verification["checks_total"],
                "output": str(output),
            }
        )
    )
    if not verification["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
