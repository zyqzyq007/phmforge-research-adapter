"""Bind PHMForge pdm_fault_013 to the pinned MFPT source.

The audit reuses A246's real MLP training and sealed external predictions.
It performs no model, Agent, or training call. The original short task only
requests classification of all MFPT test units; it does not request the
load-stratified contract that remains blocked for pdm_fault_008.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


TASK_ID = "pdm_fault_013"
QUESTION_SHA256 = "e4e0aa6b13fdbf3239950c97b75f71aa15af1bf4969960dfe5d5db12c8136c9a"
SOURCE_SHA256 = "581c6cb1fb02dcd97a31ae9785061d178101bbc146fb8713d8c510466fb086ee"
SCENARIO_SHA256 = "3d55c7acc802084deb4c3e93a648755e1d4e01322e215872720db0123860036d"
RECEIPT_SHA256 = "27b33721aa96161650bb5b03e75939a01d1d050b2f19cff501a91ca003a6e693"
SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)
SOURCE_MEMBER = "inputs/pdmb_attempt2/MFPT.csv"
RECEIPT_MEMBER = "inputs/pdmb_attempt2/MFPT.receipt.json"
MODEL_DIR = "outputs/MFPT/MFPT_mlp_6787354e15ff"
PRIMARY_PREDICTION = "outputs/MFPT/predictions_117397e57214.json"
COLD_PREDICTION = "outputs/MFPT/predictions_0e425b981347.json"
UNTRAINED_PREDICTION = "outputs/MFPT/predictions_a6faefdf0a59.json"
CHANGED_PREDICTION = "outputs/MFPT/predictions_c982447247e9.json"
CLASSES = [0, 1, 2]
LABEL_TO_CONDITION = {0: "NORMAL", 1: "IR", 2: "OR"}
EXPECTED_LABEL_COUNTS = {"0": 684, "1": 399, "2": 1083}
EXPECTED_TEST_LABEL_COUNTS = {"0": 228, "1": 57, "2": 285}
EXPECTED_MANIFEST_SHA256 = {
    "source244": "b1e5804214ae3ef2c2effdc6fa94fe75355406be3ee09fd4b494e01cd005ef19",
    "repair246": "c7138151966517de911a378cbaf3eedcf6bf6e65f4f841ae73a04a3094ee79a5",
    "runtime246": "ffd440352a5b3b0123eccb21593ed38c121905e6690b0e494bcce72a346e9ac4",
}
EXPECTED_BUNDLE_SHA256 = {
    "source244": "a9af40f07a4c3866533887db92b21575df932a71c987ed6f17fa54745fa10b0b",
    "repair246": "f3c1d3698a547eb9f8d89354cc4b3df224e0609ab9fc12bcc7c2569e3f8dfd5e",
}
REPAIR_MEMBERS = [
    "audit/MFPT_LABEL_SEMANTICS246.json",
    f"{MODEL_DIR}/model.pt",
    f"{MODEL_DIR}/untrained_model.pt",
    f"{MODEL_DIR}/training_audit.json",
    "external_labels/MFPT/labels.json",
    "logs/MFPT.stderr",
    "outputs/MFPT/PRE_SCORE_CHECKS.json",
    "outputs/MFPT/qualification_calls.json",
    "outputs/MFPT/sealed_test_predictions.json",
    "outputs/pdm_fault_013_research_adaptation.json",
    PRIMARY_PREDICTION,
    COLD_PREDICTION,
    UNTRAINED_PREDICTION,
    CHANGED_PREDICTION,
]
RUNTIME_MEMBERS = [
    "public/datasets/MFPT/dataset.json",
    "public/datasets/MFPT/train.npz",
    "public/datasets/MFPT/val.npz",
    "public/datasets/MFPT/test.npz",
    "public/datasets/MFPT/changed_input.npz",
]
LOCAL_RUNTIME_PATHS = {
    member: "replay/datasets/MFPT/" + member.rsplit("/", 1)[-1]
    for member in RUNTIME_MEMBERS
}
PREDICTION_SHA256 = {
    PRIMARY_PREDICTION: "75c7761c9479f03d69fc7b42c4daa683edf4665717cc95ee00420581ee964056",
    COLD_PREDICTION: "75c7761c9479f03d69fc7b42c4daa683edf4665717cc95ee00420581ee964056",
    UNTRAINED_PREDICTION: "987a67c09f830e2bfd2baf7748638528a8901bb773b2a7cfc876a839bd052798",
    CHANGED_PREDICTION: "594d66be510198db5518a9f260cfbc33b0509a6a2f1d6013ed068ca0d634751e",
}
CHECKPOINT_SHA256 = "33fb2f590188434630f14ce03465a912d99bcb4570573925ad3bb6ea103c8687"
UNTRAINED_CHECKPOINT_SHA256 = (
    "8ece4fbc97393ec5ef7a5b09bc8d49ecc859ea095a7e6431d0b5687c49f843f9"
)


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


def manifest_map(path):
    payload = json.loads(Path(path).read_text())
    return {row["path"]: row for row in payload["files"]}


def verify_members(root, members, manifest):
    evidence = {}
    for relative in members:
        path = Path(root) / relative
        if not path.is_file():
            raise ValueError(f"Selected archive member is missing: {relative}")
        actual = digest(path)
        if relative not in manifest:
            raise ValueError(f"Selected member is not listed in manifest: {relative}")
        if actual != manifest[relative]["sha256"]:
            raise ValueError(f"Selected member hash changed: {relative}")
        if path.stat().st_size != manifest[relative]["bytes"]:
            raise ValueError(f"Selected member byte count changed: {relative}")
        evidence[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return evidence


def parse_source(path):
    """Parse and validate every published MFPT row without storing raw arrays."""
    csv.field_size_limit(128 * 1024**2)
    columns = [
        "condition",
        "bearing_type",
        "bearing_num",
        "segment_idx",
        "features",
        "vibration_raw",
        "label",
        "sampling_rate",
        "load_condition",
        "filename",
    ]
    label_counts = Counter()
    condition_counts = Counter()
    load_counts = Counter()
    sampling_counts = Counter()
    filename_counts = Counter()
    filename_by_label = {}
    label_to_conditions = {}
    feature_lengths = Counter()
    raw_lengths = Counter()
    rows_by_index = {}
    nonfinite_rows = 0
    rows = 0

    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected MFPT columns: {reader.fieldnames}")
        for row_index, row in enumerate(reader):
            rows += 1
            label = int(row["label"])
            condition = row["condition"]
            if label not in CLASSES:
                raise ValueError(f"Unexpected MFPT label: {label}")
            if LABEL_TO_CONDITION[label] != condition:
                raise ValueError(
                    f"Label/condition conflict at row {row_index}: {label}, {condition}"
                )
            label_counts[label] += 1
            condition_counts[condition] += 1
            load_counts[row["load_condition"]] += 1
            sampling_counts[row["sampling_rate"]] += 1
            filename = row["filename"]
            filename_counts[filename] += 1
            filename_by_label.setdefault(label, set()).add(filename)
            label_to_conditions.setdefault(label, set()).add(condition)

            feature_values = np.fromstring(row["features"][1:-1], sep=",", dtype=np.float64)
            raw_values = np.fromstring(
                row["vibration_raw"][1:-1], sep=",", dtype=np.float64
            )
            feature_lengths[len(feature_values)] += 1
            raw_lengths[len(raw_values)] += 1
            finite = bool(
                np.isfinite(feature_values).all() and np.isfinite(raw_values).all()
            )
            if not finite:
                nonfinite_rows += 1
            rows_by_index[row_index] = {
                "label": label,
                "condition": condition,
                "filename": filename,
                "load_condition": row["load_condition"],
                "sampling_rate": row["sampling_rate"],
            }

    return {
        "columns": columns,
        "rows": rows,
        "label_counts": {str(key): label_counts[key] for key in CLASSES},
        "condition_counts": {
            condition: condition_counts[condition]
            for condition in sorted(condition_counts)
        },
        "label_to_conditions": {
            str(label): sorted(label_to_conditions[label]) for label in CLASSES
        },
        "feature_lengths": {
            str(length): count for length, count in sorted(feature_lengths.items())
        },
        "raw_lengths": {
            str(length): count for length, count in sorted(raw_lengths.items())
        },
        "load_conditions": {
            name: count for name, count in sorted(load_counts.items())
        },
        "sampling_rates": {
            name: count for name, count in sorted(sampling_counts.items())
        },
        "filenames": {
            name: count for name, count in sorted(filename_counts.items())
        },
        "filenames_by_label": {
            str(label): sorted(filename_by_label[label]) for label in CLASSES
        },
        "nonfinite_rows": nonfinite_rows,
        "rows_by_index": rows_by_index,
    }


def class_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    lookup = {label: index for index, label in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, estimate in zip(actual, predicted):
        if int(truth) not in lookup or int(estimate) not in lookup:
            raise ValueError("Prediction or label lies outside the frozen MFPT classes")
        matrix[lookup[int(truth)], lookup[int(estimate)]] += 1

    per_class = {}
    f1_values = []
    recalls = []
    for index, label in enumerate(CLASSES):
        true_positive = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[str(label)] = {
            "support": support,
            "true_positive": true_positive,
            "predicted_positive": predicted_positive,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        f1_values.append(f1)
        recalls.append(recall)

    return {
        "n": int(len(actual)),
        "accuracy": float(np.trace(matrix) / len(actual)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_order": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "per_class": per_class,
    }


def close(left, right, tolerance=1e-12):
    return bool(abs(float(left) - float(right)) <= tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--repair-bundle", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    if run.parent != Path("/srv/yyf-zyq/PHM_20261010").resolve():
        raise ValueError("Run must be directly under the managed experiment root")
    if not run.is_dir():
        raise ValueError("Run directory must already exist")
    if digest(args.source_manifest) != EXPECTED_MANIFEST_SHA256["source244"]:
        raise ValueError("A244 source manifest hash changed")
    if digest(args.repair_manifest) != EXPECTED_MANIFEST_SHA256["repair246"]:
        raise ValueError("A246 repair manifest hash changed")
    if digest(args.runtime_manifest) != EXPECTED_MANIFEST_SHA256["runtime246"]:
        raise ValueError("A246 runtime manifest hash changed")
    source_bundle_sha256 = digest(args.source_bundle)
    repair_bundle_sha256 = digest(args.repair_bundle)
    if source_bundle_sha256 != EXPECTED_BUNDLE_SHA256["source244"]:
        raise ValueError("A244 portable bundle hash changed")
    if repair_bundle_sha256 != EXPECTED_BUNDLE_SHA256["repair246"]:
        raise ValueError("A246 portable bundle hash changed")

    write_json(
        run / "status.json",
        {
            "run_id": run.name,
            "status": "running",
            "started_utc": utc_now(),
            "task_id": TASK_ID,
            "new_LLM_calls": 0,
            "new_Agent_sessions": 0,
            "new_training_jobs": 0,
        },
    )

    source_manifest = manifest_map(args.source_manifest)
    repair_manifest = manifest_map(args.repair_manifest)
    runtime_manifest = manifest_map(args.runtime_manifest)
    source_evidence = verify_members(
        run / "source", [SOURCE_MEMBER, RECEIPT_MEMBER, SCENARIO_MEMBER], source_manifest
    )
    repair_evidence = verify_members(
        run / "replay", REPAIR_MEMBERS, repair_manifest
    )
    runtime_evidence = {}
    for member in RUNTIME_MEMBERS:
        local_relative = LOCAL_RUNTIME_PATHS[member]
        path = run / local_relative
        actual = digest(path)
        if actual != runtime_manifest[member]["sha256"]:
            raise ValueError(f"A246 runtime member hash changed: {member}")
        runtime_evidence[member] = {
            "local_path": local_relative,
            "bytes": path.stat().st_size,
            "sha256": actual,
        }

    source_csv = run / "source" / SOURCE_MEMBER
    receipt_path = run / "source" / RECEIPT_MEMBER
    scenario_path = run / "source" / SCENARIO_MEMBER
    source_receipt = json.loads(receipt_path.read_text())
    scenario_payload = json.loads(scenario_path.read_text())
    source_summary = parse_source(source_csv)

    matches = [
        row for row in scenario_payload["pdm_scenarios"] if row["task_id"] == TASK_ID
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original pdm_fault_013 task")
    task = matches[0]
    question_sha = hashlib.sha256(task["input_question"].encode("utf-8")).hexdigest()
    expected_range = task["ground_truth"]["expected_accuracy_range"]

    dataset_info = json.loads(
        (run / "replay/datasets/MFPT/dataset.json").read_text()
    )
    training = json.loads(
        (run / f"replay/{MODEL_DIR}/training_audit.json").read_text()
    )
    pre_score = json.loads(
        (run / "replay/outputs/MFPT/PRE_SCORE_CHECKS.json").read_text()
    )
    label_semantics = json.loads(
        (run / "replay/audit/MFPT_LABEL_SEMANTICS246.json").read_text()
    )
    labels = json.loads((run / "replay/external_labels/MFPT/labels.json").read_text())
    primary = json.loads((run / "replay" / PRIMARY_PREDICTION).read_text())
    cold = json.loads((run / "replay" / COLD_PREDICTION).read_text())
    untrained = json.loads((run / "replay" / UNTRAINED_PREDICTION).read_text())
    changed = json.loads((run / "replay" / CHANGED_PREDICTION).read_text())
    sealed = json.loads(
        (run / "replay/outputs/MFPT/sealed_test_predictions.json").read_text()
    )
    calls = json.loads(
        (run / "replay/outputs/MFPT/qualification_calls.json").read_text()
    )
    a246_task_output = json.loads(
        (run / "replay/outputs/pdm_fault_013_research_adaptation.json").read_text()
    )

    train = np.load(run / "replay/datasets/MFPT/train.npz", allow_pickle=False)
    val = np.load(run / "replay/datasets/MFPT/val.npz", allow_pickle=False)
    test = np.load(run / "replay/datasets/MFPT/test.npz", allow_pickle=False)
    changed_npz = np.load(
        run / "replay/datasets/MFPT/changed_input.npz", allow_pickle=False
    )

    label_by_id = {row["sample_id"]: row for row in labels}
    prediction_rows = primary["predictions"]
    prediction_ids = [row["sample_id"] for row in prediction_rows]
    cold_ids = [row["sample_id"] for row in cold["predictions"]]
    untrained_ids = [row["sample_id"] for row in untrained["predictions"]]
    changed_ids = [row["sample_id"] for row in changed["predictions"]]
    if not (prediction_ids == cold_ids == untrained_ids == changed_ids):
        raise ValueError("Control predictions do not preserve identity and order")

    ordered_labels = [label_by_id[sample_id] for sample_id in prediction_ids]
    actual = [int(row["label"]) for row in ordered_labels]
    predicted = [int(row["predicted_label"]) for row in prediction_rows]
    cold_predicted = [int(row["predicted_label"]) for row in cold["predictions"]]
    untrained_predicted = [
        int(row["predicted_label"]) for row in untrained["predictions"]
    ]
    changed_predicted = [
        int(row["predicted_label"]) for row in changed["predictions"]
    ]
    group_by_id = {row["sample_id"]: row["group"] for row in ordered_labels}
    groups = [group_by_id[sample_id] for sample_id in prediction_ids]
    labels_by_group = {}
    for row in ordered_labels:
        labels_by_group.setdefault(row["group"], set()).add(int(row["label"]))
    group_metrics = {}
    for group in sorted(labels_by_group):
        selected = np.asarray([value == group for value in groups])
        group_actual = np.asarray(actual)[selected]
        group_predicted = np.asarray(predicted)[selected]
        group_metrics[group] = {
            "n": int(selected.sum()),
            "label": int(np.unique(group_actual).item()),
            "accuracy": float((group_actual == group_predicted).mean()),
        }

    metrics = class_metrics(actual, predicted)
    task_output = {
        "task_id": TASK_ID,
        "run_id": run.name,
        "classifications": [
            {
                "unit_id": index,
                "sample_id": sample_id,
                "fault_type": f"Fault_{predicted_label}",
            }
            for index, (sample_id, predicted_label) in enumerate(
                zip(prediction_ids, predicted)
            )
        ],
        "accuracy": {
            "overall_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": [f"Fault_{label}" for label in CLASSES],
        "per_fault_type_metrics": metrics["per_class"],
        "research_adaptation": True,
        "official_PHMForge_score": False,
    }

    recorded_metrics = a246_task_output["external_metrics"]
    metric_match = (
        close(metrics["accuracy"], recorded_metrics["accuracy"])
        and close(metrics["macro_f1"], recorded_metrics["macro_f1"])
        and close(
            metrics["balanced_accuracy"], recorded_metrics["balanced_accuracy"]
        )
        and metrics["confusion_matrix"] == recorded_metrics["confusion_matrix"]
    )

    source_rows = source_summary["rows_by_index"]
    external_source_binding = True
    for row in labels:
        source_row = source_rows[row["original_row"]]
        external_source_binding &= (
            int(row["label"]) == source_row["label"]
            and row["group"] == source_row["filename"]
        )

    prediction_hashes = {
        member: digest(run / "replay" / member) for member in PREDICTION_SHA256
    }
    calls = calls["calls"] if isinstance(calls, dict) else calls
    primary_call = next(
        row
        for row in calls
        if row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith(
            "predictions_117397e57214.json"
        )
    )
    cold_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith(
            "predictions_0e425b981347.json"
        )
    )
    untrained_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith(
            "predictions_a6faefdf0a59.json"
        )
    )
    changed_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith(
            "predictions_c982447247e9.json"
        )
    )

    split_groups = dataset_info["train_validation_groups"]
    test_groups = sorted(set(groups))
    split_ids = {
        "train": set(train["sample_ids"].tolist()),
        "val": set(val["sample_ids"].tolist()),
        "test": set(test["sample_ids"].tolist()),
    }
    split_group_sets = {
        "train": set(split_groups["train"]),
        "val": set(split_groups["val"]),
        "test": set(test_groups),
    }
    class_counts = Counter(actual)
    test_class_counts = {
        str(label): class_counts[label] for label in CLASSES
    }
    changed_input_sha = runtime_manifest["public/datasets/MFPT/changed_input.npz"][
        "sha256"
    ]
    test_input_sha = runtime_manifest["public/datasets/MFPT/test.npz"]["sha256"]

    checks = {
        "source_manifest_hash": digest(args.source_manifest)
        == EXPECTED_MANIFEST_SHA256["source244"],
        "repair_manifest_hash": digest(args.repair_manifest)
        == EXPECTED_MANIFEST_SHA256["repair246"],
        "runtime_manifest_hash": digest(args.runtime_manifest)
        == EXPECTED_MANIFEST_SHA256["runtime246"],
        "source_bundle_hash": source_bundle_sha256
        == EXPECTED_BUNDLE_SHA256["source244"],
        "repair_bundle_hash": repair_bundle_sha256
        == EXPECTED_BUNDLE_SHA256["repair246"],
        "source_csv_listed_in_manifest": SOURCE_MEMBER in source_manifest,
        "source_csv_hash": digest(source_csv) == SOURCE_SHA256,
        "source_receipt_hash": digest(receipt_path) == RECEIPT_SHA256,
        "scenario_hash": digest(scenario_path) == SCENARIO_SHA256,
        "source_bytes_match_pin": source_csv.stat().st_size
        == source_receipt["pin"]["bytes"],
        "source_receipt_complete": (
            source_receipt["actual_sha256"] == SOURCE_SHA256
            and source_receipt["pinned_hash_verified"] is True
            and source_receipt["complete_file"] is True
        ),
        "source_columns": source_summary["columns"]
        == [
            "condition",
            "bearing_type",
            "bearing_num",
            "segment_idx",
            "features",
            "vibration_raw",
            "label",
            "sampling_rate",
            "load_condition",
            "filename",
        ],
        "source_rows": source_summary["rows"] == 2166,
        "source_labels": source_summary["label_counts"] == EXPECTED_LABEL_COUNTS,
        "source_condition_mapping": source_summary["label_to_conditions"]
        == {
            "0": ["NORMAL"],
            "1": ["IR"],
            "2": ["OR"],
        },
        "source_feature_lengths": source_summary["feature_lengths"] == {"23": 2166},
        "source_raw_lengths": source_summary["raw_lengths"] == {"2560": 2166},
        "source_values_finite": source_summary["nonfinite_rows"] == 0,
        "source_file_group_counts": {
            str(label): len(source_summary["filenames_by_label"][str(label)])
            for label in CLASSES
        }
        == {"0": 3, "1": 7, "2": 10},
        "question_hash": question_sha == QUESTION_SHA256,
        "original_dataset": task["dataset"] == "MFPT",
        "original_short_prompt": task["input_question"]
        == "Classify fault types for all test units in MFPT dataset.",
        "authored_range": expected_range == [0.85, 0.98],
        "runtime_dataset_hash": runtime_evidence[
            "public/datasets/MFPT/dataset.json"
        ]["sha256"]
        == "061b247a51da4eb8f7bb08bc978a5c1117a0999ca3a93f4ca3220fb7d8bc0daa",
        "runtime_train_hash": runtime_evidence["public/datasets/MFPT/train.npz"][
            "sha256"
        ]
        == "0de1046992dfa3ebf7b955826792dd3972363138e00c5655c0a418981ce30fb1",
        "runtime_val_hash": runtime_evidence["public/datasets/MFPT/val.npz"][
            "sha256"
        ]
        == "c98d03f33675b9a4cd4b6b661e26f95559dd368a59f65ba9666f3c494a7bc2c8",
        "runtime_test_hash": runtime_evidence["public/datasets/MFPT/test.npz"][
            "sha256"
        ]
        == "8354f5bb1b41578855acf49a5fe93c5e00eca03a1183c6863ae711c853eadbbc",
        "runtime_changed_hash": runtime_evidence[
            "public/datasets/MFPT/changed_input.npz"
        ]["sha256"]
        == "bd17252fc9ff2572fe4730d2fe3958b699a76a7836d2125dea508207e2cddf02",
        "dataset_source_hash": dataset_info["source_sha256"] == SOURCE_SHA256,
        "dataset_class_names": dataset_info["class_names"] == CLASSES,
        "waveform_processing": dataset_info["waveform_processing"]
        == "2560 -> 256 nonoverlapping means of 10",
        "label_semantics": label_semantics["mapping"]
        == {"0": "NORMAL", "1": "IR", "2": "OR"},
        "test_labels_absent": "labels" not in test.files and "groups" not in test.files,
        "prepared_features_finite": all(
            np.isfinite(array["features"]).all() for array in (train, val, test)
        ),
        "prepared_signals_finite": all(
            np.isfinite(array["signals"]).all() for array in (train, val, test)
        ),
        "row_counts_match_dataset_info": {
            "train": train["features"].shape[0],
            "val": val["features"].shape[0],
            "test": test["features"].shape[0],
        }
        == dataset_info["rows_by_split"],
        "split_groups_disjoint": all(
            not (split_group_sets[left] & split_group_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "training_audit_group_isolation": training["group_overlap"] == [],
        "sample_ids_disjoint": all(
            not (split_ids[left] & split_ids[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "test_group_count": len(test_groups) == 4,
        "each_test_group_has_one_class": all(
            len(values) == 1 for values in labels_by_group.values()
        ),
        "external_source_binding": external_source_binding,
        "prediction_identity_complete": (
            len(prediction_ids) == len(set(prediction_ids))
            and set(prediction_ids) == split_ids["test"]
            and set(prediction_ids) == set(label_by_id)
        ),
        "all_three_classes_present": set(actual) == set(CLASSES),
        "test_label_counts": test_class_counts == EXPECTED_TEST_LABEL_COUNTS,
        "task_output_fields_present": all(
            field in task_output
            for field in task["ground_truth"]["required_fields"]
        ),
        "task_output_identity_complete": (
            len(task_output["classifications"]) == len(prediction_ids)
            and [
                row["sample_id"] for row in task_output["classifications"]
            ]
            == prediction_ids
            and [
                row["unit_id"] for row in task_output["classifications"]
            ]
            == list(range(len(prediction_ids)))
        ),
        "prediction_hashes": prediction_hashes == PREDICTION_SHA256,
        "primary_checkpoint_hash": digest(run / f"replay/{MODEL_DIR}/model.pt")
        == CHECKPOINT_SHA256,
        "untrained_checkpoint_hash": digest(
            run / f"replay/{MODEL_DIR}/untrained_model.pt"
        )
        == UNTRAINED_CHECKPOINT_SHA256,
        "real_parameter_updates": (
            training["epochs_executed"] == 50
            and training["optimizer_steps"] == 850
            and training["nonzero_gradient_steps"] == 850
            and training["changed_parameter_tensors"] == 6
        ),
        "training_test_labels_isolated": training["test_labels_accessed"] is False,
        "training_only_normalization": training["training_only_normalization"] is True,
        "pre_score_controls": all(pre_score.values()),
        "cold_reload_byte_identical": (
            (run / "replay" / PRIMARY_PREDICTION).read_bytes()
            == (run / "replay" / COLD_PREDICTION).read_bytes()
        ),
        "cold_reload_labels_identical": predicted == cold_predicted,
        "weight_dependency": predicted != untrained_predicted,
        "input_dependency": predicted != changed_predicted,
        "changed_input_identity_preserved": set(changed_ids) == set(prediction_ids),
        "changed_input_array_identity_preserved": (
            changed_npz["sample_ids"].tolist() == test["sample_ids"].tolist()
        ),
        "changed_input_values_differ": (
            not np.array_equal(changed_npz["features"], test["features"])
            or not np.array_equal(changed_npz["signals"], test["signals"])
        ),
        "sealed_prediction_matches_primary": sealed["predictions"]
        == primary["predictions"],
        "call_checkpoint_links": (
            primary_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
            and cold_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
            and untrained_call["result"]["checkpoint_sha256"]
            == UNTRAINED_CHECKPOINT_SHA256
            and changed_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
        ),
        "call_input_links": (
            primary_call["result"]["input_sha256"] == test_input_sha
            and cold_call["result"]["input_sha256"] == test_input_sha
            and untrained_call["result"]["input_sha256"] == test_input_sha
            and changed_call["result"]["input_sha256"] == changed_input_sha
        ),
        "metrics_match_A246": metric_match,
        "authored_range_passed": bool(
            expected_range[0] <= metrics["accuracy"] <= expected_range[1]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"pdm_fault_013 contract checks failed: {failed}")

    result = {
        "utc": utc_now(),
        "task_id": TASK_ID,
        "run_id": run.name,
        "classification": (
            "source-bound three-label research adaptation using the sealed "
            "A246 MFPT MLP run; not an official PHMForge score"
        ),
        "question_sha256": question_sha,
        "original_question": task["input_question"],
        "expected_output_fields": task["ground_truth"]["required_fields"],
        "expected_accuracy_range": expected_range,
        "source": {
            "dataset": "MFPT",
            "sha256": digest(source_csv),
            "rows": source_summary["rows"],
            "label_counts": source_summary["label_counts"],
            "condition_counts": source_summary["condition_counts"],
            "filenames_by_label": source_summary["filenames_by_label"],
            "feature_lengths": source_summary["feature_lengths"],
            "raw_lengths": source_summary["raw_lengths"],
            "load_conditions": source_summary["load_conditions"],
            "sampling_rates": source_summary["sampling_rates"],
            "nonfinite_rows": source_summary["nonfinite_rows"],
            "class_names": CLASSES,
            "class_name_policy": (
                "Published integer labels 0..2 are preserved and rendered as "
                "Fault_0..Fault_2. The source condition audit maps them to "
                "NORMAL/IR/OR; no additional physical fault names are invented."
            ),
            "group_definition": dataset_info["group_definition"],
            "metadata_caveat": (
                "A248 found that PDMBench dataset metadata lists a fourth "
                "ball class while this pinned MFPT CSV exposes only labels "
                "0..2. pdm_fault_013 does not enumerate classes or request "
                "load-stratified accuracy; this audit uses only the published "
                "three labels and does not invent a fourth class. The separate "
                "pdm_fault_008 load-stratified contract remains blocked."
            ),
        },
        "reused_A246_training": {
            "new_training_jobs": 0,
            "architecture": "mlp",
            "epochs_executed": training["epochs_executed"],
            "optimizer_steps": training["optimizer_steps"],
            "nonzero_gradient_steps": training["nonzero_gradient_steps"],
            "changed_parameter_tensors": training["changed_parameter_tensors"],
            "selected_epoch": training["selected_epoch"],
            "selected_validation_accuracy": training[
                "selected_validation_accuracy"
            ],
            "training_rows": training["train_rows"],
            "validation_rows": training["validation_rows"],
            "test_labels_accessed": training["test_labels_accessed"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "split": {
            "rows_by_split": dataset_info["rows_by_split"],
            "test_groups": test_groups,
            "group_definition": dataset_info["group_definition"],
            "group_overlap": training["group_overlap"],
        },
        "external_evaluation": {
            **metrics,
            "group_metrics": group_metrics,
            "accuracy_within_authored_range": bool(
                expected_range[0] <= metrics["accuracy"] <= expected_range[1]
            ),
            "source": "sealed A246 model predictions plus external final labels",
        },
        "research_output": "RESEARCH_OUTPUT250.json",
        "controls": {
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "cold_reload_byte_identical": checks["cold_reload_byte_identical"],
            "weight_dependency": checks["weight_dependency"],
            "input_dependency": checks["input_dependency"],
        },
        "qualification": {
            "source_bound_three_label_classifier_qualified": True,
            "research_adapted_task_contract_qualified": True,
            "official_original_benchmark_score_qualified": False,
            "fully_qualified_original_tasks": 0,
            "reason_official_false": (
                "A246 used an external hidden-label evaluator and did not replay "
                "the original visible-truth tool chain or original scorer."
            ),
        },
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
        "new_training_jobs": 0,
        "new_confirmed_PHM_problems": 0,
        "audit_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "model_calls": 0,
        },
        "evidence": {
            "source_bundle": str(args.source_bundle.resolve()),
            "source_bundle_sha256": source_bundle_sha256,
            "repair_bundle": str(args.repair_bundle.resolve()),
            "repair_bundle_sha256": repair_bundle_sha256,
            "source_manifest_sha256": digest(args.source_manifest),
            "repair_manifest_sha256": digest(args.repair_manifest),
            "runtime_manifest_sha256": digest(args.runtime_manifest),
            "source_members": source_evidence,
            "repair_members": repair_evidence,
            "runtime_members": runtime_evidence,
        },
    }
    write_json(run / "RESEARCH_OUTPUT250.json", task_output)
    write_json(run / "QUALIFICATION250.json", result)
    write_json(
        run / "retention_policy.json",
        {
            "excluded_rebuildable_directory": "clients-cache",
            "shared_dependencies": {
                "path": "/srv/yyf-zyq/PHM_20261010/_shared",
                "retention": "existing_pinned_readonly_dependencies",
            },
            "policy": (
                "retain source receipt, question scenario, frozen MFPT inputs, "
                "A246 checkpoints, predictions, calls and audit result"
            ),
            "deletion_before_archive": "none",
            "temporary_directory": "run-local only; no /tmp use",
        },
    )
    write_json(
        run / "status.json",
        {
            "run_id": run.name,
            "status": "completed",
            "completed_utc": utc_now(),
            "task_id": TASK_ID,
            "new_LLM_calls": 0,
            "new_Agent_sessions": 0,
            "new_training_jobs": 0,
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "result": "QUALIFICATION250.json",
        },
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "checks_passed": sum(bool(value) for value in checks.values()),
                "checks_total": len(checks),
                "research_adapted_task_contract_qualified": True,
                "official_original_benchmark_score_qualified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
