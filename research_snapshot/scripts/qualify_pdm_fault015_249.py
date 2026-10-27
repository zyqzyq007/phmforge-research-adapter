"""Bind PHMForge pdm_fault_015 to the pinned ElectricMotorVibrations source.

This audit reuses the real MLP training and sealed external predictions already
produced in A246. It performs no model or Agent call. The original short task
asks only for four-way fault classification, so the otherwise missing acoustic
comparison in pdm_fault_003 is not part of this task contract.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile

import numpy as np


TASK_ID = "pdm_fault_015"
QUESTION_SHA256 = "297c08679814d1e88e364999b85848d5ea5c340a180ec8dc9326e189c02f0956"
SOURCE_SHA256 = "cf87f519678ab8d9673ae2b9b94044f000edea9cd1c68a7fc39730823c048ce8"
SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)
SOURCE_MEMBER = "inputs/pdmb_attempt2/ElectricMotorVibrations.csv"
RECEIPT_MEMBER = "inputs/pdmb_attempt2/ElectricMotorVibrations.receipt.json"
PDMB_COMMIT = "a3415dd5f2b7cb45ec7f973c2f330bdafd2dd006"
PDMB_DATASET_INFO_URL = (
    "https://raw.githubusercontent.com/zshuai8/PDMBench/"
    f"{PDMB_COMMIT}/dataset/dataset_info.json"
)
PDMB_DATASET_INFO_GIT_BLOB = "7c30ea21798beaa52d14dbb005b1726280dfdd94"
PDMB_DATASET_INFO_SHA256 = (
    "a2ebc3b02b75f3e2d0f62bbb74b6b13dbb2ebc09a141507fc39c30d8b87a5050"
)
EXPECTED_MANIFEST_SHA256 = {
    "source244": "b1e5804214ae3ef2c2effdc6fa94fe75355406be3ee09fd4b494e01cd005ef19",
    "repair246": "c7138151966517de911a378cbaf3eedcf6bf6e65f4f841ae73a04a3094ee79a5",
}
EXPECTED_BUNDLE_SHA256 = {
    "source244": "a9af40f07a4c3866533887db92b21575df932a71c987ed6f17fa54745fa10b0b",
    "repair246": "f3c1d3698a547eb9f8d89354cc4b3df224e0609ab9fc12bcc7c2569e3f8dfd5e",
}
MODEL_DIR = "outputs/ELECTRICMOTORVIBRATIONS/ELECTRICMOTORVIBRATIONS_mlp_4f83024fde03"
REPAIR_MEMBERS = [
    "prepared/datasets/ELECTRICMOTORVIBRATIONS/dataset.json",
    "prepared/datasets/ELECTRICMOTORVIBRATIONS/train.npz",
    "prepared/datasets/ELECTRICMOTORVIBRATIONS/val.npz",
    "prepared/datasets/ELECTRICMOTORVIBRATIONS/test.npz",
    "prepared/datasets/ELECTRICMOTORVIBRATIONS/changed_input.npz",
    f"{MODEL_DIR}/training_audit.json",
    f"{MODEL_DIR}/model.pt",
    f"{MODEL_DIR}/untrained_model.pt",
    "outputs/ELECTRICMOTORVIBRATIONS/PRE_SCORE_CHECKS.json",
    "outputs/ELECTRICMOTORVIBRATIONS/qualification_calls.json",
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_dad57d865701.json",
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_e7f072795497.json",
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_ae0201e5e3ed.json",
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_e404aa3e5c25.json",
    "external_labels/ELECTRICMOTORVIBRATIONS/labels.json",
]
PRIMARY_PREDICTION = "outputs/ELECTRICMOTORVIBRATIONS/predictions_dad57d865701.json"
COLD_PREDICTION = "outputs/ELECTRICMOTORVIBRATIONS/predictions_e7f072795497.json"
UNTRAINED_PREDICTION = (
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_ae0201e5e3ed.json"
)
CHANGED_PREDICTION = (
    "outputs/ELECTRICMOTORVIBRATIONS/predictions_e404aa3e5c25.json"
)
PREDICTION_SHA256 = {
    PRIMARY_PREDICTION: "4101d0ecf179224509f5dce871cc6940630061aa37981f90e67f3c85a4659a24",
    COLD_PREDICTION: "4101d0ecf179224509f5dce871cc6940630061aa37981f90e67f3c85a4659a24",
    UNTRAINED_PREDICTION: "ad97eeb57348d9bf58e8fc890d7515d6e0c2a36b1b913c1ecb92d080aff07875",
    CHANGED_PREDICTION: "2acc663a6ea5772f147130e86222e544e49cd0d7014438dd0f663f8f9a5de4a2",
}
CHECKPOINT_SHA256 = "e8ab30ca3c04150faf9d4519e8849d6d17f8d539e8dc4f2d2768aed2cd575201"
UNTRAINED_CHECKPOINT_SHA256 = (
    "b3ceaf3b33298c8f6b0b1560c4d00b760a96af64e4ebe61af1ad1035918873d9"
)
CLASSES = [0, 1, 2, 3]


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


def git_blob_sha1(payload):
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\0" + payload
    ).hexdigest()


def manifest_map(path):
    payload = json.loads(Path(path).read_text())
    return {row["path"]: row for row in payload["files"]}


def extract_members(bundle, destinations):
    """Extract selected members in one sequential pass over a gzip archive."""
    remaining = {member: Path(path) for member, path in destinations.items()}
    for destination in remaining.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as stream:
        for member in stream:
            destination = remaining.get(member.name)
            if destination is None:
                continue
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(
                    f"Archive member is not a regular file: {member.name}"
                )
            with source, destination.open("wb") as output:
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            remaining.pop(member.name)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"Archive members not found: {sorted(remaining)}")
    return {member: Path(path) for member, path in destinations.items()}


def fetch_dataset_info(destination):
    result = subprocess.run(
        ["curl", "-k", "-fsSL", PDMB_DATASET_INFO_URL],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    payload = result.stdout
    if git_blob_sha1(payload) != PDMB_DATASET_INFO_GIT_BLOB:
        raise ValueError("Pinned PDMBench dataset_info Git blob changed")
    if hashlib.sha256(payload).hexdigest() != PDMB_DATASET_INFO_SHA256:
        raise ValueError("Pinned PDMBench dataset_info SHA-256 changed")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return json.loads(payload)


def parse_source(path):
    import csv

    csv.field_size_limit(128 * 1024**2)
    columns = ["AccX", "AccY", "AccZ", "torque", "ohm", "label"]
    array_columns = ["AccX", "AccY", "AccZ"]
    scalar_columns = ["torque", "ohm"]
    label_counts = Counter()
    lengths = Counter()
    rows = 0
    nonfinite_arrays = 0
    nonfinite_scalars = 0
    complete_windows = 0
    nonfinite_windows = 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected source columns: {reader.fieldnames}")
        for row in reader:
            rows += 1
            label = int(row["label"])
            if label not in CLASSES:
                raise ValueError(f"Unexpected source label: {label}")
            label_counts[label] += 1
            row_arrays = {}
            for column in array_columns:
                text = row[column].strip()
                if not (text.startswith("[") and text.endswith("]")):
                    raise ValueError(f"Malformed {column} array")
                values = np.fromstring(text[1:-1], sep=",", dtype=np.float64)
                if len(values) != text.count(",") + 1:
                    raise ValueError(f"Partially parsed {column} array")
                lengths[(column, len(values))] += 1
                nonfinite_arrays += int(not np.isfinite(values).all())
                row_arrays[column] = values
            for column in scalar_columns:
                value = float(row[column])
                nonfinite_scalars += int(not np.isfinite(value))
            window_count = min(len(values) for values in row_arrays.values()) // 2048
            complete_windows += window_count
            for index in range(window_count):
                start = index * 2048
                end = start + 2048
                nonfinite_windows += int(
                    not all(
                        np.isfinite(values[start:end]).all()
                        for values in row_arrays.values()
                    )
                )
    return {
        "columns": columns,
        "rows": rows,
        "label_counts": {str(key): label_counts[key] for key in sorted(label_counts)},
        "array_lengths": {
            column: {
                str(length): count
                for (name, length), count in sorted(lengths.items())
                if name == column
            }
            for column in array_columns
        },
        "nonfinite_arrays": nonfinite_arrays,
        "nonfinite_scalars": nonfinite_scalars,
        "complete_windows": complete_windows,
        "nonfinite_windows": nonfinite_windows,
    }


def class_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    lookup = {label: index for index, label in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, estimate in zip(actual, predicted):
        if int(truth) not in lookup or int(estimate) not in lookup:
            raise ValueError("Prediction or label lies outside the frozen four classes")
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
    source_members = extract_members(
        args.source_bundle,
        {
            SOURCE_MEMBER: run / "source" / "ElectricMotorVibrations.csv",
            RECEIPT_MEMBER: run / "source" / "ElectricMotorVibrations.receipt.json",
            SCENARIO_MEMBER: run / "source" / "phm_scenarios.json",
        },
    )
    source_csv = source_members[SOURCE_MEMBER]
    source_receipt_path = source_members[RECEIPT_MEMBER]
    scenario_payload = json.loads(source_members[SCENARIO_MEMBER].read_text())

    repair_destinations = {
        member: run / "replay" / member for member in REPAIR_MEMBERS
    }
    extract_members(args.repair_bundle, repair_destinations)
    extracted = {}
    for member in REPAIR_MEMBERS:
        destination = repair_destinations[member]
        actual_sha = digest(destination)
        expected_sha = repair_manifest[member]["sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"A246 member hash changed: {member}")
        extracted[member] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": actual_sha,
        }

    source_receipt = json.loads(source_receipt_path.read_text())
    source_summary = parse_source(source_csv)
    dataset_info = json.loads(
        (run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/dataset.json").read_text()
    )
    training = json.loads(
        (run / f"replay/{MODEL_DIR}/training_audit.json").read_text()
    )
    pre_score = json.loads(
        (run / "replay/outputs/ELECTRICMOTORVIBRATIONS/PRE_SCORE_CHECKS.json").read_text()
    )
    labels = json.loads(
        (run / "replay/external_labels/ELECTRICMOTORVIBRATIONS/labels.json").read_text()
    )
    primary = json.loads((run / "replay" / PRIMARY_PREDICTION).read_text())
    cold = json.loads((run / "replay" / COLD_PREDICTION).read_text())
    untrained = json.loads((run / "replay" / UNTRAINED_PREDICTION).read_text())
    changed = json.loads((run / "replay" / CHANGED_PREDICTION).read_text())
    calls = json.loads(
        (run / "replay/outputs/ELECTRICMOTORVIBRATIONS/qualification_calls.json").read_text()
    )

    matches = [
        row for row in scenario_payload["pdm_scenarios"] if row["task_id"] == TASK_ID
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original pdm_fault_015 task")
    task = matches[0]
    question_sha = hashlib.sha256(task["input_question"].encode("utf-8")).hexdigest()
    if question_sha != QUESTION_SHA256:
        raise ValueError("Original pdm_fault_015 question hash changed")
    if task["dataset"] != "ElectricMotorVibrations":
        raise ValueError("Original task dataset changed")
    expected_range = task["ground_truth"]["expected_accuracy_range"]

    pdmb_info = fetch_dataset_info(run / "source/PDMB_DATASET_INFO.json")
    pdmb_dataset = pdmb_info["17"]
    if pdmb_dataset["name"] != "Electric Motor Vibrations":
        raise ValueError("Pinned PDMBench dataset number no longer names the motor source")
    if len(pdmb_dataset["fault_types"]) != len(CLASSES):
        raise ValueError("Pinned PDMBench class count differs from the source labels")

    train = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/train.npz",
        allow_pickle=False,
    )
    val = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/val.npz",
        allow_pickle=False,
    )
    test = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/test.npz",
        allow_pickle=False,
    )
    changed_npz = np.load(
        run / "replay/prepared/datasets/ELECTRICMOTORVIBRATIONS/changed_input.npz",
        allow_pickle=False,
    )

    split_groups = dataset_info["train_validation_groups"]
    test_groups = sorted(set(labels_row["group"] for labels_row in labels))
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

    label_by_id = {row["sample_id"]: row for row in labels}
    prediction_rows = primary["predictions"]
    prediction_ids = [row["sample_id"] for row in prediction_rows]
    cold_ids = [row["sample_id"] for row in cold["predictions"]]
    untrained_ids = [row["sample_id"] for row in untrained["predictions"]]
    changed_ids = [row["sample_id"] for row in changed["predictions"]]
    if not (
        prediction_ids
        == cold_ids
        == untrained_ids
        == changed_ids
    ):
        raise ValueError("Control predictions do not preserve sample identity and order")

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
    labels_by_group = {}
    for row in ordered_labels:
        labels_by_group.setdefault(row["group"], set()).add(int(row["label"]))
    metrics = class_metrics(actual, predicted)
    task_output = {
        "task_id": TASK_ID,
        "run_id": run.name,
        "classifications": [
            {
                "sample_id": sample_id,
                "fault_type": f"Fault_{predicted_label}",
            }
            for sample_id, predicted_label in zip(prediction_ids, predicted)
        ],
        "accuracy": {
            "overall_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": [f"Fault_{label}" for label in CLASSES],
        "research_adaptation": True,
        "official_PHMForge_score": False,
    }

    recorded_metrics = {
        "accuracy": 0.6687306501547987,
        "macro_f1": 0.6294689343977786,
        "balanced_accuracy": 0.6585576923076923,
        "confusion_matrix": [
            [50, 0, 0, 0],
            [0, 0, 0, 69],
            [0, 37, 67, 0],
            [0, 0, 1, 99],
        ],
    }
    metric_match = (
        close(metrics["accuracy"], recorded_metrics["accuracy"])
        and close(metrics["macro_f1"], recorded_metrics["macro_f1"])
        and close(metrics["balanced_accuracy"], recorded_metrics["balanced_accuracy"])
        and metrics["confusion_matrix"] == recorded_metrics["confusion_matrix"]
    )

    prediction_hashes = {
        member: digest(run / "replay" / member) for member in PREDICTION_SHA256
    }
    primary_call = next(
        row
        for row in calls
        if row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith("predictions_dad57d865701.json")
    )
    cold_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith("predictions_e7f072795497.json")
    )
    untrained_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith("predictions_ae0201e5e3ed.json")
    )
    changed_call = next(
        row
        for row in calls
        if row.get("phase") == "cold_reload"
        and row.get("tool") == "classify_faults"
        and row.get("result", {}).get("artifact", "").endswith("predictions_e404aa3e5c25.json")
    )

    checks = {
        "source_manifest_hash": digest(args.source_manifest)
        == EXPECTED_MANIFEST_SHA256["source244"],
        "repair_manifest_hash": digest(args.repair_manifest)
        == EXPECTED_MANIFEST_SHA256["repair246"],
        "source_bundle_hash": source_bundle_sha256
        == EXPECTED_BUNDLE_SHA256["source244"],
        "repair_bundle_hash": repair_bundle_sha256
        == EXPECTED_BUNDLE_SHA256["repair246"],
        "source_member_listed_in_manifest": SOURCE_MEMBER in source_manifest,
        "source_member_manifest_hash": source_manifest[SOURCE_MEMBER]["sha256"]
        == SOURCE_SHA256,
        "source_bytes_match_pin": source_csv.stat().st_size
        == source_receipt["pin"]["bytes"],
        "source_sha256_matches_pin": digest(source_csv) == SOURCE_SHA256,
        "source_receipt_complete": source_receipt["complete_file"] is True
        and source_receipt["pinned_hash_verified"] is True,
        "source_rows": source_summary["rows"] == 30,
        "source_labels": source_summary["label_counts"]
        == {"0": 4, "1": 6, "2": 10, "3": 10},
        "source_nonfinite_values_disclosed": (
            source_summary["nonfinite_arrays"] == 4
            and source_summary["nonfinite_scalars"] == 0
            and source_summary["nonfinite_windows"] == 1
        ),
        "nonfinite_window_drop_matches_A246": (
            dataset_info["dropped_nonfinite_windows"]
            == source_summary["nonfinite_windows"]
        ),
        "retained_window_count_matches_A246": (
            source_summary["complete_windows"]
            - source_summary["nonfinite_windows"]
            == sum(dataset_info["rows_by_split"].values())
        ),
        "question_hash": question_sha == QUESTION_SHA256,
        "original_dataset": task["dataset"] == "ElectricMotorVibrations",
        "original_short_prompt": task["input_question"]
        == "Classify fault types for all test units in ElectricMotorVibrations dataset.",
        "pdmb_dataset_info_pin": len(pdmb_dataset["fault_types"]) == 4,
        "dataset_info_class_count": dataset_info["class_names"] == CLASSES,
        "test_labels_absent": "labels" not in test.files and "groups" not in test.files,
        "prepared_features_finite": all(
            np.isfinite(array["features"]).all() for array in (train, val, test)
        ),
        "train_val_test_groups_disjoint": all(
            not (split_group_sets[left] & split_group_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "test_group_count": len(test_groups)
        == dataset_info["groups_by_split"]["test"],
        "sample_ids_disjoint": all(
            not (split_ids[left] & split_ids[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "row_counts_match_dataset_info": {
            "train": train["features"].shape[0],
            "val": val["features"].shape[0],
            "test": test["features"].shape[0],
        }
        == dataset_info["rows_by_split"],
        "prediction_identity_complete": (
            len(prediction_ids) == len(set(prediction_ids))
            and set(prediction_ids) == split_ids["test"]
            and set(prediction_ids) == set(label_by_id)
        ),
        "all_four_classes_present": set(actual) == set(CLASSES),
        "each_test_group_has_one_class": all(
            len(labels_set) == 1 for labels_set in labels_by_group.values()
        ),
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
        ),
        "prediction_hashes": prediction_hashes == PREDICTION_SHA256,
        "primary_checkpoint_hash": digest(
            run / f"replay/{MODEL_DIR}/model.pt"
        ) == CHECKPOINT_SHA256,
        "untrained_checkpoint_hash": digest(
            run / f"replay/{MODEL_DIR}/untrained_model.pt"
        ) == UNTRAINED_CHECKPOINT_SHA256,
        "nonzero_parameter_updates": (
            training["epochs_executed"] == 50
            and training["optimizer_steps"] > 0
            and training["nonzero_gradient_steps"] == training["optimizer_steps"]
            and training["changed_parameter_tensors"] > 0
        ),
        "training_test_labels_isolated": training["test_labels_accessed"] is False,
        "training_group_isolation": training["group_overlap"] == [],
        "pre_score_controls": all(pre_score.values()),
        "cold_reload_byte_identical": (
            (run / "replay" / PRIMARY_PREDICTION).read_bytes()
            == (run / "replay" / COLD_PREDICTION).read_bytes()
        ),
        "cold_reload_labels_identical": predicted == cold_predicted,
        "weight_dependency": predicted != untrained_predicted,
        "input_dependency": predicted != changed_predicted,
        "changed_input_identity_preserved": set(changed_ids) == set(prediction_ids),
        "changed_input_array_identity_preserved": set(changed_npz["sample_ids"].tolist())
        == set(test["sample_ids"].tolist()),
        "changed_input_values_differ": not np.array_equal(
            changed_npz["features"], test["features"]
        ),
        "call_checkpoint_links": (
            primary_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
            and cold_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
            and untrained_call["result"]["checkpoint_sha256"]
            == UNTRAINED_CHECKPOINT_SHA256
            and changed_call["result"]["checkpoint_sha256"] == CHECKPOINT_SHA256
        ),
        "metrics_match_A246": metric_match,
        "authored_range_false": not (
            expected_range[0] <= metrics["accuracy"] <= expected_range[1]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"pdm_fault_015 contract checks failed: {failed}")

    result = {
        "utc": utc_now(),
        "task_id": TASK_ID,
        "run_id": run.name,
        "classification": (
            "source-bound four-class research adaptation using the sealed A246 "
            "MLP run; not an official PHMForge score"
        ),
        "question_sha256": question_sha,
        "original_question": task["input_question"],
        "expected_output_fields": task["ground_truth"]["required_fields"],
        "expected_accuracy_range": expected_range,
        "source": {
            "dataset": "ELECTRICMOTORVIBRATIONS",
            "sha256": digest(source_csv),
            "rows": source_summary["rows"],
            "label_counts": source_summary["label_counts"],
            "complete_2048_point_windows": source_summary["complete_windows"],
            "nonfinite_source_arrays": source_summary["nonfinite_arrays"],
            "nonfinite_complete_windows": source_summary["nonfinite_windows"],
            "dropped_nonfinite_windows": dataset_info["dropped_nonfinite_windows"],
            "class_names": CLASSES,
            "class_name_policy": (
                "Original numeric labels are preserved and rendered as the "
                "task's generic Fault_0 ... Fault_3 names; no physical fault "
                "names are invented"
            ),
            "channel_limit": (
                "The prepared classifier uses the three published accelerometer "
                "channels. This short pdm_fault_015 task does not request the "
                "acoustic/vibration comparison that blocks pdm_fault_003."
            ),
            "group_limit": (
                "Whole published records are isolated across splits, but records "
                "are not certified as independent physical motors"
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
            "selected_validation_accuracy": training["selected_validation_accuracy"],
            "training_rows": training["train_rows"],
            "validation_rows": training["validation_rows"],
            "test_labels_accessed": training["test_labels_accessed"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "split": {
            "rows_by_split": dataset_info["rows_by_split"],
            "groups_by_split": dataset_info["groups_by_split"],
            "test_groups": test_groups,
            "group_definition": dataset_info["group_definition"],
        },
        "external_evaluation": {
            **metrics,
            "accuracy_within_authored_range": bool(
                expected_range[0] <= metrics["accuracy"] <= expected_range[1]
            ),
            "source": "sealed A246 model predictions plus external final labels",
        },
        "research_output": "RESEARCH_OUTPUT249.json",
        "controls": {
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "cold_reload_byte_identical": checks["cold_reload_byte_identical"],
            "weight_dependency": checks["weight_dependency"],
            "input_dependency": checks["input_dependency"],
        },
        "qualification": {
            "source_bound_four_class_classifier_qualified": True,
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
            "selected_members": extracted,
        },
    }
    write_json(run / "RESEARCH_OUTPUT249.json", task_output)
    write_json(run / "QUALIFICATION249.json", result)
    write_json(
        run / "retention_policy.json",
        {
            "excluded_rebuildable_directory": "clients-cache",
            "shared_dependencies": {
                "path": "/srv/yyf-zyq/PHM_20261010/_shared",
                "retention": "existing_pinned_readonly_dependencies",
            },
            "policy": "retain audit, source receipt, source file, selected frozen "
            "inputs, model checkpoints, predictions, logs and result",
            "large_shared_sources": "referenced by pinned immutable archive and "
            "verified SHA-256",
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
            "result": "QUALIFICATION249.json",
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
