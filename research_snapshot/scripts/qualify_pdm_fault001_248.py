"""Qualify the PHMForge CWRU fault-classification task on its pinned source.

This is a source-bound research adaptation, not an official PHMForge score.
The original prompt asks for inner-race, outer-race, and ball-fault
classification. PDMBench's committed metadata maps CWRU source labels 0..3 to
normal, inner_race, outer_race, and ball. We keep the three faulty classes
requested by the prompt and hold out whole contiguous source blocks.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile

import numpy as np


TASK_ID = "pdm_fault_001"
QUESTION_SHA256 = "8371cf423904d0242e410f271535dfd3a74ad66bba966a412845110c6246d1d7"
SOURCE_SHA256 = "e05c2b23c7b0d7450f31982e9d10f395fe0a8f92d93dca9f6997655aa1026ff9"
SOURCE_GIT_BLOB = "605450e6e52917b6bcee8b31a62c899a4876a139"
SOURCE_MEMBER = "inputs/pdmb_attempt2/CWRU.csv"
RECEIPT_MEMBER = "inputs/pdmb_attempt2/CWRU.receipt.json"
SCENARIO_MEMBER = (
    "source/official/ReActXen/src/reactxen/demo/"
    "intent_implementation_demo/scenarios/phm_scenarios.json"
)
PDMB_COMMIT = "a3415dd5f2b7cb45ec7f973c2f330bdafd2dd006"
PDMB_DATASET_INFO_BLOB = "7c30ea21798beaa52d14dbb005b1726280dfdd94"
PDMB_DATASET_INFO_SHA256 = "a2ebc3b02b75f3e2d0f62bbb74b6b13dbb2ebc09a141507fc39c30d8b87a5050"
PDMB_DATASET_INFO_URL = (
    "https://raw.githubusercontent.com/zshuai8/PDMBenchmark/"
    f"{PDMB_COMMIT}/dataset/dataset_info.json"
)
LABEL_MAP = {0: "normal", 1: "inner_race", 2: "outer_race", 3: "ball"}
REQUESTED_CLASSES = ["inner_race", "outer_race", "ball"]
SEED = 20261020248


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


def git_blob_sha1(path):
    path = Path(path)
    value = hashlib.sha1()
    value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract_member(bundle, member, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as stream:
        source = stream.extractfile(stream.getmember(member))
        if source is None:
            raise ValueError(f"Archive member is not a file: {member}")
        with destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    return destination


def copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
    }


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


def fetch_dataset_info(destination):
    result = subprocess.run(
        ["curl", "-k", "-fsSL", PDMB_DATASET_INFO_URL],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = result.stdout
    git_blob = hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\0" + payload
    ).hexdigest()
    sha256 = hashlib.sha256(payload).hexdigest()
    if git_blob != PDMB_DATASET_INFO_BLOB or sha256 != PDMB_DATASET_INFO_SHA256:
        raise ValueError("PDMBench dataset metadata does not match its committed pin")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return json.loads(payload)


def numeric_array(text):
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Expected a complete feature array")
    value = np.fromstring(text[1:-1], sep=",", dtype=np.float64)
    if len(value) != text.count(",") + 1:
        raise ValueError("Feature array parsed only partially")
    return value


def prepare_dataset(run, source_csv):
    csv.field_size_limit(128 * 1024**2)
    features, labels, groups, identities, provenance = [], [], [], [], []
    all_label_counts = Counter()
    excluded_normal_rows = 0
    source_rows = 0
    previous = None
    group_index = -1

    with Path(source_csv).open(newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            source_rows += 1
            label_number = int(row["label"])
            fault_size = float(row["fault_size"])
            segment_idx = int(row["segment_idx"])
            all_label_counts[label_number] += 1
            current = (label_number, fault_size)
            if (
                previous is None
                or current != previous[:2]
                or segment_idx <= previous[2]
            ):
                group_index += 1
            previous = (label_number, fault_size, segment_idx)
            if label_number == 0:
                excluded_normal_rows += 1
                continue
            if label_number not in (1, 2, 3):
                raise ValueError(f"Unexpected CWRU label: {label_number}")
            vector = numeric_array(row["all_features"]).astype(np.float32)
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite published feature vector")
            identity = hashlib.sha256(
                f"PHM248:CWRU:{row_index}".encode()
            ).hexdigest()[:24]
            label_name = LABEL_MAP[label_number]
            group = f"contiguous_block_{group_index:04d}"
            features.append(vector)
            labels.append(label_name)
            groups.append(group)
            identities.append(identity)
            provenance.append(
                {
                    "sample_id": identity,
                    "source_row": row_index,
                    "segment_idx": segment_idx,
                    "fault_size": fault_size,
                    "published_label": label_number,
                    "label": label_name,
                    "group": group,
                }
            )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    g = np.asarray(groups)
    ids = np.asarray(identities)
    if not len(x) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid prepared CWRU feature matrix")
    if set(y.tolist()) != set(REQUESTED_CLASSES):
        raise ValueError("Published source does not contain all requested fault classes")

    groups_by_label = defaultdict(list)
    labels_by_group = defaultdict(set)
    for label, group in zip(y.tolist(), g.tolist()):
        labels_by_group[group].add(label)
    if any(len(values) != 1 for values in labels_by_group.values()):
        raise ValueError("A contiguous block contains multiple class labels")
    for group in sorted(labels_by_group):
        groups_by_label[next(iter(labels_by_group[group]))].append(group)

    split = {"train": [], "val": [], "test": []}
    for label in REQUESTED_CLASSES:
        available = sorted(groups_by_label[label])
        if len(available) < 5:
            raise ValueError(f"Too few source blocks for {label}: {len(available)}")
        random.Random(f"{SEED}:{label}").shuffle(available)
        val_count = max(1, round(len(available) * 0.20))
        test_count = max(1, round(len(available) * 0.20))
        train_count = len(available) - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Insufficient training blocks for {label}")
        split["val"].extend(available[:val_count])
        split["test"].extend(available[val_count : val_count + test_count])
        split["train"].extend(available[val_count + test_count :])
    assert all(
        not set(split[left]) & set(split[right])
        for left, right in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
    )

    directory = run / "dataset"
    directory.mkdir(parents=True, exist_ok=False)
    for part, selected in split.items():
        take = np.isin(g, selected)
        payload = {
            "features": x[take],
            "sample_ids": ids[take],
            "dataset": np.asarray("CWRU"),
        }
        if part != "test":
            payload.update(labels=y[take], groups=g[take])
        np.savez_compressed(directory / f"{part}.npz", **payload)

    test_indices = np.flatnonzero(np.isin(g, split["test"]))
    write_json(
        run / "hidden/test_labels248.json",
        [provenance[index] for index in test_indices],
    )
    write_json(
        run / "audit/CWRU_split248.json",
        {
            "split": split,
            "observations": provenance,
            "excluded_normal_rows": excluded_normal_rows,
            "all_source_label_counts": {
                str(key): value for key, value in sorted(all_label_counts.items())
            },
        },
    )

    info = {
        "dataset": "CWRU",
        "seed": SEED,
        "class_names": REQUESTED_CLASSES,
        "architectures": ["mlp"],
        "group_definition": (
            "Contiguous published label/fault-size runs bounded by segment_idx "
            "resets; original recording and physical-bearing IDs are absent, so "
            "groups are source blocks rather than certified independent bearings"
        ),
        "source_sha256": digest(source_csv),
        "source_rows": source_rows,
        "used_rows": int(len(x)),
        "excluded_normal_rows": excluded_normal_rows,
        "label_map": {str(key): value for key, value in LABEL_MAP.items()},
        "requested_classes": REQUESTED_CLASSES,
        "preprocessing": (
            "Use the 17 numeric features already published in all_features; "
            "no signal filtering, resampling, or target-derived feature selection"
        ),
        "units": "Original published feature scale; physical feature names are absent",
        "train_validation_test_groups": split,
        "rows_by_split": {
            key: int(np.isin(g, selected).sum()) for key, selected in split.items()
        },
        "groups_by_split": {key: len(selected) for key, selected in split.items()},
        "groups_per_class": {
            label: len(groups_by_label[label]) for label in REQUESTED_CLASSES
        },
        "labels_in_test_input": False,
        "physical_identity_limitation": (
            "Published CSV has no original CWRU recording/file or bearing identifier"
        ),
    }
    write_json(directory / "dataset.json", info)

    changed = dict(np.load(directory / "test.npz", allow_pickle=False))
    fit = np.load(directory / "train.npz", allow_pickle=False)["features"]
    changed["features"] = (
        changed["features"] + np.maximum(fit.std(axis=0), 1e-6) * 0.25
    ).astype(np.float32)
    np.savez_compressed(directory / "changed_input.npz", **changed)
    return info


def class_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    matrix = np.zeros(
        (len(REQUESTED_CLASSES), len(REQUESTED_CLASSES)), dtype=int
    )
    lookup = {label: index for index, label in enumerate(REQUESTED_CLASSES)}
    for truth, estimate in zip(actual, predicted):
        matrix[lookup[str(truth)], lookup[str(estimate)]] += 1
    per_class = {}
    f1_values, recall_values = [], []
    for label in REQUESTED_CLASSES:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        recall_values.append(recall)
        per_class[label] = {
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
    parser.add_argument("--source-bundle", type=Path, required=True)
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

    metadata = fetch_dataset_info(run / "source/PDMB_dataset_info.json")
    cwru_metadata = metadata["04"]
    if cwru_metadata["fault_types"] != ["normal", *REQUESTED_CLASSES]:
        raise ValueError("PDMBench CWRU class semantics changed")
    if cwru_metadata["features"] != ["vibration"] or cwru_metadata["channels"] != 1:
        raise ValueError("PDMBench CWRU channel semantics changed")

    source_csv = extract_member(
        args.source_bundle, SOURCE_MEMBER, run / "source/CWRU.csv"
    )
    receipt_path = extract_member(
        args.source_bundle, RECEIPT_MEMBER, run / "source/CWRU.receipt.json"
    )
    source_sha = digest(source_csv)
    source_blob = git_blob_sha1(source_csv)
    receipt = json.loads(receipt_path.read_text())
    if source_sha != SOURCE_SHA256 or source_blob != SOURCE_GIT_BLOB:
        raise ValueError("CWRU source does not match the published pin")
    if (
        receipt["actual_sha256"] != SOURCE_SHA256
        or receipt["actual_git_blob_sha1"] != SOURCE_GIT_BLOB
        or not receipt["pinned_hash_verified"]
        or not receipt["complete_file"]
    ):
        raise ValueError("CWRU receipt does not authenticate the complete source")

    info = prepare_dataset(run, source_csv)
    code = run / "code"
    code.mkdir()
    copied = [
        copy_file(
            workspace / "scripts/real_fault_model.py",
            code / "real_fault_model.py",
        ),
        copy_file(
            workspace / "scripts/real_rul_model.py",
            code / "real_rul_model.py",
        ),
        copy_file(
            workspace / "scripts/fault_model_cli_247.py",
            code / "fault_model_cli.py",
        ),
        copy_file(Path(__file__), code / Path(__file__).name),
        copy_file(
            workspace / "scripts/verify_pdm_fault001_248.py",
            code / "verify_pdm_fault001_248.py",
        ),
    ]
    write_json(
        run / "SOURCE_AND_INPUTS248.json",
        {
            "files": copied,
            "source_bundle": str(args.source_bundle.resolve()),
            "source_bundle_sha256": digest(args.source_bundle),
            "scenario_bundle": str(args.scenario_bundle.resolve()),
            "scenario_bundle_sha256": digest(args.scenario_bundle),
        },
    )

    test = np.load(run / "dataset/test.npz", allow_pickle=False)
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
            str(run / "dataset"),
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
        (model_dir / "model.pt", run / "dataset/test.npz", prediction, "predict.log"),
        (
            model_dir / "model.pt",
            run / "dataset/test.npz",
            repeat,
            "cold_reload.log",
        ),
        (
            model_dir / "untrained_model.pt",
            run / "dataset/test.npz",
            untrained,
            "untrained.log",
        ),
        (
            model_dir / "model.pt",
            run / "dataset/changed_input.npz",
            changed,
            "changed_input.log",
        ),
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
    write_json(run / "PREDICTION_SEALS248.json", seals)

    label_rows = json.loads((run / "hidden/test_labels248.json").read_text())
    by_id = {row["sample_id"]: row for row in label_rows}
    if len(by_id) != len(label_rows) or set(by_id) != set(test_ids):
        raise ValueError("External labels do not cover the sealed observations")
    ordered = [by_id[sample_id] for sample_id in test_ids]
    actual = np.asarray([row["label"] for row in ordered], dtype=object)
    groups = np.asarray([row["group"] for row in ordered])

    payload = json.loads(prediction.read_text())
    predicted = np.asarray(
        [row["predicted_label"] for row in payload["predictions"]], dtype=object
    )
    predicted_ids = [row["sample_id"] for row in payload["predictions"]]
    if predicted_ids != test_ids:
        raise ValueError("Prediction identities or order changed")
    metrics = class_metrics(actual, predicted)

    repeat_payload = json.loads(repeat.read_text())
    repeat_predictions = np.asarray(
        [row["predicted_label"] for row in repeat_payload["predictions"]],
        dtype=object,
    )
    untrained_payload = json.loads(untrained.read_text())
    untrained_predictions = np.asarray(
        [row["predicted_label"] for row in untrained_payload["predictions"]],
        dtype=object,
    )
    changed_payload = json.loads(changed.read_text())
    changed_predictions = np.asarray(
        [row["predicted_label"] for row in changed_payload["predictions"]],
        dtype=object,
    )

    group_metrics = {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_metrics[group] = {
            "n": int(selected.sum()),
            "label": Counter(actual[selected].tolist()).most_common(1)[0][0],
            "accuracy": float(np.mean(actual[selected] == predicted[selected])),
        }

    training_audit = json.loads((model_dir / "training_audit.json").read_text())
    expected_range = task["ground_truth"]["expected_accuracy_range"]
    split_groups = info["train_validation_test_groups"]
    test_observation_groups = set(groups.tolist())
    checks = {
        "source_hash_matches_public_pin": (
            source_sha == SOURCE_SHA256 and source_blob == SOURCE_GIT_BLOB
        ),
        "source_receipt_authenticates_complete_file": (
            receipt["actual_sha256"] == SOURCE_SHA256
            and receipt["actual_git_blob_sha1"] == SOURCE_GIT_BLOB
            and receipt["pinned_hash_verified"] is True
            and receipt["complete_file"] is True
        ),
        "pdmb_metadata_blob_matches": (
            digest(run / "source/PDMB_dataset_info.json") == PDMB_DATASET_INFO_SHA256
        ),
        "pdmb_metadata_maps_all_class_labels": (
            cwru_metadata["fault_types"] == ["normal", *REQUESTED_CLASSES]
        ),
        "question_hash_matches_frozen_A187": question_hash == QUESTION_SHA256,
        "labels_absent_from_test_observations": "labels" not in test.files,
        "labels_read_after_all_prediction_seals": True,
        "complete_unique_external_identity_coverage": (
            set(predicted_ids) == set(test_ids)
            and len(predicted_ids) == len(set(predicted_ids)) == len(by_id)
        ),
        "nonzero_parameter_updates": (
            training_audit["optimizer_steps"] > 0
            and training_audit["nonzero_gradient_steps"] > 0
            and training_audit["changed_parameter_tensors"] > 0
        ),
        "cold_reload_predictions_equal": bool(
            np.array_equal(predicted, repeat_predictions)
        ),
        "weight_dependency": bool(
            not np.array_equal(predicted, untrained_predictions)
        ),
        "input_dependency": bool(
            not np.array_equal(predicted, changed_predictions)
        ),
        "train_validation_test_group_isolation": (
            not set(split_groups["train"]) & set(split_groups["val"])
            and not set(split_groups["train"]) & set(split_groups["test"])
            and not set(split_groups["val"]) & set(split_groups["test"])
            and test_observation_groups == set(split_groups["test"])
        ),
        "all_requested_fault_classes_present": set(actual.tolist())
        == set(REQUESTED_CLASSES),
        "normal_class_excluded_with_prompt": info["excluded_normal_rows"] > 0,
        "finite_model_inputs": bool(np.isfinite(test["features"]).all()),
    }
    all_checks = all(bool(value) for value in checks.values())
    result = {
        "utc": utc_now(),
        "task_id": TASK_ID,
        "run_id": run.name,
        "classification": (
            "source-bound CWRU fault-classification research adaptation; "
            "official PHMForge score not claimed"
        ),
        "model_sessions": 0,
        "training_jobs": 1,
        "question_sha256": question_hash,
        "source": {
            "sha256": source_sha,
            "git_blob_sha1": source_blob,
            "public_pin": receipt["pin"],
            "source_rows": info["source_rows"],
            "used_rows": info["used_rows"],
            "normal_rows_excluded": info["excluded_normal_rows"],
            "published_label_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        json.loads(
                            (run / "audit/CWRU_split248.json").read_text()
                        )["all_source_label_counts"]
                    ).items()
                )
            },
            "label_map": info["label_map"],
            "requested_classes": REQUESTED_CLASSES,
            "group_definition": info["group_definition"],
        },
        "training": {
            "architecture": "mlp",
            "epochs_executed": training_audit["epochs_executed"],
            "optimizer_steps": training_audit["optimizer_steps"],
            "nonzero_gradient_steps": training_audit["nonzero_gradient_steps"],
            "changed_parameter_tensors": training_audit["changed_parameter_tensors"],
            "selected_epoch": training_audit["selected_epoch"],
            "selected_validation_accuracy": training_audit[
                "selected_validation_accuracy"
            ],
            "train_rows": training_audit["train_rows"],
            "validation_rows": training_audit["validation_rows"],
            "test_labels_accessed": training_audit["test_labels_accessed"],
            "groups_by_split": info["groups_by_split"],
            "rows_by_split": info["rows_by_split"],
        },
        "external_evaluation": {
            **metrics,
            "authored_accuracy_range": expected_range,
            "accuracy_within_authored_range": bool(
                expected_range[0] <= metrics["accuracy"] <= expected_range[1]
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
                "asks_inner_outer_ball": True,
                "expected_fault_types_field": task["ground_truth"].get(
                    "expected_fault_types"
                ),
                "expected_accuracy_range": expected_range,
            },
            "pdmb_metadata": {
                "commit": PDMB_COMMIT,
                "dataset_info_git_blob": PDMB_DATASET_INFO_BLOB,
                "dataset_info_sha256": PDMB_DATASET_INFO_SHA256,
                "declared_name": cwru_metadata["name"],
                "declared_features": cwru_metadata["features"],
                "declared_channels": cwru_metadata["channels"],
                "declared_fault_types": cwru_metadata["fault_types"],
            },
            "observed_source": {
                "columns": ["fault_size", "segment_idx", "all_features", "label"],
                "published_labels": sorted(
                    info["label_map"].keys(), key=lambda value: int(value)
                ),
                "rows": info["source_rows"],
                "used_faulty_rows": info["used_rows"],
                "excluded_normal_rows": info["excluded_normal_rows"],
            },
            "adaptation_scope": (
                "Classify the three faulty classes explicitly requested by the "
                "original question; normal is excluded rather than mislabeled. "
                "Contiguous published source blocks are held out as groups."
            ),
            "remaining_limitations": [
                info["physical_identity_limitation"],
                "The published all_features column has no physical feature-name dictionary",
                "The original PHMForge visible-truth tool flow and scorer are not replayed",
            ],
        },
        "qualification": {
            "source_bound_three_class_classifier_qualified": all_checks,
            "research_adapted_task_contract_qualified": all_checks,
            "official_original_benchmark_score_qualified": False,
            "fully_qualified_original_tasks": 0,
            "reason_official_score_not_claimed": (
                "The source and requested class semantics align under a defensible "
                "research adaptation, but this work does not replay the original "
                "PHMForge scorer and visible-ground-truth tool sequence."
            ),
        },
        "new_model_sessions": 0,
        "new_training_jobs": 1,
        "new_confirmed_PHM_problems": 0,
    }
    write_json(run / "QUALIFICATION248.json", result)
    write_json(
        status_path,
        {
            "run_id": run.name,
            "status": "finished" if all_checks else "finished_with_failed_controls",
            "finished_utc": utc_now(),
            "checks_passed": result["control_summary"]["checks_passed"],
            "checks_total": result["control_summary"]["checks_total"],
            "research_adapted_task_contract_qualified": all_checks,
            "official_original_benchmark_score_qualified": False,
        },
    )
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
