"""Prepare LDM bearing RUL from the authors' pinned measured feature files."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import hmac
from pathlib import Path
import secrets

import numpy as np

from .data import _catalog, download, resource_json
from .io import file_sha256, write_json


# The published experiment table does not qualify these three runs as having
# the same documented threshold endpoint as the other 14 runs.
ELIGIBLE = [f"B{i:02}" for i in range(1, 18) if i not in (1, 5, 16)]
TARGET = "Remaining author-feature acquisition records until the recorded sequence endpoint; not hours."


def _numeric_csv(path, columns):
    times, rows = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != columns + 1 or header[0] != "Time":
            raise ValueError(f"Unexpected LDM CSV columns: {path}")
        for row in reader:
            if len(row) != columns + 1:
                raise ValueError(f"Unexpected LDM CSV row: {path}")
            times.append(row[0])
            rows.append([float(value) for value in row[1:]])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"Invalid LDM CSV values: {path}")
    return times, values


def _trajectory(source, name):
    fft = np.load(source / "fft" / (name + ".npy"), allow_pickle=False)
    op_times, operating = _numeric_csv(source / "op" / (name + ".csv"), 6)
    temp_times, temperature = _numeric_csv(source / "temp" / (name + ".csv"), 3)
    if fft.ndim != 2 or fft.shape[1] != 200 or len(fft) < 101 or not np.isfinite(fft).all():
        raise ValueError(f"Invalid author-derived LDM FFT features: {name}")
    if len(fft) != len(operating) or len(fft) != len(temperature) or op_times != temp_times:
        raise ValueError(f"LDM FFT/operating/temperature rows are not aligned: {name}")
    return np.column_stack((fft.reshape(len(fft), 20, 10).max(axis=2), operating, temperature,
                            np.arange(len(fft), dtype=np.float64))).astype(np.float32)


def _payload(rows, include_labels):
    result = {"dataset": np.asarray("LDM_RTF"),
              "features": np.asarray([r["features"] for r in rows], dtype=np.float32),
              "sample_ids": np.asarray([r["sample_id"] for r in rows]),
              "unit_ids": np.asarray([r["unit_id"] for r in rows], dtype=np.int64),
              "scenario_ids": np.asarray([r["scenario_id"] for r in rows], dtype=np.int64),
              "elapsed_acquisitions": np.asarray([r["elapsed_acquisitions"] for r in rows], dtype=np.int64)}
    if include_labels:
        result["labels"] = np.asarray([r["actual_rul"] for r in rows], dtype=np.float32)
    return result


def prepare_ldm(destination, source=None, workers=4):
    """Download/verify author features and split the runs by whole bearing."""
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source).resolve() if source else dest / "raw"
    pin = resource_json("ldm_sources.json")

    def acquire(item):
        path = source_dir / item["path"]
        if not source:
            download(item["url"], path, item["sha256"], item["bytes"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"LDM author feature file is missing or does not match the pinned SHA-256: {path}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        list(pool.map(acquire, pin["files"]))
    trajectories = {name: _trajectory(source_dir, name) for name in ELIGIBLE}
    order = np.random.default_rng(20261024).permutation(ELIGIBLE).tolist()
    split = {"train": order[:7], "val": order[7:9], "test": order[9:]}
    if len(set(sum(split.values(), []))) != 14:
        raise ValueError("LDM bearing split contains an overlap")

    private = dest / "private"
    private.mkdir()
    public = dest / "public"
    data = public / "datasets/LDM_RTF"
    data.mkdir(parents=True)
    (dest / "holdout").mkdir()
    identity_key = secrets.token_bytes(32)
    (private / "holdout_identity_key.hex").write_text(identity_key.hex() + "\n")
    mapping = dict(zip(split["train"] + split["val"], range(1, 10)))
    mapping.update(dict(zip(split["test"], secrets.SystemRandom().sample(range(1000000, 2000000), 5))))
    rows = {"train": [], "val": [], "test": []}
    gold = []
    for kind, names in split.items():
        for name in names:
            features = trajectories[name]
            n, unit = len(features), mapping[name]
            indices = (range(n) if kind == "train" else
                       sorted({max(0, min(n - 2, round((n - 1) * f))) for f in (.3, .6, .9)}) if kind == "val" else
                       sorted(secrets.SystemRandom().sample(range(max(0, int(.15 * n)), max(5, int(.9 * n))), 5)))
            for index in indices:
                sample = hmac.new(identity_key, f"{kind}:{name}:{index}".encode(), hashlib.sha256).hexdigest()[:24]
                row = {"features": features[index], "sample_id": sample, "unit_id": unit,
                       "scenario_id": 1, "elapsed_acquisitions": index, "actual_rul": n - 1 - index}
                rows[kind].append(row)
                if kind == "test":
                    gold.append({key: row[key] for key in ("sample_id", "unit_id", "scenario_id", "actual_rul")})
    pairs = list(zip(rows["test"], gold))
    secrets.SystemRandom().shuffle(pairs)
    rows["test"], gold = map(list, zip(*pairs))
    np.savez_compressed(data / "train.npz", **_payload(rows["train"], True))
    np.savez_compressed(data / "val.npz", **_payload(rows["val"], True))
    np.savez_compressed(dest / "holdout/test.npz", **_payload(rows["test"], False))
    write_json(private / "labels.json", gold)
    write_json(private / "source_audit.json", {"split": split, "unit_mapping": mapping,
        "sequence_lengths": {name: len(value) for name, value in trajectories.items()}})
    write_json(data / "training_config.json", {"seed": 321, "epochs": 30, "batch_size": 128,
        "learning_rate": .001, "cpu_threads": 2, "fit_unit_ids": list(range(1, 8)), "target_definition": TARGET})
    write_json(data / "dataset.json", {"dataset": "LDM_RTF", "task": "rul", "feature_count": 30,
        "target_unit": "acquisitions", "label": TARGET,
        "source": "https://github.com/alireza-javanmardi/bearing-RUL", "source_revision": pin["revision"],
        "preprocessing": "20 maxima of ten author FFT bins, six operating values, three temperatures and acquisition index.",
        "split": "7 train, 2 validation and 5 test whole-bearing runs; B01, B05 and B16 excluded.",
        "limitations": ["The original 152 GB of raw waveforms are not reprocessed; features are author-provided.",
                        "Acquisition spacing is irregular; endpoint and acquisition count are not field failure time or hours.",
                        "Five test bearings and 25 correlated prefix queries; public data may be memorized.",
                        "This is LDM, not the diagnostic Paderborn KAt dataset or Mendeley data used in some original questions."]})
    write_json(public / "dataset_registry.json", {"LDM_RTF": {"task": "rul", "format": "measured_endpoint_npz",
        "path": "datasets/LDM_RTF", "architectures": ["mlp"], "available_splits": ["train", "val"],
        "test_inputs_visible": False}})
    _catalog(public)
    write_json(dest / "SOURCE_BINDINGS.json", pin)
    report = {"task_id": "research_ldm_bearing_rul", "dataset": "LDM_RTF", "task_kind": "rul", "status": "prepared",
        "training_bearings": 7, "development_bearings": 2, "test_bearings": 5, "test_queries": len(gold),
        "labels_in_test_input": False, "measured_or_simulated": "author-derived laboratory measurements",
        "scope": "LDM measured endpoint numerical research adaptation and source substitution; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    return report
