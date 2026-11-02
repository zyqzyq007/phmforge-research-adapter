"""Prepare measured XJTU-SY bearing lifetimes from the pinned native archives."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys

import numpy as np

from .data import _catalog, download, features, resource_json
from .io import file_sha256, write_json


# Original tutorial, Table 3: terminal acquisition counts, not per-record faults.
COUNTS = {
    "Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158,
    "Bearing1_4": 122, "Bearing1_5": 52, "Bearing2_1": 491,
    "Bearing2_2": 161, "Bearing2_3": 533, "Bearing2_4": 42,
    "Bearing2_5": 339, "Bearing3_1": 2538, "Bearing3_2": 2496,
    "Bearing3_3": 371, "Bearing3_4": 1515, "Bearing3_5": 114,
}
CONDITIONS = {"35Hz12kN": 1, "37.5Hz11kN": 2, "40Hz10kN": 3}
HEADER = b"Horizontal_vibration_signals,Vertical_vibration_signals"


def _record(item):
    path, name, minute = item
    with path.open("rb") as stream:
        if stream.readline().strip() != HEADER:
            raise ValueError(f"Unexpected XJTU CSV header: {path}")
        raw = np.loadtxt(stream, delimiter=",", dtype=np.float64)
    if raw.shape != (32768, 2) or not np.isfinite(raw).all() or np.any(raw.std(axis=0) <= 0):
        raise ValueError(f"Invalid XJTU measured recording: {path}")
    value = features(raw)
    if value.shape != (30,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid XJTU feature values: {path}")
    return name, minute, value


def _source_records(root, workers):
    """Validate identities and compute features from every measured recording."""
    csvs, extras = [], []
    pattern = re.compile(r"XJTU-SY_Bearing_Datasets/([^/]+)/(Bearing[123]_[1-5])/(\d+)\.csv")
    seen = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = pattern.fullmatch(relative)
        if match is None:
            extras.append(relative)
            continue
        condition, name, minute = match.group(1), match.group(2), int(match.group(3))
        if condition not in CONDITIONS or int(name[7]) != CONDITIONS[condition] or not 1 <= minute <= COUNTS[name]:
            raise ValueError(f"Unexpected XJTU identity: {relative}")
        if (name, minute) in seen:
            raise ValueError(f"Duplicate XJTU recording: {relative}")
        seen.add((name, minute))
        csvs.append((path, name, minute))
    expected = {(name, minute) for name, count in COUNTS.items() for minute in range(1, count + 1)}
    if seen != expected or extras != ["XJTU-SY_Bearing_Datasets/Introduction_to_XJTU-SY_Bearing_Dataset.pdf"]:
        raise ValueError(f"Incomplete or unexpected XJTU archive members: {len(seen)}/{len(expected)}, extras={extras[:5]}")
    values = {name: np.empty((count, 30), dtype=np.float32) for name, count in COUNTS.items()}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        for done, (name, minute, value) in enumerate(pool.map(_record, csvs), 1):
            values[name][minute - 1] = value
            if done % 1000 == 0 or done == len(csvs):
                print(f"XJTU measured records processed: {done}/{len(csvs)}", file=sys.stderr, flush=True)
    return values


def _payload(rows, labels):
    result = {"dataset": np.asarray("XJTU"),
              "features": np.asarray([r["features"] for r in rows], dtype=np.float32),
              "sample_ids": np.asarray([r["sample_id"] for r in rows]),
              "unit_ids": np.asarray([r["unit_id"] for r in rows], dtype=np.int64),
              "scenario_ids": np.asarray([r["scenario_id"] for r in rows], dtype=np.int64),
              "elapsed_minutes": np.asarray([r["elapsed_minutes"] for r in rows], dtype=np.int64)}
    if labels:
        result["labels"] = np.asarray([r["actual_rul"] for r in rows], dtype=np.float32)
    return result


def prepare_xjtu(destination, source=None, workers=4, extractor=None):
    """Download/verify six volumes, reconstruct real features and make bearing-level splits."""
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    executable = str(Path(extractor).resolve()) if extractor else (shutil.which("7zz") or shutil.which("7z"))
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError("XJTU multi-volume RAR extraction requires a compatible 7zz/7z; install 7-Zip or pass --extractor")
    dest.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source).resolve() if source else dest / "raw"
    pin = resource_json("xjtu_sources.json")
    volumes = []
    for item in pin["files"]:
        path = source_dir / item["filename"]
        if not source:
            download(item["url"], path, item["publisher_sha256"], item["bytes"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["publisher_sha256"]:
            raise ValueError(f"XJTU source volume is missing or fails the published SHA-256 pin: {path}")
        volumes.append(path)

    private = dest / "private"
    private.mkdir()
    scratch = private / "native_extraction"
    scratch.mkdir()
    command = [executable, "x", "-y", "-bb0", "-bsp0", "-mmt=2", f"-o{scratch}", str(volumes[0])]
    with (private / "extraction.log").open("wb") as log:
        status = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
    if status.returncode:
        raise RuntimeError(f"XJTU extraction failed (exit {status.returncode}); inspect {private / 'extraction.log'}")
    records = _source_records(scratch, workers)

    # The bearing split is fixed before training and never mixes recordings of a bearing.
    rng = np.random.default_rng(278)
    split = {"train": [], "val": [], "test": []}
    for scenario in (1, 2, 3):
        names = rng.permutation([f"Bearing{scenario}_{i}" for i in range(1, 6)]).tolist()
        split["train"].extend(names[:3])
        split["val"].append(names[3])
        split["test"].append(names[4])
    if len(set(sum(split.values(), []))) != 15:
        raise ValueError("XJTU bearing split contains an overlap")

    public = dest / "public"
    data = public / "datasets/XJTU"
    data.mkdir(parents=True)
    (dest / "holdout").mkdir()
    identity_key = secrets.token_bytes(32)
    (private / "holdout_identity_key.hex").write_text(identity_key.hex() + "\n")
    test_units = secrets.SystemRandom().sample(range(1000000, 2000000), 3)
    mappings = dict(zip(split["train"] + split["val"], range(1, 13)))
    mappings.update(dict(zip(split["test"], test_units)))
    rows = {"train": [], "val": [], "test": []}
    gold = []
    for kind, names in split.items():
        for name in names:
            n, scenario, unit = COUNTS[name], int(name[7]), mappings[name]
            # Random held-out prefixes avoid publishing a fixed query fraction
            # from which a test bearing's terminal time could be derived.
            indices = (range(1, n + 1) if kind == "train" else
                       sorted({max(1, min(n - 1, round(n * f))) for f in (.3, .6, .9)}) if kind == "val" else
                       sorted(secrets.SystemRandom().sample(range(max(1, int(.15 * n)), max(2, int(.9 * n))), 5)))
            for minute in indices:
                token = f"{kind}:{name}:{minute}".encode()
                sample = hmac.new(identity_key, token, hashlib.sha256).hexdigest()[:24]
                vector = np.r_[records[name][minute - 1], minute, np.eye(3)[scenario - 1]]
                row = {"features": vector, "sample_id": sample, "unit_id": unit,
                       "scenario_id": scenario, "elapsed_minutes": minute, "actual_rul": n - minute}
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
    write_json(private / "source_audit.json", {"split": split, "unit_mapping": mappings,
                                               "counts": COUNTS, "total_records": sum(COUNTS.values())})
    write_json(data / "training_config.json", {"seed": 278, "epochs": 30, "batch_size": 128,
        "learning_rate": .001, "cpu_threads": 2, "fit_unit_ids": list(range(1, 10))})
    write_json(data / "dataset.json", {"dataset": "XJTU", "task": "rul", "feature_count": 34,
        "target_unit": "nominal minutes", "source": pin["origin"], "source_revision": pin["mirror_revision"],
        "label": "Terminal acquisition index minus current index; nominal one-minute interval, not an exact physical failure timestamp.",
        "preprocessing": "30 current dual-axis signal statistics, observed elapsed minute and three condition indicators; no future observations.",
        "split": "9 train, 3 validation and 3 test bearings; each split has all three operating conditions.",
        "limitations": ["Accelerated laboratory experiments; 3 test bearings and 15 correlated prefix queries.",
                        "Terminal damage is not a per-minute fault state.",
                        "Public data can be memorized; pseudonymous IDs cannot prevent it."]})
    write_json(public / "dataset_registry.json", {"XJTU": {"task": "rul", "format": "xjtu_npz",
        "path": "datasets/XJTU", "architectures": ["mlp"], "available_splits": ["train", "val"],
        "test_inputs_visible": False}})
    _catalog(public)
    write_json(dest / "SOURCE_BINDINGS.json", pin)
    report = {"task_id": "research_xjtu_bearing_rul", "dataset": "XJTU", "task_kind": "rul",
              "status": "prepared", "training_bearings": 9, "development_bearings": 3,
              "test_bearings": 3, "test_queries": len(gold), "native_records": sum(COUNTS.values()),
              "labels_in_test_input": False, "measured_or_simulated": "laboratory measurements",
              "scope": "measured XJTU-SY RUL numerical research adaptation; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    shutil.rmtree(scratch)
    return report
