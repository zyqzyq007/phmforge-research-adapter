"""Prepare source-bound PDMBench fault-classification research tasks.

These are numerical adaptations, not complete PHMForge natural-language,
maintenance, or safety tasks. Test observations and labels remain outside the
public training/development directory.
"""
from collections import defaultdict
import csv
import hashlib
import hmac
from pathlib import Path
import random
import secrets

import numpy as np

from .data import _catalog, download, features, resource_json
from .io import file_sha256, write_json


SEED = 20261020246
WINDOW = 2048


def _numeric_array(text):
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Expected a complete source array")
    value = np.fromstring(text[1:-1], sep=",", dtype=np.float64)
    if not len(value) or len(value) != text.count(",") + 1:
        raise ValueError("Source array is empty or was only partially parsed")
    return value


def _source_label(value):
    value = value.strip()
    if not value:
        raise ValueError("A source label is empty")
    return int(value) if value.lstrip("-").isdigit() else value


def prepare_pdmb_fault(destination, dataset, source=None):
    pins = resource_json("pdmb_fault_sources.json")
    if dataset not in pins:
        raise ValueError(f"Unsupported PDMBench classification source: {dataset}")
    pin = pins[dataset]
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    raw_path = Path(source).resolve() if source else dest / "raw/PdM.csv"
    if not source:
        download(pin["url"], raw_path, pin["sha256"], pin["bytes"])
    if not raw_path.is_file() or raw_path.stat().st_size != pin["bytes"] or file_sha256(raw_path) != pin["sha256"]:
        raise ValueError(f"Published source missing or does not match the pinned size and SHA-256: {raw_path}")

    # CSV fields hold full waveforms. Stream rows rather than load a multi-GB
    # source into memory, and bound individual fields at one GiB.
    csv.field_size_limit(1024 * 1024 * 1024)
    observations, labels, groups, source_ids = [], [], [], []
    dropped, row_count = [], 0
    block, previous_segment = -1, None
    with raw_path.open(encoding="utf-8-sig", newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            row_count += 1
            label = _source_label(row["label"])
            if pin["format"] == "cwru_features":
                segment = int(row["segment_idx"])
                if previous_segment is None or segment <= previous_segment:
                    block += 1
                previous_segment = segment
                group = f"contiguous_block_{block:04d}"
                vector = _numeric_array(row["all_features"]).astype(np.float32)
                if not np.isfinite(vector).all():
                    raise ValueError(f"Nonfinite original CWRU features at row {row_index}")
                windows = [(0, vector)]
            else:
                group_field = pin.get("group_field")
                group = (row[group_field].strip() if group_field else
                         f"{'published_sample' if pin['dataset'] == 'UOC' else 'published_record'}_{row_index:05d}")
                if not group:
                    raise ValueError(f"Missing source recording identity at row {row_index}")
                arrays = [_numeric_array(row[column]) for column in pin["channels"]]
                if len({len(array) for array in arrays}) != 1:
                    raise ValueError(f"Unequal channel lengths at row {row_index}")
                signal = np.stack(arrays, axis=1)
                width = min(pin.get('window_length', WINDOW), len(signal))
                if width < 32:
                    raise ValueError(f"Insufficient source observations at row {row_index}")
                windows = []
                starts = range(0, len(signal) - width + 1, width)
                limit = pin.get("max_windows_per_record")
                if limit and len(starts) > limit:
                    # Fixed, label-independent coverage of long recordings;
                    # never choose windows using held-out scores or predictions.
                    starts = [int(i) * width for i in np.linspace(0, len(starts) - 1, limit, dtype=int)]
                for start in starts:
                    samples = signal[start:start + width]
                    if not np.isfinite(samples).all():
                        dropped.append({"source_row": row_index, "start": start, "reason": "nonfinite source samples"})
                        continue
                    vector = features(samples)
                    if not np.isfinite(vector).all():
                        raise ValueError(f"Nonfinite derived features at row {row_index}, start {start}")
                    windows.append((start, vector))
            for start, vector in windows:
                observations.append(vector)
                labels.append(label)
                groups.append(group)
                source_ids.append(f"{pin['dataset']}:{row_index}:{start}")
    if not observations:
        raise ValueError("No finite source observations were prepared")
    if len({type(label) for label in labels}) != 1:
        raise ValueError("Mixed source label types require a dataset-specific contract")

    by_group = defaultdict(set)
    by_class = defaultdict(set)
    for label, group in zip(labels, groups):
        by_group[group].add(label)
        by_class[label].add(group)
    splits = {"train": [], "val": [], "test": []}
    for label, available in by_class.items():
        if len(available) < 3:
            raise ValueError(f"Fewer than three separate source groups for class {label!r}")
    if any(len(values) != 1 for values in by_group.values()):
        # Some published identifiers are reused across fault categories. Keep
        # the entire identifier together, even when that conservatively merges
        # different devices; never assign one identifier to multiple splits.
        ordered = sorted(by_group)
        count = max(1, len(ordered) // 5)
        rng = random.Random(f"{SEED}:{pin['dataset']}:mixed-groups")
        all_classes = set(by_class)
        for _ in range(4096):
            shuffled = ordered.copy()
            rng.shuffle(shuffled)
            candidate = {"val": shuffled[:count], "test": shuffled[count:2 * count],
                         "train": shuffled[2 * count:]}
            if all(set().union(*(by_group[g] for g in values)) == all_classes
                   for values in candidate.values()):
                splits = candidate
                break
        else:
            raise ValueError("Could not form source-group-disjoint splits containing every source class")
    else:
        for label in sorted(by_class, key=str):
            available = sorted(by_class[label])
            random.Random(f"{SEED}:{pin['dataset']}:{label}").shuffle(available)
            count = max(1, len(available) // 5)
            splits["val"].extend(available[:count])
            splits["test"].extend(available[count:2 * count])
            splits["train"].extend(available[2 * count:])
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Source groups overlap between splits")

    public = dest / "public"
    data = public / "datasets" / pin["dataset"]
    data.mkdir(parents=True)
    holdout, private = dest / "holdout", dest / "private"
    holdout.mkdir()
    private.mkdir()
    identity_key = secrets.token_bytes(32)
    (private / "holdout_identity_key.hex").write_text(identity_key.hex() + "\n")
    x = np.asarray(observations, dtype=np.float32)
    y, g = np.asarray(labels), np.asarray(groups)
    counts = {}
    for split, selected in splits.items():
        indices = np.flatnonzero(np.isin(g, selected)).tolist()
        if split == "test":
            secrets.SystemRandom().shuffle(indices)
        ids = [(hmac.new(identity_key, source_ids[i].encode(), hashlib.sha256).hexdigest()[:24]
                if split == "test" else hashlib.sha256(source_ids[i].encode()).hexdigest()[:24]) for i in indices]
        payload = {"features": x[indices], "sample_ids": np.asarray(ids), "dataset": np.asarray(pin["dataset"])}
        if split == "test":
            np.savez_compressed(holdout / "test.npz", **payload)
            write_json(private / "labels.json", [{"sample_id": identity, "label": labels[i], "group": groups[i]}
                                                  for i, identity in zip(indices, ids)])
        else:
            payload.update(labels=y[indices], groups=g[indices])
            np.savez_compressed(data / f"{split}.npz", **payload)
        counts[split] = len(indices)
    classes = sorted(by_class, key=str)
    source_info = {"repository": f"https://huggingface.co/datasets/submission096/{pin['source_name']}",
                   "revision": pin["revision"], "sha256": pin["sha256"], "bytes": pin["bytes"]}
    if "source_git_blob_sha1" in pin:
        source_info["source_git_blob_sha1"] = pin["source_git_blob_sha1"]
    info = {"dataset": pin["dataset"], "seed": SEED, "class_names": classes, "architectures": ["mlp"],
            "group_definition": pin["group_definition"], "source": source_info,
            "label_definition": "Original published labels; no inferred physical fault or severity names.",
            "preprocessing": ("Original published all_features vectors." if pin["format"] == "cwru_features" else
                              f"Nonoverlapping up-to-{pin.get('window_length', WINDOW)}-sample windows; seven time and eight FFT-band statistics per channel."
                              + (f" At most {pin['max_windows_per_record']} evenly spaced windows per recording."
                                 if pin.get("max_windows_per_record") else "")),
            "feature_count": x.shape[1], "train_rows": counts["train"], "validation_rows": counts["val"],
            "mixed_label_source_groups": any(len(values) != 1 for values in by_group.values()),
            "dropped_nonfinite_windows": len(dropped), "measured_or_simulated": "published laboratory benchmark data"}
    write_json(data / "dataset.json", info)
    write_json(public / "dataset_registry.json", {pin["dataset"]: {"task": "classification",
        "path": f"datasets/{pin['dataset']}", "architectures": ["mlp"],
        "group_definition": pin["group_definition"], "available_splits": ["train", "val"]}})
    _catalog(public)
    write_json(private / "SOURCE_SPLIT.json", {"groups": splits, "dropped_windows": dropped, "source_rows": row_count})
    write_json(dest / "SOURCE_BINDINGS.json", source_info)
    report = {"task_id": pin["task_id"], "dataset": pin["dataset"], "task_kind": "classification",
        "status": "prepared", "source": source_info, "rows": counts,
        "recordings": {name: len(value) for name, value in splits.items()}, "group_overlap": False,
        "labels_in_test_input": False, "holdout_ids": "random private HMAC; different for each preparation",
        "scope": "source-bound numerical research adaptation; not a complete original PHMForge task or official score"}
    if pin['dataset'] == 'UOC':
        report['published_sample_groups'] = report.pop('recordings')
        report['physical_recording_independence_verified'] = False
        report['scope'] += '; UoC split is exploratory at published-sample level only'
    write_json(dest / "READY.json", report)
    return report
