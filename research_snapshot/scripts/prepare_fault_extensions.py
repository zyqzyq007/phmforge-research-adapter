"""Source-bound record splits for additional real PHMForge classifier backends.

No window from a source group crosses splits. Label names and identity fields
are preserved; anonymous pre-segmented tables are not guessed into recordings.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from acquire_forge_pdmb_sources244 import sha256, save


SEED = 20261020246
WINDOW = 2048
CHANNELS = {
    "ElectricMotorVibrations": ["AccX", "AccY", "AccZ"],
    "RotorBrokenBar": ["Vib_acpe", "Vib_acpi", "Vib_axial", "Vib_base", "Vib_carc"],
    "HUST": ["vibration_raw"],
    "Padeborn": ["vibration_raw"],
    "XJTU": ["horizontal_raw", "vertical_raw"],
    "PlanetaryPdM": ["Acc_Carrier"],
}


def numeric_array(text):
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Expected a complete array")
    value = np.fromstring(text[1:-1], sep=",", dtype=np.float64)
    if len(value) != text.count(",") + 1:
        raise ValueError("Array parsed only partially")
    return value


def features(x):
    centered = x - x.mean(axis=0)
    variance = np.square(centered).mean(axis=0)
    scale = np.sqrt(variance)
    moments = [x.mean(0), scale, np.sqrt(np.square(x).mean(0)),
               np.max(np.abs(x), axis=0), np.ptp(x, axis=0),
               np.mean(centered**3, axis=0) / np.maximum(scale, 1e-12)**3,
               np.mean(centered**4, axis=0) / np.maximum(variance, 1e-12)**2]
    spectrum = np.abs(np.fft.rfft(centered, axis=0))**2 / len(x)**2
    bands = [np.log1p(band.sum(axis=0)) for band in np.array_split(spectrum[1:], 8)]
    return np.concatenate(moments + bands).astype(np.float32)


def prepare(run, name):
    candidates = sorted(run.glob(f"inputs/pdmb_attempt*/{name}.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Exactly one complete source required for {name}")
    source = candidates[0]
    receipt = json.loads(source.with_suffix(".receipt.json").read_text())
    source_sha = sha256(source)
    if not receipt["pinned_hash_verified"] or source_sha != receipt["actual_sha256"]:
        raise ValueError("Source content is not bound to the public pin")
    if name not in CHANNELS:
        raise ValueError("No defensible recording map/observation contract implemented")
    csv.field_size_limit(128*1024**2)
    observations, labels, groups, samples, provenance = [], [], [], [], []
    dropped = []
    full_rows = 0
    with source.open(newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            full_rows += 1
            label = int(row["label"]) if row["label"].lstrip("-").isdigit() else row["label"]
            if name in ["ElectricMotorVibrations", "RotorBrokenBar", "PlanetaryPdM"]:
                group = f"published_record_{row_index:05d}"
            elif name == "HUST":
                group = row["filename"]
            elif name == "Padeborn":
                group = row["bearing_num"]
            else:
                group = row["bearing"]
            arrays = [numeric_array(row[column]) for column in CHANNELS[name]]
            if len({len(array) for array in arrays}) != 1:
                raise ValueError("Sensor channels have unequal lengths")
            signal = np.stack(arrays, axis=1)
            # Every complete non-overlapping 2048-point window is used. Short
            # already-segmented rows remain one complete observation.
            width = min(WINDOW, len(signal))
            if width < 32:
                raise ValueError("Insufficient observed signal")
            for start in range(0, len(signal)-width+1, width):
                x = signal[start:start+width]
                if not np.isfinite(x).all():
                    dropped.append(dict(row=row_index, start=start, reason="nonfinite measured samples"))
                    continue
                vector = features(x)
                if not np.isfinite(vector).all():
                    raise ValueError("Nonfinite derived features")
                identity = hashlib.sha256(f"PHM246:{name}:{row_index}:{start}".encode()).hexdigest()[:24]
                observations.append(vector)
                labels.append(label)
                groups.append(group)
                samples.append(identity)
                provenance.append(dict(sample_id=identity, original_row=row_index, start=start,
                                       length=width, group=group, label=label))
    y, g = np.asarray(labels), np.asarray(groups)
    by_class = defaultdict(set)
    group_labels = defaultdict(set)
    for label, group in zip(labels, groups):
        by_class[label].add(group)
        group_labels[group].add(label)
    if any(len(v) != 1 for v in group_labels.values()):
        raise ValueError("One recording group has multiple labels; requires a source-specific temporal contract")
    split = dict(train=[], val=[], test=[])
    for label in sorted(by_class, key=str):
        available = sorted(by_class[label])
        if len(available) < 3:
            raise ValueError(f"Fewer than three distinct source groups for label {label}: {len(available)}")
        random.Random(str(SEED)+":"+name+":"+str(label)).shuffle(available)
        count = max(1, len(available)//5)
        split["val"] += available[:count]
        split["test"] += available[count:2*count]
        split["train"] += available[2*count:]
    assert all(not set(split[a]) & set(split[b]) for a,b in [("train","val"),("train","test"),("val","test")])
    directory = run/"prepared/datasets"/name.upper()
    directory.mkdir(parents=True, exist_ok=False)
    x = np.asarray(observations, np.float32)
    ids = np.asarray(samples)
    for part, selected in split.items():
        take = np.isin(g, selected)
        payload = dict(features=x[take], sample_ids=ids[take], dataset=np.asarray(name.upper()))
        if part != "test":
            payload.update(labels=y[take], groups=g[take])
        np.savez_compressed(directory/(part+".npz"), **payload)
    test_indices = np.flatnonzero(np.isin(g, split["test"]))
    save(run/"external_labels"/name.upper()/"labels.json", [provenance[i] for i in test_indices])
    save(run/"audit"/(name+"_split.json"), dict(split=split, observations=provenance, excluded=dropped))
    group_definition = (
        "Entire published raw-signal row; records may share physical hardware; no cross-device claim"
        if name in ["ElectricMotorVibrations","RotorBrokenBar","PlanetaryPdM"] else
        "Published source filename; not independently certified physical bearings"
        if name == "HUST" else "Published bearing identity, kept intact across all operating conditions")
    info = dict(dataset=name.upper(), seed=SEED, class_names=sorted(by_class, key=str),
                architectures=["mlp"], group_definition=group_definition,
                source_sha256=source_sha, source_pin=receipt["pin"],
                source_rows=full_rows, channels=CHANNELS[name],
                preprocessing="Nonoverlapping up-to-2048-point windows; 7 time-domain and 8 FFT-band statistics per channel",
                feature_names=["mean","std","rms","peak_abs","peak_to_peak","skewness","kurtosis"]+
                              [f"log1p_fft_band_{i}" for i in range(8)],
                feature_order="statistic-major then channel",
                units="Original published signal scale; physical sampling rate not inferred",
                dropped_nonfinite_windows=len(dropped), label_definition="Original published labels; no invented fault-name/severity mapping",
                train_validation_groups={k:v for k,v in split.items() if k != "test"},
                rows_by_split={k:int(np.isin(g,v).sum()) for k,v in split.items()},
                groups_by_split={k:len(v) for k,v in split.items()},
                groups_per_class={str(k):len(v) for k,v in by_class.items()},
                labels_in_test_input=False)
    save(directory/"dataset.json", info)
    # Input-dependence probe: a separate artificial perturbation, never scored as
    # measured data and never used for fitting or model selection.
    changed = dict(np.load(directory/"test.npz", allow_pickle=False))
    fit = np.load(directory/"train.npz", allow_pickle=False)["features"]
    changed["features"] = changed["features"] + np.maximum(fit.std(0),1e-6)*0.25
    np.savez_compressed(directory/"changed_input.npz", **changed)
    save(run/"audit"/(name+"_prepared.json"), info)
    return {key:info[key] for key in ["dataset","rows_by_split","groups_by_split","dropped_nonfinite_windows"]}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    a=p.parse_args()
    for name in a.datasets:
        try:
            result=prepare(a.run.resolve(),name)
        except Exception as error:
            result=dict(dataset=name,status="not_prepared",error_type=type(error).__name__,error=str(error))
            save(a.run/"audit"/(name+"_prepare_failure.json"),result)
        print(json.dumps(result),flush=True)
