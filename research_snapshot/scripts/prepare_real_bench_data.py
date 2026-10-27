"""Freeze source-bound data splits for the explicitly adapted benchmarks."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np

from real_rul_model import write_json


CACHE = Path("/srv/yyf-zyq/PHM_20261010/_shared/forge_A187_f6edf732_runtime_recovery")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run):
    public, hidden = run / "public", run / "hidden"
    registry, bindings = {}, []
    known = {x["cache_relative_path"]: x for x in
             json.loads((CACHE / "CACHE_VERIFIED.json").read_text())["files"]}
    for index in range(1, 5):
        dataset = f"CMAPSS_FD00{index}"
        target = public / "datasets" / dataset
        target.mkdir(parents=True, exist_ok=False)
        for prefix, destination in [("train", target / "train.txt"),
                                    ("test", target / "test.txt"),
                                    ("RUL", hidden / dataset / "rul.txt")]:
            name = f"forge_data/{prefix}_FD00{index}.txt"
            source = CACHE / name
            assert sha(source) == known[name]["sha256"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            bindings.append(dict(source=str(source), sha256=sha(source),
                                 destination=str(destination)))
        raw = np.loadtxt(target / "train.txt")
        ids = sorted(set(raw[:, 0].astype(int).tolist()))
        random.Random(20261020238).shuffle(ids)
        split = int(len(ids) * .8)
        write_json(target / "training_config.json", dict(
            seed=20261020238, train_engine_ids=sorted(ids[:split]),
            validation_engine_ids=sorted(ids[split:]), window=30, training_stride=5,
            validation_prefix_fractions=[.3, .45, .6, .75, .9], training_rul_cap=125.,
            epochs=40, batch_size=256, learning_rate=.001, cpu_threads=2))
        registry[dataset] = dict(task="rul", path=f"datasets/{dataset}",
                                 data_origin="NASA physical simulator; not field measurements",
                                 architectures=["mlp", "lstm", "transformer"],
                                 split="official test; training engines split 80/20")
    csv.field_size_limit(10_000_000)
    for name in ["CWRU", "MFPT"]:
        with (run / "sources" / f"{name}.csv").open() as f:
            rows = list(csv.DictReader(f))
        labels = np.array([int(x["label"]) for x in rows])
        if name == "MFPT":
            groups = np.array([x["filename"] for x in rows])
            group_definition = "Published source filename, not certified independent physical bearings"
            feature_key = "features"
        else:
            # Published feature table omits original recording identity.
            # Preserve contiguous segment runs; do not claim bearing independence.
            group, previous, groups = -1, None, []
            for row in rows:
                current = int(row["segment_idx"])
                if previous is None or current <= previous:
                    group += 1
                groups.append(f"contiguous_block_{group:03d}")
                previous = current
            groups = np.array(groups)
            group_definition = "Inferred contiguous segment blocks; original recording/bearing IDs absent"
            feature_key = "all_features"
        by_split = dict(train=[], val=[], test=[])
        for label in sorted(set(labels)):
            group_ids = sorted(set(groups[labels == label]))
            random.Random(240 + int(label)).shuffle(group_ids)
            assert len(group_ids) >= 3
            nval = max(1, len(group_ids) // 5)
            by_split["val"] += group_ids[:nval]
            by_split["test"] += group_ids[nval:2*nval]
            by_split["train"] += group_ids[2*nval:]
        assert not (set(by_split["train"]) & set(by_split["test"]))
        arrays = dict(features=np.asarray([json.loads(x[feature_key]) for x in rows], np.float32))
        if name == "MFPT":
            # Non-overlapping block averaging, fixed before training; retains actual waveform.
            waveforms = np.asarray([json.loads(x["vibration_raw"]) for x in rows], np.float32)
            assert waveforms.shape[1] == 2560
            arrays["signals"] = waveforms.reshape(len(rows), 256, 10).mean(-1)
        directory = public / "datasets" / name
        directory.mkdir(parents=True, exist_ok=False)
        for split_name, ids in by_split.items():
            take = np.isin(groups, ids)
            part = {k: v[take] for k, v in arrays.items()}
            indices = np.where(take)[0]
            identities = ([hashlib.sha256(f"PHM240:{name}:{i}".encode()).hexdigest()[:20] for i in indices]
                          if split_name == "test" else [str(i) for i in indices])
            part.update(sample_ids=np.asarray(identities))
            if split_name != "test":
                part["labels"] = labels[take]
                part["groups"] = groups[take]
            np.savez_compressed(directory / f"{split_name}.npz", **part)
            if split_name == "test":
                write_json(hidden / name / "labels.json", [
                    dict(sample_id=identity, label=int(labels[i]), group=str(groups[i]), original_row=int(i))
                    for i, identity in zip(indices, identities)])
        info = dict(dataset=name, seed=240, class_names=sorted(set(labels.tolist())),
                    architectures=["mlp", "lstm", "transformer"] if name == "MFPT" else ["mlp"],
                    group_definition=group_definition,
                    train_validation_groups={k: v for k,v in by_split.items() if k != "test"},
                    source_sha256=sha(run / "sources" / f"{name}.csv"),
                    waveform_processing="2560 -> 256 nonoverlapping means of 10" if name == "MFPT" else None,
                    label_definition="Original published integer labels; no invented physical mapping",
                    rows_by_split={k: int(np.isin(groups, ids).sum()) for k, ids in by_split.items()})
        write_json(directory / "dataset.json", info)
        write_json(hidden / name / "recording_split.json", by_split)
        registry[name] = dict(task="classification", path=f"datasets/{name}",
                               data_origin="Published benchmark-derived vibration data",
                               **{k: info[k] for k in ["architectures", "group_definition", "rows_by_split"]})
    write_json(public / "dataset_registry.json", registry)
    write_json(run / "audit/DATA_BINDINGS240.json", bindings)
    for name,item in registry.items():
        d=public/item["path"]
        if item["task"]=="rul":
            raw=np.loadtxt(d/"test.txt")
            raw[:,5:]+=np.std(raw[:,5:],axis=0)*.25
            np.savetxt(d/"changed_input.txt",raw,fmt="%.7f")
        else:
            data=dict(np.load(d/"test.npz",allow_pickle=False))
            for key in ["features","signals"]:
                if key in data:
                    data[key]=data[key]+data[key].std(axis=0)*.25
            np.savez_compressed(d/"changed_input.npz",**data)
    prepare_asset(run)


def prepare_asset(run):
    public, hidden = run / "public", run / "hidden"
    source = run / "restored/current/source/asset/src/couchdb/scenarios_data/shared/iot/chiller_6.json"
    original = json.loads(source.read_text())
    # Native source appends irregular-time observations. Gold must remain measured
    # exact-grid values; public inputs may use explicitly recorded causal alignment.
    rows, off_grid = [], []
    for row in original:
        t = datetime.fromisoformat(row["timestamp"])
        (rows if t.minute % 15 == 0 and t.second == 0 and t.microsecond == 0 else off_grid).append(row)
    rows.sort(key=lambda x: x["timestamp"])
    exact_rows = rows
    column = "Chiller 6 Power Input"
    assert all(rows[i]["timestamp"] < rows[i+1]["timestamp"] for i in range(len(rows)-1))
    positive_runs, current = [], []
    for row in rows:
        value = row.get(column)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0):
            if (current and datetime.fromisoformat(row["timestamp"])
                    - datetime.fromisoformat(current[-1]["timestamp"]) != timedelta(minutes=15)):
                positive_runs.append(current)
                current = []
            current.append(row)
        elif current:
            positive_runs.append(current)
            current = []
    if current:
        positive_runs.append(current)
    eligible_runs = [run_rows for run_rows in positive_runs if len(run_rows) >= 28]
    if not eligible_runs:
        raise ValueError("No contiguous 28-point positive target window exists")
    selected_run = eligible_runs[-1]
    future = selected_run[-28:]
    for a, b in zip(future, future[1:]):
        assert (datetime.fromisoformat(b["timestamp"])-datetime.fromisoformat(a["timestamp"])).total_seconds() == 900
    origin = datetime.fromisoformat(future[0]["timestamp"]) - timedelta(minutes=15)
    missing_target = [x for x in original if not isinstance(x.get(column), (int, float)) or not math.isfinite(x[column])]
    past = sorted([x for x in original if datetime.fromisoformat(x["timestamp"]) <= origin
                   and isinstance(x.get(column), (int, float)) and math.isfinite(x[column])],
                  key=lambda x: x["timestamp"])
    aligned, alignment, index = [], [], 0
    tick = datetime.fromisoformat(rows[0]["timestamp"])
    while tick <= origin:
        while index + 1 < len(past) and datetime.fromisoformat(past[index+1]["timestamp"]) <= tick:
            index += 1
        observed = past[index]
        assert datetime.fromisoformat(observed["timestamp"]) <= tick
        aligned.append(dict(timestamp=tick.isoformat(), **{column: observed[column]}))
        if observed["timestamp"] != tick.isoformat():
            alignment.append(dict(grid_timestamp=tick.isoformat(), source_timestamp=observed["timestamp"],
                                  operation="last observation at or before timestamp; no future interpolation"))
        tick += timedelta(minutes=15)
    rows = aligned + future
    split = len(aligned)
    asset = public / "asset"
    asset.mkdir(exist_ok=True)
    assert not (asset / "train.csv").exists(), "Do not overwrite prepared data"
    for a, b in zip(rows[split-96:], rows[split-95:]):
        assert (datetime.fromisoformat(b["timestamp"])-datetime.fromisoformat(a["timestamp"])).total_seconds() == 900
    for filename, selected in [
        ("train.csv", rows[:split]),
        ("context.csv", rows[split-96:split]),
        ("serving_context.csv", rows[split-124:split]),
    ]:
        with (asset / filename).open("w") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", column])
            w.writeheader()
            for row in selected:
                w.writerow({k: row[k] for k in w.fieldnames})
    write_json(hidden / "asset/chiller6_future.json", [
        dict(timestamp=x["timestamp"], value=x[column]) for x in rows[split:]])
    target_values = np.asarray([x[column] for x in rows[split:]], float)
    later = [x for x in exact_rows if x["timestamp"] > future[-1]["timestamp"]]
    later_values = [x[column] for x in later]
    write_json(run / "audit/ASSET_HOLDOUT_SELECTION240.json", dict(
        selection_rule=(
            "latest contiguous exact-15-minute run of finite strictly positive target "
            "values with at least 28 points; use its last 28 points"
        ),
        selection_used_model_scores=False,
        source_first_timestamp=exact_rows[0]["timestamp"],
        source_last_exact_timestamp=exact_rows[-1]["timestamp"],
        eligible_positive_runs=len(eligible_runs),
        selected_run_start=selected_run[0]["timestamp"],
        selected_run_rows=len(selected_run),
        forecast_origin=origin.isoformat(),
        held_out_first=future[0]["timestamp"],
        held_out_last=future[-1]["timestamp"],
        held_out_rows=len(future),
        held_out_nonzero=int((target_values != 0).sum()),
        held_out_distinct=len(set(target_values.tolist())),
        held_out_min=float(target_values.min()),
        held_out_max=float(target_values.max()),
        held_out_mean=float(target_values.mean()),
        held_out_std=float(target_values.std()),
        later_exact_grid_rows_excluded=len(later),
        later_exact_grid_values_all_zero=bool(later and all(v == 0 for v in later_values)),
        rationale=(
            "The source tail contains a long all-zero target segment. A final-tail split "
            "would evaluate trivial zero prediction and cannot qualify forecasting."
        )))
    write_json(asset / "dataset.json", dict(
        source=str(source), source_sha256=sha(source), target=column,
        training_rows=split, held_out_rows=28, context_length=96,
        current_main_serving_context_rows=124,
        forecast_origin=rows[split-1]["timestamp"], frequency="15min",
        data_origin="Published AssetOps telemetry fixture; field-measurement provenance not independently verified",
        original_rows=len(original), off_grid_rows=len(off_grid),
        rows_missing_target=len(missing_target),
        causally_aligned_input_rows=len(alignment),
        preparation="sort unique timestamps; causal last-observation alignment of public input to 15min; final 28 labels are untouched exact-grid observations",
        holdout_selection="latest contiguous positive 15min run with at least28 points; no model-score selection",
        qualification_scope="One Chiller 6 forecast on a nondegenerate published-fixture window"))
    write_json(run / "audit/ASSET_OFF_GRID_ROWS240.json", off_grid)
    write_json(run / "audit/ASSET_CAUSAL_ALIGNMENT240.json", alignment)
    write_json(run / "audit/ASSET_MISSING_TARGET240.json", missing_target)
    with (asset / "context.csv").open() as f:
        reader = csv.DictReader(f)
        fields, context = reader.fieldnames, list(reader)
    with (asset / "changed_context.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in context:
            row[column] = float(row[column]) + 1.
            w.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    prepare(p.parse_args().run.resolve())
