#!/usr/bin/env python3
"""Bounded 2x2 CWRU transform falsification; fit, cold predict, external score."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def transform(features, variant):
    bands = features.astype(np.float64)[:, 7:15]
    if variant[0] == "1":
        bands = np.expm1(bands)
    epsilon = 1e-7 if variant[1] == "0" else 1e-7 * bands.mean(axis=1)
    assert np.isfinite(bands).all() and (bands > 0).all()
    return np.column_stack([
        np.log((bands[:, i] + epsilon) / (bands[:, j] + epsilon))
        for i in range(8) for j in range(i + 1, 8)
    ])


def fit(run, variant):
    from sklearn.linear_model import LogisticRegression
    import sklearn
    config = json.loads((run / "FIT_CONFIG.json").read_text())
    started = time.perf_counter()
    write(run / f"models/{variant}.started.json", {"utc": now(), "variant": variant})
    data = arrays(config["train_path"])
    features = transform(data["features"], variant)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (features - mean) / scale
    estimator = LogisticRegression(**config["estimator"])
    estimator.fit(normalized, data["labels"])
    np.savez_compressed(run / f"models/{variant}.npz", mean=mean, scale=scale,
                        coefficients=estimator.coef_, intercepts=estimator.intercept_,
                        classes=estimator.classes_)
    write(run / f"models/{variant}.fit.json", {
        "variant": variant, "utc": now(), "seconds": time.perf_counter() - started,
        "train_sha256": digest(config["train_path"]), "rows": len(features),
        "source_groups": sorted(set(data["groups"].tolist())),
        "estimator": config["estimator"], "sklearn": sklearn.__version__,
        "iterations": estimator.n_iter_.tolist(),
        "coefficient_norm": float(np.linalg.norm(estimator.coef_)),
        "fitted_model_sha256": digest(run / f"models/{variant}.npz"),
        "fit_calls": 1, "development_or_test_read": False,
    })


def predict(run, variant, split):
    started = time.perf_counter()
    model_path = run / f"models/{variant}.npz"
    input_path = run / f"inputs/{split}.npz"
    model, data = arrays(model_path), arrays(input_path)
    assert set(data) == {"features", "unit_ids", "sample_ids"}
    representation = "00" if variant == "archived" else variant
    x = (transform(data["features"], representation) - model["mean"]) / model["scale"]
    logits = x @ model["coefficients"].T + model["intercepts"]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(run / f"predictions/{variant}_{split}.npz",
                        probabilities=probabilities, classes=model["classes"],
                        predicted=model["classes"][probabilities.argmax(axis=1)],
                        unit_ids=data["unit_ids"], sample_ids=data["sample_ids"])
    write(run / f"predictions/{variant}_{split}.receipt.json", {
        "utc": now(), "seconds": time.perf_counter() - started,
        "model_sha256": digest(model_path), "observation_sha256": digest(input_path),
        "rows": len(x), "labels_in_input": False,
    })


def scores(prediction, truth):
    labels = np.asarray([truth[str(unit)] for unit in prediction["unit_ids"]])
    unit_rows = []
    for unit in sorted(set(prediction["unit_ids"].tolist())):
        mask = prediction["unit_ids"] == unit
        classes = prediction["classes"].tolist()
        votes = [(prediction["predicted"][mask] == label).sum() for label in classes]
        tied = [i for i in range(len(classes)) if votes[i] == max(votes)]
        prob = prediction["probabilities"][mask].mean(axis=0)
        choice = min(tied, key=lambda i: (-float(prob[i]), classes[i]))
        unit_rows.append({"unit_id": unit, "actual": truth[unit],
                          "predicted": classes[choice], "windows": int(mask.sum())})
    correct = int((prediction["predicted"] == labels).sum())
    unit_correct = sum(row["actual"] == row["predicted"] for row in unit_rows)
    return {"window_correct": correct, "window_total": len(labels),
            "window_accuracy": correct / len(labels), "record_correct": unit_correct,
            "record_total": len(unit_rows), "record_accuracy": unit_correct / len(unit_rows),
            "records": unit_rows}


def coordinate(run):
    protocol = json.loads((run / "PROTOCOL412.json").read_text())
    assert digest(__file__) == protocol["script_sha256"]
    started = time.perf_counter()
    source = Path(protocol["source"])
    for item in protocol["source_bindings"]:
        assert digest(source / item["path"]) == item["sha256"]
    info = json.loads((source / "c/restored/CWRU/data/dataset.json").read_text())
    config = {"train_path": str(source / "c/restored/CWRU/data/train.npz"),
              "estimator": dict(C=1.0, class_weight="balanced", max_iter=5000,
                                solver="lbfgs", random_state=int(info["seed"]))}
    write(run / "FIT_CONFIG.json", config)
    shutil.copy2(source / "c/workspaces/expand307_cost002_initial/artifacts/fault_model.npz",
                 run / "models/archived.npz")
    records = json.loads((source / "audit/CWRU_NATIVE_RECORDS.json").read_text())
    truth = {row["unit_id"]: row["label"] for row in records}
    for split in protocol["splits"]:
        data = arrays(source / f"c/restored/CWRU/data/{split}.npz")
        if "labels" in data:
            assert data["labels"].tolist() == [truth[str(u)] for u in data["unit_ids"]]
        np.savez_compressed(run / f"inputs/{split}.npz",
                            **{key: data[key] for key in ("features", "unit_ids", "sample_ids")})

    def child(arguments):
        name = "_".join(arguments)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--run", str(run), *arguments]
        with (run / f"logs/{name}.log").open("w") as stream:
            process = subprocess.run(command, cwd=run, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=protocol["child_timeout_seconds"])
        assert process.returncode == 0, f"Child failed: {arguments}"

    variants = protocol["variants"]
    with ThreadPoolExecutor(max_workers=protocol["parallel"]) as executor:
        list(executor.map(child, [["fit", v] for v in variants]))
    old, replay = arrays(run / "models/archived.npz"), arrays(run / "models/00.npz")
    differences = {}
    for key in ("mean", "scale", "coefficients", "intercepts"):
        differences[key] = float(np.max(np.abs(old[key] - replay[key])))
        assert np.allclose(old[key], replay[key], rtol=1e-6, atol=1e-8), key
    assert np.array_equal(old["classes"], replay["classes"])
    write(run / "ORIGINAL_REPRODUCTION.json", {"passed": True, "max_abs_differences": differences})
    model_names = ["archived", *variants]
    sealed = {v: digest(run / f"models/{v}.npz") for v in model_names}
    write(run / "MODELS_SEALED.json", {"utc": now(), "models": sealed,
                                       "protocol_sha256": digest(run / "PROTOCOL412.json")})
    with ThreadPoolExecutor(max_workers=protocol["parallel"]) as executor:
        list(executor.map(child, [["predict", v, p] for v in model_names for p in protocol["splits"]]))
    assert all(digest(run / f"models/{v}.npz") == sha for v, sha in sealed.items())
    result = {v: {p: scores(arrays(run / f"predictions/{v}_{p}.npz"), truth)
                  for p in protocol["splits"]} for v in model_names}
    for split in protocol["splits"]:
        a = arrays(run / f"predictions/archived_{split}.npz")
        b = arrays(run / f"predictions/00_{split}.npz")
        assert np.array_equal(a["predicted"], b["predicted"])
        assert np.allclose(a["probabilities"], b["probabilities"], rtol=1e-6, atol=1e-8)
    assert result["00"]["val"]["window_correct"] == 177
    assert result["00"]["test"]["window_correct"] == 0
    before, primary = result["00"], result[protocol["primary_variant"]]
    gate = all(primary["test"][k] > before["test"][k]
               for k in ("window_accuracy", "record_accuracy")) and all(
                   primary["val"][k] >= before["val"][k]
                   for k in ("window_accuracy", "record_accuracy"))
    output = {"utc": now(), "seconds": time.perf_counter() - started, "scores": result,
              "primary_mechanism_gate": gate,
              "primary_reaches_historical_MLP": primary["test"]["window_correct"] >= 118
                  and primary["test"]["record_correct"] >= 2,
              "fit_calls": 4, "fitted_normalizers": 4, "cold_prediction_processes": 15,
              "LLM_calls": 0, "new_independent_scenarios": 0, "confirmation_opened": False,
              "scope": "Already-exposed CWRU discovery records; no new Agent comparison or causal-selection claim"}
    write(run / "RESULT412.json", output)
    print(json.dumps({"seconds": output["seconds"], "primary_gate": gate,
                      "correct": {v: {p: r["window_correct"] for p, r in row.items()}
                                  for v, row in result.items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("mode", choices=("coordinate", "fit", "predict"))
    parser.add_argument("variant", nargs="?")
    parser.add_argument("split", nargs="?")
    args = parser.parse_args()
    if args.mode == "fit":
        fit(args.run, args.variant)
    elif args.mode == "predict":
        predict(args.run, args.variant, args.split)
    else:
        try:
            coordinate(args.run)
        except BaseException:
            write(args.run / "FAILURE412.json", {"utc": now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
