#!/usr/bin/env python3
"""Frozen-model ATC qualification, separating unlabeled choice from scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predict(run, split):
    import torch
    from real_fault_model import FaultNetwork
    torch.set_num_threads(1)
    start = time.perf_counter()
    model_path = run / "source/model.pt"
    cp = torch.load(model_path, map_location="cpu", weights_only=True)
    model = FaultNetwork(cp["architecture"], cp["features"], cp["classes"])
    model.load_state_dict(cp["state_dict"], strict=True)
    data = load(run / f"inputs/{split}.npz")
    assert set(data) == {"features", "sample_ids", "unit_ids"}
    values = (data["features"].astype(np.float32) - cp["mean"].numpy()) / cp["scale"].numpy()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(b) for b in torch.from_numpy(values).split(128)])
        probs = logits.softmax(-1).numpy()
    classes = np.asarray(cp["class_names"])
    np.savez_compressed(run / f"predictions/mlp_{split}.npz", probabilities=probs,
                        classes=classes, predicted=classes[probs.argmax(1)],
                        sample_ids=data["sample_ids"], unit_ids=data["unit_ids"])
    save(run / f"predictions/mlp_{split}.receipt.json", {
        "utc": now(), "seconds": time.perf_counter() - start,
        "model_sha256": sha(model_path), "input_sha256": sha(run / f"inputs/{split}.npz"),
        "rows": len(values), "labels_read": False, "fit_calls": 0,
    })


def exact_threshold(scores, correct):
    # Finite-sample minimization of Eq.1 in ATC; include zero/all rejection.
    candidates = np.r_[np.unique(scores), np.nextafter(np.max(scores), np.inf)]
    desired = int(np.sum(correct))
    threshold = min(candidates, key=lambda t: (abs(int(np.sum(scores >= t)) - desired), float(t)))
    return float(threshold), int(abs(int(np.sum(scores >= threshold)) - desired))


def probabilities(data):
    p = data["probabilities"].astype(float)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    return p / p.sum(1, keepdims=True)


def normalized_features(run, model, data):
    features = data["features"]
    if model == "mlp":
        import torch
        cp = torch.load(run / "source/model.pt", map_location="cpu", weights_only=True)
        return ((features.astype(np.float32) - cp["mean"].numpy()) / cp["scale"].numpy()).astype(float)
    cp = load(run / "source/ratio_model.npz")
    bands = features.astype(float)[:, 7:15]
    transform = np.column_stack([np.log((bands[:, i] + 1e-7) / (bands[:, j] + 1e-7))
                                 for i in range(8) for j in range(i + 1, 8)])
    return (transform - cp["mean"]) / cp["scale"]


def choose(run):
    from scipy.spatial.distance import cdist
    protocol = json.loads((run / "PROTOCOL413.json").read_text())
    helper = import_file(run / "source/ATC_ATC_helper.py", "atc_author")
    calibration = load(run / "source/val_labeled.npz")
    observations = {s: load(run / f"inputs/{s}.npz") for s in ("train", "val", "test")}
    table = {}
    for name in protocol["models"]:
        val, target = [load(run / f"predictions/{name}_{s}.npz") for s in ("val", "test")]
        assert np.array_equal(val["sample_ids"], calibration["sample_ids"])
        vp, tp = probabilities(val), probabilities(target)
        correct = val["classes"][vp.argmax(1)] == calibration["labels"]
        row = {"development_accuracy": float(np.mean(correct)),
               "average_confidence": float(tp.max(1).mean()), "calibrations": {}}
        for score_name, function in (("entropy", helper.get_entropy), ("maxconf", helper.get_max_conf)):
            vs, ts = function(vp), function(tp)
            threshold, mismatch = exact_threshold(vs, correct)
            _, official_threshold = helper.find_ATC_threshold(vs, correct)
            row["paper_" + score_name] = float(np.mean(ts >= threshold))
            row["official_" + score_name] = float(helper.get_ATC_acc(official_threshold, ts) / 100)
            row["calibrations"][score_name] = {
                "paper_threshold": threshold, "paper_mismatch_count": mismatch,
                "official_threshold": float(official_threshold),
                "official_actual_mismatch_count": int(abs(np.sum(vs >= official_threshold) - np.sum(correct))),
                "calibration_rows": len(vs), "calibration_correct": int(correct.sum()),
                "source_score_min": float(vs.min()), "source_score_max": float(vs.max()),
                "target_score_min": float(ts.min()), "target_score_max": float(ts.max()),
            }
        train = normalized_features(run, name, observations["train"])
        nearest = {}
        for split in ("val", "test"):
            x = normalized_features(run, name, observations[split])
            distances = cdist(x, train).min(1) / np.sqrt(train.shape[1])
            nearest[split] = float(distances.mean())
        assert nearest["val"] > 0
        ratio = nearest["test"] / nearest["val"]
        row["relative_support"] = 1 / (1 + ratio)
        row["support_diagnostic"] = {"mean_nearest_RMS_distance": nearest,
                                     "target_to_development_distance_ratio": ratio,
                                     "is_accuracy_estimate": False}
        table[name] = row
    rules = ["development_accuracy", "paper_entropy", "paper_maxconf", "official_entropy",
             "official_maxconf", "average_confidence", "relative_support"]
    selected = {rule: max(protocol["models"], key=lambda name: table[name][rule]) for rule in rules}
    # Predeclared model order is reference first; a tie is reported, not a strict win.
    strict = table["mlp"][protocol["primary_rule"]] > table["ratio"][protocol["primary_rule"]]
    save(run / "DECISIONS_SEALED413.json", {
        "utc": now(), "table": table, "selected": selected,
        "primary_strictly_prefers_reference": strict,
        "selection_scope": "Transductive exploratory control, not a retrospective Agent decision",
        "final_labels_read": False, "new_PHM_fit_calls": 0,
        "empirical_threshold_calibrations": 4, "author_helper_calls": 4,
        "protocol_sha256": sha(run / "PROTOCOL413.json"),
    })


def score(run):
    choice = json.loads((run / "DECISIONS_SEALED413.json").read_text())
    truth = {x["unit_id"]: x["label"] for x in json.loads(
        (run / "source/native_truth_external_only.json").read_text())}
    metrics = {}
    for name in ("mlp", "ratio"):
        data = load(run / f"predictions/{name}_test.npz")
        actual = np.asarray([truth[x] for x in data["unit_ids"]])
        units = []
        for unit in sorted(set(data["unit_ids"])):
            mask = data["unit_ids"] == unit
            labels, counts = np.unique(data["predicted"][mask], return_counts=True)
            predicted = str(labels[counts.argmax()])
            units.append({"unit_id": str(unit), "predicted": predicted, "actual": truth[unit]})
        metrics[name] = {"window_correct": int(np.sum(data["predicted"] == actual)),
                         "window_total": len(actual), "record_correct": sum(
                             x["predicted"] == x["actual"] for x in units),
                         "record_total": len(units), "records": units}
    assert metrics["mlp"]["window_correct"] == 118
    assert metrics["ratio"]["window_correct"] == 0
    selected = {rule: metrics[name]["window_correct"] / 177 for rule, name in choice["selected"].items()}
    save(run / "RESULT413.json", {"utc": now(), "models": metrics,
        "rule_accuracy": selected, "primary_discrimination_gate": choice["primary_strictly_prefers_reference"],
        "new_native_Agent_sessions": 0, "new_LLM_calls": 0, "new_PHM_fits": 0,
        "new_cold_prediction_processes": 3, "reused_cold_prediction_artifacts": 3,
        "new_independent_scenarios": 0, "confirmation_opened": False,
        "interpretation": "Known CWRU bad case only, no gain opportunity for keep; not a new Agent method superiority claim"})


def coordinate(run):
    start = time.perf_counter()
    protocol = json.loads((run / "PROTOCOL413.json").read_text())
    assert sha(__file__) == protocol["script_sha256"]
    for name, expected in protocol["inputs_sha256"].items():
        assert sha(run / name) == expected
    def invoke(args):
        with (run / ("logs/" + "_".join(args) + ".log")).open("w") as log:
            process = subprocess.run([sys.executable, "-B", __file__, "--run", str(run), *args],
                                     cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert process.returncode == 0, args
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(invoke, [["predict", split] for split in ("train", "val", "test")]))
    invoke(["choose"])
    invoke(["score"])
    result = json.loads((run / "RESULT413.json").read_text())
    result["science_seconds"] = time.perf_counter() - start
    save(run / "RESULT413.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("mode", choices=["coordinate", "predict", "choose", "score"])
    parser.add_argument("split", nargs="?")
    args = parser.parse_args()
    try:
        if args.mode == "predict":
            predict(args.run, args.split)
        else:
            globals()[args.mode](args.run)
    except BaseException:
        save(args.run / ("FAILURE_" + args.mode + ".json"), {"utc": now(), "error": traceback.format_exc()})
        raise
