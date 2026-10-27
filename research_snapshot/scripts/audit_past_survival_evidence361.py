"""Qualify past-error lower bounds without reading a future event label.

This is the known censored MAE-hinge, not a new estimator or a selector.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    root = parser.parse_args().run.resolve()
    config = json.loads((root / "PROTOCOL361.json").read_text())
    source = Path(config["source"])
    assert sha(source / "SOURCE_MANIFEST.json") == config["source_manifest_sha256"]
    inventory = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    files = inventory["files"]
    if isinstance(files, list):
        files = {row.get("path", row.get("relative_path")): row for row in files}

    def bound(path):
        relative = str(path.relative_to(source))
        entry = files[relative]
        assert sha(path) == entry["sha256"], relative
        bindings[relative] = entry["sha256"]

    started, bindings, rows = time.monotonic(), {}, []
    receipt_path = source / "PREDICTED357.json"
    bound(receipt_path)
    cases = json.loads(receipt_path.read_text())["cases"]
    for task, spec in config["tasks"].items():
        records = []
        for case in cases:
            if case["task"] != task or case["kind"] != "query":
                continue
            observation = source / case["input"]
            prediction = source / case["output"] / "predictions.json"
            bound(observation)
            bound(prediction)
            assert sha(prediction) == case["prediction_sha256"]
            with np.load(observation, allow_pickle=False) as data:
                assert not any(k in data.files for k in ["labels", "actual_rul", "rul", "targets"])
                records.append(dict(sample_id=str(data["sample_ids"][0]),
                    unit=int(data["unit_ids"][0]), clock=float(data[spec["clock"]][0]),
                    feature=data["features"][0].copy(), history=data["history_features"].copy(),
                    predictions=json.loads(prediction.read_text())["banks"]["full"]))
        for current in sorted(records, key=lambda x: (x["unit"], x["clock"])):
            past = [x for x in records if x["unit"] == current["unit"] and x["clock"] < current["clock"]]
            for previous in past:
                mask = np.isclose(current["history"][:, spec["feature_clock"]], previous["clock"], rtol=0, atol=1e-6)
                assert mask.any(), "Past query must be in the legally visible current prefix"
                assert any(np.allclose(x, previous["feature"], rtol=1e-6, atol=1e-6)
                           for x in current["history"][mask]), "Past features differ from available history"
            bounds = None if not past else {
                name: float(np.mean([max(0, current["clock"] - x["clock"] - x["predictions"][name])
                                     for x in past])) for name in current["predictions"]}
            rows.append(dict(task=task, sample_id=current["sample_id"], unit=current["unit"],
                current_clock=current["clock"], earlier_queries=len(past),
                earlier_ids=[x["sample_id"] for x in past], past_error_lower_bounds=bounds,
                positive=bool(bounds and max(bounds.values()) > 1e-6),
                differentiates_models=bool(bounds and max(bounds.values()) - min(bounds.values()) > 1e-6)))
    assert time.monotonic() - started < config["budget"]["seconds"]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status="complete", rows=rows,
        tasks={task: dict(queries=sum(x["task"] == task for x in rows),
             with_past=sum(x["task"] == task and x["earlier_queries"] > 0 for x in rows),
             positive=sum(x["task"] == task and x["positive"] for x in rows),
             differentiates_models=sum(x["task"] == task and x["differentiates_models"] for x in rows))
             for task in config["tasks"]}, source_bindings=bindings,
        future_truth_reads=0, new_fits=0, new_predictions=0, new_LLM_calls=0,
        ordinary_Agent_comparison=False, novelty_established=False, majority_goal_complete=False,
        note="Known one-sided past-error bounds; zero is not correctness; not current or future MAE.")
    (root / "RESULT361.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["tasks"]))


if __name__ == "__main__":
    main()
