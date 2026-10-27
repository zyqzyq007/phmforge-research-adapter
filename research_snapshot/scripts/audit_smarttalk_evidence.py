"""Audit a frozen SMARTTalk artifact without invoking any LLM or loading weights.

The audit distinguishes published metrics, reconstructed counts, saved predictions,
and oracle-conditioned explanations. No device-level independence is inferred from
window indices. Run outputs are deterministic and support a cold numerical replay.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((args.run / "smarttalk_blob_sources68.json").read_text())
    manifests = json.loads((args.run / "smarttalk_acquired_manifest68.json").read_text())
    manifest = {x["id"]: x for x in manifests}
    locations = {}
    verified = []
    rejected = []
    for source in sources:
        item = manifest[source["id"]]
        if item["status"] != "saved":
            rejected.append({"path": source["repo_path"], "status": item["status"],
                             "reason": "Acquisition/integrity gate; not a scientific outcome."})
            continue
        path = args.run / item["snapshot_directory"] / item["file"]
        body = path.read_bytes()
        sha1 = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        sha256 = hashlib.sha256(body).hexdigest()
        assert len(body) == source["expected_bytes"], source["repo_path"]
        assert sha1 == source["git_blob_sha1"], source["repo_path"]
        assert sha256 == item["sha256"], source["repo_path"]
        locations[source["repo_path"]] = path
        verified.append({"path": source["repo_path"], "bytes": len(body),
                         "git_blob_sha1": sha1, "sha256": sha256})
    save(args.output / "verified_sources.json", verified)
    save(args.output / "rejected_sources.json", rejected)

    index_rows = rows(locations["data/splits/sampled_test_1to23/sampled_test_indices.csv"])
    selected = defaultdict(list)
    for row in index_rows:
        selected[f"{row['dataset']}_round{row['round']}"].append(row)
    arrays = {}
    audit_arrays = []
    for name, path in sorted(locations.items()):
        if not (name.startswith("data/splits/") and name.endswith("/test.npz")):
            continue
        group = Path(name).parent.name
        with zipfile.ZipFile(path) as zipped, zipped.open("X.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _, dtype = np.lib.format._read_array_header(handle, version)
        with np.load(path, allow_pickle=False) as bundle:
            y, ttf = bundle["y"], bundle["ttf"]
            fields, features = bundle.files, bundle["features"].tolist()
        assert len(y) == len(ttf) == shape[0]
        assert np.isin(y, [0, 1]).all()
        arrays[group] = (y, ttf)
        ix = np.array([int(r["test_index"]) for r in selected[group]], dtype=int)
        assert len(ix) == len(np.unique(ix))
        assert ((ix >= 0) & (ix < len(y))).all()
        label_column = next((k for k in (selected[group][0] if len(ix) else {})
                             if k in ("y", "label", "status", "true_status")), None)
        mismatches = None
        if label_column:
            given = np.array([int(r[label_column]) for r in selected[group]])
            mismatches = int((given != y[ix]).sum())
        audit_arrays.append({
            "group": group, "window_shape": shape, "X_dtype": str(dtype),
            "fields": fields, "features": features,
            "healthy_windows": int((y == 0).sum()), "risk_windows": int((y == 1).sum()),
            "positive_outside_0_to_30_days": int(((y == 1) & ((ttf < 0) | (ttf > 30))).sum()),
            "negative_with_ttf_0_to_30_days": int(((y == 0) & (ttf >= 0) & (ttf <= 30)).sum()),
            "negative_without_recorded_failure": int(((y == 0) & (ttf == -1)).sum()),
            "negative_with_failure_after_30_days": int(((y == 0) & (ttf > 30)).sum()),
            "sampled_windows": len(ix), "sampled_risk": int(y[ix].sum()),
            "sampled_healthy": int((y[ix] == 0).sum()),
            "sampled_index_label_column": label_column,
            "sampled_index_label_mismatches": mismatches,
            "device_ids_and_window_dates_available": all(k in fields for k in ("disk_id", "ds")),
        })
    save(args.output / "array_label_audit.json", audit_arrays)

    # This deliberately exercises the author's published-metric fallback only.
    # It does NOT convert the fallback into empirical prediction observations.
    name = "smarttalk_sampled_audit_module"
    spec = importlib.util.spec_from_file_location(
        name, locations["smarttalk/_legacy/scripts/status_sampled_utils.py"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    table = pd.read_csv(locations["configs/paper_tables/table5_status.csv"])
    supports = module.load_sampled_supports(
        locations["data/splits/sampled_test_1to23/sampling_summary.csv"])
    _, detail = module.derive_sampled_status_table(table, supports)
    detail.to_csv(args.output / "reconstructed_counts.csv", index=False)
    cached = pd.read_csv(locations[
        "artifacts/cached_ablation_results/cached_outputs/confusion_matrices.csv"])
    keys = ["method", "backbone", "dataset"]
    merged = detail.merge(cached, on=keys, suffixes=("_recomputed", "_cached"))
    count_equal = all((merged[k + "_recomputed"] == merged[k + "_cached"]).all()
                      for k in ("tp", "fp", "tn", "fn", "source"))
    save(args.output / "table_lineage_audit.json", {
        "rows": len(detail), "matched_cached_rows": len(merged),
        "count_and_source_columns_equal": bool(count_equal),
        "source_counts": dict(Counter(detail["source"])),
        "inference": "Matching reconstructed counts verify code lineage, not observed model predictions.",
        "current_LLM_calls": 0,
    })

    pred_paths = [k for k in locations if k.endswith("predictions.jsonl")]
    predictions = []
    for name in pred_paths:
        for row in map(json.loads, locations[name].read_text().splitlines()):
            group = "MB2_round1"  # The only shipped prediction JSONL at this frozen commit.
            assert Path(name).name == "mb2_round1_llm_predictions.jsonl", name
            y, ttf = arrays[group]
            idx = int(row["index"])
            predictions.append({
                "file": name, "index": idx,
                "saved_true_status": row["true_status"],
                "array_label": int(y[idx]), "array_ttf": int(ttf[idx]),
                "pred_status": row["pred_status"],
                "sampled_test_member": idx in {int(r["test_index"]) for r in selected[group]},
                "explanation": row["explanation"], "recommendation": row["recommendation"],
            })
    save(args.output / "saved_prediction_audit.json", predictions)

    explanation_audit = []
    for name, path in sorted(locations.items()):
        if not (name.startswith("artifacts/checkpoints/") and "/mb2_exp_" in name):
            continue
        data = rows(path)
        group = Path(name).parent.name
        available = group in arrays
        if available:
            y, ttf = arrays[group]
        explanation_audit.append({
            "file": name, "rows": len(data),
            "status_label_counts": dict(Counter(r["status_label"] for r in data)),
            "distinct_window_indices": len({r["idx"] for r in data}),
            "array_available": available,
            "label_mismatches": sum(int(y[int(r["idx"])]) != 1 for r in data) if available else None,
            "ttf_mismatches": sum(float(r["ttf_label_days"]) != int(ttf[int(r["idx"])])
                                  for r in data) if available else None,
            "role": "Ground-truth-conditioned explanation generation; not blind diagnosis.",
        })
    save(args.output / "explanation_cache_audit.json", explanation_audit)
    save(args.output / "summary.json", {
        "commit": json.loads((args.run / "smarttalk_commit68.json").read_text())["sha"],
        "verified_files": len(verified), "verified_bytes": sum(x["bytes"] for x in verified),
        "selected_files": len(sources), "rejected_files": len(rejected),
        "test_arrays": len(audit_arrays),
        "test_windows": sum(x["window_shape"][0] for x in audit_arrays),
        "sampled_windows": len(index_rows),
        "sampled_windows_with_array_verified": sum(x["sampled_windows"] for x in audit_arrays),
        "cached_prediction_rows": len(predictions),
        "oracle_explanation_rows": sum(x["rows"] for x in explanation_audit),
        "reconstructed_count_rows": len(detail),
        "reconstructed_counts_match": bool(count_equal),
        "current_LLM_calls": 0, "confirmed_independent_Agent_failure_tasks": 0,
        "qualification": "Current-health exclusion task not yet qualified: no device/date linkage or empirical negative-advice denominator in the inspected cache.",
    })
    print((args.output / "summary.json").read_text())


if __name__ == "__main__":
    main()
