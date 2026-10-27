#!/usr/bin/env python3
"""Reconstruct A13 errors without relabelling diagnosis as health permission.

Read-only, posthoc attribution audit. Semantic judgments are recorded separately
from mechanically reproduced labels and code/report locators.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path

JOBS = ("EngineTransfer_r1", "EngineTransfer_r2",
        "EngineMagnitude_r1", "EngineMagnitude_r2")
MANIFEST_SHA = "11d932406a12690477eb76bc387793b768fd8d29e2242eb7a18861d8fe73c1f0"
REPORT_RANGES = {
    "EngineTransfer_r1": [(100, 112)],
    "EngineTransfer_r2": [(107, 121)],
    "EngineMagnitude_r1": [(97, 104), (123, 128), (132, 145)],
    "EngineMagnitude_r2": [(58, 63)],
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(out, name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def category(truth, prediction):
    if prediction in (5, None):
        return "unknown"
    if prediction == truth:
        return "correct"
    if prediction == 0:
        return "false_NF"
    if truth == 0:
        return "false_alarm"
    return "wrong_fault_class"


def branch(job, evidence):
    if job == "EngineTransfer_r1":
        return evidence["decision_rule"]
    if job == "EngineTransfer_r2":
        # Saved log_deviations are d_eff, as bound by source analysis.py.
        guard = 0.10 if evidence["prefix_time_seconds"] <= 100 else 0.05
        maximum = max(abs(v) for v in evidence["log_deviations"].values())
        return "NF_guard" if maximum < guard else "classifier_argmax_NF"
    if job == "EngineMagnitude_r1":
        assert evidence["class_ranking"][0] == 0
        return "Gaussian_log_likelihood_argmax_NF"
    return evidence["rule"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.archive.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError("Output must be outside the immutable source archive")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "SOURCE_MANIFEST.json"
    assert sha(manifest_path) == MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    expected = {f["path"]: f for f in manifest["files"]}
    needed = ["controller_only/query_gold.json", "formal_plan.json",
              "full_scientific_replay_audit.json", "future_prefix_audit.json"]
    for job in JOBS:
        needed.append(f"tasks/{job}_prompt.txt")
        needed += [f"tasks/{job}/{name}" for name in
                   ("predictions.json", "report.md", "analysis.py", "validation.json",
                    "data_dictionary.json", "queries.json")]
    integrity = []
    for name in needed:
        p, item = root / name, expected[name]
        size, digest = p.stat().st_size, sha(p)
        assert size == item["bytes"] and digest == item["sha256"], name
        integrity.append({"path": name, "bytes": size, "sha256": digest})
    save(out, "source_integrity70.json", {"source_archive": str(root),
         "manifest_sha256": MANIFEST_SHA, "selected_files_verified": integrity,
         "scope": "Selected audit inputs, not a new full 833-file archive verification"})

    gold = json.loads((root / "controller_only/query_gold.json").read_text())
    by_id = {x["case_id"]: x for x in gold}
    assert len(by_id) == len(gold) == 81
    rows, misses, counts, anchors = [], [], {}, {}
    for job in JOBS:
        task = root / "tasks" / job
        predictions = json.loads((task / "predictions.json").read_text())["results"]
        queries = json.loads((task / "queries.json").read_text())
        if isinstance(queries, dict):
            queries = queries["queries"]
        assert [p["case_id"] for p in predictions] == [q["case_id"] for q in queries]
        expected_ids = {g["case_id"] for g in gold
                        if g["condition"] == job.rsplit("_", 1)[0]}
        assert {p["case_id"] for p in predictions} == expected_ids
        counter = collections.Counter()
        for pred in predictions:
            truth = by_id[pred["case_id"]]
            kind = category(truth["class_code"], pred["class_code"])
            counter[kind] += 1
            row = {"job_id": job, "case_id": pred["case_id"],
                   "record_id": truth["record_id"], "source_member": truth["source_member"],
                   "sample_index": truth["sample_index"], "time_seconds": truth["time_seconds"],
                   "driving_regime": truth["driving_regime"], "gold_class": truth["class_code"],
                   "predicted_class": pred["class_code"], "diagnosis_result": kind,
                   "goal_mechanism_confirmed": False,
                   "maintenance_action_observed": False}
            rows.append(row)
            if kind == "false_NF":
                misses.append(dict(row, code_branch=branch(job, pred["evidence"]),
                                   original_evidence=pred["evidence"],
                                   attribution="Real diagnosis error; no scoped exclusion or maintenance permission was evaluated by the original task."))
        counts[job] = dict(counter)
        code = (task / "analysis.py").read_text()
        lines, report = code.splitlines(), (task / "report.md").read_text().splitlines()
        functions = []
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.FunctionDef) and node.name in ("classify", "predict_dvec", "decide"):
                functions.append({"name": node.name, "start_line": node.lineno,
                                  "end_line": node.end_lineno,
                                  "text": "\n".join(lines[node.lineno - 1:node.end_lineno])})
        anchors[job] = {"code_file": f"tasks/{job}/analysis.py", "decision_functions": functions,
                        "report_file": f"tasks/{job}/report.md", "report_passages": [
                            {"start_line": lo, "end_line": hi, "text": "\n".join(report[lo-1:hi])}
                            for lo, hi in REPORT_RANGES[job]],
                        "semantic_review": "All four complete reports were read by the research assistant; each preserves evidence limitations and denies maintenance authorization. NF remains a diagnosis under the original task, not an evaluated health certificate.",
                        "review_provenance": "AI manual posthoc review, not independent expert judgment or keyword-based automatic proof"}
    with (out / "all_diagnosis_rows70.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(out, "false_nf_trace70.json", misses)
    save(out, "code_report_anchors70.json", anchors)
    total = collections.Counter(row["diagnosis_result"] for row in rows)
    summary = {"source_archive": str(root), "source_family": "LiU-ICE, one physical engine",
               "phase": "posthoc attribution of old discovery data; no new confirmation task",
               "workflows": 4, "presentations": len(rows), "unique_checkpoints": len(by_id),
               "unique_query_records": len({row['record_id'] for row in rows}),
               "by_job": counts, "counts": dict(total),
               "false_nf_unique_checkpoints": len({r['case_id'] for r in misses}),
               "false_nf_unique_records": len({r['record_id'] for r in misses}),
               "false_nf_source_members": sorted({r['source_member'] for r in misses}),
               "false_nf_code_branches": dict(collections.Counter(r['code_branch'] for r in misses)),
               "new_model_calls": 0, "new_goal_specific_confirmed_failures": 0,
               "E3": 0, "goal_status": "active",
               "claim_boundary": ["37 wrong answered diagnoses and one unknown remain original-task results.",
                 "NF classification alone does not establish unsupported maintenance permission.",
                 "No concrete downstream maintenance/pruning action was requested or measured in A13.",
                 "Reports preserve caveats; this does not prove every possible downstream consumer would retain them.",
                 "Possible label-to-action loss is unmeasured; not a confirmed failure mechanism.",
                 "Shared prefix baseline assumption is disclosed and not violated by the actual source onsets; a hypothetical earlier onset is not an observed failure.",
                 "No pooled failure-rate confidence interval across correlated checkpoints and repeats."]}
    assert len(rows) == 162 and sum(total.values()) == 162
    save(out, "summary70.json", summary)
    print(json.dumps({k: summary[k] for k in ("counts", "false_nf_unique_checkpoints",
                     "false_nf_unique_records", "new_goal_specific_confirmed_failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
