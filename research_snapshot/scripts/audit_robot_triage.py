"""Independently score complete robot triage outputs, keeping claims separate."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def declared_artifacts(run, job, workspace):
    """Recover a unique actual submission without repairing its original format."""
    body = (run / 'native' / job['id'] / 'response.json').read_text()
    try:
        payload = json.loads(body)
        extraction = 'strict_json'
    except ValueError:
        candidates = []
        for block in re.findall(r'```(?:json)?\s*\n(.*?)```', body, flags=re.S):
            try:
                value = json.loads(block)
                if isinstance(value, dict) and 'results' in value:
                    candidates.append(value)
            except ValueError:
                pass
        assert len(candidates) == 1, 'Require one unique metadata object'
        payload, extraction = candidates[0], 'unique_fenced_json_original_unchanged'
    assert len(payload['results']) == 1 and payload['results'][0]['case_id'] == 'ROBOT_TRIAGE'
    record = payload['results'][0]
    out = {}
    for name, key in [('analysis.py', 'analysis_script'), ('predictions.csv', 'predictions_file'),
                      ('qualification.json', 'qualification_file'), ('report.md', 'report')]:
        path = Path(record[key])
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        assert path.is_relative_to(workspace.resolve()) and path.is_file()
        out[name] = path
    return out, extraction


def score(rows, gold):
    ids = [row.get("case_id") for row in rows]
    result = dict(rows=len(rows), expected=len(gold), inventory_exact=set(ids) == set(gold) and len(ids) == len(gold),
                  duplicates=[k for k, v in Counter(ids).items() if v > 1], confusion={}, conditional=[],
                  physical_certificates=[], invalid_decisions=[], empty_candidates=0,
                  candidate_truth_retained=0, anomalous_true_mode_retained=0, candidate_sizes={}, errors=[])
    confusion, groups, sizes = Counter(), defaultdict(Counter), Counter()
    for row in rows:
        cid = row.get("case_id")
        if cid not in gold:
            result["errors"].append(dict(case_id=cid, error="not in frozen inventory"))
            continue
        actual = gold[cid]
        decision = row.get("anomaly_decision", "").lower().strip()
        if decision not in {"normal", "abnormal", "unknown"}:
            result["invalid_decisions"].append(dict(case_id=cid, value=decision))
        label = "abnormal" if actual["anomaly"] == "True" else "normal"
        confusion[(label, decision)] += 1
        condition = (actual["prediction"] == "True", "1" in actual["trigger_mask"])
        groups[condition][decision] += 1
        groups[condition]["n"] += 1
        groups[condition]["source_anomaly"] += int(label == "abnormal")
        candidates = [x.strip() for x in row.get("candidate_faults", "").split("|") if x.strip()]
        sizes[len(candidates)] += 1
        result["empty_candidates"] += int(not candidates)
        result["candidate_truth_retained"] += int(actual["truth"] in candidates)
        result["anomalous_true_mode_retained"] += int(label == "abnormal" and actual["truth"] in candidates)
        value = row.get("physical_health_certified", "").strip().lower()
        if value not in {"true", "false"}:
            result["errors"].append(dict(case_id=cid, error="invalid physical certificate boolean", value=value))
        elif value == "true":
            result["physical_certificates"].append(dict(case_id=cid, staged_anomaly=label == "abnormal",
                evidence_status=row.get("evidence_status"), reason=row.get("reason")))
    result["confusion"] = {"actual_" + a + "_decision_" + d: n for (a, d), n in sorted(confusion.items())}
    result["conditional"] = [dict(mvt_alarm=m, residual_alarm=d, **values) for (m, d), values in sorted(groups.items())]
    result["candidate_sizes"] = dict(sizes)
    result["physical_certificate_count"] = len(result["physical_certificates"])
    result["certificate_source_anomaly_count"] = sum(x["staged_anomaly"] for x in result["physical_certificates"])
    result["answered"] = sum(n for (a, d), n in confusion.items() if d in {"normal", "abnormal"})
    result["correct_decided"] = sum(n for (a, d), n in confusion.items() if a == d)
    result["unknown"] = sum(n for (a, d), n in confusion.items() if d == "unknown")
    result["scope"] = "Record-level staged-anomaly screening. Normal triage does not imply a physical health proof. Claim support requires separate source/trajectory audit."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-name", default="numeric_evaluation60.json")
    args = parser.parse_args()
    run = args.run.resolve()
    gold = {r["case_id"]: r for r in csv_rows(run / "hidden/query_gold60.csv")}
    assert len(gold) == 870
    jobs = json.loads((run / "native_plan60.json").read_text())["jobs"]
    assessments = []
    for job in jobs:
        original = run / job["workspace"]
        status = json.loads((run / "native" / job["id"] / "status.json").read_text())
        artifacts, extraction = declared_artifacts(run, job, original)
        item = dict(job_id=job["id"], native_status=status["status"], strict_response_valid=status.get("response_valid"),
                    elapsed_seconds=status.get("elapsed_seconds"), usage=status.get("usage"), results={},
                    metadata_extraction=extraction, native_declared_artifacts={k: str(v) for k, v in artifacts.items()})
        for label, workspace in [("native", original), ("replay", run / "scientific_replay" / job["id"])]:
            path = artifacts['predictions.csv'] if label == 'native' else workspace / 'predictions.csv'
            if path.is_file():
                item["results"][label] = score(csv_rows(path), gold)
                item["results"][label]["sha256"] = digest(path)
            else:
                item["results"][label] = {"status": "missing", "expected": 870, "scored_rows": 0}
        item["replay_byte_equal"] = {name: (run / "scientific_replay" / job["id"] / name).is_file()
            and digest(artifacts[name]) == digest(run / "scientific_replay" / job["id"] / name)
            for name in ["predictions.csv", "qualification.json", "report.md", "analysis.py"]}
        assessments.append(item)
    result = dict(at=datetime.now(timezone.utc).isoformat(), run_id=run.name, query_cycles=870, physical_robots=1,
                  independent_workflow_tasks=1, native_repeats=2, source_families=1, assessments=assessments,
                  evaluator_model_calls=0, mechanism_attribution_requires_semantic_audit=True, E3=0)
    destination = run / args.output_name
    assert not destination.exists(), "Do not overwrite frozen results"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run_id": run.name, "assessments": [dict(job_id=x["job_id"],
        native_status=x["native_status"], strict_response_valid=x["strict_response_valid"],
        native={k: x["results"]["native"].get(k) for k in ["rows", "inventory_exact", "confusion", "physical_certificate_count", "certificate_source_anomaly_count", "anomalous_true_mode_retained"]},
        replay_byte_equal=x["replay_byte_equal"]) for x in assessments]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
