#!/usr/bin/env python3
"""Independently verify A252/A253 external classification scores and isolation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def load_json(path: Path):
    return json.loads(path.read_text())


def metric(actual, predicted):
    labels = sorted(set(actual) | set(predicted), key=lambda x: (str(type(x)), x))
    return {
        "n": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "class_order": labels,
    }


def close(a, b, tolerance=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = load_json(run / "PLAN252.json")
    report = load_json(run / "EVALUATION252.json")
    controller = load_json(run / "CONTROLLER_STATUS252.json")
    rows = {x["job"]: x for x in report["rows"]}

    checks = {
        "controller_all_terminal": controller.get("status")
        == "all_selected_sessions_terminal_no_scoring",
        "report_has_three_jobs": len(plan["jobs"]) == report["sessions"] == 3,
        "no_session_still_running": all(
            load_json(run / "primary" / job["id"] / job["id"] / "status.json")["status"]
            != "running"
            for job in plan["jobs"]
        ),
        "no_official_score_claimed": (
            report["official_original_benchmark_score_qualified"] is False
            and report["fully_qualified_original_tasks"] == 0
        ),
    }
    rows_out = []
    for job in plan["jobs"]:
        ident = job["id"]
        workspace = run / job["workspace"]
        label_rows = load_json(run / "hidden" / ident / "labels.json")
        gold = {str(x["sample_id"]): x.get("label", x.get("published_label"))
                for x in label_rows}
        hidden_npz = np.load(run / "hidden" / ident / "test.npz", allow_pickle=False)
        checks[f"{ident}_test_observation_has_no_labels"] = (
            "labels" not in hidden_npz.files and "groups" not in hidden_npz.files
        )
        checks[f"{ident}_workspace_has_no_test_artifact"] = not any(
            p.name == "test.npz" for p in workspace.rglob("*") if p.is_file()
        )
        checks[f"{ident}_workspace_has_no_hidden_dir"] = not (workspace / "hidden").exists()
        events_text = "\n".join(
            p.read_text(errors="replace")
            for p in (run / "primary" / ident / ident).glob("events.jsonl")
        )
        checks[f"{ident}_events_do_not_name_run_hidden_path"] = (
            str(run / "hidden" / ident) not in events_text
        )
        checks[f"{ident}_events_do_not_name_credentials_file"] = (
            "/home/jupyter-yyf-zyq/.bashrc" not in events_text
        )

        independent = {}
        for arm in ("baseline", "candidate"):
            prediction_path = run / "evaluation" / ident / arm / "output/predictions.json"
            if not prediction_path.is_file():
                checks[f"{ident}_{arm}_predictions_exist"] = False
                continue
            checks[f"{ident}_{arm}_predictions_exist"] = True
            value = load_json(prediction_path)
            predictions = value.get("predictions", value)
            by_id = {str(x["sample_id"]): x["predicted_label"] for x in predictions}
            checks[f"{ident}_{arm}_identities_unique"] = len(by_id) == len(predictions)
            checks[f"{ident}_{arm}_identities_complete"] = set(by_id) == set(gold)
            ids = sorted(gold)
            independent[arm] = metric([gold[x] for x in ids], [by_id[x] for x in ids])
            primary = rows[ident].get(arm + "_metrics")
            checks[f"{ident}_{arm}_accuracy_recomputed"] = (
                primary is not None
                and close(primary["accuracy"], independent[arm]["accuracy"])
            )
            checks[f"{ident}_{arm}_macro_f1_recomputed"] = (
                primary is not None
                and close(primary["macro_f1"], independent[arm]["macro_f1"])
            )
            checks[f"{ident}_{arm}_balanced_accuracy_recomputed"] = (
                primary is not None
                and close(primary["balanced_accuracy"], independent[arm]["balanced_accuracy"])
            )
            checks[f"{ident}_{arm}_confusion_matrix_recomputed"] = (
                primary is not None
                and primary["confusion_matrix"] == independent[arm]["confusion_matrix"]
            )
        baseline_known = job["baseline_metrics"]
        checks[f"{ident}_baseline_matches_prior_external_accuracy"] = (
            "baseline" in independent
            and close(independent["baseline"]["accuracy"], baseline_known["accuracy"])
        )
        checks[f"{ident}_baseline_matches_prior_external_macro_f1"] = (
            "baseline" in independent
            and close(independent["baseline"]["macro_f1"], baseline_known["macro_f1"])
        )
        rows_out.append({
            "job": ident,
            "session_status": rows[ident]["session_status"],
            "baseline": independent.get("baseline"),
            "candidate": independent.get("candidate"),
            "delta_accuracy": (
                independent["candidate"]["accuracy"] - independent["baseline"]["accuracy"]
                if set(independent) == {"baseline", "candidate"} else None
            ),
            "delta_macro_f1": (
                independent["candidate"]["macro_f1"] - independent["baseline"]["macro_f1"]
                if set(independent) == {"baseline", "candidate"} else None
            ),
        })

    result = {
        "audit_id": "A252-independent",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "rows": rows_out,
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
    }
    (run / "audit/INDEPENDENT_VERIFY252.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": result["passed"], "total": result["total"],
                      "all_passed": result["all_passed"]}))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
