#!/usr/bin/env python3
"""Audit explicit as-of tasks against their original tool traces.

This is a zero-model, read-only analysis of an already closed archive.  It
does not infer whole-task PHM accuracy or re-run the original evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = {
    "413": {
        "job": "asset_0077",
        "cut": "2018-01-03T00:00:00",
        "cut_label": "2018-01-02",
        "family": "work-order probability",
        "asset_id": "CWC04009",
    },
    "414": {
        "job": "asset_0078",
        "cut": "2018-01-03T00:00:00",
        "cut_label": "2018-01-02",
        "family": "work-order probability",
        "asset_id": "CWC04009",
    },
    "420": {
        "job": "asset_0084",
        "cut": "2020-07-01T00:00:00",
        "cut_label": "early July 2020 / June review",
        "family": "chiller decision support",
        "asset_id": "CWC04009",
    },
    "422": {
        "job": "asset_0086",
        "cut": "2020-07-01T00:00:00",
        "cut_label": "end of June 2020",
        "family": "chiller decision support",
        "asset_id": "CWC04009",
    },
    "423": {
        "job": "asset_0087",
        "cut": "2020-07-01T00:00:00",
        "cut_label": "end of June 2020",
        "family": "chiller decision support",
        "asset_id": "CWC04009",
    },
    "431": {
        "job": "asset_0095",
        "cut": "2020-07-01T00:00:00",
        "cut_label": "early July 2020 / June review",
        "family": "chiller decision support",
        "asset_id": "CWC04009",
    },
    "435": {
        "job": "asset_0099",
        "cut": "2020-07-01T00:00:00",
        "cut_label": "early July 2020 / June review",
        "family": "chiller decision support",
        "asset_id": "CWC04009",
    },
}

ANSWER_EVIDENCE = {
    "413": "Prediction tool called with end_date=2018-01-02; no later record returned.",
    "414": "Prediction and work-order calls use end_date=2018-01-02; no later record returned.",
    "420": "No post-cut work-order ID or date appears in the final answer.",
    "422": (
        "Answer cites the post-cut preventive addition WO367298 on 2020-07-07 and "
        "calls it part of the yearly plan."
    ),
    "423": (
        "Answer calls two post-cut actual records (WO367298, WO369088) the July "
        "work-order slate and assigns exact due dates."
    ),
    "431": (
        "Answer labels post-cut actual records WO371657 (2020-08-24) and "
        "WO366934 (2020-11-19) as scheduled, and cites eight July 1-7 alerts."
    ),
    "435": (
        "Answer reuses post-cut actual record WO367298 and its exact 2020-07-07 "
        "19:00 timestamp as a generated work order and calls it a historical match."
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nested_payloads(value: Any) -> list[Any]:
    """Return JSON objects embedded in text blocks of tool responses."""
    found: list[Any] = []
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            try:
                found.append(json.loads(text))
            except json.JSONDecodeError:
                pass
        for child in value.values():
            found.extend(nested_payloads(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_payloads(child))
    return found


def objects_with(value: Any, key: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get(key), list):
            found.append(value)
        for child in value.values():
            found.extend(objects_with(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(objects_with(child, key))
    return found


def records_from_call(call: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    work_orders: list[dict] = []
    events: list[dict] = []
    for payload in nested_payloads(call.get("response", {})):
        for container in objects_with(payload, "work_orders"):
            work_orders.extend(container.get("work_orders", []))
        for container in objects_with(payload, "events"):
            events.extend(container.get("events", []))
    return work_orders, events


def date_arg_cutoff(arguments: dict[str, Any]) -> str | None:
    for key in ("end_date", "final", "end"):
        value = arguments.get(key)
        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value):
            return value + ("T23:59:59" if len(value) == 10 else "")
    return None


def load_archive_members(archive: Path) -> dict[str, bytes]:
    wanted = {"source/asset_all_utterance.jsonl"}
    for task in TASKS.values():
        job = task["job"]
        wanted.update(
            {
                f"primary/{job}/{job}/response.txt",
                f"primary/{job}/{job}/status.json",
                f"traces/{job}/calls.jsonl",
            }
        )
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if member.isfile() and member.name in wanted:
                extracted = handle.extractfile(member)
                if extracted is not None:
                    members[member.name] = extracted.read()
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=Path(
        "results/fullbench187/A190_native_results.json"
    ))
    parser.add_argument("--ledger", type=Path, default=Path(
        "results/fullbench187/final_source_ledger214.jsonl"
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    members = load_archive_members(args.archive)
    source_rows = {
        str(row["id"]): row
        for row in map(
            json.loads,
            members["source/asset_all_utterance.jsonl"].decode().splitlines(),
        )
    }
    native = {
        row["job"]: row
        for row in json.loads(args.native.read_text())["rows"]
    }
    ledger = {}
    for line in args.ledger.read_text().splitlines():
        row = json.loads(line)
        if row.get("contract") == "AssetOps_HF/scenarios":
            ledger[str(row.get("original_task_id"))] = row

    rows = []
    for task_id, spec in TASKS.items():
        job = spec["job"]
        answer = members[f"primary/{job}/{job}/response.txt"].decode("utf-8", "replace")
        calls_bytes = members[f"traces/{job}/calls.jsonl"]
        calls = [json.loads(line) for line in calls_bytes.splitlines()]

        future_calls = []
        future_records: dict[str, dict[str, Any]] = {}
        business_calls = 0
        for line_number, call in enumerate(calls, 1):
            tool = call.get("tool", "")
            if tool in {"list_tools", "tool_schema", "close_worker"}:
                continue
            business_calls += 1
            arguments = call.get("arguments", {})
            end_cutoff = date_arg_cutoff(arguments)
            work_orders, events = records_from_call(call)
            work_orders = [
                record
                for record in work_orders
                if record.get("equipment_id") in {None, spec["asset_id"]}
            ]
            events = [
                record
                for record in events
                if record.get("equipment_id") in {None, spec["asset_id"]}
            ]
            local_future = []
            for record in work_orders:
                timestamp = record.get("actual_finish")
                record_id = record.get("wo_id")
                if timestamp and timestamp >= spec["cut"] and record_id:
                    future_records[record_id] = record
                    local_future.append(record_id)
            for record in events:
                timestamp = record.get("event_time")
                record_id = record.get("event_id")
                if timestamp and timestamp >= spec["cut"] and record_id:
                    future_records[record_id] = record
                    local_future.append(record_id)
            if (end_cutoff and end_cutoff > spec["cut"]) or local_future:
                future_calls.append(
                    {
                        "line": line_number,
                        "tool": tool,
                        "arguments": arguments,
                        "declared_end_crosses_cut": bool(
                            end_cutoff and end_cutoff > spec["cut"]
                        ),
                        "returned_future_record_count": len(set(local_future)),
                    }
                )

        future_work_order_ids = sorted(
            record_id
            for record_id, record in future_records.items()
            if record.get("wo_id")
            or (
                record.get("event_group") == "WORK_ORDER"
                and str(record.get("event_id", "")).startswith("WO")
            )
        )
        future_event_groups = Counter(
            record.get("event_group", "UNKNOWN")
            for record in future_records.values()
            if record.get("event_id")
        )
        cited_future_records = sorted(
            record_id for record_id in future_work_order_ids if record_id in answer
        )
        future_timestamps = sorted(
            {
                record.get("actual_finish") or record.get("event_time")
                for record in future_records.values()
            }
        )
        boundary = (
            "post_cut_records_cited"
            if cited_future_records
            else "post_cut_records_exposed_not_cited"
            if future_calls
            else "cutoff_respected"
        )

        question = source_rows[task_id]
        original = ledger.get(task_id, {})
        native_row = native[job]
        answer_hash = sha256_bytes(answer.encode())
        calls_hash = sha256_bytes(calls_bytes)
        rows.append(
            {
                "original_task_id": task_id,
                "job": job,
                "family": spec["family"],
                "decision_cut": spec["cut_label"],
                "question": question["text"],
                "original_gold_or_characteristic_form": question.get(
                    "characteristic_form"
                ),
                "question_sha256": original.get("question_sha256"),
                "answer_sha256": answer_hash,
                "calls_sha256": calls_hash,
                "hash_matches_ledger": (
                    original.get("evidence", {}).get("answer_sha256") == answer_hash
                    and original.get("evidence", {}).get("calls_sha256") == calls_hash
                ),
                "business_tool_calls": business_calls,
                "post_cut_calls": future_calls,
                "unique_post_cut_records": len(future_records),
                "post_cut_work_order_count": len(future_work_order_ids),
                "nearest_post_cut_work_orders": [
                    {
                        "work_order": record_id,
                        "timestamp": future_records[record_id].get("actual_finish")
                        or future_records[record_id].get("event_time"),
                        "primary_code": future_records[record_id].get("primary_code")
                        or future_records[record_id].get("event_type"),
                        "preventive": future_records[record_id].get("preventive"),
                    }
                    for record_id in sorted(
                        future_work_order_ids,
                        key=lambda item: future_records[item].get("actual_finish")
                        or future_records[item].get("event_time")
                        or "",
                    )[:10]
                ],
                "post_cut_event_group_counts": dict(sorted(future_event_groups.items())),
                "post_cut_timestamp_min": future_timestamps[0] if future_timestamps else None,
                "post_cut_timestamp_max": future_timestamps[-1] if future_timestamps else None,
                "post_cut_work_order_ids_cited_in_answer": cited_future_records,
                "answer_evidence": ANSWER_EVIDENCE[task_id],
                "boundary_behavior": boundary,
                "original_comparator": {
                    "scorer": "AssetOpsBench LLM judge / original six-dimension rubric",
                    "same_model_dependent": True,
                    "visible_trace_char_limit": original.get("original_scoring", {}).get(
                        "scored_trace_chars"
                    ),
                    "visible_call_names": original.get("original_scoring", {}).get(
                        "visible_call_names"
                    ),
                    "passed": native_row["native_score"].get("passed"),
                    "score": native_row["native_score"].get("score"),
                    "rationale": native_row["native_score"].get("rationale"),
                },
                "scope_note": (
                    "No post-cut actual record exposure or citation was observed."
                    if boundary == "cutoff_respected"
                    else "Evidence of post-decision-cut actual record exposure or citation; "
                    "not a whole-task PHM accuracy label and not an independent case."
                ),
            }
        )

    behavior_counts = Counter(row["boundary_behavior"] for row in rows)
    result = {
        "analysis": "explicit_as_of_task_temporal_boundary_audit",
        "archive": str(args.archive),
        "new_agent_sessions": 0,
        "new_llm_calls": 0,
        "new_phm_predictions": 0,
        "retrospective_not_blinded": True,
        "selection_scope": (
            "All seven AssetOps HF scenarios in the closed 152-task cohort whose "
            "question text explicitly states a current date or an end-of-period "
            "information cut. This is a boundary-audit subset, not the full "
            "152-task accuracy denominator."
        ),
        "complete_subgroup_denominator": len(rows),
        "same_asset_fixture_family": "CWC04009 work orders/alerts",
        "independent_physical_cases": 1,
        "boundary_behavior_counts": dict(sorted(behavior_counts.items())),
        "post_cut_record_exposure_rows": sum(
            bool(row["post_cut_calls"]) for row in rows
        ),
        "post_cut_record_citing_rows": sum(
            bool(row["post_cut_work_order_ids_cited_in_answer"]) for row in rows
        ),
        "record_date_semantics": {
            "workorders_csv_sha256": (
                "5d81ae9783c5012ea03f52046692f400c8e395007ca37f39c12115c861671789"
            ),
            "event_csv_sha256": (
                "19313c21eb73eb942888842c9b0dfa23399a53be9128144ef6e74b4c8c210f61"
            ),
            "workorder_fields": [
                "wo_id",
                "equipment_id",
                "preventive",
                "work_priority",
                "actual_finish",
                "duration",
                "actual_labor_hours",
            ],
            "event_fields": ["event_id", "event_group", "event_time"],
            "scheduled_or_created_timestamp_present": False,
            "note": (
                "The detected post-cut values are actual_finish or actual event_time. "
                "The source snapshot has no scheduled, created, or record-availability "
                "timestamp, so the answers' scheduled/historical labels are not "
                "supported by these records."
            ),
        },
        "availability_limitation": (
            "The immutable snapshot cannot prove the exact wall-clock time at which a "
            "future-dated record became visible to a CMMS. The confirmed finding is "
            "post-cut actual-record use and schedule/outcome relabeling, not a proven "
            "database-access violation."
        ),
        "original_evaluator_caught_temporal_violation_rows": 0,
        "important_counterexamples": [
            "413 and 414 pass end_date=2018-01-02 to the prediction and work-order "
            "tools; no later record is returned.",
            "420 receives unbounded post-cut history but does not cite a post-cut "
            "work-order record in its final answer; exposure and dependence are separate.",
        ],
        "ordinary_control": (
            "Reject or clamp tool calls whose declared end exceeds the question's "
            "decision cut, reject unbounded work-order/event queries in as-of tasks, "
            "and require separate scheduled versus actual-completion fields. This "
            "generic check identifies all five exposure rows."
        ),
        "research_problem_status": (
            "Observed temporal evidence-binding error with potential duplicate or "
            "unsupported maintenance action; not confirmed as a hard or novel "
            "research problem because the date/schema controls are direct and the "
            "record-availability time is absent from the source."
        ),
        "new_confirmed_problems": 0,
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows": len(rows),
                    "boundary_behavior_counts": result["boundary_behavior_counts"],
                    "future_access_rows": result["future_access_rows"],
                    "future_record_citing_rows": result["future_record_citing_rows"],
                }
            )
        )
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
