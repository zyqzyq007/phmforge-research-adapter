"""Assemble the PHMForge source-recovery and validity matrix for A244."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


PDMB_DATASETS = (
    "Azure",
    "CWRU",
    "ElectricMotorVibrations",
    "FEMTO",
    "HUST",
    "IMS",
    "MAFAULDA",
    "MFPT",
    "Mendeley",
    "Padeborn",
    "PlanetaryPdM",
    "RotorBrokenBar",
    "UoC",
    "XJTU",
)
MIRROR_FILE_NAMES = {
    "PlanetaryPdM": "Planetary.csv",
    "UoC": "UoC.csv",
}
LOCAL_FULL_DATASETS = {
    "CWRU",
    "MFPT",
    "RotorBrokenBar",
    "Azure",
    "ElectricMotorVibrations",
    "UoC",
}
CMAPSS_LOCAL = {
    "CMAPSS_FD001",
    "CMAPSS_FD002",
    "CMAPSS_FD003",
    "CMAPSS_FD004",
}
DATASET_ALIASES = {
    "GearboxUoC": "UoC",
    "PlanetaryPdM": "PlanetaryPdM",
    "Paderborn": "Padeborn",
    "Padeborn": "Padeborn",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def load_qa_ids(source):
    streams = {}
    all_ids = set()
    all_record_keys = set()
    for split in ("train", "test"):
        record_keys = []
        with (source / f"engine_{split}_qa.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                values = record["id"]
                if not isinstance(values, list):
                    values = [values]
                key = tuple(str(value) for value in values)
                record_keys.append(key)
                all_record_keys.add(key)
                all_ids.update(key)
        streams[split] = record_keys
    return streams, all_ids, all_record_keys


def source_pins(source):
    catalog = read_json(source / "hf_submission096.json")
    catalog_by_id = {item["id"]: item for item in catalog}
    pins = {}
    for dataset in PDMB_DATASETS:
        item = catalog_by_id[f"submission096/{dataset}"]
        primary = read_json(source / f"pdmb_{dataset}_tree.json")
        mirror = read_json(source / f"odyssey_{dataset}_tree.json")
        primary_file = next(row for row in primary if row["path"] == "PdM.csv")
        mirror_name = MIRROR_FILE_NAMES.get(dataset, f"{dataset}.csv")
        mirror_file = next(row for row in mirror if row["path"] == mirror_name)
        primary_lfs = primary_file.get("lfs") or {}
        mirror_lfs = mirror_file.get("lfs") or {}
        primary_hash = primary_lfs.get("oid") or primary_file["oid"]
        mirror_hash = mirror_lfs.get("oid") or mirror_file["oid"]
        pins[dataset] = {
            "repo": f"submission096/{dataset}",
            "revision": item["sha"],
            "path": "PdM.csv",
            "bytes": primary_file["size"],
            "hash_type": "sha256" if primary_lfs else "git_blob_sha1",
            "hash": primary_hash,
            "mirror_repo": f"odysseywt/{dataset}",
            "mirror_path": mirror_name,
            "mirror_bytes_match": mirror_file["size"] == primary_file["size"],
            "mirror_hash_match": mirror_hash == primary_hash,
            "license_tag": next(
                (
                    tag.split(":", 1)[1]
                    for tag in item.get("tags", [])
                    if tag.startswith("license:")
                ),
                None,
            ),
        }
    return pins


def prompt_dataset(task):
    text = task["input_question"]
    explicit = (
        ("CMAPSS FD001", "CMAPSS_FD001"),
        ("CMAPSS FD002", "CMAPSS_FD002"),
        ("CMAPSS FD003", "CMAPSS_FD003"),
        ("CMAPSS FD004", "CMAPSS_FD004"),
        ("CWRU", "CWRU"),
        ("FEMTO", "FEMTO"),
        ("XJTU", "XJTU"),
        ("IMS", "IMS"),
        ("HUST", "HUST"),
        ("MFPT", "MFPT"),
        ("Paderborn", "Padeborn"),
        ("Mendeley", "Mendeley"),
        ("Azure", "Azure"),
        ("Planetary", "PlanetaryPdM"),
        ("UoC", "UoC"),
        ("ElectricMotorVibrations", "ElectricMotorVibrations"),
        ("Rotor Broken Bar", "RotorBrokenBar"),
    )
    for marker, dataset in explicit:
        if marker.lower() in text.lower():
            return dataset
    if task["task_id"] == "pdm_fault_003":
        return "ElectricMotorVibrations"
    if task["task_id"] == "pdm_fault_004":
        return "RotorBrokenBar"
    return None


def engine_refs(task):
    if isinstance(task.get("source_engine_ids"), list):
        return [str(value) for value in task["source_engine_ids"]]
    value = task.get("source_engine_id")
    if value and not str(value).startswith("synthetic_"):
        return [str(value)]
    return []


def engine_core(ground_truth):
    components = ground_truth.get(
        "component_health",
        ground_truth.get("component_health_detail", {}),
    )
    return {
        "component_health": dict(sorted(components.items())),
        "faulty_components": ground_truth.get("faulty_components"),
        "overall_health_state": ground_truth.get("overall_health_state"),
        "timing_issue_root_cause": ground_truth.get("timing_issue_root_cause"),
        "qualitative_condition": ground_truth.get("qualitative_condition"),
        "failure_probability_range": ground_truth.get("failure_probability_range"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    project = Path(__file__).resolve().parents[1]
    result_directory = (
        args.result_directory or project / "results/forgesources244"
    ).resolve()
    source = run / "source"
    scenario_path = (
        source
        / "official/ReActXen/src/reactxen/demo/intent_implementation_demo"
        / "scenarios/phm_scenarios.json"
    )
    scenario_doc = read_json(scenario_path)
    tasks = scenario_doc["pdm_scenarios"]
    if len(tasks) != 75:
        raise ValueError(f"Expected 75 scenarios, found {len(tasks)}")

    expected_hashes = {}
    with (project / "results/fullbench187/final_source_ledger214.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("contract") == "PHMForge_public":
                expected_hashes[row["original_task_id"]] = row["question_sha256"]
    question_hash_matches = {
        task["task_id"]: sha256_text(task["input_question"])
        == expected_hashes[task["task_id"]]
        for task in tasks
    }

    pins = source_pins(source)
    acquisition = read_json(run / "OUTPUTS244_ATTEMPT2.json")
    downloaded = {
        receipt["dataset"]: receipt for receipt in acquisition["receipts"]
    }
    h5 = read_json(run / "outputs/engine_signals_all/RESULT.json")
    h5_ids = {row["id"] for row in h5["records"]}
    qa_splits, qa_ids, qa_record_keys = load_qa_ids(source)
    a242_units = read_json(project / "results/forgeaudit242/SCENARIO_CHECK242.json")

    engine_summary_groups = defaultdict(list)
    task_rows = []
    engine_rows = []
    for task in tasks:
        task_id = task["task_id"]
        category = task["classification_type"]
        configured = task["dataset"]
        normalized_dataset = DATASET_ALIASES.get(configured, configured)
        mentioned = prompt_dataset(task) if category in {
            "RUL Prediction",
            "Fault Classification",
        } else None
        row = {
            "task_id": task_id,
            "category": category,
            "configured_dataset": configured,
            "prompt_dataset": mentioned,
            "question_sha256": sha256_text(task["input_question"]),
            "question_hash_matches_frozen_A187_ledger": question_hash_matches[task_id],
            "required_tools": task.get("required_tools", []),
        }
        if category == "RUL Prediction":
            row["identity_conflict"] = mentioned not in {
                None,
                normalized_dataset,
            }
            if row["identity_conflict"]:
                row["contract_state"] = "blocked_prompt_config_identity_conflict"
            elif task_id == "pdm_rul_001":
                row["contract_state"] = "source_available_original_metadata_defect"
            else:
                row["contract_state"] = "source_public_backend_not_integrated"
        elif category == "Fault Classification":
            row["identity_conflict"] = mentioned not in {
                None,
                normalized_dataset,
            }
            if normalized_dataset in LOCAL_FULL_DATASETS:
                row["contract_state"] = "source_verified_backend_not_yet_qualified"
            else:
                row["contract_state"] = "source_public_budget_blocked"
        elif category in {"Cost-Benefit Analysis", "Safety/Policy Evaluation"}:
            row["identity_conflict"] = False
            row["contract_state"] = "blocked_missing_validated_decision_policy"
        else:
            refs = engine_refs(task)
            in_h5 = [source_id for source_id in refs if source_id in h5_ids]
            direct_qa = [source_id for source_id in refs if source_id in qa_ids]
            exact_qa_record = bool(refs) and tuple(refs) in qa_record_keys
            summary = task["input_question"].split(
                "ENGINE SIGNAL DATA SUMMARY", 1
            )[-1]
            summary_hash = sha256_text(summary)
            engine_summary_groups[summary_hash].append(
                {
                    "task_id": task_id,
                    "core": engine_core(task["ground_truth"]),
                    "question_sha256": row["question_sha256"],
                }
            )
            if refs and len(refs) == 1 and exact_qa_record:
                source_state = "single_record_source_and_QA_label_recovered"
                contract_state = "source_recovered_tool_semantics_unresolved"
            elif len(refs) > 1 and exact_qa_record:
                source_state = "multi_record_source_and_QA_label_recovered"
                contract_state = "authored_summary_not_fully_source_bound"
            elif refs:
                source_state = "multi_record_H5_signals_present_incomplete_QA_binding"
                contract_state = "authored_summary_not_fully_source_bound"
            else:
                source_state = "explicit_synthetic_template_no_source_record"
                contract_state = "explicit_synthetic_scenario"
            row.update(
                source_ids=refs,
                source_id_count=len(refs),
                source_ids_found_in_H5=len(in_h5),
                all_source_ids_found_in_H5=len(in_h5) == len(refs),
                source_ids_with_QA_records=len(direct_qa),
                exact_ordered_source_tuple_has_QA_record=exact_qa_record,
                source_state=source_state,
                contract_state=contract_state,
                input_summary_sha256=summary_hash,
            )
            engine_rows.append(row | {"ground_truth_core": engine_core(task["ground_truth"])})
            row["identity_conflict"] = False

        if category != "Engine Health Analysis":
            if normalized_dataset in CMAPSS_LOCAL:
                source_state = "full_source_verified_in_A240_runtime"
            elif normalized_dataset in LOCAL_FULL_DATASETS:
                source_state = "full_source_verified_A244"
            elif normalized_dataset in pins:
                source_state = "public_pinned_source_not_downloaded_A244_budget"
            else:
                source_state = "source_mapping_unresolved"
            row["source_state"] = source_state
            row["dataset_pin"] = pins.get(normalized_dataset)
        task_rows.append(row)

    source_group_rows = []
    for summary_hash, members in engine_summary_groups.items():
        cores = {
            json.dumps(member["core"], sort_keys=True) for member in members
        }
        questions = {member["question_sha256"] for member in members}
        source_group_rows.append(
            {
                "summary_sha256": summary_hash,
                "task_ids": [member["task_id"] for member in members],
                "task_count": len(members),
                "distinct_full_questions": len(questions),
                "distinct_core_ground_truths": len(cores),
                "same_summary_conflicting_core_ground_truths": len(cores) > 1,
            }
        )
    source_group_rows.sort(key=lambda row: row["task_ids"][0])

    counts = {
        "categories": dict(Counter(row["category"] for row in task_rows)),
        "task_contract_states": dict(
            Counter(row["contract_state"] for row in task_rows)
        ),
        "source_states": dict(Counter(row["source_state"] for row in task_rows)),
        "identity_conflicts": sum(row["identity_conflict"] for row in task_rows),
        "rul_identity_conflicts": sum(
            row["identity_conflict"] and row["category"] == "RUL Prediction"
            for row in task_rows
        ),
        "fault_identity_conflicts": sum(
            row["identity_conflict"] and row["category"] == "Fault Classification"
            for row in task_rows
        ),
        "questions_hash_matching_frozen_ledger": sum(question_hash_matches.values()),
        "engine_tasks": len(engine_rows),
        "engine_single_record_QA_bound_tasks": sum(
            row["source_state"]
            == "single_record_source_and_QA_label_recovered"
            for row in engine_rows
        ),
        "engine_multi_record_tasks": sum(
            row["source_id_count"] > 1
            for row in engine_rows
        ),
        "engine_exact_tuple_QA_bound_tasks": sum(
            row["exact_ordered_source_tuple_has_QA_record"] for row in engine_rows
        ),
        "engine_synthetic_tasks": sum(
            row["source_state"] == "explicit_synthetic_template_no_source_record"
            for row in engine_rows
        ),
        "engine_source_reference_instances": sum(
            row["source_id_count"] for row in engine_rows
        ),
        "engine_unique_source_ids": len(
            {source_id for row in engine_rows for source_id in row["source_ids"]}
        ),
        "engine_source_ids_found_in_H5": len(h5_ids & {
            source_id for row in engine_rows for source_id in row["source_ids"]
        }),
        "engine_multi_record_QA_bound_tasks": sum(
            row["exact_ordered_source_tuple_has_QA_record"]
            for row in engine_rows
            if row["source_id_count"] > 1
        ),
        "engine_multi_record_QA_ID_reference_matches": sum(
            row["source_ids_with_QA_records"]
            for row in engine_rows
            if row["source_id_count"] > 1
        ),
        "engine_multi_record_source_reference_instances": sum(
            row["source_id_count"]
            for row in engine_rows
            if row["source_id_count"] > 1
        ),
        "engine_multi_record_unique_source_ids": len({
            source_id
            for row in engine_rows
            if row["source_id_count"] > 1
            for source_id in row["source_ids"]
        }),
        "engine_unique_summary_groups": len(engine_summary_groups),
        "engine_summary_groups_with_conflicting_core_ground_truth": sum(
            row["same_summary_conflicting_core_ground_truths"]
            for row in source_group_rows
        ),
    }

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "reused_acquisition_utc": acquisition["utc"],
        "run_id": run.name,
        "scope": "PHMForge public 75 scenarios, public source recovery, and failure attribution",
        "upstream": {
            "official_main_commit": "28430199f7b8ccba924d49e06c2d59b761846038",
            "official_latest_release": None,
            "official_tags": [],
            "scenario_path": str(scenario_path),
            "scenario_git_blob_sha1": git_blob_sha1(scenario_path),
            "fix_scenarios_git_blob_sha1": git_blob_sha1(
                source / "official/fix_scenarios.py"
            ),
            "extract_ground_truth_git_blob_sha1": git_blob_sha1(
                source / "official/extract_ground_truth.py"
            ),
            "later_fork_checked": {
                "repo": "RobbieFeng/PHMForge",
                "commit": "922e256d4ab47d5d8c5ee97977b6d4575581eb4f",
                "changes_scenario_generator_or_tools": False,
                "changed_paths": [
                    "README.md",
                    "ReActXen/env/.env_template",
                    "ReActXen/src/reactxen/demo/intent_implementation_demo/README.md",
                    "ReActXen/src/reactxen/demo/intent_implementation_demo/benchmark_pass1.py",
                    "ReActXen/src/reactxen/demo/intent_implementation_demo/run_pass3.py",
                    "ReActXen/src/reactxen/utils/model_inference.py",
                ],
            },
        },
        "counts": counts,
        "pdmb_sources": {
            "dataset_count": len(pins),
            "public_original_bytes": sum(pin["bytes"] for pin in pins.values()),
            "independent_mirror_size_matches": sum(
                pin["mirror_bytes_match"] for pin in pins.values()
            ),
            "independent_mirror_hash_matches": sum(
                pin["mirror_hash_match"] for pin in pins.values()
            ),
            "downloaded_and_hash_verified": sorted(downloaded),
            "downloaded_bytes": acquisition["source_bytes_verified"],
            "pins": pins,
        },
        "transport_history": [
            {
                "attempt": 1,
                "status": "failed_before_source_acceptance",
                "error_type": "SSLCertVerificationError",
                "error": "self-signed certificate in certificate chain",
                "accepted_source_files": 0,
            },
            {
                "attempt": 2,
                "status": "completed_with_pinned_content_authentication",
                "tls_verification_disabled_for_transport_only": True,
                "accepted_source_files": len(downloaded),
                "content_rule": (
                    "Accept only exact byte count plus LFS SHA-256 or Git blob "
                    "SHA-1, cross-checked against an independent public mirror pin."
                ),
            },
        ],
        "engine_source": {
            "source_sha256_claim": (
                "e35bfa367df18dc4365ec96b7ef8844de1dde8fc1060e6ebb40c842e270d403c"
            ),
            "whole_file_sha256_verified": False,
            "whole_file_bytes": h5["whole_file_bytes"],
            "unique_ids_in_public_H5": h5["unique_ids"],
            "requested_record_references": sum(
                row["source_id_count"] for row in engine_rows
            ),
            "unique_requested_record_ids": len(
                {value for row in engine_rows for value in row["source_ids"]}
            ),
            "all_unique_requested_ids_present": not h5["missing_requested_ids"],
            "exact_records_extracted": len(h5["records"]),
            "all_extracted_records_finite": all(
                row["finite"] for row in h5["records"]
            ),
            "range_transport_bytes": h5["downloaded_bytes"],
            "baseline_comparison": h5["baseline_comparison"],
            "train_QA_records": len(qa_splits["train"]),
            "test_QA_records": len(qa_splits["test"]),
            "unique_QA_signal_ids": len(qa_ids),
            "unique_ordered_QA_record_keys": len(qa_record_keys),
            "tasks_with_direct_single_record_QA_binding": counts[
                "engine_single_record_QA_bound_tasks"
            ],
            "tasks_with_exact_ordered_QA_binding": counts[
                "engine_exact_tuple_QA_bound_tasks"
            ],
            "summary_groups": source_group_rows,
            "exact_same_question_conflicting_ground_truth": [
                {
                    "task_ids": ["engine_health_combined_013", "engine_health_combined_017"],
                    "question_sha256": "905c7509db19c1346a81ed96cebca1c8909bcecdbd47b877d04a538ef1dd69bc",
                    "ground_truth": {
                        "engine_health_combined_013": "Good Condition / 10%-30%",
                        "engine_health_combined_017": "Moderate Condition / 10%-30%",
                    },
                }
            ],
        },
        "source_generation_defects": [
            {
                "id": "zero_based_cmapss_unit_ids",
                "evidence": (
                    "fix_scenarios.py uses range(len(rul_values)); the current "
                    "pdm_rul_001 sample_correct_units are the zero-based indices "
                    "of the actual one-based unit IDs."
                ),
                "current_units": a242_units["rul_top_units"]["sample_correct_units"],
                "actual_one_based_units": a242_units["rul_top_units"][
                    "computed_smallest_actual_RUL_units"
                ],
            },
            {
                "id": "dataset_reassignment_without_prompt_rewrite",
                "evidence": (
                    "fix_scenarios.py changes scenario['dataset'] for task-type "
                    "compatibility but does not rewrite input_question; 13 of 15 "
                    "RUL prompts still name a different asset/dataset."
                ),
            },
            {
                "id": "non_cmapss_ground_truth_placeholders",
                "evidence": (
                    "The retained extraction log reports that RUL extraction is "
                    "not implemented for CWRU, FEMTO, IMS, XJTU, HUST, MFPT, "
                    "Paderborn, Mendeley and Azure, then falls back to placeholder "
                    "units/default ranges. Commit b4bbba40 describes these as "
                    "actual replacements, so commit text cannot authenticate them."
                ),
            },
            {
                "id": "engine_combined_scenarios_not_source_copies",
                "evidence": (
                    "generation_info states the 30 engine-health scenarios are "
                    "combined with embedded summaries and are not copies from "
                    "EngineMTQA. Eighteen tasks have one source ID and matching QA "
                    "labels; all ten multi-ID tasks match complete ordered ID "
                    "tuples in the public QA files. Their authored input summaries "
                    "still require separate validation. Two tasks are explicitly "
                    "synthetic."
                ),
            },
            {
                "id": "same_engine_question_different_label",
                "evidence": (
                    "engine_health_combined_013 and _017 have byte-identical input "
                    "questions but ground truths Good Condition and Moderate "
                    "Condition respectively."
                ),
            },
        ],
        "task_matrix": task_rows,
        "engine_task_matrix": engine_rows,
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
        "new_training_jobs": 0,
        "new_scientific_scores": 0,
        "problem_confirmation_increment": 0,
        "fully_qualified_original_tasks": 0,
        "all_original_tasks_ready": False,
        "all_referenced_PDMB_and_non_synthetic_Engine_data_sources_identified": True,
        "explicitly_synthetic_tasks_without_source": counts[
            "engine_synthetic_tasks"
        ],
        "caution": (
            "Public source availability does not repair task identity conflicts, "
            "authored engine summaries, missing decision models, text-only scoring "
            "or absent physical validation."
        ),
    }
    save(run / "SOURCE_MATRIX244.json", result)
    save(result_directory / "SOURCE_MATRIX244.json", result)
    summary = {
        "run_id": run.name,
        "status": "source_audit_complete_pending_archive",
        "source_matrix": "SOURCE_MATRIX244.json",
        "result_sha256": hashlib.sha256(
            (run / "SOURCE_MATRIX244.json").read_bytes()
        ).hexdigest(),
        "counts": counts,
        "pdmb_public_original_bytes": result["pdmb_sources"][
            "public_original_bytes"
        ],
        "pdmb_full_files_hash_verified": result["pdmb_sources"][
            "downloaded_and_hash_verified"
        ],
        "engine_whole_file_bytes": h5["whole_file_bytes"],
        "engine_whole_file_sha256_verified": False,
        "engine_exact_records_extracted": len(h5["records"]),
        "transport_attempts": result["transport_history"],
        "fully_qualified_original_tasks": 0,
        "new_LLM_calls": 0,
        "new_Agent_sessions": 0,
        "new_training_jobs": 0,
        "new_scientific_scores": 0,
        "problem_confirmation_increment": 0,
        "overall_goal_complete": False,
    }
    save(run / "SUMMARY244.json", summary)
    save(result_directory / "SUMMARY244.json", summary)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
