"""Build reversible task corrections and actual engine-input bindings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256


def build(run, runtime, matrix_path):
    matrix=json.loads(matrix_path.read_text())
    source=next((run/"source/official").rglob("phm_scenarios.json"))
    original=json.loads(source.read_text())["pdm_scenarios"]
    source_hash=sha256(source)
    repairs=[]
    for task in matrix["task_matrix"]:
        item=dict(task_id=task["task_id"], original_dataset=task["configured_dataset"],
                  original_question_sha256=task["question_sha256"],
                  original_state=task["contract_state"])
        if task.get("identity_conflict"):
            item.update(corrected_dataset=task["prompt_dataset"],
                        correction="Match the explicit dataset named in the unchanged question",
                        runnable=False,
                        remaining="Requires a source-specific lifetime target contract; never substitute CMAPSS")
        else:
            item.update(corrected_dataset=task["configured_dataset"],
                        correction=None, runnable=None)
        repairs.append(item)
    assert len(repairs)==75
    # Engine source arrays are kept label-free and explicitly anonymous in their
    # channel semantics. This recovers actual inputs, not health certification.
    directory=runtime/"public/engine"
    directory.mkdir(exist_ok=True)
    signals=directory/"signals"
    signals.mkdir(exist_ok=True)
    inputs=[]
    originals={row["task_id"]:row for row in original}
    seen={}
    references=0
    for task in matrix["engine_task_matrix"]:
        if (not task["source_ids"] or not task["all_source_ids_found_in_H5"]
                or not task["exact_ordered_source_tuple_has_QA_record"]):
            continue
        paths=[]
        for identity in task["source_ids"]:
            source_array=run/"outputs/engine_signals_all"/(str(identity)+".npy")
            array=np.load(source_array,allow_pickle=False)
            assert array.shape==(600,33) and np.isfinite(array).all()
            target=signals/source_array.name
            if not target.exists():
                shutil.copy2(source_array,target)
            assert sha256(target)==sha256(source_array)
            seen[str(identity)]=sha256(target)
            paths.append("/app/engine/signals/"+source_array.name)
            references+=1
        inputs.append(dict(task_id=task["task_id"], ordered_source_ids=task["source_ids"],
                           sequence_files=paths, shape_per_sequence=[600,33],
                           original_question=originals[task["task_id"]]["input_question"],
                           original_question_sha256=task["question_sha256"],
                           generated_summary_verified=False,
                           input_adaptation="Explicit original source identity and actual sequences appended to question",
                           channel_semantics_verified=False, physical_health_contract_qualified=False))
    assert len(inputs)==28 and len(seen)==114
    save(directory/"task_inputs.json",inputs)
    save(directory/"source_manifest.json",dict(
        arrays=seen, source="pandalin98/EngineMT-QA", revision="ea2051a15512212976c22c1a075ec549c7d48eef",
        origin="N-CMAPSS simulator-derived source; not field measurements",
        whole_HDF5_downloaded=False, whole_HDF5_hash_verified=False,
        binding="Hash-matched archived arrays obtained through exact HDF5 byte ranges",
        channel_semantics_verified=False, labels_included=False))
    registry_path=runtime/"public/dataset_registry.json"
    registry=json.loads(registry_path.read_text())
    registry["ENGINEMTQA"]=dict(task="sequence_qa",path="engine",architectures=[],
        source_tasks=28,unique_sequences=114, labels_available=False,
        physical_health_contract_qualified=False,
        data_origin="Actual public N-CMAPSS-derived source arrays; unverified physical channel map")
    # Recover MFPT's fault-name dictionary from actual published columns.
    csv.field_size_limit(128*1024**2)
    mapping={}
    with (run/"inputs/pdmb_attempt2/MFPT.csv").open() as stream:
        for row in csv.DictReader(stream):
            label=row["label"]
            if label in mapping:
                assert mapping[label]==row["condition"]
            mapping[label]=row["condition"]
    registry["MFPT"]["source_label_meanings"]=mapping
    registry_path.unlink()
    save(registry_path,registry)
    save(run/"audit/MFPT_LABEL_SEMANTICS246.json",dict(
        mapping=mapping, source_sha256=sha256(run/"inputs/pdmb_attempt2/MFPT.csv"),
        rule="All rows for each label must agree in the published condition column",
        undefined_load_categories_not_invented=True))
    # External task outputs: predictions were sealed before accessing test
    # labels. Their identity-to-fault-name translation does not refit a model.
    case=json.loads((run/"audit/MFPT_QUALIFICATION246.json").read_text())
    pred=json.loads((run/"outputs/MFPT/sealed_test_predictions.json").read_text())
    save(run/"outputs/pdm_fault_013_research_adaptation.json",dict(
        original_task_id="pdm_fault_013", task=originals["pdm_fault_013"]["input_question"],
        predictions=[dict(sample_id=row["sample_id"], label=row["predicted_label"],
                          source_condition=mapping[str(row["predicted_label"])])
                     for row in pred["predictions"]],
        external_metrics=case["external_score"], model_parameter_updates=case["training_audit"]["optimizer_steps"],
        external_verification_complete=True, official_benchmark_score=False,
        boundaries="Published file-group split and external final-label evaluation; no Agent run"))
    save(run/"audit/TASK_REPAIRS246.json",dict(
        original_file_sha256=source_hash, original_file_unchanged=sha256(source)==source_hash,
        task_count=75, dataset_identity_corrections=sum(r["correction"] is not None for r in repairs),
        repairs=repairs, engine_input_tasks=len(inputs), engine_sequence_references=references,
        engine_unique_sequences=len(seen), engine_physical_health_qualified=False,
        missing_acoustic_modality=["pdm_fault_003"], unknown_load_groups=["pdm_fault_008"],
        anonymous_presegmented_record_identity=["pdm_fault_010","pdm_fault_009","pdm_fault_014"],
        explicit_synthetic_tasks=["engine_health_combined_029","engine_health_combined_030"],
        corrections_are_not_new_official_scores=True))
    print(json.dumps(dict(engine_tasks=len(inputs),unique_sequences=len(seen),
                          identity_corrections=13,MFPT_label_meanings=mapping)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--matrix",type=Path,required=True)
    a=p.parse_args()
    build(a.run,a.runtime,a.matrix)
