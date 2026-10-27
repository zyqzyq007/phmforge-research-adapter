"""Final checks and lightweight result export for the A240 verified environments."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "run_verified_benchmarks.py",
    "prepare_real_bench_data.py",
    "qualify_asset_current.py",
    "prepare_asset_current_runtime.py",
    "score_verified_qualification.py",
    "finalize_verified_qualification.py",
]
DOCUMENTS = [
    "docs/real_bench_environment240.md",
    "docs/benchmark_training_evaluation_audit239.md",
]


def load(path):
    return json.loads(path.read_text())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(run):
    qualification = load(run / "audit/QUALIFICATION_RESULTS240.json")
    holdout = load(run / "audit/ASSET_HOLDOUT_SELECTION240.json")
    runtime = load(run / "audit/INSTALLED_RUNTIME240.json")
    runtime_dir = Path(runtime["runtime"])
    runtime_manifest = runtime_dir / "FILE_MANIFEST.json"
    assert sha(runtime_manifest) == runtime["manifest_sha256"]
    shutil.copy2(runtime_manifest, run / "audit/INSTALLED_RUNTIME_FILE_MANIFEST240.json")
    shutil.copy2(runtime_dir / "READY.json", run / "audit/INSTALLED_RUNTIME_READY240.json")
    installed = load(run / "audit/INSTALLED_MCP240.json")
    contracts = load(run / "audit/CONTRACT_CHECKS240.json")
    couch = load(run / "audit/COUCH_RESOURCE240.json")
    legacy_stages = load(run / "outputs/asset/job_status.json")
    current_stages = load(run / "outputs/asset_current/job_status.json")
    old_qualification = load(
        run / "superseded/attempt0_final_tail_all_zero/audit_snapshot/"
        "QUALIFICATION_RESULTS240.json"
    )

    forge = qualification["forge"]
    assert len(forge) == 10 and all(row["qualification_pass"] for row in forge)
    asset = qualification["asset"]
    assert asset["qualification_pass"] and asset["controls"]["nondegenerate_target"]
    assert asset["current_main"]["forecast_rows"] == 28
    assert [row["stage"] for row in legacy_stages] == [
        "train", "predict", "repeat", "pretrained", "changed_input"
    ]
    assert all(row["exit_code"] == 0 for row in legacy_stages)
    assert [row["mode"] for row in current_stages] == ["train", "reload", "forecast"]
    assert all(row["exit_code"] == 0 for row in current_stages)
    assert installed == [
        dict(family="forge", tools=22, passed=True),
        dict(family="asset_legacy", tools=6, passed=True),
        dict(family="asset_current", tools=85, passed=True),
    ]
    assert contracts["all_passed"]
    assert couch["status"] == "stopped_and_removed_after_qualification"
    assert couch["cleanup_verified"]
    assert holdout["held_out_rows"] == 28
    assert holdout["held_out_nonzero"] == 28
    assert holdout["held_out_distinct"] == 28
    assert not holdout["selection_used_model_scores"]
    assert old_qualification["asset"]["trained"]["n"] == 28
    old_gold = load(
        run / "superseded/attempt0_final_tail_all_zero/hidden_asset_live/"
        "chiller6_future.json"
    )
    old_target = np.asarray([row["value"] for row in old_gold], float)
    assert old_target.size == 28
    assert np.unique(old_target).size == 1 and old_target[0] == 0

    context = list(csv.DictReader((run / "public/asset/serving_context.csv").open()))
    train = list(csv.DictReader((run / "public/asset/train.csv").open()))
    gold = load(run / "hidden/asset/chiller6_future.json")
    assert len(context) == 124
    assert context[-1]["timestamp"] == holdout["forecast_origin"]
    assert train[-1]["timestamp"] == holdout["forecast_origin"]
    first_expected = str(np.datetime64(context[-1]["timestamp"], "s") + np.timedelta64(15, "m"))
    assert gold[0]["timestamp"] == first_expected
    assert all(
        np.datetime64(b["timestamp"], "s") - np.datetime64(a["timestamp"], "s")
        == np.timedelta64(15, "m")
        for a, b in zip(gold, gold[1:])
    )
    target = np.asarray([row["value"] for row in gold], float)
    past = np.asarray([float(row["Chiller 6 Power Input"]) for row in context], float)
    naive = {
        "persistence_mae": float(np.abs(target - past[-1]).mean()),
        "context_mean_mae": float(np.abs(target - past.mean()).mean()),
        "zero_mae": float(np.abs(target).mean()),
    }

    catalog = load(run / "public/tool_catalog.json")
    unavailable = sorted(
        name for name, item in catalog.items()
        if item["description"].startswith("UNAVAILABLE")
    )
    script_hashes = {
        name: sha(PROJECT / "scripts" / name)
        for name in SCRIPTS
    }
    result = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        run_id=run.name,
        classification=(
            "research-adapted, source-bound qualification of real training/inference; "
            "not completion of either full original benchmark"
        ),
        status="qualified_with_documented_limitations",
        original_benchmark_scores_replaced=False,
        new_LLM_sessions=0,
        forge_numeric_jobs=len(forge),
        forge_qualification_passes=sum(row["qualification_pass"] for row in forge),
        forge_metrics={
            row["job"]: row["evaluation"]["predict"] for row in forge
        },
        asset=asset,
        nondegenerate_holdout=holdout,
        simple_reference_errors=naive,
        installed_runtime=runtime,
        installed_mcp=installed,
        couch_resource=couch,
        tool_catalog=dict(
            total=len(catalog),
            adapted_real_or_source_bound=sum(
                item["description"].startswith("Research-adapted")
                for item in catalog.values()
            ),
            explicitly_unavailable=unavailable,
            unavailable_count=len(unavailable),
            no_fabricated_returns_for_unavailable=True,
        ),
        contract_negative_controls=contracts,
        stages=dict(
            legacy_asset=legacy_stages,
            current_asset=current_stages,
        ),
        preserved_failures=[
            dict(
                attempt="initial final-tail holdout",
                reason="28 hidden targets were all zero",
                consequence=(
                    "near-zero MAE measured a trivial zero forecast; result invalidated "
                    "and retained under superseded/attempt0_final_tail_all_zero"
                ),
            ),
            dict(
                attempt="first corrected current-main training",
                reason="fixed initial_window=2800 exceeded 939 visible rows",
                consequence=(
                    "training did not start; failure retained under "
                    "superseded/attempt1_current_initial_window_overflow"
                ),
            ),
        ],
        script_hashes=script_hashes,
        code_and_document_snapshot=str(run / "code_snapshot"),
        limitations=[
            "C-MAPSS is a published simulator, not field run-to-failure telemetry.",
            "CWRU original recording/bearing identities are absent; contiguous blocks are disclosed.",
            "MFPT filenames are not certified as independent physical bearings.",
            "AssetOps Chiller 6 is one published fixture and one nondegenerate 28-step holdout; field provenance is unverified.",
            "The current-main Chiller result is a qualification example, not evidence of strong forecasting accuracy.",
            "Nine PHMForge tools have no valid source contract and return explicit unavailable status.",
            "This run does not replace original benchmark scores or certify the full original task suites.",
        ],
    )
    for name in SCRIPTS:
        destination = run / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / "scripts" / name, destination)
    for name in DOCUMENTS:
        destination = run / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, destination)
    put(run / "audit/FINAL_VERIFICATION240.json", result)

    output = PROJECT / "results/realbench240"
    output.mkdir(parents=True, exist_ok=True)
    for source, name in [
        (run / "audit/QUALIFICATION_RESULTS240.json", "QUALIFICATION_RESULTS240.json"),
        (run / "audit/ASSET_HOLDOUT_SELECTION240.json", "ASSET_HOLDOUT_SELECTION240.json"),
        (run / "audit/INSTALLED_RUNTIME240.json", "INSTALLED_RUNTIME240.json"),
        (run / "audit/INSTALLED_MCP240.json", "INSTALLED_MCP240.json"),
        (run / "audit/CONTRACT_CHECKS240.json", "CONTRACT_CHECKS240.json"),
        (run / "audit/FINAL_VERIFICATION240.json", "FINAL_VERIFICATION240.json"),
    ]:
        shutil.copy2(source, output / name)
    put(output / "current_run.json", dict(
        run_id=run.name,
        path=str(run),
        status="qualified_with_documented_limitations",
        result=str(output / "FINAL_VERIFICATION240.json"),
        original_benchmark_scores_replaced=False,
        new_LLM_sessions=0,
    ))
    print(json.dumps(dict(
        status=result["status"],
        forge=f"{len(forge)}/{len(forge)}",
        asset_qualification=asset["qualification_pass"],
        installed_tools=sum(x["tools"] for x in installed),
        unavailable_tools=len(unavailable),
    )))


if __name__ == "__main__":
    finalize(Path("/srv/yyf-zyq/PHM_20261010/20261020T111512Z_realbench240"))
