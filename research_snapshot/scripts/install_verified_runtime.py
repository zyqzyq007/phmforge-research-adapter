"""Install source-bound code/data/dependencies without external evaluation labels."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request


def install(run, destination):
    destination.mkdir(exist_ok=False)
    shutil.copytree(run/"public", destination/"public")
    shutil.copytree(run/"restored/legacy/runtime/asset_deps", destination/"deps")
    shutil.copytree(run/"adapted/asset_current/servers", destination/"asset_current/servers")
    shutil.copytree(run/"adapted/asset_current/llm", destination/"asset_current/llm")
    shutil.copytree(run/"adapted/asset/servers/tsfm", destination/"asset_legacy/servers/tsfm")
    shutil.copytree(run/"adapted/asset/servers/tsfm/artifacts/tsfm_models/ttm_96_28",
                    destination/"models/ttm_96_28")
    code = destination/"code"
    code.mkdir()
    scripts = Path(__file__).parent
    for name in ["real_rul_model.py","real_fault_model.py","verified_phmforge.py",
                 "verified_benchmark_worker.py","verified_asset_audit.py","verified_asset_forecast.py",
                 "qualify_asset_runtime.py","verified_benchmark_scoring.py"]:
        shutil.copy2(scripts/name,code/name)
    resource=json.loads((run/"audit/COUCH_RESOURCE240.json").read_text())
    auth="Basic "+base64.b64encode(b"phm240:isolated-benchmark-only").decode()
    def request(path, value=None, method=None):
        data=None if value is None else json.dumps(value).encode()
        req=urllib.request.Request(resource["url"]+path,data=data,method=method,
                                   headers={"Authorization":auth,"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.load(r)
    seed,indexes={},{}
    for name in ["workorder","iot","asset","model_catalog","feature_catalog","failure_code"]:
        rows=[r["doc"] for r in request("/"+name+"/_all_docs?include_docs=true")["rows"]]
        rows=[r for r in rows if not r["_id"].startswith("_design/")]
        if name=="model_catalog":
            for row in rows:
                row.update(params=dict(model_path="/models/ttm_96_28",fit_strategy="zero-shot",
                                       use_source_package=True,device="cpu"),
                           model_checkpoint="/models/ttm_96_28", artifact_path="/models/ttm_96_28",
                           hf_repo=None, version="published-local-config-96-28",
                           trained_on=[], description="Actual shipped local 96/28 checkpoint. Pretraining corpus and domain tuning unverified.")
                request("/model_catalog/"+row["_id"],row,"PUT")
        for row in rows:
            row.pop("_rev",None)
        seed[name]=rows
        indexes[name]=[dict(index=x["def"],name=x["name"],type=x["type"])
                       for x in request("/"+name+"/_index")["indexes"] if x["type"]=="json"]
    (destination/"database_seed.json").write_text(json.dumps(seed,indent=2))
    (destination/"database_indexes.json").write_text(json.dumps(indexes,indent=2))
    ready=dict(classification="research-adapted runtime, not the complete original benchmark",
               run_id=run.name, final_labels_included=False, dataset_groups=["CMAPSS_FD001..004","CWRU","MFPT","Chiller 6"],
               source_dependencies_archive="/data/yyf-zyq/PHM_20261010/20261018T185541Z_assethf190",
               original_main_scenarios_available=3, original_main_scenarios_missing=229,
               synthetic_motor_fixture_enabled=False,
               FMSR_generation="not enabled in numerical qualification; no hidden API credentials passed",
               note="database seed is the qualified forecasting profile with future observations withheld")
    (destination/"READY.json").write_text(json.dumps(ready,indent=2))
    records=[]
    for p in sorted(destination.rglob("*")):
        if p.is_file():
            records.append(dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(files=records,files_count=len(records),bytes=sum(x["bytes"] for x in records))
    (destination/"FILE_MANIFEST.json").write_text(json.dumps(manifest,indent=2))
    (run/"audit/INSTALLED_RUNTIME240.json").write_text(json.dumps(dict(
        runtime=str(destination),manifest_sha256=hashlib.sha256((destination/"FILE_MANIFEST.json").read_bytes()).hexdigest(),
        files=len(records),bytes=manifest["bytes"],external_labels_included=False),indent=2))
    print(json.dumps(dict(runtime=str(destination),files=len(records),bytes=manifest["bytes"])))


if __name__=="__main__":
    install(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
