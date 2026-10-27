"""Bring up an isolated real database and exercise latest AssetOps main."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import base64

from run_verified_benchmarks import asset_command, execute


def run_current_jobs(run, url, modes=None):
    source = run / "adapted/asset_current"
    code = Path(__file__).parent
    for filename in ["qualify_asset_current.py", "verified_asset_audit.py", "qualify_asset_runtime.py"]:
        shutil.copy2(code / filename, run/"public/code"/filename)
    output = run / "outputs/asset_current"
    output.mkdir(exist_ok=True)
    for d in ["tmp", "mpl", "xdg", "hf"]:
        (output/".runtime"/d).mkdir(parents=True, exist_ok=True)
    status_path = output / "job_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else []
    modes = modes or ["train", "reload", "forecast"]
    added = []
    for mode in modes:
        argv = asset_command(run, ["/code/qualify_asset_current.py", mode])
        argv.remove("--unshare-net")
        i = argv.index(str(run/"adapted/asset/servers/tsfm"))
        argv[i:i+2] = [str(source/"servers"), "/opt/asset/servers"]
        i = argv.index(str(run/"outputs/asset"))
        argv[i] = str(output)
        i = argv.index("--proc")
        argv[i:i] = ["--ro-bind", str(run/"adapted/asset/servers/tsfm/artifacts/tsfm_models"), "/models"]
        i = argv.index(str(Path("/home/jupyter-yyf-zyq/.local/bin/python3.12").resolve()))
        argv[i:i] = ["--setenv", "COUCHDB_URL", url,
                     "--setenv", "COUCHDB_USERNAME", "phm240",
                     "--setenv", "COUCHDB_PASSWORD", "isolated-benchmark-only",
                     "--setenv", "TSFM_STORE", "couch"]
        result = execute(run, "asset_current_"+mode, argv)
        added.append(dict(mode=mode, **result))
        status.append(dict(mode=mode, **result))
        print(mode, result["exit_code"], flush=True)
        if result["exit_code"] != 0:
            break
    status_path.write_text(json.dumps(status, indent=2))
    if any(row["exit_code"] != 0 for row in added) or len(added) != len(modes):
        raise RuntimeError("Current AssetOps native stage failed; inspect retained logs")
    return status


def run_current(run):
    original = run / "restored/current/source/asset/src"
    source = run / "adapted/asset_current"
    shutil.copytree(original, source)
    # Latest main only returns the first five values, insufficient to evaluate its full horizon.
    file = source / "servers/tsfm/engine/composition.py"
    text = file.read_text()
    old = '"forecast_head": forecast[:5],'
    assert text.count(old) == 2
    new = '"forecast": forecast, "forecast_head": forecast[:5],'
    modified = text.replace(old, new)
    modified = modified.replace('    model = getattr(fc, "model", None)\n',
                                '    model = getattr(fc, "model_", None)\n'
                                '    if model is None:\n'
                                '        model = getattr(fc, "model", None)\n')
    file.write_text(modified)
    (run / "audit/CURRENT_ASSET_PATCHES240.json").write_text(json.dumps([dict(
        file="servers/tsfm/engine/composition.py", before=hashlib.sha256(text.encode()).hexdigest(),
        after=hashlib.sha256(file.read_bytes()).hexdigest(), removed=old, added=new,
        additional_change="Save fitted sktime model_ (fallback model for older wrappers)")], indent=2))
    file = source / "servers/tsfm/io/refs.py"
    text = file.read_text()
    assert 'os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work")' in text
    file.write_text(text.replace('os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work")',
                                'os.environ["TSFM_WORKDIR"]'))
    fixture = run / "filtered_db"
    fixture.mkdir()
    with (run / "public/asset/train.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["asset_id"] = "Chiller 6"
        for k in row.keys()-{"timestamp", "asset_id"}:
            row[k] = float(row[k])
    (fixture / "iot.json").write_text(json.dumps(rows))
    shared = source / "couchdb/scenarios_data/shared"
    manifest = dict(workorder=str(shared / "work_order/workorders.csv"),
                    iot=str(fixture / "iot.json"),
                    asset=str(shared / "iot/asset_profile_sample.json"),
                    model_catalog=str(shared / "tsfm/model_catalog.json"),
                    feature_catalog=str(shared / "tsfm/feature_catalog.json"),
                    failure_code=str(shared / "failure_code/failure_code_sample.csv"))
    (fixture / "manifest.json").write_text(json.dumps(manifest, indent=2))
    name = "phm240_real_couch"
    password = "isolated-benchmark-only"
    subprocess.run(["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::5984",
                    "-e", "COUCHDB_USER=phm240", "-e", "COUCHDB_PASSWORD="+password,
                    "couchdb:3.5"], check=True, capture_output=True)
    port = subprocess.check_output(["docker", "port", name, "5984/tcp"], text=True).strip().split(":")[-1]
    url = "http://127.0.0.1:" + port
    resource = dict(container=name, url=url, purpose="isolated, disposable source-bound local benchmark database",
                    final_future_rows_excluded=28, synthetic_vibration_fixture_loaded=False)
    (run / "audit/COUCH_RESOURCE240.json").write_text(json.dumps(resource, indent=2))
    authorization = "Basic " + base64.b64encode(("phm240:"+password).encode()).decode()
    for attempt in range(30):
        try:
            with urllib.request.urlopen(urllib.request.Request(url+"/_up", headers={"Authorization":authorization}), timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Local CouchDB did not become ready")
    from benchmark_tool_bridge import clean_environment
    env = clean_environment()
    python = str(Path("/home/jupyter-yyf-zyq/.local/bin/python3.12").resolve())
    env.update(PYTHONPATH=f"{run}/restored/legacy/runtime/asset_deps:{source}",
               DEFAULT_MANIFEST=str(fixture/"manifest.json"),
               COUCHDB_URL=url, COUCHDB_USERNAME="phm240", COUCHDB_PASSWORD=password,
               TMPDIR=str(run/"runtime/tmp"), TMP=str(run/"runtime/tmp"), TEMP=str(run/"runtime/tmp"),
               PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(run/"runtime/mpl"))
    command = [python, "-B", "-c", "from couchdb.init_data import init_data; print(init_data())"]
    result = execute(run, "current_db_init", command, env)
    if result["exit_code"] != 0:
        raise RuntimeError("Database initialization failed; inspect log")
    run_current_jobs(run, url)


if __name__ == "__main__":
    run_current(Path(sys.argv[1]).resolve())
