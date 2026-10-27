"""Reproducible isolated qualification launcher; no LLM credentials or hidden gold."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare_real_training_campaign import environment, numeric_namespace


SCRIPTS = Path(__file__).resolve().parent
MATRIX = [(f"CMAPSS_FD00{i}", "mlp", 40) for i in range(1, 5)] + [
    ("CMAPSS_FD001", "lstm", 40), ("CMAPSS_FD001", "transformer", 40),
    ("CWRU", "mlp", 30), ("MFPT", "mlp", 30),
    ("MFPT", "lstm", 20), ("MFPT", "transformer", 20)]


def execute(run, name, command, env=None, timeout=900):
    start = time.monotonic()
    with (run / "logs" / (name + ".log")).open("w") as log:
        try:
            r = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
            result = dict(exit_code=r.returncode)
        except subprocess.TimeoutExpired:
            result = dict(exit_code=None, error="timeout")
    result.update(command=command, elapsed_seconds=time.monotonic()-start)
    (run / "logs" / (name + ".status.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def asset_command(run, command):
    output = run / "outputs/asset"
    output.mkdir(parents=True, exist_ok=True)
    for d in ["tmp", "mpl", "xdg", "hf"]:
        (output / ".runtime" / d).mkdir(parents=True, exist_ok=True)
    python = Path("/home/jupyter-yyf-zyq/.local/bin/python3.12").resolve()
    argv = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-net", "--die-with-parent", "--new-session", "--clearenv"]
    for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", str(python.parent.parent)]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    # Only forecasting source/model files; original CouchDB fixtures include hidden future.
    for source, target in [(run / "restored/legacy/runtime/asset_deps", "/deps"),
                            (run / "adapted/asset/servers/tsfm", "/opt/asset/servers/tsfm"),
                            (run / "public/asset", "/data"), (run / "public/code", "/code")]:
        argv += ["--ro-bind", str(source), target]
    argv += ["--bind", str(output), "/output", "--proc", "/proc", "--dev", "/dev",
             "--chdir", "/output"]
    variables = dict(PYTHONPATH="/deps:/opt/asset:/code", PYTHONDONTWRITEBYTECODE="1",
                     TMPDIR="/output/.runtime/tmp", TMP="/output/.runtime/tmp", TEMP="/output/.runtime/tmp",
                     MPLCONFIGDIR="/output/.runtime/mpl", XDG_CACHE_HOME="/output/.runtime/xdg",
                     HF_HOME="/output/.runtime/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                     TSFM_WORKDIR="/output/artifacts",
                     PATH_TO_MODELS_DIR="/opt/asset/servers/tsfm/artifacts/tsfm_models",
                     PATH_TO_DATASETS_DIR="/data", PATH_TO_OUTPUTS_DIR="/output",
                     OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", HF_HUB_DISABLE_TELEMETRY="1")
    for k, v in variables.items():
        argv += ["--setenv", k, v]
    return argv + ["--setenv", "PATH", "/usr/bin:/bin", str(python), "-B"] + command


def asset_job(run):
    results = []
    cases = [
        ("train", ["train", "--name", "native_train"]),
        ("predict", ["predict", "--name", "native_predict"]),
        ("repeat", ["predict", "--name", "native_repeat"]),
        ("pretrained", ["predict", "--name", "native_pretrained", "--model", "ttm_96_28"]),
        ("changed_input", ["predict", "--name", "native_changed_input", "--input", "/data/changed_context.csv"]),
    ]
    for name, args in cases:
        command = asset_command(run, ["/code/qualify_asset_runtime.py"] + args)
        result = execute(run, "asset_" + name, command)
        results.append(dict(stage=name, **result))
        if result["exit_code"] != 0:
            break
    (run / "outputs/asset/job_status.json").write_text(json.dumps(results, indent=2))
    return dict(job="asset", stages=len(results), final_exit=results[-1]["exit_code"])


def forge_job(run, item):
    dataset, arch, epochs = item
    job = f"{dataset}_{arch}"
    output = run / "outputs" / job
    output.mkdir()
    results = []
    for mode in ["train", "predict", "repeat", "untrained", "changed_input"]:
        args = ["python3", "-B", "/app/code/qualify_forge_runtime.py", mode,
                "--dataset", dataset, "--arch", arch, "--epochs", str(epochs)]
        if mode == "changed_input":
            ext = ".txt" if dataset.startswith("CMAPSS") else ".npz"
            args += ["--input", f"/app/datasets/{dataset}/changed_input{ext}"]
        result = execute(run, job + "_" + mode,
                         numeric_namespace(run, run / "public", output, args),
                         environment(run))
        results.append(dict(stage=mode, **result))
        if result["exit_code"] != 0:
            break
    (output / "job_status.json").write_text(json.dumps(results, indent=2))
    return dict(job=job, stages=len(results), final_exit=results[-1]["exit_code"])


def main(run):
    code = run / "public/code"
    code.mkdir(exist_ok=True)
    for name in ["real_rul_model.py", "real_fault_model.py", "verified_phmforge.py",
                 "qualify_forge_runtime.py", "verified_asset_audit.py", "verified_asset_forecast.py",
                 "qualify_asset_runtime.py"]:
        shutil.copy2(SCRIPTS / name, code / name)
    (run / "audit/QUALIFICATION_MATRIX240.json").write_text(json.dumps(dict(
        forge=MATRIX, asset="native 4 epochs, 1e-4 learning rate, chronological development split",
        concurrency=4, cpus_per_job=2, per_stage_timeout_seconds=900,
        criteria=["parameters updated", "cold reload equal", "weights affect prediction",
                  "changed observations affect prediction", "hidden gold not mounted"],
        performance="report independent errors; no invented pass range or test tuning"), indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(asset_job, run)] + [pool.submit(forge_job, run, x) for x in MATRIX]
        for future in futures:
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run.resolve())
