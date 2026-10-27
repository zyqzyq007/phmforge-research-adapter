"""Start a qualified profile over MCP/JSONL, with a private writable workspace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def command(runtime, workspace, family, transport, couch_resource=None):
    runtime, workspace = Path(runtime).resolve(), Path(workspace).resolve()
    base = Path("/srv/yyf-zyq/PHM_20261010")
    if not workspace.is_relative_to(base) or workspace.is_relative_to(base/"_shared"):
        raise ValueError("Writable job workspace must be under the managed /srv run directory")
    if not (runtime/"READY.json").is_file():
        raise FileNotFoundError("Verified runtime installation is missing READY.json")
    workspace.mkdir(parents=True, exist_ok=True)
    for x in ["tmp", "mpl", "xdg", "hf"]:
        (workspace/".runtime"/x).mkdir(parents=True, exist_ok=True)
    python = Path("/home/jupyter-yyf-zyq/.local/bin/python3.12").resolve()
    argv = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent", "--new-session", "--clearenv"]
    if family != "asset_current":
        argv += ["--unshare-net"]
    for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", str(python.parent.parent)]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    bindings = [(runtime/"deps","/deps"), (runtime/"code","/code"), (runtime/"public","/app"),
                (runtime/"public/asset","/data"), (runtime/"models","/models")]
    if family == "asset_current":
        bindings += [(runtime/"asset_current/servers","/opt/asset/servers"),
                     (runtime/"asset_current/llm","/opt/asset/llm")]
    elif family == "asset_legacy":
        bindings += [(runtime/"asset_legacy/servers/tsfm","/opt/asset/servers/tsfm")]
    for src, dst in bindings:
        argv += ["--ro-bind",str(src),dst]
    argv += ["--bind",str(workspace),"/output","--proc","/proc","--dev","/dev","--chdir","/output"]
    env = dict(PATH="/usr/bin:/bin", PYTHONPATH="/deps:/code:/opt/asset", PYTHONDONTWRITEBYTECODE="1",
               TMPDIR="/output/.runtime/tmp", TMP="/output/.runtime/tmp", TEMP="/output/.runtime/tmp",
               XDG_CACHE_HOME="/output/.runtime/xdg", MPLCONFIGDIR="/output/.runtime/mpl",
               HF_HOME="/output/.runtime/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TSFM_WORKDIR="/output/artifacts", PATH_TO_MODELS_DIR="/models", PATH_TO_DATASETS_DIR="/data",
               PATH_TO_OUTPUTS_DIR="/output", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               FMSR_MODEL_ID="tokenrouter/deepseek-flash")
    if family == "asset_current":
        if not couch_resource:
            raise ValueError("A private CouchDB resource is required; do not share mutable scenario state")
        data = json.loads(Path(couch_resource).read_text())
        env.update(COUCHDB_URL=data["url"], COUCHDB_USERNAME="phm240",
                   COUCHDB_PASSWORD="isolated-benchmark-only", TSFM_STORE="couch")
    for k,v in env.items():
        argv += ["--setenv",k,v]
    worker_family = {"forge":"forge","asset_current":"asset","asset_legacy":"asset_tsfm"}[family]
    return argv + ["--remount-ro","/",str(python),"-B","/code/verified_benchmark_worker.py",
                   "--family",worker_family,"--data","/app","--output","/output","--transport",transport]


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--family",choices=["forge","asset_current","asset_legacy"],required=True)
    p.add_argument("--transport",choices=["mcp","jsonl"],default="mcp")
    p.add_argument("--couch-resource",type=Path)
    args=p.parse_args()
    raise SystemExit(subprocess.call(command(args.runtime,args.workspace,args.family,args.transport,args.couch_resource)))
