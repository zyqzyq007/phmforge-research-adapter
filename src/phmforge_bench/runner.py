"""Build real reference submissions and score frozen programs in a Linux sandbox."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .data import file_sha256, write_json
from verified_benchmark_scoring import classification, regression


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ready(root):
    root = Path(root).resolve()
    meta = json.loads((root / "READY.json").read_text())
    if meta.get("status") != "prepared" or meta.get("dataset") not in ("MFPT", "CMAPSS_FD001"):
        raise ValueError("Expected a prepared, supported source dataset")
    return root, meta


def workspace(data_root, destination):
    """Expose permitted training/development material, not holdout files or labels."""
    root, meta = _ready(data_root)
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if dest.is_relative_to(root) or root.is_relative_to(dest):
        raise ValueError("Agent workspace and private data root must be separate directories")
    dest.mkdir(parents=True)
    shutil.copytree(root / "public", dest / "data")
    dataset = meta["dataset"]
    task = ("Predict healthy (0), inner-race fault (1), or outer-race fault (2) from actual MFPT vibration features. "
            "Records, rather than windows, are the split unit; do not infer independent physical bearings.") if dataset == "MFPT" else (
            "Predict remaining cycles at each FD001 test engine's final observed cycle. Source is NASA simulation, not field data. "
            "Training and development engines are disjoint; use only past observations for each query.")
    input_contract = ("The input is an NPZ with `features` (N x 15) and string `sample_ids` (plus a dataset identifier)."
                      if dataset == "MFPT" else
                      "The input is the original 26-column FD001 format; column 1 is a pseudonymous integer engine ID, column 2 is cycle, then settings and sensors.")
    text = f"""# PHMForge research-adapted task: {meta['task_id']}

{task}

{input_contract} Test identities are random, not original file names or original NASA unit numbers; never assume IDs are consecutive.

`data/` contains permitted training and development material. Final test observations and labels are not supplied during development.
Produce a self-contained directory containing `solution.py` and any needed fitted model files. It will run as:

`python /app/solution.py --input /input/observations --output /output/predictions.json`

Write JSON with a `predictions` list. For MFPT each row needs `sample_id` and integer `predicted_label` (0, 1, or 2).
For FD001 each row needs integer `unit_id` and finite nonnegative `predicted_rul` in cycles.
Return exactly one prediction for each input identity. Use only the given training/development material to fit and select models.
The test input, labels, original filename mappings and final scores are unavailable until external scoring.
Use the supplied source metadata and report assumptions; do not invent physical risk, maintenance, or compliance labels.
"""
    (dest / "TASK.md").write_text(text, encoding="utf-8")
    write_json(dest / "WORKSPACE.json", {"task_id": meta["task_id"], "dataset": dataset,
        "allowed_information": "training/development data only", "contains_final_labels": False,
        "contains_final_observations": False, "created_utc": _now()})
    return {"status": "prepared", "task_id": meta["task_id"], "workspace": str(dest)}


def train_reference(data_root, destination, epochs=10):
    """Fit a genuine CPU MLP; the resulting submission is evaluated separately."""
    root, meta = _ready(data_root)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be positive")
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if dest.is_relative_to(root) or root.is_relative_to(dest):
        raise ValueError("Submission and private data root must be separate directories")
    dest.mkdir(parents=True)
    dataset = root / "public/datasets" / meta["dataset"]
    if meta["dataset"] == "MFPT":
        import real_fault_model as model
        audit = model.train(dataset, dest / "model", "mlp", epochs)
        module = "real_fault_model"
    else:
        import real_rul_model as model
        audit = model.train(dataset, dest / "model", "mlp", epochs_override=epochs)
        module = "real_rul_model"
    if not (audit["optimizer_steps"] > 0 and audit["nonzero_gradient_steps"] > 0 and audit["changed_parameter_tensors"] > 0):
        raise ValueError("Reference did not perform a real parameter update")
    (dest / "solution.py").write_text(f'''"""Fitted MLP reference for a research-adapted PHM task."""
import argparse
from pathlib import Path
from {module} import infer
p = argparse.ArgumentParser()
p.add_argument('--input', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
infer(Path(__file__).resolve().parent / 'model/model.pt', a.input, a.output)
''', encoding="utf-8")
    write_json(dest / "SUBMISSION.json", {"task_id": meta["task_id"], "dataset": meta["dataset"],
        "kind": "genuine fitted MLP reference", "epochs": epochs, "optimizer_steps": audit["optimizer_steps"],
        "nonzero_gradient_steps": audit["nonzero_gradient_steps"],
        "changed_parameter_tensors": audit["changed_parameter_tensors"], "final_labels_read_during_fit": False})
    return {"status": "trained", "submission": str(dest), "optimizer_steps": audit["optimizer_steps"],
        "nonzero_gradient_steps": audit["nonzero_gradient_steps"], "changed_parameter_tensors": audit["changed_parameter_tensors"]}


def _tree_hashes(directory):
    directory = Path(directory)
    values = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Submission symlinks are not supported: {path}")
        if path.is_file():
            values.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    if not values:
        raise ValueError("Empty submission")
    return values


def _sandbox_command(submission, data_root, holdout, output):
    """Expose selected runtime and inputs, never raw source or evaluation labels."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise FileNotFoundError("bubblewrap (bwrap) is required for isolated external evaluation")
    args = [bwrap, "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
            "--die-with-parent", "--new-session", "--clearenv"]
    system = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc") if Path(p).exists()]
    mounted = set()
    def bind(path):
        path = Path(os.path.abspath(path))
        resolved = path.resolve()
        if str(path) in mounted:
            return
        if resolved == Path("/") or Path(data_root).is_relative_to(resolved) or resolved.is_relative_to(data_root):
            raise ValueError(f"Unsafe runtime mount would expose the evaluation data root: {path}")
        args.extend(["--ro-bind", str(path), str(path)])
        mounted.add(str(path))
    for p in system:
        bind(p)
    for p in (Path(sys.prefix), Path(sys.base_prefix)):
        if p.exists(): bind(p)
    extra = []
    for value in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not value:
            continue
        path = Path(value).resolve()
        if not path.is_dir():
            raise ValueError(f"PYTHONPATH entry is not a directory: {path}")
        bind(path)
        extra.append(str(path))
    tools = Path(__file__).resolve().parents[1]
    # Preserve the venv interpreter path. Resolving its symlink would make Python
    # start outside the venv and silently lose the freshly installed packages.
    executable = Path(os.path.abspath(sys.executable))
    # If the executable is a system or venv interpreter, its prefix is already mounted.
    args += ["--ro-bind", str(tools), "/tools", "--ro-bind", str(submission), "/app",
             "--dir", "/data", "--dir", "/input",
             "--ro-bind", str(holdout), "/input/observations", "--bind", str(output), "/output",
             "--proc", "/proc", "--dev", "/dev", "--chdir", "/app"]
    environment = {"HOME": "/output/home", "PATH": "/usr/bin:/bin", "PYTHONPATH": os.pathsep.join(["/tools"] + extra),
        "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": "/output/tmp", "TMP": "/output/tmp", "TEMP": "/output/tmp",
        "MPLCONFIGDIR": "/output/mpl", "XDG_CACHE_HOME": "/output/xdg", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    for k, v in environment.items():
        args += ["--setenv", k, v]
    return args + [str(executable), "-B", "/app/solution.py", "--input", "/input/observations",
                   "--output", "/output/predictions.json"]


def _recording_metrics(predictions, gold):
    p = {str(x["sample_id"]): x["predicted_label"] for x in predictions}
    by_group, by_load = defaultdict(list), defaultdict(list)
    for row in gold:
        ok = p[str(row["sample_id"])] == row["label"]
        by_group[row["group"]].append((p[str(row["sample_id"])], row["label"], ok))
        by_load[str(row["load_lb"])].append(ok)
    groups = []
    for name, rows in sorted(by_group.items()):
        labels = {r[1] for r in rows}
        if len(labels) != 1:
            raise ValueError(f"A native recording has inconsistent labels: {name}")
        votes = Counter(r[0] for r in rows)
        maximum = max(votes.values())
        winners = [k for k, n in votes.items() if n == maximum]
        groups.append({"recording": name, "windows": len(rows), "actual": next(iter(labels)),
            "majority_prediction": winners[0] if len(winners) == 1 else None,
            "correct": len(winners) == 1 and winners[0] == next(iter(labels)), "tie": len(winners) != 1})
    return {"recordings": groups, "recording_accuracy": sum(x["correct"] for x in groups) / len(groups),
            "by_load_lb": {k: {"count": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_load.items())}}


def evaluate(data_root, submission, destination, timeout=120):
    """Freeze a submission, run it without network/labels, and score by exact ID."""
    root, meta = _ready(data_root)
    submission, dest = Path(submission).resolve(), Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if not (submission / "solution.py").is_file():
        raise FileNotFoundError("Submission must contain solution.py")
    if root.is_relative_to(submission) or submission.is_relative_to(root):
        raise ValueError("Submission and private data root must be separate directories")
    if dest.is_relative_to(root) or root.is_relative_to(dest) or dest.is_relative_to(submission) or submission.is_relative_to(dest):
        raise ValueError("Evaluation output must be separate from data and submission directories")
    dest.mkdir(parents=True)
    for folder in ("tmp", "mpl", "xdg", "home"):
        (dest / folder).mkdir()
    frozen = _tree_hashes(submission)
    write_json(dest / "submission_manifest.json", {"frozen_utc": _now(), "files": frozen})
    holdout = root / "holdout" / ("test.npz" if meta["dataset"] == "MFPT" else "test.txt")
    command = _sandbox_command(submission, root, holdout, dest)
    started = time.monotonic()
    try:
        with (dest / "stdout.log").open("wb") as out, (dest / "stderr.log").open("wb") as err:
            proc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, check=False)
        if proc.returncode:
            raise RuntimeError(f"Isolated prediction exited with status {proc.returncode}; inspect stderr.log")
        if _tree_hashes(submission) != frozen:
            raise RuntimeError("Submission changed after freezing")
        predictions_path = dest / "predictions.json"
        if predictions_path.is_symlink() or not predictions_path.is_file() or predictions_path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("Expected a regular prediction JSON file of at most 50 MiB")
        value = json.loads(predictions_path.read_text())
        rows = value.get("predictions") if isinstance(value, dict) else value
        gold = json.loads((root / "private/labels.json").read_text())
        if meta["dataset"] == "MFPT":
            score = classification(rows, gold, declared_classes=[0, 1, 2])
            score.update(_recording_metrics(rows, gold))
        else:
            score = regression(rows, gold)
            if any(float(row["predicted_rul"]) < 0 for row in rows):
                raise ValueError("RUL predictions must be nonnegative")
        score.update(task_id=meta["task_id"], dataset=meta["dataset"],
            source_scope=meta["scope"], execution_isolation="bubblewrap: no network, training/development data, private labels or raw sources mounted",
            prediction_sha256=file_sha256(dest / "predictions.json"), truth_sha256=file_sha256(root / "private/labels.json"),
            holdout_input_sha256=file_sha256(holdout), elapsed_seconds=time.monotonic() - started,
            official_benchmark_score=False)
        write_json(dest / "score.json", score)
        write_json(dest / "STATUS.json", {"status": "completed", "task_id": meta["task_id"], "scored_at_utc": _now()})
        return {"status": "completed", "task_id": meta["task_id"], "score": str(dest / "score.json"),
                "n": score["n"], "accuracy": score.get("accuracy"), "mae": score.get("mae")}
    except Exception as exc:
        write_json(dest / "STATUS.json", {"status": "failed", "task_id": meta["task_id"],
            "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        raise
