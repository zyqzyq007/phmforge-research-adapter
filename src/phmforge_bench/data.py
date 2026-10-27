"""Acquire pinned public sources and make split-aware benchmark inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import secrets
import ssl
import time
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat
from .io import file_sha256, write_json


NASA_URL = "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1"
NASA_ARCHIVE_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
NASA_MEMBERS = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}


def resource_json(name):
    return json.loads(files("phmforge_bench.resources").joinpath(name).read_text(encoding="utf-8"))


def download(url, path, expected_hash, expected_size=None):
    """Stream a pinned public file with bounded retries; retain failed partials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != expected_hash or (expected_size is not None and path.stat().st_size != expected_size):
            raise ValueError(f"Cached source does not match the pinned digest: {path}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "PHMForge-research-benchmark/0.2", "Accept-Encoding": "identity"})
    ca = Path("/etc/ssl/certs/ca-certificates.crt")
    context = ssl.create_default_context(cafile=str(ca)) if ca.exists() else ssl.create_default_context()
    last_error = None
    for attempt in range(3):
        serial = 1
        while path.with_suffix(path.suffix + f".partial.{serial}").exists():
            serial += 1
        partial = path.with_suffix(path.suffix + f".partial.{serial}")
        count, h = 0, hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=90, context=context) as response, partial.open("xb") as target:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(block)
                    if count > (expected_size if expected_size is not None else 32 * 1024 * 1024):
                        raise ValueError(f"Download exceeds expected size: {url}")
                    h.update(block)
                    target.write(block)
            if h.hexdigest() != expected_hash or (expected_size is not None and count != expected_size):
                raise ValueError(f"Downloaded source failed hash/size verification: {url}")
            partial.rename(path)
            return
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Source download failed after three attempts: {url}; last error: {last_error}") from last_error


def features(x):
    """The original A306 seven time statistics and eight spectral bands."""
    centered = x - x.mean(axis=0)
    variance = np.square(centered).mean(axis=0)
    scale = np.sqrt(variance)
    moments = [x.mean(0), scale, np.sqrt(np.square(x).mean(0)),
               np.max(np.abs(x), axis=0), np.ptp(x, axis=0),
               np.mean(centered**3, axis=0) / np.maximum(scale, 1e-12)**3,
               np.mean(centered**4, axis=0) / np.maximum(variance, 1e-12)**2]
    spectrum = np.abs(np.fft.rfft(centered, axis=0))**2 / len(x)**2
    bands = [np.log1p(band.sum(axis=0)) for band in np.array_split(spectrum[1:], 8)]
    return np.concatenate(moments + bands).astype(np.float32)


def _catalog(destination):
    text = files("phmforge_bench.resources").joinpath("tool_catalog.json").read_text(encoding="utf-8")
    (destination / "tool_catalog.json").write_text(text, encoding="utf-8")


def prepare_mfpt(destination, source=None, workers=4):
    """Create fixed train/validation/holdout splits from 20 native measured files."""
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    manifest = resource_json("mfpt_manifest.json")
    dest.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source).resolve() if source else dest / "raw"
    if not source:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
            jobs = [pool.submit(download, row["url"], source_dir / row["path"], row["sha256"], row["bytes"]) for row in manifest["files"]]
            for job in as_completed(jobs):
                job.result()
    for row in manifest["files"]:
        path = source_dir / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"MFPT source file missing or mismatched: {path}")
    write_json(dest / "SOURCE_BINDINGS.json", {"repository": manifest["source_repository"],
        "source_tree": manifest["source_tree"], "files": manifest["files"], "sha256_verified": True})

    def label(path):
        name = path.name
        if name.startswith("InnerRace"): return 1
        if name.startswith("OuterRace"): return 2
        if name.startswith("baseline_"): return 0
        raise ValueError(f"Unexpected source filename: {name}")

    native = sorted((source_dir / row["path"] for row in manifest["files"]), key=lambda p: p.as_posix())
    development = {min(p for p in native if p.parent.name == "train_data" and label(p) == c) for c in (0, 1, 2)}
    split_rows = {"train": [], "val": [], "test": []}
    group_sets = {"train": set(), "val": set(), "test": set()}
    holdout_identity_key = secrets.token_bytes(32)
    for path in native:
        split = "test" if path.parent.name == "test_data" else "val" if path in development else "train"
        source_rel = f"{path.parent.name}/{path.name}"
        group_sets[split].add(source_rel)
        bearing = loadmat(path, simplify_cells=True)["bearing"]
        signal = np.asarray(bearing["gs"], dtype=float).reshape(-1)
        load = float(bearing["load"])
        sample_rate, rotation = float(bearing["sr"]), float(bearing["rate"])
        if not np.isfinite(signal).all() or len(signal) < 2560 or sample_rate not in (48828, 97656) or rotation != 25 or load < 0:
            raise ValueError(f"Invalid source observation or metadata: {source_rel}")
        for start in range(0, len(signal) - 2560 + 1, 2560):
            identity_input = f"mfpt-native-v1:{source_rel}:{start}".encode()
            identity = (hmac.new(holdout_identity_key, identity_input, hashlib.sha256).hexdigest()[:24]
                        if split == "test" else hashlib.sha256(identity_input).hexdigest()[:24])
            split_rows[split].append(dict(sample_id=identity, features=features(signal[start:start + 2560, None]),
                label=label(path), group=source_rel, load_lb=load, sampling_hz=sample_rate, shaft_frequency_hz=rotation))
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("A source recording appears in multiple splits")
    public = dest / "public"
    dataset = public / "datasets/MFPT"
    dataset.mkdir(parents=True)
    holdout = dest / "holdout"
    holdout.mkdir()
    private = dest / "private"
    private.mkdir()
    (private / "holdout_identity_key.hex").write_text(holdout_identity_key.hex() + "\n")
    for split, rows in split_rows.items():
        payload = {"features": np.asarray([r["features"] for r in rows], dtype=np.float32),
                   "sample_ids": np.asarray([r["sample_id"] for r in rows]), "dataset": np.asarray("MFPT")}
        if split != "test":
            payload["labels"] = np.asarray([r["label"] for r in rows], dtype=int)
            payload["groups"] = np.asarray([r["group"] for r in rows])
            np.savez_compressed(dataset / f"{split}.npz", **payload)
        else:
            np.savez_compressed(holdout / "test.npz", **payload)
            write_json(private / "labels.json", [{k: v for k, v in r.items() if k != "features"} for r in rows])
    info = {"dataset": "MFPT", "seed": 2026102306, "class_names": [0, 1, 2], "architectures": ["mlp"],
        "group_definition": "Whole native recording; recordings may share physical hardware.",
        "source_url": manifest["source_repository"], "source_tree": manifest["source_tree"],
        "data_license": manifest["source_license"], "measured_or_simulated": "measured laboratory recordings",
        "preprocessing": "Nonoverlapping 2560-sample windows; 7 time and 8 FFT-band features.",
        "train_rows": len(split_rows["train"]), "validation_rows": len(split_rows["val"]),
        "label_definition": {"0": "healthy", "1": "inner race", "2": "outer race"}}
    write_json(dataset / "dataset.json", info)
    write_json(public / "dataset_registry.json", {"MFPT": {"task": "classification", "path": "datasets/MFPT",
        "architectures": ["mlp"], "data_origin": info["measured_or_simulated"], "available_splits": ["train", "val"]}})
    _catalog(public)
    report = {"task_id": "pdm_fault_008", "dataset": "MFPT", "status": "prepared", "source": manifest["source_repository"],
        "source_revision": manifest["source_tree"], "rows": {k: len(v) for k, v in split_rows.items()},
        "recordings": {k: len(v) for k, v in group_sets.items()}, "group_overlap": False,
        "labels_in_test_input": False, "holdout_ids": "random private HMAC; different for each preparation",
        "scope": "research adaptation; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    return report


def prepare_cmapss(destination, archive=None):
    """Bind NASA FD001 train/test to original uncapped RUL labels."""
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(archive).resolve() if archive else dest / "raw/CMAPSSData.zip"
    if not archive:
        download(NASA_URL, source, NASA_ARCHIVE_SHA256)
    if not source.is_file() or file_sha256(source) != NASA_ARCHIVE_SHA256:
        raise ValueError(f"NASA archive missing or digest mismatch: {source}")
    public, holdout, private = dest / "public", dest / "holdout", dest / "private"
    data = public / "datasets/CMAPSS_FD001"
    data.mkdir(parents=True)
    holdout.mkdir()
    private.mkdir()
    with zipfile.ZipFile(source) as z:
        selected = {}
        for name, expected in NASA_MEMBERS.items():
            matches = [n for n in z.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one NASA archive member: {name}")
            raw = z.read(matches[0])
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"NASA member hash mismatch: {name}")
            selected[name] = raw
    (data / "train.txt").write_bytes(selected["train_FD001.txt"])
    config = resource_json("cmapss_fd001_config.json")
    write_json(data / "training_config.json", config)
    targets = np.fromstring(selected["RUL_FD001.txt"].decode(), sep=" ")
    if len(targets) != 100 or not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Invalid official FD001 endpoint labels")
    pseudonyms = secrets.SystemRandom().sample(range(1000000, 2000000), len(targets))
    mapping = {unit: pseudonyms[unit - 1] for unit in range(1, len(targets) + 1)}
    lines, seen = [], set()
    for line in selected["test_FD001.txt"].decode("ascii").splitlines(keepends=True):
        values = line.split(maxsplit=1)
        if len(values) != 2:
            raise ValueError("Invalid original FD001 test row")
        unit = int(values[0])
        if unit not in mapping:
            raise ValueError("Unexpected original FD001 test engine identity")
        seen.add(unit)
        lines.append(f"{mapping[unit]} {values[1]}")
    if seen != set(mapping):
        raise ValueError("Original FD001 test engine coverage differs from labels")
    (holdout / "test.txt").write_text("".join(lines), encoding="ascii")
    write_json(private / "holdout_identity_mapping.json", mapping)
    write_json(private / "labels.json", [{"unit_id": mapping[n + 1], "actual_rul": float(v)} for n, v in enumerate(targets)])
    write_json(public / "dataset_registry.json", {"CMAPSS_FD001": {"task": "rul", "path": "datasets/CMAPSS_FD001",
        "architectures": ["mlp", "lstm", "transformer"], "data_origin": "NASA C-MAPSS simulator, not field measurements",
        "available_splits": ["train", "val"]}})
    _catalog(public)
    report = {"task_id": "pdm_rul_001", "dataset": "CMAPSS_FD001", "status": "prepared",
        "source": "NASA C-MAPSS via the mirror used by nasa/progpy", "archive_sha256": NASA_ARCHIVE_SHA256,
        "train_engines": len(config["train_engine_ids"]), "validation_engines": len(config["validation_engine_ids"]),
        "test_engines": 100, "group_overlap": False, "measured_or_simulated": "simulated",
        "holdout_ids": "private randomized engine pseudonyms; different for each preparation",
        "scope": "research adaptation; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    return report
