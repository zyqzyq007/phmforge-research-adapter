"""Acquire pinned public sources and make split-aware benchmark inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
import hashlib
import hmac
import http.client
import json
from pathlib import Path
import random
import re
import shutil
import secrets
import ssl
import sys
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
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}
NASA_TEST_ENGINES = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}


def resource_json(name):
    return json.loads(files("phmforge_bench.resources").joinpath(name).read_text(encoding="utf-8"))


def _download_ranges(url, path, expected_hash, expected_size, chunk_size=16 * 1024 * 1024):
    """Resume a large pinned source using checked ranges and an atomic final rename."""
    partial = path.with_name(path.name + ".partial.resume")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError(f"Existing partial exceeds the pinned source size: {partial}")
    ca = Path("/etc/ssl/certs/ca-certificates.crt")
    context = ssl.create_default_context(cafile=str(ca)) if ca.exists() else ssl.create_default_context()
    failures = 0
    while not partial.exists() or partial.stat().st_size < expected_size:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(start + chunk_size, expected_size) - 1
        request = urllib.request.Request(url, headers={
            "User-Agent": "PHMForge-research-benchmark/0.4", "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=90, context=context) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    bounds = tuple(map(int, match.groups())) if match else None
                    if not bounds or bounds[0] != start or not start <= bounds[1] <= end or bounds[2] != expected_size:
                        raise ValueError(f"Server returned the wrong byte range for {url}")
                    expected_count = bounds[1] - start + 1
                elif response.status == 200 and start == 0:
                    # Some public servers ignore Range; a complete initial
                    # response remains acceptable but cannot be resumed.
                    expected_count = expected_size
                else:
                    raise ValueError(f"Server cannot resume the requested source range for {url}")
                received = 0
                with partial.open("ab") as target:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        received += len(block)
                        if received > expected_count:
                            raise ValueError(f"Server exceeded the requested byte range for {url}")
                        target.write(block)
                if received != expected_count:
                    raise OSError(f"Incomplete source range ({received}/{expected_count} bytes)")
            failures = 0
            completed = partial.stat().st_size
            if completed == expected_size or completed * 10 // expected_size > start * 10 // expected_size:
                print(f"Downloading {path.name}: {completed}/{expected_size} bytes", file=sys.stderr, flush=True)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            failures += 1
            if failures >= 3:
                raise RuntimeError(f"Source transfer interrupted; partial file retained at {partial} for later verification: {exc}") from exc
            time.sleep(2 ** (failures - 1))
    if partial.stat().st_size != expected_size or file_sha256(partial) != expected_hash:
        raise ValueError(f"Complete downloaded source failed pinned size/SHA-256 verification: {partial}")
    partial.rename(path)


def download(url, path, expected_hash, expected_size=None):
    """Stream a pinned public file with bounded retries; retain failed partials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != expected_hash or (expected_size is not None and path.stat().st_size != expected_size):
            raise ValueError(f"Cached source does not match the pinned digest: {path}")
        return
    if expected_size is not None and expected_size > 32 * 1024 * 1024:
        return _download_ranges(url, path, expected_hash, expected_size)
    request = urllib.request.Request(url, headers={"User-Agent": "PHMForge-research-benchmark/0.4", "Accept-Encoding": "identity"})
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
    # HMAC identifiers alone do not hide source identity if rows remain in
    # the same publicly known source-file order.
    secrets.SystemRandom().shuffle(split_rows["test"])
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


def prepare_cmapss(destination, archive=None, subset="FD001"):
    """Bind one NASA C-MAPSS subset to its original uncapped RUL labels."""
    subset = subset.upper()
    if subset not in NASA_TEST_ENGINES:
        raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
    dataset_name = f"CMAPSS_{subset}"
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
    data = public / "datasets" / dataset_name
    data.mkdir(parents=True)
    holdout.mkdir()
    private.mkdir()
    with zipfile.ZipFile(source) as z:
        selected = {}
        for name in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
            expected = NASA_MEMBERS[name]
            matches = [n for n in z.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one NASA archive member: {name}")
            raw = z.read(matches[0])
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"NASA member hash mismatch: {name}")
            selected[name] = raw
    (data / "train.txt").write_bytes(selected[f"train_{subset}.txt"])
    config = resource_json("cmapss_fd001_config.json")
    if subset != "FD001":
        # Select by complete engine, never individual windows. These source
        # IDs are training identities, not the held-out NASA test identities.
        import io
        raw_train = np.loadtxt(io.BytesIO(selected[f"train_{subset}.txt"]))
        if raw_train.ndim != 2 or raw_train.shape[1] != 26 or not np.isfinite(raw_train).all():
            raise ValueError(f"Invalid original {subset} training observations")
        ids = sorted(set(raw_train[:, 0].astype(int).tolist()))
        if not ids or set(ids) != set(range(1, len(ids) + 1)):
            raise ValueError(f"Unexpected original {subset} training engine identities")
        random.Random(config["seed"]).shuffle(ids)
        boundary = int(len(ids) * .8)
        config["train_engine_ids"] = sorted(ids[:boundary])
        config["validation_engine_ids"] = sorted(ids[boundary:])
    write_json(data / "training_config.json", config)
    targets = np.fromstring(selected[f"RUL_{subset}.txt"].decode(), sep=" ")
    if len(targets) != NASA_TEST_ENGINES[subset] or not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError(f"Invalid official {subset} endpoint labels")
    pseudonyms = secrets.SystemRandom().sample(range(1000000, 2000000), len(targets))
    mapping = {unit: pseudonyms[unit - 1] for unit in range(1, len(targets) + 1)}
    lines, seen = {}, set()
    for line in selected[f"test_{subset}.txt"].decode("ascii").splitlines(keepends=True):
        values = line.split(maxsplit=1)
        if len(values) != 2:
            raise ValueError(f"Invalid original {subset} test row")
        unit = int(values[0])
        if unit not in mapping:
            raise ValueError(f"Unexpected original {subset} test engine identity")
        seen.add(unit)
        lines.setdefault(unit, []).append(f"{mapping[unit]} {values[1]}")
    if seen != set(mapping):
        raise ValueError(f"Original {subset} test engine coverage differs from labels")
    order = list(lines)
    secrets.SystemRandom().shuffle(order)
    (holdout / "test.txt").write_text("".join(line for unit in order for line in lines[unit]), encoding="ascii")
    write_json(private / "holdout_identity_mapping.json", mapping)
    write_json(private / "labels.json", [{"unit_id": mapping[n + 1], "actual_rul": float(v)} for n, v in enumerate(targets)])
    write_json(public / "dataset_registry.json", {dataset_name: {"task": "rul", "path": f"datasets/{dataset_name}",
        "architectures": ["mlp", "lstm", "transformer"], "data_origin": "NASA C-MAPSS simulator, not field measurements",
        "available_splits": ["train", "val"]}})
    _catalog(public)
    task_id = "pdm_rul_001" if subset == "FD001" else f"research_{subset.lower()}_rul"
    report = {"task_id": task_id, "dataset": dataset_name, "task_kind": "rul", "status": "prepared",
        "source": "NASA C-MAPSS via the mirror used by nasa/progpy", "archive_sha256": NASA_ARCHIVE_SHA256,
        "train_engines": len(config["train_engine_ids"]), "validation_engines": len(config["validation_engine_ids"]),
        "test_engines": len(targets), "group_overlap": False, "measured_or_simulated": "simulated",
        "holdout_ids": "private randomized engine pseudonyms; different for each preparation",
        "scope": "research adaptation; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    return report
