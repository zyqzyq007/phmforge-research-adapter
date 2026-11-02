"""Prepare a real PRONOSTIA/FEMTO bearing RUL task from public sources.

Only the original learning runs and observed challenge prefixes are used.
The challenge PDF and full source archive remain outside the public workspace.
"""
import hashlib
import hmac
import io
from pathlib import Path
import re
import secrets
import shutil
import zipfile

import numpy as np
from pypdf import PdfReader

from .data import _catalog, download
from .io import file_sha256, write_json


FEMTO_URL = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
FEMTO_SHA256 = "e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f"
FEMTO_BYTES = 1157035288
CHALLENGE_URL = ("https://raw.githubusercontent.com/aref9069/"
                 "IEEE-PHM-2012-Prognostic-Challenge-/9c476de6405b8c31b53bba8d6b0955ea57172206/IEEEPHM2012-Challenge-Details.pdf")
CHALLENGE_SHA256 = "13bd43eb2dc4c8455e36584094552ccbd8101b9a0b99bd10ca82103453585ba5"
CHALLENGE_BYTES = 2279692
LEARNING = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
TEST = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3"]


def _features(raw):
    result = []
    for a in raw.T:
        mean = float(a.mean())
        centered = a - mean
        std = float(a.std())
        power = np.abs(np.fft.rfft(a)) ** 2
        probability = power / max(float(power.sum()), 1e-20)
        result.extend([mean, std, float(np.sqrt(np.mean(a * a))), float(np.max(np.abs(a))),
                       float(np.ptp(a)), float(np.mean(centered ** 3) / max(std, 1e-12) ** 3),
                       float(np.mean(centered ** 4) / max(std, 1e-12) ** 4),
                       float(np.mean(np.diff(np.signbit(a)) != 0)),
                       float(-(probability * np.log(np.maximum(probability, 1e-30))).sum())])
        result.extend(float(np.log1p(b.mean())) for b in np.array_split(power, 8))
    return result


def _copy_member(archive, member, destination):
    # Fixed destination filenames avoid treating archive member paths as paths
    # on the host. Reading the member to completion also verifies its ZIP CRC.
    with archive.open(member) as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def prepare_femto(destination, source=None, challenge_pdf=None):
    dest = Path(destination).resolve()
    if dest.exists() and (dest / "READY.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve() if source else dest / "raw/NASA_FEMTO.zip"
    challenge = Path(challenge_pdf).resolve() if challenge_pdf else dest / "raw/IEEEPHM2012-Challenge-Details.pdf"
    if not source.is_file() and source == dest / "raw/NASA_FEMTO.zip":
        download(FEMTO_URL, source, FEMTO_SHA256, FEMTO_BYTES)
    if not challenge.is_file() and challenge == dest / "raw/IEEEPHM2012-Challenge-Details.pdf":
        download(CHALLENGE_URL, challenge, CHALLENGE_SHA256, CHALLENGE_BYTES)
    for path, digest, size in ((source, FEMTO_SHA256, FEMTO_BYTES),
                               (challenge, CHALLENGE_SHA256, CHALLENGE_BYTES)):
        if not path.is_file() or path.stat().st_size != size or file_sha256(path) != digest:
            raise ValueError(f"FEMTO source missing or pinned digest does not match: {path}")

    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(challenge).pages)
    table = re.findall(r"(Bearing\d_\d)\s+(\d+)\s*s", pdf_text)
    annotations = {name: float(value) for name, value in table}
    if len(table) != len(TEST) or set(annotations) != set(TEST):
        raise ValueError("The pinned challenge document does not contain exactly the expected 11 RUL annotations")

    private = dest / "private"
    private.mkdir()
    scratch = private / "extraction"
    scratch.mkdir()
    # The outer NASA ZIP contains another ZIP containing three ZIPs. Do not
    # extract the validation continuation ZIP into the Agent workspace.
    inner = scratch / "dataset.zip"
    with zipfile.ZipFile(source) as outer:
        candidates = [item for item in outer.infolist() if Path(item.filename).name == "FEMTOBearingDataSet.zip"]
        if len(candidates) != 1:
            raise ValueError("Expected one NASA FEMTO dataset archive")
        _copy_member(outer, candidates[0], inner)
    with zipfile.ZipFile(inner) as archive:
        for name in ("Training_set.zip", "Test_set.zip"):
            matches = [item for item in archive.infolist() if Path(item.filename).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one FEMTO {name}")
            _copy_member(archive, matches[0], scratch / name)

    public = dest / "public"
    data = public / "datasets/FEMTO"
    data.mkdir(parents=True)
    (dest / "holdout").mkdir()
    identity_key = secrets.token_bytes(32)
    (private / "holdout_identity_key.hex").write_text(identity_key.hex() + "\n")
    test_ids = secrets.SystemRandom().sample(range(1000000, 2000000), len(TEST))
    test_mapping = dict(zip(TEST, test_ids))
    training, validation, heldout, gold, audit = [], [], [], [], []
    for is_learning, bearings in ((True, LEARNING), (False, TEST)):
        with zipfile.ZipFile(scratch / ("Training_set.zip" if is_learning else "Test_set.zip")) as archive:
            for bearing in bearings:
                paths = sorted(item for item in archive.namelist()
                               if re.search(r"/" + re.escape(bearing) + r"/acc_\d+\.csv$", item))
                if len(paths) < 31 or len(paths) != len(set(paths)):
                    raise ValueError(f"Incomplete or duplicated FEMTO source recordings: {bearing}")
                samples, logger = [], []
                for name in paths:
                    content = archive.read(name)
                    delimiter = "," if b"," in content.splitlines()[0] else ";"
                    raw = np.loadtxt(io.BytesIO(content), delimiter=delimiter)
                    if raw.shape != (2560, 6) or not np.isfinite(raw).all():
                        raise ValueError(f"Invalid FEMTO vibration source member: {name}")
                    logger.append(float(raw[0, :4] @ np.array([3600, 60, 1, 1e-6])))
                    samples.append(_features(raw[:, 4:]))
                base = np.asarray(samples)
                if base.shape != (len(paths), 34) or not np.isfinite(base).all():
                    raise ValueError(f"Invalid FEMTO source features: {bearing}")
                elapsed = np.arange(len(paths), dtype=np.float64) * 10
                vectors = []
                for i, row in enumerate(base):
                    history = base[max(0, i - 29):i + 1]
                    vectors.append(np.r_[row, history.mean(axis=0), row - history[0], elapsed[i] / 3600])
                scenario = int(bearing[7])
                unit = LEARNING.index(bearing) + 1 if is_learning else test_mapping[bearing]
                for i, vector in enumerate(vectors):
                    original_id = f"{bearing}:{i + 1}"
                    sample_id = (hashlib.sha256(original_id.encode()).hexdigest()[:24] if is_learning else
                                 hmac.new(identity_key, original_id.encode(), hashlib.sha256).hexdigest()[:24])
                    row = {"features": vector, "sample_id": sample_id, "unit_id": unit,
                           "scenario_id": scenario, "time_sec": float(elapsed[i])}
                    if is_learning:
                        row["label"] = float(elapsed[-1] - elapsed[i])
                        training.append(row)
                        if bearing in LEARNING[1::2] and i in {int((len(paths) - 1) * f) for f in (.3, .6, .9)}:
                            validation.append(row)
                    elif i == len(paths) - 1:
                        heldout.append(row)
                        gold.append({"sample_id": sample_id, "unit_id": unit, "scenario_id": scenario,
                                     "actual_rul": annotations[bearing], "bearing": bearing,
                                     "ambiguous_challenge_annotation": bearing == "Bearing1_4"})
                logger_elapsed = np.asarray(logger) - logger[0]
                logger_elapsed[logger_elapsed < 0] += 86400
                audit.append({"bearing": bearing, "unit_id": unit, "scenario_id": scenario,
                    "source_records": len(paths), "nominal_interval_seconds": 10,
                    "nonmonotonic_logger_steps": int((np.diff(logger_elapsed) <= 0).sum()),
                    "logger_vs_nominal_max_error_seconds": float(np.abs(logger_elapsed - elapsed).max())})

    def payload(rows, include_labels):
        value = {"dataset": np.asarray("FEMTO"),
                 "features": np.asarray([row["features"] for row in rows], dtype=np.float32),
                 "sample_ids": np.asarray([row["sample_id"] for row in rows]),
                 "unit_ids": np.asarray([row["unit_id"] for row in rows], dtype=np.int64),
                 "scenario_ids": np.asarray([row["scenario_id"] for row in rows], dtype=np.int64),
                 "time_sec": np.asarray([row["time_sec"] for row in rows], dtype=np.float64)}
        if include_labels:
            value["labels"] = np.asarray([row["label"] for row in rows], dtype=np.float32)
        return value
    if len(validation) != 9 or len(heldout) != len(TEST):
        raise ValueError("FEMTO development or held-out query coverage differs from the declared task")
    pairs = list(zip(heldout, gold))
    secrets.SystemRandom().shuffle(pairs)
    heldout, gold = map(list, zip(*pairs))
    np.savez_compressed(data / "train.npz", **payload(training, True))
    np.savez_compressed(data / "val.npz", **payload(validation, True))
    np.savez_compressed(dest / "holdout/test.npz", **payload(heldout, False))
    write_json(private / "labels.json", gold)
    write_json(private / "source_audit.json", {"bearings": audit, "test_pseudonyms": test_mapping})
    config = {"seed": 253, "epochs": 30, "batch_size": 128, "learning_rate": .001, "cpu_threads": 2,
              "fit_unit_ids": [1, 3, 5], "validation_unit_ids": [2, 4, 6],
              "validation_prefix_fractions": [.3, .6, .9], "feature_count": 103}
    write_json(data / "training_config.json", config)
    info = {"dataset": "FEMTO", "data_origin": "Original PRONOSTIA laboratory measurements from NASA",
            "architectures": ["mlp"], "label_unit": "seconds", "feature_count": 103,
            "source_url": FEMTO_URL, "source_sha256": FEMTO_SHA256,
            "preprocessing": "Original-scale dual-axis features plus causal 30-observation history; nominal 10-second acquisition interval.",
            "validation_scope": "One learning bearing per condition held out for epoch selection; final model refits both learning bearings.",
            "limitations": ["Six learning and eleven challenge bearings from one laboratory source.",
                            "Nominal time is used because two source logger clocks are nonmonotonic.",
                            "One challenge annotation conflicts with its later continuation; report 11 and consistent 10 separately."]}
    write_json(data / "dataset.json", info)
    write_json(public / "dataset_registry.json", {"FEMTO": {"task": "rul", "format": "femto_npz",
        "path": "datasets/FEMTO", "architectures": ["mlp"], "available_splits": ["train", "val"]}})
    _catalog(public)
    write_json(dest / "SOURCE_BINDINGS.json", {"archive_url": FEMTO_URL, "archive_sha256": FEMTO_SHA256,
        "archive_hash_scope": "locally established full-file pin; not an asserted publisher-provided checksum",
        "challenge_document_url": CHALLENGE_URL, "challenge_document_sha256": CHALLENGE_SHA256})
    report = {"task_id": "research_femto_bearing_rul", "dataset": "FEMTO", "task_kind": "rul",
        "status": "prepared", "training_bearings": len(LEARNING), "development_queries": len(validation),
        "test_bearings": len(TEST), "test_annotations_with_documented_ambiguity": 1,
        "labels_in_test_input": False, "measured_or_simulated": "laboratory measurements",
        "scope": "measured bearing RUL numerical research adaptation; not an official PHMForge score"}
    write_json(dest / "READY.json", report)
    # These files were created only for nested ZIP access and can be rebuilt
    # from the verified source; failed preparations intentionally retain them.
    for path in scratch.iterdir():
        path.unlink()
    scratch.rmdir()
    return report
