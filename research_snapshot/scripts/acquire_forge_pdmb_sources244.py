"""Verify a bounded set of public PHMForge/PDMBench source files.

This prepares source data only. It makes no model or Agent calls, does not
modify the frozen original benchmark, and never substitutes one dataset for
another.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import ssl
import urllib.parse
import urllib.request


DATASETS = (
    "CWRU",
    "MFPT",
    "RotorBrokenBar",
    "Azure",
    "ElectricMotorVibrations",
    "UoC",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv(path):
    csv.field_size_limit(64 * 1024 * 1024)
    labels = Counter()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("CSV has no header")
        label_column = next(
            (column for column in columns if column.lower() == "label"),
            None,
        )
        for row in reader:
            rows += 1
            if label_column is not None:
                labels[str(row[label_column])] += 1
    return {
        "columns": columns,
        "rows": rows,
        "label_column": label_column,
        "label_counts": dict(sorted(labels.items())),
    }


def load_pin(source_dir, dataset, mirror_dataset=None):
    catalog = json.loads(
        (source_dir / "hf_submission096.json").read_text(encoding="utf-8")
    )
    item = next(entry for entry in catalog if entry["id"] == f"submission096/{dataset}")
    tree = json.loads(
        (source_dir / f"pdmb_{dataset}_tree.json").read_text(encoding="utf-8")
    )
    entry = next(row for row in tree if row["path"] == "PdM.csv")
    lfs = entry.get("lfs") or {}
    pin = {
        "dataset": dataset,
        "repo": "submission096/" + dataset,
        "revision": item["sha"],
        "path": "PdM.csv",
        "bytes": entry["size"],
        "lfs_sha256": lfs.get("oid"),
        "git_blob_sha1": None if lfs else entry["oid"],
        "url": (
            f"https://huggingface.co/datasets/submission096/{dataset}/resolve/"
            f"{item['sha']}/PdM.csv?download=true"
        ),
    }
    if mirror_dataset:
        mirror_tree = json.loads(
            (source_dir / f"odyssey_{mirror_dataset}_tree.json").read_text(
                encoding="utf-8"
            )
        )
        mirror_name = "Planetary.csv" if mirror_dataset == "PlanetaryPdM" else f"{mirror_dataset}.csv"
        mirror_entry = next(row for row in mirror_tree if row["path"] == mirror_name)
        mirror_lfs = mirror_entry.get("lfs") or {}
        mirror_hash = mirror_lfs.get("oid") or mirror_entry["oid"]
        expected_hash = pin["lfs_sha256"] or pin["git_blob_sha1"]
        pin["independent_mirror"] = {
            "repo": "odysseywt/" + mirror_dataset,
            "path": mirror_name,
            "bytes": mirror_entry["size"],
            "pinned_hash": mirror_hash,
            "same_size": mirror_entry["size"] == pin["bytes"],
            "same_pinned_hash": mirror_hash == expected_hash,
        }
    return pin


def download(pin, output_dir, insecure_tls):
    target = output_dir / pin.get('filename', f"{pin['dataset']}.csv")
    partial = target.with_name(target.name + ".partial")
    receipt_path = output_dir / f"{pin['dataset']}.receipt.json"
    if target.exists():
        if target.stat().st_size != pin["bytes"]:
            raise ValueError(f"Existing source has wrong byte count: {target}")
        actual_sha256 = sha256(target)
        actual_git_blob = git_blob_sha1(target) if pin["git_blob_sha1"] else None
        expected_matches = (
            actual_sha256 == pin["lfs_sha256"]
            if pin["lfs_sha256"]
            else actual_git_blob == pin["git_blob_sha1"]
        )
        if not expected_matches:
            raise ValueError(f"Existing source has wrong source hash: {target}")
        receipt = {
            "utc": now(),
            "pin": pin,
            "actual_sha256": actual_sha256,
            "actual_git_blob_sha1": actual_git_blob,
            "pinned_hash_verified": True,
            "complete_file": True,
            "download_reused": True,
        }
    else:
        if partial.exists():
            raise FileExistsError(f"Preserving pre-existing partial file: {partial}")
        request = urllib.request.Request(
            pin["url"],
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": pin.get('user_agent', "PHMForge-source-verification/1.0"),
            },
        )
        digest = hashlib.sha256()
        git_digest = hashlib.sha1()
        if pin["git_blob_sha1"]:
            git_digest.update(f"blob {pin['bytes']}\0".encode())
        downloaded = 0
        started = now()
        context = ssl._create_unverified_context() if insecure_tls else ssl.create_default_context()
        route = pin.get('transport_route', 'direct')
        if route not in ('direct', 'proxy'):
            raise ValueError('Unknown public download transport route')
        proxy_handler = urllib.request.ProxyHandler() if route == 'proxy' else urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler,
            urllib.request.HTTPSHandler(context=context))
        with opener.open(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            with partial.open("xb") as destination:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    downloaded += len(block)
                    if downloaded > pin["bytes"]:
                        raise ValueError("Response exceeds pinned source byte count")
                    destination.write(block)
                    digest.update(block)
                    if pin["git_blob_sha1"]:
                        git_digest.update(block)
            final_url = response.url
        if downloaded != pin["bytes"]:
            raise ValueError(
                f"Response ended at {downloaded} bytes, expected {pin['bytes']}"
            )
        actual_sha256 = digest.hexdigest()
        actual_git_blob = git_digest.hexdigest() if pin["git_blob_sha1"] else None
        expected_matches = (
            actual_sha256 == pin["lfs_sha256"]
            if pin["lfs_sha256"]
            else actual_git_blob == pin["git_blob_sha1"]
        )
        if not expected_matches:
            raise ValueError("Downloaded source does not match the pinned tree hash")
        partial.rename(target)
        receipt = {
            "utc": now(),
            "started_utc": started,
            "final_url": final_url,
            "final_host": urllib.parse.urlparse(final_url).netloc,
            "pin": pin,
            "actual_sha256": actual_sha256,
            "actual_git_blob_sha1": actual_git_blob,
            "pinned_hash_verified": True,
            "complete_file": True,
            "download_reused": False,
            "tls_verification_disabled_for_transport_only": insecure_tls,
            "transport_route": route,
            "content_authenticated_by_independent_pinned_hashes": True,
        }
    if pin.get('format', 'csv') == 'csv':
        receipt["csv"] = parse_csv(target)
    else:
        receipt['format'] = pin['format']
    save(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempt", default="2")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--insecure-tls", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = json.loads((run / "PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol.get("new_LLM_calls") != 0:
        raise ValueError("This source-only action requires a zero-LLM protocol")
    source_dir = run / "source"
    output_dir = run / f"inputs/pdmb_attempt{args.attempt}"
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(run / "runtime/tmp")
    os.environ["TMP"] = str(run / "runtime/tmp")
    os.environ["TEMP"] = str(run / "runtime/tmp")

    datasets = tuple(dict.fromkeys(args.datasets))
    mirror_names = {dataset: dataset for dataset in datasets}
    pins = [
        load_pin(source_dir, dataset, mirror_names.get(dataset))
        for dataset in datasets
    ]
    planned_bytes = sum(pin["bytes"] for pin in pins)
    prior_bytes = sum(
        path.stat().st_size for path in run.rglob("*") if path.is_file()
    )
    budget = protocol["budget_bytes"]
    if prior_bytes + planned_bytes > budget:
        raise ValueError(
            f"Source plan exceeds frozen budget: {prior_bytes}+{planned_bytes}>{budget}"
        )
    save(
        run / f"audit/PDMB_ACQUISITION_PLAN244_ATTEMPT{args.attempt}.json",
        {
            "utc": now(),
            "datasets": list(datasets),
            "pins": pins,
            "planned_bytes": planned_bytes,
            "prior_run_bytes": prior_bytes,
            "budget_bytes": budget,
            "within_budget": True,
            "model_calls": 0,
            "agent_sessions": 0,
            "tls_verification_disabled_for_transport_only": args.insecure_tls,
        },
    )

    receipts = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, pin, output_dir, args.insecure_tls): pin["dataset"]
            for pin in pins
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                receipt = future.result()
            except BaseException as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                    }
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "error_type": type(error).__name__,
                            "error": str(error)[:500],
                        }
                    ),
                    flush=True,
                )
                continue
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "bytes": receipt["pin"]["bytes"],
                        "rows": receipt["csv"]["rows"],
                        "pinned_hash_verified": True,
                    }
                ),
                flush=True,
            )

    receipts.sort(key=lambda receipt: receipt["pin"]["dataset"])
    retained_bytes = sum(
        path.stat().st_size for path in run.rglob("*") if path.is_file()
    )
    summary = {
        "utc": now(),
        "status": (
            "selected_public_sources_fully_verified"
            if not failures
            else "selected_public_sources_partially_verified"
        ),
        "datasets": [receipt["pin"]["dataset"] for receipt in receipts],
        "dataset_count": len(receipts),
        "source_bytes_verified": sum(receipt["pin"]["bytes"] for receipt in receipts),
        "run_bytes_after_verification": retained_bytes,
        "budget_bytes": budget,
        "within_budget": retained_bytes <= budget,
        "pinned_hashes_verified": all(
            receipt["pinned_hash_verified"] for receipt in receipts
        ),
        "all_files_complete": all(receipt["complete_file"] for receipt in receipts),
        "model_calls": 0,
        "agent_sessions": 0,
        "new_training_jobs": 0,
        "scientific_scores": 0,
        "physical_correctness_qualified": False,
        "original_benchmark_modified": False,
        "problem_confirmation_increment": 0,
        "failures": failures,
        "tls_verification_disabled_for_transport_only": args.insecure_tls,
        "content_authentication": (
            "Each accepted complete file matched a pinned SHA-256 or Git blob SHA-1"
        ),
        "receipts": [
            {
                "dataset": receipt["pin"]["dataset"],
                "bytes": receipt["pin"]["bytes"],
                "sha256": receipt["actual_sha256"],
                "git_blob_sha1": receipt["actual_git_blob_sha1"],
                "rows": receipt["csv"]["rows"],
                "columns": receipt["csv"]["columns"],
                "label_counts": receipt["csv"]["label_counts"],
            }
            for receipt in receipts
        ],
    }
    save(run / f"OUTPUTS244_ATTEMPT{args.attempt}.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
