"""Fork a qualified runtime with selected PHMForge fixes and a fresh manifest.

Unchanged files are hard-linked from the source runtime. Every replaced file is
copied, so the source runtime's bytes and manifest remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


REPLACEMENTS = {
    "code/verified_phmforge.py": "verified_phmforge.py",
    "public/code/verified_phmforge.py": "verified_phmforge.py",
    "code/real_fault_model.py": "real_fault_model.py",
    "public/code/real_fault_model.py": "real_fault_model.py",
    "code/verified_benchmark_scoring.py": "verified_benchmark_scoring.py",
    "public/code/verified_benchmark_scoring.py": "verified_benchmark_scoring.py",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(source, destination, scripts, report_path, overlay=None, revision="A246"):
    source = source.resolve()
    destination = destination.resolve()
    report_path = report_path.resolve()
    if source == destination or destination.exists():
        raise ValueError("Destination must be a new path distinct from the source runtime")
    if report_path.is_relative_to(destination):
        raise ValueError("Refresh receipt must remain outside the runtime manifest")
    if report_path.exists():
        raise FileExistsError(report_path)
    source_manifest = source / "FILE_MANIFEST.json"
    source_manifest_sha = sha256(source_manifest)
    shutil.copytree(source, destination, copy_function=os.link)
    replaced = []
    for relative, script_name in REPLACEMENTS.items():
        target = destination / relative
        target.unlink(missing_ok=True)
        shutil.copy2(scripts / script_name, target)
        replaced.append(dict(
            path=relative,
            source_script=str(scripts / script_name),
            sha256=sha256(target),
        ))
    additions = []
    if overlay is not None:
        registry_path = destination / "public/dataset_registry.json"
        registry = json.loads(registry_path.read_text())
        for metadata in sorted((overlay / "datasets").glob("*/dataset.json")):
            info = json.loads(metadata.read_text())
            name = info["dataset"]
            if name in registry:
                raise ValueError("Overlay may not replace an existing dataset")
            target = destination / "public/datasets" / name
            shutil.copytree(metadata.parent, target)
            registry[name] = dict(
                task="classification", path=f"datasets/{name}",
                data_origin="Published source-bound vibration records; no field-measurement certification",
                **{key:info[key] for key in ["architectures","group_definition","rows_by_split"]})
            additions.append(name)
        registry_path.unlink()
        registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    catalog_path = destination / "public/tool_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    descriptions = {
        "load_ground_truth": "Development labels only. Explicit file='train' or 'val'; empty or final-test requests are unavailable.",
        "verify_ground_truth": "Checks exact sample/unit IDs and numeric RUL metrics over JSON record lists or allowed JSON artifact paths. Does not certify caller-supplied truth.",
        "verify_classification": "Checks exact sample IDs and actual classification metrics over JSON records or allowed JSON artifact paths. Does not certify caller-supplied truth.",
    }
    for name, description in descriptions.items():
        catalog[name]["description"] = description
    catalog_path.unlink()
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    ready_path = destination / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready_path.unlink()
    ready.update(
        runtime_revision=revision,
        supersedes=str(source),
        superseded_manifest_sha256=source_manifest_sha,
        phmforge_interface_fixes=[
            "source-bound additional classification recordings and complete-group splits",
            "original string/noncontiguous class labels are reversibly encoded for training",
            "development label access and identity-aligned verification without hidden test labels",
            "model/input dataset identity checked before classification",
        ],
        added_classification_datasets=additions,
        official_original_benchmark_score_qualified=False,
    )
    ready_path.write_text(json.dumps(ready, indent=2) + "\n")
    manifest_path = destination / "FILE_MANIFEST.json"
    manifest_path.unlink()
    records = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            records.append(dict(
                path=str(path.relative_to(destination)),
                bytes=path.stat().st_size,
                sha256=sha256(path),
            ))
    manifest = dict(
        files=records,
        files_count=len(records),
        bytes=sum(row["bytes"] for row in records),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = dict(
        source_runtime=str(source),
        runtime=str(destination),
        source_manifest_sha256=source_manifest_sha,
        manifest_sha256=sha256(manifest_path),
        files=len(records),
        bytes=manifest["bytes"],
        replaced_files=replaced,
        external_labels_included=False,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--public-overlay", type=Path)
    parser.add_argument("--revision", default="A246")
    args = parser.parse_args()
    print(json.dumps(refresh(
        args.source,
        args.destination,
        args.scripts.resolve(),
        args.report,
        args.public_overlay,
        args.revision,
    )))
