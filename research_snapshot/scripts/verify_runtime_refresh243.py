"""Verify source and derived runtime manifests and enumerate intentional changes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
    manifest_path = root / "FILE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    listed = {}
    for record in manifest["files"]:
        relative = Path(record["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
        listed[str(relative)] = record
    assert len(listed) == manifest["files_count"]
    assert sum(x["bytes"] for x in listed.values()) == manifest["bytes"]
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert actual == set(listed), (
        sorted(actual - set(listed))[:10],
        sorted(set(listed) - actual)[:10],
    )
    return manifest, sha256(manifest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, derived = args.source.resolve(), args.derived.resolve()
    source_manifest, source_hash = verify_manifest(source)
    derived_manifest, derived_hash = verify_manifest(derived)
    source_files = {row["path"]: row["sha256"] for row in source_manifest["files"]}
    derived_files = {row["path"]: row["sha256"] for row in derived_manifest["files"]}
    assert set(source_files) == set(derived_files)
    changed = sorted(path for path in source_files if source_files[path] != derived_files[path])
    assert changed == ["READY.json", "code/verified_phmforge.py",
                       "public/code/verified_phmforge.py"]
    result = dict(
        status="passed",
        source_runtime=str(source),
        source_manifest_sha256=source_hash,
        source_files=source_manifest["files_count"],
        source_bytes=source_manifest["bytes"],
        derived_runtime=str(derived),
        derived_manifest_sha256=derived_hash,
        derived_files=derived_manifest["files_count"],
        derived_bytes=derived_manifest["bytes"],
        unchanged_file_count=len(source_files) - len(changed),
        changed_files=changed,
        every_listed_file_hash_verified=True,
        unlisted_runtime_files=[],
        external_labels_included=False,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(
        status="passed",
        files=derived_manifest["files_count"],
        changed=changed,
    )))


if __name__ == "__main__":
    main()
