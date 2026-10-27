"""Materialize hash-verified source/runtime inputs for the adapted environments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ARCHIVES = Path("/data/yyf-zyq/PHM_20261010")


def extract_bound(archive, destination, prefixes):
    manifest_path = archive / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    wanted = {x["path"]: x for x in manifest["files"]
              if any(x["path"].startswith(p) for p in prefixes)}
    bindings = []
    if (archive / "RUN_CONTENTS.tar.gz").exists():
        with tarfile.open(archive / "RUN_CONTENTS.tar.gz", "r|gz") as stream:
            for item in stream:
                name = item.name.removeprefix("./")
                if name not in wanted:
                    continue
                target = destination / name
                assert not Path(name).is_absolute() and ".." not in Path(name).parts
                data = stream.extractfile(item).read()
                reference = wanted[name]
                assert len(data) == reference["bytes"]
                assert hashlib.sha256(data).hexdigest() == reference["sha256"], name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                bindings.append(dict(reference, archive=str(archive),
                                     destination=str(target)))
                if len(bindings) == len(wanted):
                    break
    else:
        for name, reference in wanted.items():
            source = archive / name
            assert source.is_file() and not source.is_symlink()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["sha256"]
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            bindings.append(dict(reference, archive=str(archive), destination=str(target)))
    assert len(bindings) == len(wanted), (len(bindings), len(wanted))
    return bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    r = args.run.resolve()
    report = r / "audit/RUNTIME_SOURCE_BINDINGS240.json"
    if report.exists():
        raise FileExistsError(report)
    bindings = []
    # Identical current-main and legacy runtime dependencies: restore once.
    bindings += extract_bound(ARCHIVES / "20261018T185541Z_assethf190",
                              r / "restored/legacy",
                              ["runtime/asset_deps/", "runtime/asset_public/",
                               "runtime/fixture", "runtime/DOCKER_RESOURCES",
                               "SCORING_AUTHOR_SOURCE.json"])
    bindings += extract_bound(ARCHIVES / "20261018T170243Z_fullbench187",
                              r / "restored/current", ["source/asset/"])
    for dataset, pointer in [("CWRU", "data/phmforge_recovered_inputs203.json"),
                             ("MFPT", "data/phmforge_recovered_inputs207.json")]:
        info = json.loads(Path(pointer).read_text())
        entry = next(x for x in info["prepared_input_files"]
                     if x["path"].endswith(f"{dataset}_train.csv"))
        source = Path(entry["path"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        target = r / "sources" / (dataset + ".csv")
        shutil.copy2(source, target)
        bindings.append(dict(entry, destination=str(target), published_dataset=dataset))
    report.write_text(json.dumps(bindings, indent=2) + "\n")
    print(json.dumps({"bound_files":len(bindings), "bytes":sum(x["bytes"] for x in bindings)}))


if __name__ == "__main__":
    main()
