"""Recover only the missing HUST source bytes with the existing range reader."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import zipfile


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def run(root):
    config = json.loads((root / "PROTOCOL360.json").read_text())
    transport, seed = config["transport"], config["prefix"]
    prefix = Path(seed["path"])
    assert prefix.stat().st_size == seed["bytes"]
    assert digest(prefix) == seed["sha256"]
    command = [sys.executable, "-B", str(root / "code/acquire_range_source.py"),
               "--run", str(root), "--url", config["url"],
               "--bytes", str(config["publisher_expected_bytes"]),
               "--etag", config["ETag"], "--name", "hust_tail360",
               "--destination", "raw/our_data.tail", "--format", "raw",
               "--offset", str(seed["bytes"]), "--drop-completed-cache",
               "--refresh-public-request"]
    for key in ["workers", "chunk_bytes", "max_attempts", "max_failures",
                "connect_timeout", "read_timeout", "route"]:
        command += ["--" + key.replace("_", "-"), str(transport[key])]
    if config.get("reuse_interval_manifest"):
        command += ["--reuse-interval-manifest", config["reuse_interval_manifest"]]
    if transport.get("proxy_wait_seconds"):
        command += ["--proxy-wait-seconds", str(transport["proxy_wait_seconds"])]
    save(root / "COMMAND360.json", {"command": command, "utc": now()})
    with (root / "audit/range.stdout").open("w") as output, \
            (root / "audit/range.stderr").open("w") as error:
        child = subprocess.Popen(command, stdout=output, stderr=error,
                                 start_new_session=True)
        save(root / "HANDLE360.json", {"utc": now(), "pid": os.getpid(),
             "range_pid": child.pid, "run": str(root), "status": "running"})
        try:
            code = child.wait(timeout=transport["global_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError("Frozen whole-transport deadline reached")
    if code:
        raise RuntimeError(f"Range reader exit {code}; no implicit restart")
    status = json.loads((root / "hust_tail360_status.json").read_text())
    assert status["status"] == "finished"
    tail, final = root / "raw/our_data.tail", root / "raw/our_data.zip"
    assert not final.exists()
    sha = hashlib.sha256()
    with final.open("xb") as output:
        for source in [prefix, tail]:
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    output.write(block)
                    sha.update(block)
    assert final.stat().st_size == config["publisher_expected_bytes"]
    assert sha.hexdigest() == config["publisher_expected_sha256"], "Stop on SHA mismatch"
    save(root / "PUBLISHER_SHA360.json", {"utc": now(), "passed": True,
         "bytes": final.stat().st_size, "sha256": sha.hexdigest(),
         "old_prefix_now_qualified_by_full_publisher_SHA": True})
    members = []
    with zipfile.ZipFile(final) as archive:
        assert sum(x.file_size for x in archive.infolist()) <= transport["uncompressed_limit_bytes"]
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and ".." not in path.parts and not entry.flag_bits & 1
            if entry.is_dir():
                continue
            member_sha, size = hashlib.sha256(), 0
            with archive.open(entry) as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    member_sha.update(block)
                    size += len(block)
            assert size == entry.file_size
            members.append({"name": entry.filename, "bytes": size,
                 "compressed_bytes": entry.compress_size,
                 "crc32": f"{entry.CRC:08x}", "sha256": member_sha.hexdigest()})
            save(root / "CRC_PROGRESS360.json", {"utc": now(), "members_done": len(members)})
    result = {"utc": now(), "passed": True, "members": members,
         "raw_bytes": final.stat().st_size, "sha256": sha.hexdigest(),
         "all_member_CRCS_verified": True, "pickle_deserializations": 0,
         "semantic_task_qualified": False, "new_fits": 0,
         "new_LLM_calls": 0, "new_predictions": 0}
    save(root / "INTEGRITY360.json", result)
    # The tail is reconstructable from the retained full publisher-bound ZIP.
    tail.unlink()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    root = parser.parse_args().run.resolve()
    started = time.monotonic()
    try:
        result = run(root)
    except Exception as error:
        save(root / "TERMINAL360.json", {"utc": now(), "exit_code": 1,
             "error_type": type(error).__name__, "error": str(error),
             "new_fits": 0, "new_LLM_calls": 0})
        raise
    receipt = {"utc": now(), "exit_code": 0, "source_integrity_passed": result["passed"],
         "members_verified": len(result["members"]), "elapsed_seconds": time.monotonic() - started,
         "new_fits": 0, "new_LLM_calls": 0, "semantic_task_qualified": False}
    save(root / "TERMINAL360.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
