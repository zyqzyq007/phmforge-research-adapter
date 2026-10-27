"""Fetch the exact public HUST-LFP archive; inventory bytes without unpickling.

No model fitting, trajectory parsing, or benchmark score is performed here.
Budgets and publisher checksums come from the prospectively frozen protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def qualify(root):
    protocol = json.loads((root / "PROTOCOL359.json").read_text())
    spec, budget = protocol["raw"], protocol["budget"]
    started = time.monotonic()
    partial = root / "raw/our_data.zip.part"
    final = root / "raw/our_data.zip"
    assert not partial.exists() and not final.exists(), "Do not restart this run."
    attempts, received = [], 0
    for attempt in range(budget["raw_transport_attempts_max"]):
        before = partial.stat().st_size if partial.exists() else 0
        remaining = budget["total_seconds"] - (time.monotonic() - started)
        if remaining <= 0 or received >= budget["received_bytes_max"]:
            raise RuntimeError("Frozen transport budget exhausted")
        command = ["curl", "--silent", "--show-error", "--fail", "--location",
                   "--retry", "0", "--connect-timeout", "30", "--max-time",
                   str(min(budget["transport_seconds_per_attempt"], int(remaining))),
                   "--max-filesize", str(spec["bytes"]), "--output", str(partial),
                   "--write-out", "%{json}"]
        if before:
            command += ["--continue-at", "-"]
        command.append(spec["url"])
        with (root / f"audit/curl_{attempt}.stdout").open("w") as stdout, \
                (root / f"audit/curl_{attempt}.stderr").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                     timeout=min(remaining, 920))
        after = partial.stat().st_size if partial.exists() else 0
        received += max(after - before, 0)
        attempts.append({"utc": stamp(), "attempt": attempt + 1,
                         "exit_code": process.returncode, "before": before,
                         "after": after, "resumed": bool(before)})
        write(root / "TRANSPORT359.json", {"attempts": attempts,
              "retained_download_bytes": after, "received_raw_bytes": received})
        if process.returncode == 0:
            break
        if after == spec["bytes"]:
            break
    if not partial.exists() or partial.stat().st_size != spec["bytes"]:
        raise RuntimeError("Exact publisher size not obtained; partial retained")
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise RuntimeError("Publisher SHA256 mismatch; stop without retry")
    partial.rename(final)
    members, total = [], 0
    with zipfile.ZipFile(final) as archive:
        declared = sum(x.file_size for x in archive.infolist())
        if declared > budget["uncompressed_bytes_max"]:
            raise RuntimeError("Declared ZIP expansion exceeds frozen limit")
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or entry.flag_bits & 1:
                raise RuntimeError("Unsafe or encrypted ZIP member")
            if entry.is_dir():
                continue
            sha, size = hashlib.sha256(), 0
            # ZipExtFile verifies each member CRC on complete streaming read.
            with archive.open(entry) as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    sha.update(block)
                    size += len(block)
            assert size == entry.file_size
            total += size
            members.append({"name": entry.filename, "bytes": size,
                            "compressed_bytes": entry.compress_size,
                            "crc32": f"{entry.CRC:08x}", "sha256": sha.hexdigest()})
            write(root / "PROGRESS359.json", {"utc": stamp(),
                  "phase": "member_crc_and_hash", "members_done": len(members),
                  "uncompressed_bytes_read": total})
    result = {"utc": stamp(), "status": "exact_raw_bytes_and_all_member_crc_passed",
              "raw_bytes": final.stat().st_size, "sha256": digest.hexdigest(),
              "members": members, "uncompressed_bytes": total,
              "elapsed_seconds": time.monotonic() - started,
              "new_fits": 0, "new_LLM_calls": 0, "new_predictions": 0,
              "pickle_deserializations": 0, "semantic_task_qualified": False}
    write(root / "INTEGRITY359.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    write(root / "HANDLE359.json", {"pid": os.getpid(), "utc": stamp(),
                                   "run": str(root), "status": "running"})
    try:
        result = qualify(root)
    except Exception as error:
        write(root / "TERMINAL359.json", {"utc": stamp(), "exit_code": 1,
              "error_type": type(error).__name__, "error": str(error),
              "partial_retained": True, "new_fits": 0, "new_LLM_calls": 0})
        raise
    write(root / "TERMINAL359.json", {"utc": stamp(), "exit_code": 0,
          "status": result["status"], "new_fits": 0, "new_LLM_calls": 0})
    print(json.dumps({k: v for k, v in result.items() if k != "members"}))


if __name__ == "__main__":
    main()
