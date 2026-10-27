"""Resume interrupted, pinned CSV transfers with exact ranged reads.

Partial bytes never qualify as a dataset. Only the complete source hash admits
the final file. The transcript records every requested range and retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
import urllib.request

from acquire_forge_pdmb_sources244 import load_pin, save, parse_csv, sha256


def resume(run, name, workers, seconds, chunk_bytes=1024**2):
    directory = run / "inputs/pdmb_attempt246"
    pin = load_pin(run / "source", name, name)
    partial, target = directory / f"{name}.csv.partial", directory / f"{name}.csv"
    if target.exists():
        return {"dataset": name, "status": "already_complete"}
    started = time.monotonic()
    offset = partial.stat().st_size if partial.exists() else 0
    initial = {"dataset": name, "prior_partial_bytes": offset,
               "prior_partial_sha256": sha256(partial) if offset else None,
               "pin": pin, "workers": workers, "wall_budget_seconds": seconds}
    save(run / f"audit/RESUME_{name}.json", initial)
    chunk = chunk_bytes
    log = directory / f"{name}.ranges.jsonl"
    def fetch(start):
        end = min(start + chunk, pin["bytes"]) - 1
        for attempt in range(3):
            tick = time.monotonic()
            record = dict(start=start, end=end, attempt=attempt)
            try:
                request = urllib.request.Request(
                    pin["url"] + f"&byte_range={start}-{end}",
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=45,
                                            context=ssl._create_unverified_context()) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{pin['bytes']}":
                        raise ValueError("Server did not honor the exact range")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise ValueError(f"Short range: {len(data)}")
                record.update(ok=True, sha256=hashlib.sha256(data).hexdigest())
            except Exception as error:
                record.update(ok=False, error_type=type(error).__name__, error=str(error)[:300])
                data = None
            record["seconds"] = time.monotonic() - tick
            # Independent append with one write per compact record.
            with log.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if data is not None:
                return data
            if attempt < 2:
                time.sleep(1 + attempt)
        raise RuntimeError(f"Range {start}-{end} failed after three attempts")
    try:
        with partial.open("ab") as output, ThreadPoolExecutor(max_workers=workers) as pool:
            while offset < pin["bytes"]:
                if time.monotonic() - started > seconds:
                    raise TimeoutError("Bounded transfer stopped; partial file retained for continuation")
                starts = list(range(offset, min(offset + workers*chunk, pin["bytes"]), chunk))
                for data in pool.map(fetch, starts):
                    output.write(data)
                    offset += len(data)
                output.flush()
                save(directory / f"{name}.progress.json", dict(
                    dataset=name, bytes=offset, total=pin["bytes"],
                    elapsed_seconds=time.monotonic()-started))
        actual = sha256(partial)
        if pin["lfs_sha256"] != actual or partial.stat().st_size != pin["bytes"]:
            raise ValueError("Complete content hash differs from the pinned source")
        partial.rename(target)
        receipt = dict(pin=pin, actual_sha256=actual, pinned_hash_verified=True,
                       complete_file=True, download_reused=False, ranged_resume=True,
                       tls_verification_disabled_for_transport_only=True,
                       csv=parse_csv(target))
        save(directory / f"{name}.receipt.json", receipt)
        result = dict(dataset=name, status="complete", bytes=offset, sha256=actual)
    except Exception as error:
        result = dict(dataset=name, status="incomplete", bytes=partial.stat().st_size,
                      error_type=type(error).__name__, error=str(error))
    result["seconds"] = time.monotonic() - started
    save(run / f"audit/RESUME_RESULT_{name}.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seconds-per-file", type=int, default=900)
    p.add_argument("--parallel-files", type=int, default=1)
    p.add_argument("--chunk-bytes", type=int, default=1024**2)
    a = p.parse_args()
    run = a.run.resolve()
    protocol = json.loads((run/"PROTOCOL.json").read_text())
    if protocol["new_LLM_calls"] != 0:
        raise ValueError("Source continuation has no model calls")
    planned = sum(load_pin(run/"source",name,name)["bytes"] for name in a.datasets)
    if planned > protocol["budget_bytes"]:
        raise ValueError("Pinned complete source bytes exceed frozen disk budget")
    temporary = run/"runtime/tmp"
    temporary.mkdir(parents=True,exist_ok=True)
    for name in ["TMPDIR","TMP","TEMP"]:
        os.environ[name]=str(temporary)
    if a.parallel_files == 1:
        results=[resume(run,name,a.workers,a.seconds_per_file,a.chunk_bytes) for name in a.datasets]
    else:
        with ThreadPoolExecutor(max_workers=a.parallel_files) as pool:
            results=list(pool.map(lambda name: resume(run,name,a.workers,a.seconds_per_file,a.chunk_bytes),a.datasets))
    save(run/"SOURCE_CONTINUATION_RESULT.json",dict(
        results=results,complete=all(x["status"] in ["complete","already_complete"] for x in results),
        new_LLM_calls=0, scientific_scores=0))
