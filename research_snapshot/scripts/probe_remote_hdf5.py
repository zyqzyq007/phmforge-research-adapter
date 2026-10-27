"""Inspect selected public HDF5 records using verified, bounded HTTP ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import ssl
import time
import urllib.request

import h5py
import numpy as np


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, output, budget, insecure_tls=False, transport_attempts=1):
        self.url, self.size, self.output = url, size, output
        self.position, self.used, self.budget = 0, 0, budget
        self.cache, self.receipts = {}, []
        self.context = ssl._create_unverified_context() if insecure_tls else None
        self.transport_attempts = transport_attempts

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else
                         self.position + offset if whence == 1 else self.size + offset)
        if self.position < 0:
            raise ValueError("Negative offset")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        begin = self.position
        if end <= begin:
            return b""
        key = (begin, end)
        if key not in self.cache:
            if end - begin > 16 * 1024 * 1024 or self.used + end - begin > self.budget:
                raise ValueError("Bounded range budget exceeded")
            url = self.url + ("&" if "?" in self.url else "?") + f"phm_range={begin}-{end-1}"
            request = urllib.request.Request(url, headers={
                "Range": f"bytes={begin}-{end-1}",
                "Accept-Encoding": "identity",
                "User-Agent": "PHM-source-audit/1.0",
            })
            for attempt in range(1, self.transport_attempts + 1):
                try:
                    with urllib.request.urlopen(request, timeout=30, context=self.context) as response:
                        expected = f"bytes {begin}-{end-1}/{self.size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Server did not honor exact byte range")
                        body = response.read(end - begin + 1)
                        if len(body) != end - begin:
                            raise ValueError("Incorrect response size")
                    break
                except (OSError, ValueError) as error:
                    with (self.output / 'transport_failures.jsonl').open('a') as log:
                        log.write(json.dumps(dict(begin=begin,end_exclusive=end,attempt=attempt,
                            error_type=type(error).__name__,error=str(error)))+'\n')
                    if attempt == self.transport_attempts:
                        raise
                    time.sleep(min(2.0, 0.5 * attempt))
            name = f"{begin}_{end}.bin"
            (self.output / name).write_bytes(body)
            self.cache[key] = body
            self.used += len(body)
            self.receipts.append(dict(begin=begin, end_exclusive=end, bytes=len(body),
                                      sha256=hashlib.sha256(body).hexdigest(), file=name))
            (self.output / "ranges.json").write_text(json.dumps(self.receipts, indent=2) + "\n")
        self.position = end
        return self.cache[key]

    def readinto(self, buffer):
        body = self.read(len(buffer))
        buffer[:len(body)] = body
        return len(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-records", type=int, default=4)
    parser.add_argument("--time-rows", type=int, default=0,
                        help="Read only this native prefix; zero retains the full record")
    parser.add_argument("--whole-file-sha256")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--insecure-tls", action="store_true")
    parser.add_argument("--transport-attempts", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ranges = args.output / "ranges"
    ranges.mkdir()
    reader = RangeReader(
        args.url,
        args.size,
        ranges,
        args.budget,
        insecure_tls=args.insecure_tls,
        transport_attempts=args.transport_attempts,
    )
    requested = [str(x) for x in json.loads(args.ids.read_text())]
    result = dict(
        url=args.url,
        whole_file_bytes=args.size,
        claimed_whole_file_sha256=args.whole_file_sha256,
        full_file_downloaded=False,
        whole_file_sha256_verified=False,
        requested_ids=requested,
        requested_count=len(requested),
        tls_verification_disabled_for_transport_only=args.insecure_tls,
    )
    try:
        with h5py.File(reader, "r") as source:
            result["datasets"] = {
                key: dict(shape=list(value.shape), dtype=str(value.dtype),
                          chunks=value.chunks, compression=value.compression,
                          attributes={k: str(v) for k, v in value.attrs.items()})
                for key, value in source.items() if isinstance(value, h5py.Dataset)
            }
            ids = source["data_ID"][:].reshape(-1)
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
            lookup = {key: i for i, key in enumerate(ids)}
            result["unique_ids"] = len(lookup)
            result["missing_requested_ids"] = [key for key in requested if key not in lookup]
            result["requested_ids_found"] = len(requested) - len(result["missing_requested_ids"])
            rows = []
            selected = requested if args.max_records == 0 else requested[:args.max_records]
            for position, key in enumerate(selected, start=1):
                index = lookup[key]
                values = (source["seq_data"][index, :args.time_rows, :]
                          if args.time_rows > 0 else source["seq_data"][index])
                np.save(args.output / (key + ".npy"), values, allow_pickle=False)
                rows.append(dict(id=key, source_index=index, shape=list(values.shape),
                                 dtype=str(values.dtype), finite=bool(np.isfinite(values).all()),
                                 min_per_channel=np.min(values, axis=0).tolist(),
                                 max_per_channel=np.max(values, axis=0).tolist(),
                                 sha256=hashlib.sha256(values.tobytes()).hexdigest()))
                (args.output / "PROGRESS.json").write_text(
                    json.dumps(
                        {
                            "processed_records": position,
                            "requested_records": len(selected),
                            "last_record_id": key,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            result["records"] = rows
        if args.baseline and args.baseline.is_file():
            baseline = json.loads(args.baseline.read_text())
            baseline_hashes = {
                row["id"]: row["sha256"] for row in baseline.get("records", [])
            }
            current_hashes = {row["id"]: row["sha256"] for row in rows}
            compared = sorted(set(baseline_hashes) & set(current_hashes))
            result["baseline_comparison"] = {
                "path": str(args.baseline),
                "records_compared": len(compared),
                "all_matching": all(
                    baseline_hashes[key] == current_hashes[key] for key in compared
                ),
                "mismatches": [
                    key
                    for key in compared
                    if baseline_hashes[key] != current_hashes[key]
                ],
            }
        result["status"] = (
            "all_requested_records_read"
            if len(rows) == len(requested)
            else "selected_records_read"
        )
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(downloaded_bytes=reader.used, range_requests=len(reader.receipts),
                      h5py_version=h5py.__version__, numpy_version=np.__version__)
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ["records", "requested_ids", "datasets"]}), flush=True)


if __name__ == "__main__":
    main()
