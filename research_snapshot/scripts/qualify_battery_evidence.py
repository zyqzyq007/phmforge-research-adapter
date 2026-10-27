"""Audit public battery workbooks; descriptive proxies are NOT ISC ground truth.

Uses cached OOXML values without recalculating source formulas. Temperature and
voltage use their own time columns; no unverified cross-DAQ alignment is made.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TAG = "{" + NS["s"] + "}"


def workbook_rows(path, sheet="xl/worksheets/sheet1.xml"):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in e.findall(".//s:t", NS))
                       for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        with archive.open(sheet) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != TAG + "row":
                    continue
                values = {}
                for cell in row.findall("s:c", NS):
                    value = cell.find("s:v", NS)
                    if value is None:
                        continue
                    value = value.text
                    if cell.get("t") == "s":
                        value = strings[int(value)]
                    values["".join(c for c in cell.get("r") if c.isalpha())] = value
                yield int(row.get("r")), values
                row.clear()


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def timing(series):
    deltas = [b[0] - a[0] for a, b in zip(series, series[1:])]
    return {"n": len(series), "start_s": series[0][0], "end_s": series[-1][0],
            "nonpositive_dt": sum(d <= 0 for d in deltas),
            "median_positive_dt_s": statistics.median([d for d in deltas if d > 0])
            if any(d > 0 for d in deltas) else None}


def schema(row):
    if ("time" in row.get("F", "").lower()
            and "voltage" in row.get("H", "").lower()
            and "time" in row.get("T", "").lower()):
        return ("formatted_thermocouples", "F", "H", "T",
                [c for c in ("U", "V", "W", "X") if "tc" in row.get(c, "").lower()])
    if ("time" in row.get("A", "").lower() and "vcell" in row.get("D", "").lower()
            and "[v]" in row.get("D", "").lower()):
        return ("sandia_thermocouples", "A", "D", "A",
                [c for c in ("F", "G", "H", "I", "J", "K")
                 if row.get(c, "").lower().startswith("tc") and "[c]" in row[c].lower()])
    if ("time" in row.get("A", "").lower() and "voltage" in row.get("C", "").lower()
            and "time" in row.get("E", "").lower()
            and (any(unit in row.get("F", "").lower() for unit in ("[c]", "(c)", "°c"))
                 or row.get("F", "").lower().strip() == "c")):
        return ("published_temperature_channel", "A", "C", "E", ["F"])
    return None


def audit_one(job):
    root, entry, author = job
    path = Path(root) / entry.get("relative_path", "battery_raw/" + entry["id"] + ".xlsx")
    result = {"filename": entry["filename"], "id": entry["id"], "author_summary": author}
    if not path.exists():
        return dict(result, status="download_unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result.update(sha256=digest, bytes=path.stat().st_size,
                  publisher_hash_match=digest == entry["publisher_sha256"])
    if not result["publisher_hash_match"]:
        return dict(result, status="hash_mismatch")
    try:
        voltage, temperatures, headers = [], {}, None
        duplicate_voltage_comparisons, duplicate_voltage_mismatches = 0, 0
        with zipfile.ZipFile(path) as archive:
            sheets = sorted(s for s in archive.namelist()
                            if s.startswith("xl/worksheets/sheet") and s.endswith(".xml"))
        candidates, sheet_headers = [], {}
        for sheet in sheets:
            iterator = workbook_rows(path, sheet)
            first = next(iterator, None)
            iterator.close()
            sheet_headers[sheet] = first[1] if first else None
            if first and first[0] == 1 and schema(first[1]):
                candidates.append(sheet)
        result["worksheet_headers"] = sheet_headers
        if len(candidates) != 1:
            return dict(result, status="ambiguous_worksheets" if len(candidates) > 1
                        else "unsupported_schema" if any(sheet_headers.values()) else "missing_signal")
        selected = candidates[0]
        result["worksheet"] = selected
        for rownum, row in workbook_rows(path, selected):
            if rownum == 1:
                headers = row
                name, vt_col, v_col, tt_col, temp_cols = schema(row)
                temperatures = {c: [] for c in temp_cols}
                result["schema"] = name
                continue
            tv, v, tt = number(row.get(vt_col)), number(row.get(v_col)), number(row.get(tt_col))
            if tv is not None and v is not None:
                voltage.append((tv, v))
            if tt is not None:
                for col, series in temperatures.items():
                    temp = number(row.get(col))
                    if temp is not None:
                        series.append((tt, temp))
            raw_v = number(row.get("C"))
            if v_col == "H" and raw_v is not None and v is not None:
                duplicate_voltage_comparisons += 1
                duplicate_voltage_mismatches += not math.isclose(raw_v, v, abs_tol=1e-8)
        result.update(headers=headers, voltage_duplicate_comparisons=duplicate_voltage_comparisons,
                      voltage_duplicate_mismatches=duplicate_voltage_mismatches)
        if not voltage or not any(temperatures.values()):
            return dict(result, status="missing_signal")
        vmin = min(v for _, v in voltage)
        vmax = max(v for _, v in voltage)
        first = voltage[0][1]
        baseline = statistics.median([v for t, v in voltage if t <= voltage[0][0] + 1])
        peak = -math.inf
        drawdown = 0.0
        for _, value in voltage:
            peak = max(peak, value)
            drawdown = max(drawdown, peak - value)
        min_index = min(range(len(voltage)), key=lambda i: voltage[i][1])
        temp_stats = {}
        for col, series in temperatures.items():
            if not series:
                continue
            rates = [(b[1] - a[1]) / (b[0] - a[0])
                     for a, b in zip(series, series[1:]) if b[0] > a[0]]
            temp_stats[col] = dict(timing(series), minimum_C=min(v for _, v in series),
                                   maximum_C=max(v for _, v in series),
                                   max_adjacent_rate_C_s=max(rates) if rates else None)
        tmax = max(d["maximum_C"] for d in temp_stats.values())
        result.update(status="parsed", voltage=dict(timing(voltage),
                      first_V=first, baseline_first_second_median_V=baseline,
                      minimum_V=vmin, maximum_V=vmax, end_V=voltage[-1][1],
                      range_V=vmax-vmin, first_to_min_drop_V=first-vmin,
                      baseline_to_min_drop_V=baseline-vmin,
                      maximum_causal_drawdown_V=drawdown,
                      minimum_time_s=voltage[min_index][0],
                      observed_time_after_min_s=voltage[-1][0]-voltage[min_index][0]),
                      temperatures=temp_stats, recorded_temperature_max_C=tmax,
                      recorded_temperature_below_40_C=tmax < 40,
                      baseline_drop_ge_25mV=baseline-vmin >= .025 - 1e-9,
                      end_deficit_below_25mV=baseline-voltage[-1][1] < .025 - 1e-9,
                      end_deficit_equal_25mV_numeric_tolerance=abs(baseline-voltage[-1][1]-.025) <= 1e-9)
        return result
    except Exception as error:
        return dict(result, status="parser_error", error_type=type(error).__name__, error=str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", default="battery_raw_sources67.json")
    parser.add_argument("--metadata", default="battery_metadata/main.xlsx")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = json.loads((args.run / args.manifest).read_text())
    author = {}
    for rownum, row in workbook_rows(args.run / args.metadata,
                                     "xl/worksheets/sheet2.xml"):
        if rownum > 1 and row.get("A"):
            author[row["A"].casefold()] = {"observed_text": row.get("B"),
                "observed_category_raw": row.get("C"), "observed_score": number(row.get("C")), "calculated_score": number(row.get("D")),
                "capacity_mAh": number(row.get("E")), "SOC_percent": number(row.get("F"))}
    jobs = [(str(args.run), e, author.get(Path(e["filename"]).stem.casefold())) for e in entries]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit_one, jobs))
    good = [r for r in results if r["status"] == "parsed"]
    quiet = [r for r in good if r["recorded_temperature_below_40_C"]]
    quiet_drop = [r for r in quiet if r["baseline_drop_ge_25mV"]]
    summary = {"files_listed": len(entries), "status_counts": dict(Counter(r["status"] for r in results)),
        "matching_author_summary_rows": sum(r["author_summary"] is not None for r in results),
        "source_manifest": args.manifest, "source_metadata": args.metadata,
        "recorded_temperature_below_40_C": len(quiet), "quiet_with_baseline_drop_ge_25mV": len(quiet_drop),
        "quiet_drop_end_deficit_below_25mV": sum(r["end_deficit_below_25mV"] for r in quiet_drop),
        "quiet_drop_files": [r["filename"] for r in quiet_drop],
        "no_agent_experiment": True,
        "scope": "Retrospective sensor descriptions, not ground-truth ISC or health labels. Recorded temperatures are not core temperature. First-second baseline is a researcher-defined descriptive reference. Voltage thresholds have a 1e-9 V numerical tolerance to avoid floating-point equality errors; this is not sensor uncertainty. No cross-DAQ alignment or causal thermal attribution. Workbooks are not assumed to be distinct physical cells.",
        "source_gate": "No currently verified fault-exclusion calibration or maintenance release rule; do not launch a health-certification Agent experiment on these proxies."}
    for name, value in [("per_record.json", results), ("summary.json", summary)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
