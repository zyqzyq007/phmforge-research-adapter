"""Qualify a real Drive Stats cohort for the frozen negative-evidence task.

Reads the original quarter ZIP, verifies CRCs, preserves measured SMART fields,
and keeps missing follow-up separate from observed events. The risk rule is a
declared research control, not a manufacturer health certificate or new method.
No model/API calls, downloads, training, or test-dependent threshold selection.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import pandas as pd
from scipy.stats import beta

SMART = [f"smart_{n}_raw" for n in (5, 187, 188, 197, 198)]
META = ["date", "serial_number", "model", "capacity_bytes", "failure"]
ANCHORS = ("2016-01-01", "2016-02-01")
ENDPOINTS = ("2016-01-08", "2016-02-08")
LIMITS = (0.001, 0.005)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def csv_gzip(frame, path):
    # Stable gzip headers make cold source replays byte-comparable.
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            frame.to_csv(text, index=False)


def partition(model, serial):
    return "calibration" if hashlib.sha256(f"PHM69|{model}|{serial}".encode()).digest()[0] < 179 else "query"


def check(values):
    valid = [bool(re.fullmatch(r"\+?\d+(?:\.0+)?", str(v).strip())) for v in values]
    positive = any(ok and int(str(v).strip().split(".")[0]) > 0 for ok, v in zip(valid, values))
    return "positive" if positive else ("negative" if all(valid) else "unknown")


def upper(n, k):
    return float(beta.ppf(0.95, k + 1, n - k)) if n and k < n else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    visible = args.output / "visible_source"
    visible.mkdir()
    base_day = date(2016, 1, 1).toordinal()
    anchors, presence, events = {}, defaultdict(int), defaultdict(list)
    model_history = defaultdict(list)
    manifest, quality = [], []
    all_models, total_rows = Counter(), 0
    failure_checks = Counter()
    with zipfile.ZipFile(args.zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Original ZIP CRC failure: {bad}")
        members = sorted((m for m in archive.infolist()
                          if re.fullmatch(r"2016-\d{2}-\d{2}\.csv", Path(m.filename).name)),
                         key=lambda m: Path(m.filename).name)
        assert len({Path(m.filename).name for m in members}) == len(members)
        for member in members:
            day = Path(member.filename).stem
            ordinal = date.fromisoformat(day).toordinal()
            keep_full = ANCHORS[0] <= day <= ANCHORS[1]
            with archive.open(member) as handle:
                frame = pd.read_csv(handle, dtype=str, keep_default_na=False,
                                    usecols=None if keep_full else lambda c: c in META + SMART)
            assert all(c in frame for c in META), (day, frame.columns.tolist())
            assert frame["date"].eq(day).all(), day
            assert frame["failure"].isin(["0", "1"]).all(), day
            absent_columns = [c for c in SMART if c not in frame]
            for column in absent_columns:
                frame[column] = ""
            keys = list(zip(frame["model"], frame["serial_number"]))
            duplicate = frame.duplicated(["model", "serial_number"], keep=False)
            quality.append({"date": day, "rows": len(frame), "duplicate_key_rows": int(duplicate.sum()),
                            "empty_serial_rows": int(frame["serial_number"].eq("").sum()),
                            "missing_SMART_columns": absent_columns,
                            "missing_SMART_values": {c: int(frame[c].eq("").sum()) for c in SMART}})
            bit = 1 << (ordinal - base_day)
            for key, failure in zip(keys, frame["failure"]):
                presence[key] |= bit
                if failure == "1":
                    events[key].append(ordinal)
                serial = key[1]
                if not model_history[serial] or model_history[serial][-1][0] != key[0]:
                    model_history[serial].append((key[0], ordinal))
            all_models.update(frame["model"])
            total_rows += len(frame)
            flagged = frame.loc[frame["failure"].eq("1")]
            for values in flagged[SMART].itertuples(index=False, name=None):
                failure_checks[check(values)] += 1
            if day in ANCHORS:
                anchors[day] = frame.copy()
            if ANCHORS[0] <= day < ANCHORS[1]:
                csv_gzip(frame, visible / (day + ".csv.gz"))
            manifest.append({"member": member.filename, "bytes": member.file_size,
                             "compressed_bytes": member.compress_size, "crc32": f"{member.CRC:08x}",
                             "rows": len(frame), "date": day})
            if len(manifest) % 15 == 0:
                print(json.dumps({"days_read": len(manifest), "rows_read": total_rows}), flush=True)

    if not all(day in anchors for day in ANCHORS):
        raise ValueError("Frozen anchor date missing; do not substitute another date after inspection")
    known_days = {m["date"] for m in manifest}
    cohort_rows = []
    for role, anchor, endpoint in zip(("calibration", "query"), ANCHORS, ENDPOINTS):
        frame = anchors[anchor]
        ao, eo = date.fromisoformat(anchor).toordinal(), date.fromisoformat(endpoint).toordinal()
        duplicate_keys = set(zip(frame.loc[frame.duplicated(["model", "serial_number"], keep=False), "model"],
                                 frame.loc[frame.duplicated(["model", "serial_number"], keep=False), "serial_number"]))
        for row in frame.to_dict("records"):
            key = (row["model"], row["serial_number"])
            if partition(*key) != role:
                continue
            past_models = {m for m, t in model_history[key[1]] if t <= ao}
            prior_event = any(t <= ao for t in events[key])
            admissible = bool(key[1]) and key not in duplicate_keys and len(past_models) == 1 and not prior_event
            occurred = any(ao < t <= eo for t in events[key])
            seen_end = bool(presence[key] & (1 << (eo - base_day)))
            outcome = "event" if occurred else ("observed_no_event" if seen_end else "unknown_followup")
            cohort_rows.append({"role": role, "anchor": anchor, "endpoint": endpoint,
                                "model": key[0], "serial_number": key[1], "admissible_at_anchor": admissible,
                                "current_recorded_event": row["failure"], "prior_recorded_event": prior_event,
                                "ambiguous_serial_model_at_anchor": len(past_models) != 1,
                                "check": check([row[c] for c in SMART]), "outcome": outcome})
        if role == "query":
            mask = [partition(m, s) == role for m, s in zip(frame["model"], frame["serial_number"])]
            csv_gzip(frame.loc[mask], visible / "query_2016-02-01.csv.gz")

    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(args.output / "cohorts_hidden_outcomes.csv", index=False)
    calibration = cohorts[(cohorts.role == "calibration") & cohorts.admissible_at_anchor & (cohorts.check == "negative")]
    counts = {}
    for model, group in calibration.groupby("model", sort=True):
        freq = Counter(group.outcome)
        n = len(group)
        k = freq["event"] + freq["unknown_followup"]
        counts[model] = {"model": model, "n": n, "events": freq["event"],
                         "unknown_followup": freq["unknown_followup"],
                         "observed_no_event": freq["observed_no_event"], "upper_95": upper(n, k),
                         "naive_unknown_as_no_event_upper_95": upper(n, freq["event"])}
    pd.DataFrame(list(counts.values())).to_csv(args.output / "reference_calibration.csv", index=False)
    reference = []
    for row in cohorts[cohorts.role == "query"].to_dict("records"):
        c = counts.get(row["model"])
        for limit in LIMITS:
            permission = row["admissible_at_anchor"] and row["check"] == "negative" and c is not None and c["upper_95"] <= limit
            reference.append({**row, "risk_limit": limit, "calibration_n": c["n"] if c else 0,
                              "upper_95": c["upper_95"] if c else None,
                              "permission_under_control_policy": bool(permission),
                              "required_monitoring": "unchanged routine monitoring",
                              "scope": "provider_recorded_event_within_7_days"})
    pd.DataFrame(reference).to_csv(args.output / "reference_decisions_hidden.csv", index=False)
    dump(args.output / "members.json", manifest)
    dump(args.output / "quality.json", quality)
    decision_counts = Counter(str(r["permission_under_control_policy"]) for r in reference)
    role_summary = {}
    for role, group in cohorts.groupby("role", sort=True):
        role_summary[role] = {"anchor_records": len(group), "unique_model_serials": len(group[["model", "serial_number"]].drop_duplicates()),
                              "admissible": int(group.admissible_at_anchor.sum()),
                              "checks": dict(Counter(group.check)), "outcomes": dict(Counter(group.outcome))}
    dump(args.output / "summary.json", {
        "zip_bytes": args.zip.stat().st_size, "zip_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        "all_zip_members_crc_verified": True, "daily_csv_members": len(manifest), "source_rows": total_rows,
        "distinct_model_serials": len(presence), "distinct_serials": len(model_history),
        "models": len(all_models), "anchor_cohorts": role_summary,
        "all_followup_endpoints_present": all(d in known_days for d in ENDPOINTS),
        "failure_day_rows_by_check": dict(failure_checks), "calibration_negative_models": len(counts),
        "control_policy_decisions": dict(decision_counts),
        "both_action_controls_present": bool(decision_counts["True"] and decision_counts["False"]),
        "source_task_families": 1, "LLM_calls": 0,
        "qualification_limitations": ["Exchangeability and unchanged monitoring are explicit benchmark assumptions, not field guarantees.",
                                      "Provider events include proactive removal; not necessarily catastrophic physical failure.",
                                      "One-sided confidence bounds are per-model; no simultaneous all-model guarantee."]})
    print((args.output / "summary.json").read_text())


if __name__ == "__main__":
    main()
