"""Audit public physical sources for the negative-evidence research question.

Exploratory source qualification, not an Agent evaluation. Never changes sources.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from scipy.stats import nct, norm


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bounds(values):
    """Existing exact normal tolerance bound; log transform for lognormal."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    k = nct.ppf(.95, n - 1, norm.ppf(.9) * math.sqrt(n)) / math.sqrt(n)
    out = {}
    for name, z, inverse in [("normal", x, float), ("lognormal", np.log(x), np.exp)]:
        out[name] = {
            "a90_sample_sd": float(inverse(z.mean() + norm.ppf(.9) * z.std(ddof=1))),
            "a90_ML_sd": float(inverse(z.mean() + norm.ppf(.9) * z.std(ddof=0))),
            "a9095_exact": float(inverse(z.mean() + k * z.std(ddof=1))),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    run, out = args.run, args.output
    package = run / "RPOD/RPOD"
    md5_checks = []
    for line in (package / "MD5").read_text().splitlines():
        expected, name = line.split(" ", 1)
        name = name.strip().lstrip("*").strip('"')
        actual = hashlib.md5((package / name).read_bytes()).hexdigest()
        md5_checks.append({"path": name, "matched": actual == expected})
    assert all(x["matched"] for x in md5_checks)
    save(out / "package_integrity65.json", {"checks": md5_checks, "all_match": True})

    html = BeautifulSoup((run / "sources_default/shm_pod_2026_html.html").read_text(), "html.parser")
    tex = (run / "tex/SHM_POD.tex").read_text()
    families = [
        ("PZT", "CleanPZTData.csv", .05, 2, 17),
        ("CNT", "CNT_SHM_data.csv", 1., 3, 21),
        ("CVM", "CVM_SHM_data.csv", 1.5, 2, 24),
    ]
    results, all_curves, source_rows = [], [], []
    for (family, filename, threshold, digits, page), table in zip(families, html.select("figure.ltx_table")[:3]):
        reported = {}
        for tr in table.select("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if len(cells) != 3:
                continue
            specimen, lad, slope = cells
            assert re.search(re.escape(specimen) + r"\s*&\s*" + re.escape(lad) + r"\s*&\s*" + re.escape(slope), tex)
            reported[specimen] = float(lad)
            source_rows.append({"family": family, "specimen": specimen, "lad_mm": float(lad),
                                "slope": float(slope), "pdf_page": page, "html_tex_match": True})
        with (package / "RPODData" / filename).open() as handle:
            raw = list(csv.DictReader(handle))
        groups = {}
        for row_number, row in enumerate(raw, 2):
            if family == "PZT":
                specimen = row["Specimen - Path"].split()[0]
                x, y = float(row["Crack Length"]), float(row["DI Value"])
            else:
                specimen, x, y = row["unit"], float(row["x"]), float(row["y"])
            assert math.isfinite(x) and math.isfinite(y)
            groups.setdefault(specimen, []).append({"raw_line": row_number, "x_mm": x, "y": y})
        assert set(groups) == set(reported)
        curves, mismatches = [], []
        for specimen, observations in groups.items():
            assert all(a["x_mm"] <= b["x_mm"] for a, b in zip(observations, observations[1:]))
            positives = [i for i, r in enumerate(observations) if r["y"] > threshold]
            assert positives
            first = positives[0]
            after_negative = [r for r in observations[first+1:] if r["y"] <= threshold]
            lad = observations[first]["x_mm"]
            matched = abs(lad - reported[specimen]) <= .5 * 10 ** (-digits) + 1e-9
            curve = {"family": family, "specimen": specimen, "source_rows": len(observations),
                     "threshold": threshold, "first_observed_exceedance": observations[first],
                     "published_lad_mm": reported[specimen], "published_lad_matches_raw": matched,
                     "later_nonexceedances": after_negative,
                     "threshold_unit": "source damage index; not crack length", "source_file": filename}
            curves.append(curve)
            if not matched:
                mismatches.append(curve)
        all_curves.extend(curves)
        x = [r["first_observed_exceedance"]["x_mm"] for r in curves]
        results.append({"family": family, "raw_rows": len(raw), "specimens": len(groups),
                        "zero_zero_rows": sum(float(r.get("x", r.get("Crack Length", 0))) == 0 and float(r.get("y", r.get("DI Value", 0))) == 0 for r in raw),
                        "published_lad_matches": len(curves) - len(mismatches),
                        "mismatches": mismatches, "later_nonexceedance_specimens": sum(bool(r["later_nonexceedances"]) for r in curves),
                        "from_published_lad": bounds(list(reported.values())),
                        "from_raw_first_exceedance": bounds(x)})
    save(out / "shm_published_lad65.json", source_rows)
    save(out / "shm_curve_audit65.json", all_curves)
    save(out / "shm_numeric_audit65.json", {
        "analysis_type": "exploratory_source_qualification",
        "methods": "Threshold crossing in original row order; exact one-sided normal tolerance bound using noncentral t; lognormal via log transform. Not RPOD/Stan execution.",
        "limitations": "Raw package and paper versions are not assumed identical; discrepancies are source questions, not Agent errors. Normality, representativeness, independent specimen sampling and deployment equivalence are not established by arithmetic. No pooled commonality or field-frequency inference.",
        "families": results,
    })

    trials = []
    for method, page in [("A", 13), ("D", 16)]:
        rows = json.loads((run / f"extracted/nasa_table_page{page}_0.json").read_text())
        for cells in rows[1:]:
            if not cells[0]:
                continue
            assert len(cells) == 10
            date, inspector, penetrant, level, flaws, hits, misses, false_calls, logit, point = cells
            assert int(hits) + int(misses) == int(flaws)
            trials.append({"method": method, "date": date, "inspector": inspector,
                           "penetrant": penetrant, "sensitivity_level": int(level),
                           "flaws": int(flaws), "hits": int(hits), "misses": int(misses),
                           "false_calls": int(false_calls), "logit_a9095_inches": None if logit == "N/A" else float(logit),
                           "point_a9095_inches": None if point == "N/A" else float(point), "pdf_page": page})
    assert len(trials) == 44
    save(out / "nasa_physical_trials65.json", trials)
    source_summaries = []
    for threshold in [.05, .15]:
        point_pass = [r for r in trials if r["point_a9095_inches"] is not None and r["point_a9095_inches"] <= threshold]
        curve_pass = [r for r in trials if r["logit_a9095_inches"] is not None and r["logit_a9095_inches"] <= threshold]
        source_summaries.append({"threshold_inches": threshold, "trials": len(trials), "point_pass": len(point_pass),
                                 "point_above": sum(r["point_a9095_inches"] is not None and r["point_a9095_inches"] > threshold for r in trials),
                                 "point_NA": sum(r["point_a9095_inches"] is None for r in trials), "curve_pass": len(curve_pass),
                                 "curve_pass_without_point_pass": [r for r in curve_pass if r not in point_pass]})
    save(out / "nasa_qualification_audit65.json", {"scope": "Historical source-specific capability demonstration. Current health/release is not inferred. Original 44 trials are not 44 independent devices.",
                                                  "summaries": source_summaries,
                                                  "raw_per_crack_measurements_acquired": False})
    print(json.dumps({"shm_rows": sum(r["raw_rows"] for r in results), "shm_specimens": sum(r["specimens"] for r in results),
                      "shm_matches": {r["family"]: r["published_lad_matches"] for r in results},
                      "later_nonexceedance_specimens": {r["family"]: r["later_nonexceedance_specimens"] for r in results},
                      "nasa_summary": [{k: v for k, v in r.items() if k != "curve_pass_without_point_pass"} for r in source_summaries],
                      "LLM_calls": 0}))


if __name__ == "__main__":
    main()
