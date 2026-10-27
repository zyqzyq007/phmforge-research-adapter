"""Independent cell-address extraction for two illustrative physical records."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract(path, sheet, xcol, ycols):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        tree = ET.fromstring(archive.read(sheet))
    cells = {}
    for cell in tree.iter(ns + "c"):
        value = cell.find(ns + "v")
        if value is not None and cell.get("t") != "s":
            try:
                cells[cell.get("r")] = float(value.text)
            except (ValueError, TypeError):
                pass
    out = {}
    for col in ycols:
        pairs = []
        for addr, value in cells.items():
            prefix = "".join(c for c in addr if c.isalpha())
            suffix = "".join(c for c in addr if c.isdigit())
            if prefix == col and int(suffix) > 1 and xcol + suffix in cells:
                pairs.append((int(suffix), cells[xcol + suffix], value))
        out[col] = [(x, y) for _, x, y in sorted(pairs)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    source = {r["filename"]: r for r in json.loads((args.run / "battery_v2_all_sources67.json").read_text())}
    audit = {r["filename"]: r for r in json.loads((args.run / "analysis_v2_repaired/per_record.json").read_text())}
    cases = [("SNL_LFP_Graphite_10Ah_0SOC_a.xlsx", "A", "D", "A", list("FGHIJK")),
             ("NMC_10000mAh-20SOC_cell1_MAX.xlsx", "A", "C", "E", ["F"])]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    checks = []
    for i, (name, vt, vc, tt, tc) in enumerate(cases):
        record = audit[name]
        path = args.run / source[name]["relative_path"]
        voltage = extract(path, record["worksheet"], vt, [vc])[vc]
        temps = extract(path, record["worksheet"], tt, tc)
        base = statistics.median([v for t, v in voltage if t <= voltage[0][0] + 1])
        peak_temp = max(t for channel in temps.values() for _, t in channel)
        expected = record["voltage"]
        check = dict(filename=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     n_voltage=len(voltage), baseline_V=base,
                     minimum_V=min(v for _, v in voltage), recorded_temperature_max_C=peak_temp)
        check["agrees_with_main_audit"] = (len(voltage) == expected["n"]
            and abs(base - expected["baseline_first_second_median_V"]) < 1e-12
            and abs(check["minimum_V"] - expected["minimum_V"]) < 1e-12
            and abs(peak_temp - record["recorded_temperature_max_C"]) < 1e-12)
        assert check["agrees_with_main_audit"], check
        checks.append(check)
        axs[i, 0].plot(*zip(*voltage), lw=1, color="#225ea8")
        axs[i, 0].axhline(base - .025, ls="--", lw=1, color="#b2182b", label="Reference baseline − 25 mV")
        for col, series in temps.items():
            if series:
                axs[i, 1].plot(*zip(*series), lw=.8, label=col)
        axs[i, 1].axhline(40, ls="--", lw=1, color="#b2182b", label="40 °C CHS boundary")
        axs[i, 0].set_title(name.removesuffix(".xlsx"), fontsize=9)
        axs[i, 1].set_title(f"Recorded channels: maximum {peak_temp:.2f} °C", fontsize=9)
        axs[i, 0].set_ylabel("Recorded voltage (V)")
        axs[i, 1].set_ylabel("Recorded temperature (°C)")
        for ax in axs[i]:
            ax.set_xlabel("Signal's recorded time (s)")
            ax.grid(alpha=.2)
        axs[i, 0].legend(fontsize=7)
        axs[i, 1].legend(fontsize=7, ncol=3)
    fig.suptitle("Thermally quiet records can contain electrical changes", fontsize=12)
    fig.savefig(args.run / "battery_scope67.png", dpi=180)
    fig.savefig(args.run / "battery_scope67.pdf")
    (args.run / "independent_cell_audit67.json").write_text(json.dumps({
        "checks": checks, "figure_scope": "Two illustrative, post-analysis selections; no ISC truth or maintenance clearance inferred. Each channel retains its source clock; no cross-DAQ alignment asserted."}, indent=2))


if __name__ == "__main__":
    main()
