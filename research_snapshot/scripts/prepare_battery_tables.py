"""Preserve every original discharge index, including missing NASA capacities."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, seen = [], set()
    for zip_path in sorted(args.run.glob('battery_raw/*/*.zip')):
        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist()
                     if Path(n).name.startswith('B00') and n.endswith('.mat')]
            for name in names:
                bid = Path(name).stem
                if bid in seen:
                    raise ValueError('Multiple original versions of cell: ' + bid)
                seen.add(bid)
                raw = archive.read(name)
                root = loadmat(io.BytesIO(raw), simplify_cells=True)[bid]
                rows, missing = [], []
                for source_cycle, cycle in enumerate(root['cycle'], 1):
                    if cycle['type'] != 'discharge':
                        continue
                    values = np.asarray(cycle['data']['Capacity']).ravel()
                    if values.size not in (0, 1):
                        raise ValueError('Unexpected nonscalar Capacity: ' + bid)
                    capacity = float(values[0]) if values.size else float('nan')
                    if values.size and not np.isfinite(capacity):
                        raise ValueError('Unexpected nonfinite recorded Capacity: ' + bid)
                    if not values.size:
                        missing.append(len(rows) + 1)
                    rows.append(dict(discharge_cycle=len(rows) + 1,
                                     source_cycle=source_cycle,
                                     ambient_temperature_C=float(cycle['ambient_temperature']),
                                     capacity_Ah=capacity))
                table = pd.DataFrame(rows)
                path = args.output / (bid + '.csv')
                table.to_csv(path, index=False)
                crossings = table.loc[table.capacity_Ah <= 1.4, 'discharge_cycle'].tolist()
                event = int(crossings[0]) if crossings else None
                records.append(dict(
                    battery=bid, cohort_zip=str(zip_path.relative_to(args.run)), MAT_member=name,
                    MAT_sha256=hashlib.sha256(raw).hexdigest(),
                    CSV_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    discharge_records=len(table), missing_capacity_cycles=missing,
                    first_threshold_cycle=event,
                    rebound_after_first_crossing=bool(event is not None and
                        (table.loc[table.discharge_cycle > event, 'capacity_Ah'] > 1.4).any()),
                    event_observed_by20=bool(event is not None and event <= 20),
                    all_capacity_observed_first20=bool(len(table) >= 20 and
                        table.iloc[:20].capacity_Ah.notna().all()),
                    ambient_temperature_values=sorted(table.ambient_temperature_C.unique().tolist())))
    result = dict(status='finished', checked_utc=datetime.now(timezone.utc).isoformat(),
                  source_cell_count=len(seen), records=records,
                  missing_policy='Retain discharge index; blank capacity; no imputation or reindexing',
                  event_definition='Strict first measured Capacity<=1.4Ah; not permanent physical failure')
    (args.output / 'TRANSFORM_MANIFEST.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='finished', physical_cells=len(seen),
                         missing_measurements=sum(len(x['missing_capacity_cycles']) for x in records),
                         rebound_cells=sum(x['rebound_after_first_crossing'] for x in records))), flush=True)


if __name__ == '__main__':
    main()
