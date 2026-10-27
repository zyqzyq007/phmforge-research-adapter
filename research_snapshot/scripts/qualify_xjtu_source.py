"""Verify complete, publisher-hash-bound XJTU measured life histories.

No model fitting, Agent call, fault relabeling, or prediction scoring occurs.
Features are computed within each native 1.28-second recording. The source
archives, native values, nominal minute clock, and terminal failure modes remain
separate from any later train/development/test allocation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from native_archive_members import members
from prepare_fault_extensions import features

# Author's 2019 tutorial, Table 3. These are complete recording counts and
# inspected terminal damage locations, not per-minute fault-state labels.
COUNTS = {
    'Bearing1_1': 123, 'Bearing1_2': 161, 'Bearing1_3': 158,
    'Bearing1_4': 122, 'Bearing1_5': 52, 'Bearing2_1': 491,
    'Bearing2_2': 161, 'Bearing2_3': 533, 'Bearing2_4': 42,
    'Bearing2_5': 339, 'Bearing3_1': 2538, 'Bearing3_2': 2496,
    'Bearing3_3': 371, 'Bearing3_4': 1515, 'Bearing3_5': 114,
}
TERMINAL_DAMAGE = {
    'Bearing1_1': ['outer'], 'Bearing1_2': ['outer'], 'Bearing1_3': ['outer'],
    'Bearing1_4': ['cage'], 'Bearing1_5': ['inner', 'outer'],
    'Bearing2_1': ['inner'], 'Bearing2_2': ['outer'], 'Bearing2_3': ['cage'],
    'Bearing2_4': ['outer'], 'Bearing2_5': ['outer'],
    'Bearing3_1': ['outer'], 'Bearing3_2': ['inner', 'rolling_element', 'cage', 'outer'],
    'Bearing3_3': ['inner'], 'Bearing3_4': ['inner'], 'Bearing3_5': ['outer'],
}
CONDITIONS = {'35Hz12kN': (1, 2100, 12), '37.5Hz11kN': (2, 2250, 11),
              '40Hz10kN': (3, 2400, 10)}
HEADER = 'Horizontal_vibration_signals,Vertical_vibration_signals'


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_record(name, payload):
    m = re.fullmatch(r'XJTU-SY_Bearing_Datasets/([^/]+)/(Bearing[123]_[1-5])/(\d+)\.csv', name)
    if m is None:
        raise ValueError('Unexpected native CSV identity: ' + name)
    condition, bearing, minute_text = m.groups()
    minute = int(minute_text)
    cid, speed, force = CONDITIONS[condition]
    assert int(bearing[7]) == cid and 1 <= minute <= COUNTS[bearing]
    assert payload.splitlines()[0].decode().strip() == HEADER
    signal = np.loadtxt(io.BytesIO(payload), delimiter=',', skiprows=1, dtype=np.float64)
    assert signal.shape == (32768, 2) and np.isfinite(signal).all()
    assert np.all(signal.std(axis=0) > 0)
    independent = minute in (1, COUNTS[bearing])
    if independent:
        alternate = np.asarray([[float(x) for x in row]
            for row in list(csv.reader(io.StringIO(payload.decode())))[1:]], dtype=np.float64)
        assert np.array_equal(signal, alternate)
    f = features(signal)
    assert f.shape == (30,) and np.isfinite(f).all()
    row = dict(native_member=name, bearing=bearing, sample_index=minute,
        condition=cid, speed_rpm=speed, radial_force_kn=force,
        samples_per_channel=32768, channels=2, source_bytes=len(payload),
        csv_sha256=hashlib.sha256(payload).hexdigest(),
        float64_values_sha256=hashlib.sha256(signal.astype('<f8').tobytes()).hexdigest(),
        independent_python_float_parser_exact=independent,
        channel_rms=np.sqrt(np.mean(signal**2, axis=0)).tolist(),
        channel_max_abs=np.max(np.abs(signal), axis=0).tolist())
    return row, f


def main(root, workers, extracted_root=None):
    pins = json.loads((root / 'audit/XJTU_NATIVE_PIN277.json').read_text())['files']
    volumes = [root / 'source/native_xjtu' / p['filename'] for p in sorted(pins, key=lambda x:x['filename'])]
    for p, path in zip(sorted(pins, key=lambda x:x['filename']), volumes):
        assert path.stat().st_size == p['bytes'] and sha256(path) == p['publisher_sha256']
    destination = root / 'qualified_xjtu'
    destination.mkdir(exist_ok=False)
    status_path = root / 'xjtu_qualification_status.json'
    status = dict(status='running', utc=utc(), pid=os.getpid(), workers=workers,
        expected_native_records=sum(COUNTS.values()), verified_records=0,
        new_LLM_calls=0, training_jobs=0, final_scores_read=False)
    save(status_path, status)
    rows, feature_rows, ignored = [], [], []
    prior_path = root / 'audit/XJTU_NATIVE_RECORDS277_attempt1.jsonl'
    prior = {x['native_member']: x['csv_sha256'] for x in
        (json.loads(line) for line in prior_path.read_text().splitlines())} if prior_path.exists() else {}
    if extracted_root is not None:
        receipt = json.loads((root / 'audit/XJTU_EXTRACTION277.json').read_text())
        assert receipt['passed'] and receipt['archive_integrity_checked_during_extraction']
        assert extracted_root.resolve() == Path(receipt['extracted_root']).resolve()
        native_members = ((str(p.relative_to(extracted_root)), p.read_bytes())
                          for p in sorted(extracted_root.rglob('*')) if p.is_file())
    else:
        native_members = members(volumes)
    log = (root / 'audit/XJTU_NATIVE_RECORDS277.jsonl').open('w')

    def consume(future):
        row, f = future.result()
        if row['native_member'] in prior:
            assert row['csv_sha256'] == prior[row['native_member']]
        rows.append(row)
        feature_rows.append(f)
        log.write(json.dumps(row) + '\n')
        if len(rows) % 100 == 0:
            log.flush()
            status.update(utc=utc(), verified_records=len(rows))
            save(status_path, status)
            print(json.dumps(dict(verified_records=len(rows), expected=sum(COUNTS.values()))), flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            for name, payload in native_members:
                if not name.endswith('.csv'):
                    ignored.append(dict(name=name, bytes=len(payload)))
                    continue
                pending.add(pool.submit(parse_record, name, payload))
                if len(pending) >= 2 * workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume(future)
            for future in pending:
                consume(future)
        assert len(rows) == sum(COUNTS.values())
        assert len({x['native_member'] for x in rows}) == len(rows)
        assert sum(x['independent_python_float_parser_exact'] for x in rows) == 30
        saved = []
        for bearing, n in COUNTS.items():
            take = sorted([i for i,x in enumerate(rows) if x['bearing'] == bearing],
                          key=lambda i:rows[i]['sample_index'])
            indices = np.asarray([rows[i]['sample_index'] for i in take])
            assert np.array_equal(indices, np.arange(1, n+1))
            path = destination / (bearing+'.npz')
            np.savez_compressed(path, features=np.asarray([feature_rows[i] for i in take]),
                native_sample_index=indices, nominal_elapsed_minutes=indices,
                nominal_remaining_minutes=n-indices,
                channel_rms=np.asarray([rows[i]['channel_rms'] for i in take]),
                channel_max_abs=np.asarray([rows[i]['channel_max_abs'] for i in take]))
            saved.append(dict(bearing=bearing, recordings=n, nominal_end_minute=n,
                terminal_damage_locations=TERMINAL_DAMAGE[bearing], path=str(path.relative_to(root)),
                sha256=sha256(path)))
        duplicate_hashes = {h:n for h,n in Counter(x['csv_sha256'] for x in rows).items() if n>1}
        report = dict(utc=utc(), passed=True, native_bearings=15, native_records=len(rows),
            values_checked=len(rows)*32768*2, publisher_volume_sha256_all_match=True,
            native_multivolume_archive_read_completed=True, all_native_record_shapes=(32768,2),
            all_values_finite=True, all_record_numbers_contiguous=True,
            independent_double_parser_records=30, independent_double_parser_all_exact=True,
            previous_decoder_complete_records_matched=len(prior),
            native_decoder='7-Zip26.05 with extractionCRC' if extracted_root else 'libarchive',
            duplicate_native_csv_hashes=duplicate_hashes, ignored_archive_entries=ignored,
            nominal_sampling_hz=25600, nominal_recording_duration_seconds=1.28,
            nominal_recording_interval_minutes=1, channels=['horizontal acceleration','vertical acceleration'],
            conditions=CONDITIONS, bearings=saved,
            labels='N-i minutes to published terminal recording, on the nominal one-minute schedule. Original endpoint retained; no new threshold or percentage labels.',
            failure_definition='Author tutorial section3.2: stop when either measured direction exceeds10 times its normal-stage maximum amplitude.',
            terminal_damage_is_not_every_minute_fault_state=True,
            classification_task_qualified=False,
            rul_measured_source_qualified=True, original_PHMForge_score_qualified=False,
            limitations=['Nominal minute labels; exact physical failure time between acquisitions is unavailable.',
                'Laboratory accelerated degradation, not field maintenance outcomes.',
                'Public PHMForge XJTU RUL prompt/configuration mismatch must be explicitly adapted.',
                'No train/development/test allocation or fit has occurred in this source stage.'],
            new_LLM_calls=0, training_jobs=0, prediction_scores=0)
        save(root / 'audit/XJTU_SOURCE_QUALIFICATION277.json', report)
        project = Path(__file__).resolve().parents[1]
        save(project / 'results/source277/XJTU_SOURCE_QUALIFICATION277.json', report)
        status.update(status='finished', verified_records=len(rows), passed=True)
        print(json.dumps({k:report[k] for k in ['passed','native_bearings','native_records','values_checked','training_jobs','prediction_scores']}),flush=True)
    except BaseException as exc:
        status.update(status='execution_error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        log.close()
        status['ended_utc'] = utc()
        save(status_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--workers', default=2, type=int, choices=range(1,5))
    parser.add_argument('--extracted-root', type=Path)
    args = parser.parse_args()
    main(args.run.resolve(), args.workers, args.extracted_root)
