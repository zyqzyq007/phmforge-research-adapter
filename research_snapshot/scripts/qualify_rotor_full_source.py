"""Qualify complete measured Rotor MAT7.3 records without fitting or scoring.

Whole-record hashes bind the previously opened short source. New raw exports
are losslessly checked; sampling rates remain publisher declarations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import traceback
import zipfile

import h5py
import numpy as np


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(array):
    array = np.asarray(array, dtype='<f8').reshape(-1)
    return [array.size, hashlib.sha256(array.tobytes()).hexdigest()]


def key(fingerprints):
    return tuple(tuple(x) for x in fingerprints)


def audit_member(run_name, member, deadline, audit_missing=False):
    run = Path(run_name)
    protocol = json.loads((run / 'protocol.json').read_text())
    previous = json.loads((run / 'inputs/INDEPENDENT_SOURCE_CHECK271.json').read_text())
    physical = json.loads((run / 'inputs/PHYSICAL_INFO271.json').read_text())
    opened = {key(x['channel_fingerprints']): x
              for x in previous['rows'] + previous['unused_upstream_experiments']}
    severity, = {x['severity'] for x in opened.values() if x['member'] == member}
    channels = physical['channels']
    status_path = run / 'audit/members' / (member + '.status.json')
    def update(phase, **extra):
        save(status_path, dict(status='running', phase=phase, member=member,
                              utc=datetime.now(timezone.utc).isoformat(), **extra))
    try:
        update('extracting')
        native = run / 'clients-cache/native' / member
        native.parent.mkdir(parents=True, exist_ok=True)
        source = Path(protocol['source_archive']) / protocol['source_ZIP_relative']
        with zipfile.ZipFile(source) as archive:
            info = archive.getinfo(member)
            if native.exists():
                first = json.loads((run / 'audit/FIRST_NATIVE_EXTRACTION304.json').read_text())
                if first['member'] == member:
                    assert first['read_completed_with_CRC']
                    assert native.stat().st_size == first['bytes'] == info.file_size
                    assert digest(native) == first['native_sha256']
                else:
                    # A failed traversal may leave a complete MAT but no receipt.
                    # Re-read the ZIP with CRC and compare the whole cached file.
                    h = hashlib.sha256()
                    with archive.open(info) as src:
                        while block := src.read(8 << 20):
                            assert time.time() < deadline
                            h.update(block)
                    assert native.stat().st_size == info.file_size
                    assert digest(native) == h.hexdigest()
            else:
                temporary = native.with_suffix('.extracting')
                with archive.open(info) as src, temporary.open('wb') as dst:
                    while block := src.read(8 << 20):
                        assert time.time() < deadline, 'Native audit deadline exceeded'
                        dst.write(block)
                assert temporary.stat().st_size == info.file_size
                temporary.replace(native)
            member_receipt = dict(member=member, native_bytes=info.file_size,
                native_sha256=digest(native), ZIP_CRC32=f'{info.CRC:08x}',
                extraction_CRC_verified=True)
            save(run / 'audit/members' / (member + '.extraction.json'), member_receipt)
        records = []
        with h5py.File(native, 'r') as handle:
            root, = [x for x in handle if x != '#refs#']
            loads = sorted(handle[root])
            assert loads == ['torque05', 'torque10', 'torque15', 'torque20',
                             'torque25', 'torque30', 'torque35', 'torque40']
            for load in loads:
                group = handle[root][load]
                fields = sorted(group)
                assert fields == sorted(channels + ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', 'Trigger'])
                assert all(group[x].shape == (10, 1) and
                           h5py.check_dtype(ref=group[x].dtype) is not None for x in fields)
                for ordinal in range(10):
                    assert time.time() < deadline, 'Native audit deadline exceeded'
                    vibs, field_info = {}, {}
                    for field in fields:
                        target = handle[group[field][ordinal, 0]]
                        assert target.ndim == 2 and 1 in target.shape
                        assert target.dtype.kind == 'f' and target.dtype.itemsize == 8
                        values = np.asarray(target[...], dtype='<f8').reshape(-1)
                        assert values.size > 0
                        finite = np.isfinite(values)
                        if not audit_missing:
                            assert finite.all(), f'{member}/{load}/{ordinal + 1}/{field} nonfinite'
                        field_info[field] = dict(target=target.name, shape=list(target.shape),
                            dtype=str(target.dtype), fingerprint=fingerprint(values),
                            finite=bool(finite.all()), finite_count=int(finite.sum()),
                            nan_count=int(np.isnan(values).sum()),
                            positive_infinity_count=int(np.isposinf(values).sum()),
                            negative_infinity_count=int(np.isneginf(values).sum()),
                            nonfinite_indices_first16=np.flatnonzero(~finite)[:16].tolist())
                        if field in channels:
                            vibs[field] = values
                    fingerprints = [field_info[x]['fingerprint'] for x in channels]
                    assert len({x[0] for x in fingerprints}) == 1
                    old = opened.get(key(fingerprints))
                    if old:
                        assert old['member'] == member and old['load'] == load
                        assert old['severity'] == severity
                    record = dict(member=member, root=root, load=load, acquisition=ordinal + 1,
                        broken_bars=severity, load_nm=int(load[6:]) / 10,
                        points_per_vibration_channel=fingerprints[0][0],
                        channel_fingerprints=fingerprints, fields=field_info,
                        previously_opened=old is not None,
                        old_source_identity=({k: old[k] for k in ['member', 'load', 'experiment']}
                                             if old else None))
                    if old is None:
                        output = run / 'prepared/ROTOR_FULL_NEW' / (
                            f'{Path(member).stem}_{load}_acquisition{ordinal + 1:02d}.npz')
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if not output.exists():
                            np.savez_compressed(output,
                                signals=np.stack([vibs[x] for x in channels], axis=1),
                                channels=np.asarray(channels), broken_bars=np.asarray(severity),
                                load_nm=np.asarray(record['load_nm']), acquisition=np.asarray(ordinal + 1),
                                sampling_hz=np.asarray(physical['vibration_sampling_hz_publisher_declared']))
                        with np.load(output, allow_pickle=False) as reloaded:
                            assert reloaded['channels'].tolist() == channels
                            assert [fingerprint(reloaded['signals'][:, i]) for i in range(5)] == fingerprints
                            assert reloaded['broken_bars'].item() == severity
                        record.update(export_path=str(output.relative_to(run)),
                                      export_sha256=digest(output), export_bytes=output.stat().st_size,
                                      export_reload_all_values_hash_equal=True)
                    records.append(record)
                    update('reading_and_exporting', records_complete=len(records),
                           old_records=sum(x['previously_opened'] for x in records))
                    print(json.dumps(dict(member=member, records=len(records),
                                          old=records[-1]['previously_opened'])), flush=True)
        result = dict(status='completed', member=member, receipt=member_receipt,
                      records=records, record_count=len(records))
        save(run / 'audit/members' / (member + '.json'), result)
        save(status_path, dict(status='completed', member=member, records_complete=len(records)))
        return result
    except Exception:
        save(status_path, dict(status='failed', member=member, error=traceback.format_exc()))
        raise


def main(run, audit_missing=False):
    protocol = json.loads((run / 'protocol.json').read_text())
    started = datetime.fromisoformat(protocol['utc']).timestamp()
    deadline = started + protocol['time_budget_seconds']
    assert time.time() < deadline
    previous = json.loads((run / 'inputs/INDEPENDENT_SOURCE_CHECK271.json').read_text())
    old_rows = previous['rows'] + previous['unused_upstream_experiments']
    old_keys = {key(x['channel_fingerprints']) for x in old_rows}
    assert previous['passed'] and len(old_rows) == len(old_keys) == 80
    source = Path(protocol['source_archive'])
    receipt = json.loads((source / 'ARCHIVE_VERIFIED.json').read_text())
    assert receipt['status'] == 'verified' and receipt['source_cleanup'] == 'complete'
    assert digest(source / 'SOURCE_MANIFEST.json') == protocol['source_manifest_sha256']
    inventory = json.loads((source / 'SOURCE_MANIFEST.json').read_text())['files']
    bound, = [x for x in inventory if x['path'] == protocol['source_ZIP_relative']]
    assert bound['sha256'] == protocol['source_ZIP_sha256']
    assert bound['bytes'] == (source / bound['path']).stat().st_size == protocol['source_ZIP_bytes']
    members = sorted({x['member'] for x in old_rows})
    assert len(members) == 5
    assert shutil.disk_usage(run).free > 12 * (1 << 30)
    save(run / 'status.json', dict(status='running', phase='native_record_audit',
        new_fits=0, new_LLM_calls=0, members=members))
    results = []
    with ProcessPoolExecutor(max_workers=protocol['native_member_workers'],
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = []
        for member in members:
            cached = run / 'audit/members' / (member + '.json')
            if audit_missing and cached.exists():
                result = json.loads(cached.read_text())
                assert result['status'] == 'completed' and result['record_count'] == 80
                assert result['member'] == member
                for row in result['records']:
                    if 'export_path' in row:
                        assert digest(run / row['export_path']) == row['export_sha256']
                results.append(result)
            else:
                futures.append(pool.submit(audit_member, str(run), member, deadline, audit_missing))
        for future in as_completed(futures):
            results.append(future.result())
    rows = [x for result in results for x in result['records']]
    all_keys = [key(x['channel_fingerprints']) for x in rows]
    actual_old = {key(x['channel_fingerprints']) for x in rows if x['previously_opened']}
    assert len(rows) == 400 and len(set(all_keys)) == 400
    assert actual_old == old_keys
    new_rows = [x for x in rows if not x['previously_opened']]
    assert len(new_rows) == 320
    channels = json.loads((run / 'inputs/PHYSICAL_INFO271.json').read_text())['channels']
    clean = [x for x in new_rows if all(x['fields'][c]['finite'] for c in channels)]
    missing = [dict(member=x['member'], load=x['load'], acquisition=x['acquisition'],
                    previously_opened=x['previously_opened'],
                    fields={c: v for c, v in x['fields'].items() if not v['finite']})
               for x in rows if any(not v['finite'] for v in x['fields'].values())]
    all_finite = not missing
    result = dict(iteration='A304', source_identity_qualified=True,
        source_qualified=all_finite, original_all_fields_finite_criterion_passed=all_finite,
        native_records=len(rows),
        unique_five_vibration_channel_records=len(set(all_keys)), old_opened_records_matched=80,
        new_whole_acquisitions=320, exported_records=len(new_rows),
        all12fields_finite=all_finite,
        quality_audit_after_original_gate_failure=audit_missing,
        newly_exported_clean_vibration_records=len(clean),
        new_records_with_nonfinite_vibration=len(new_rows) - len(clean),
        nonfinite_record_details=missing, missing_values_replaced_or_imputed=False,
        no_modeling_population_automatically_selected=True,
        all_selected_vibration_exports_reload_hash_identical=True,
        all80old_records_bit_identical_across_shortMAT_and_fullHDF5=True,
        new_acquisition_ordinals=sorted({x['acquisition'] for x in new_rows}),
        old_to_full_index_mapping=sorted({(x['old_source_identity']['experiment'], x['acquisition'])
                                        for x in rows if x['previously_opened']}),
        values_checked_all_fields=sum(sum(y['fingerprint'][0] for y in x['fields'].values())
                                            for x in rows),
        new_export_bytes=sum(x['export_bytes'] for x in new_rows),
        new_vibration_values=sum(x['points_per_vibration_channel'] * 5 for x in new_rows),
        calibrated_amplitude_units='unverified; raw scale retained',
        sampling_hz_publisher_declared=dict(vibration=7600, electrical=50000, Trigger=None),
        no_independent_physical_motor_count_claim=True,
        independent_reader_scope='Old80 match SciPy-decoded source; new320 use h5py plus NPZ reload, not a second independent HDF5 engine.',
        no_new_source_family_claim=True, new_LLM_calls=0, new_fits=0,
        new_model_predictions=0, new_test_scores=0, confirmed_Agent_mechanisms=0,
        formal_training_and_Agent_protocol_still_required=True,
        utc=datetime.now(timezone.utc).isoformat())
    save(run / 'audit/FULL_NATIVE_RECORD_INDEX304.json', dict(records=sorted(rows,
        key=lambda x: (x['member'], x['load'], x['acquisition']))))
    save(run / 'audit' / ('SOURCE_IDENTITY_AND_QUALITY304.json' if audit_missing
                         else 'SOURCE_QUALIFICATION304.json'), result)
    save(run / 'status.json', dict(status='completed', **result))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit-missing-with-reuse', action='store_true',
                        help='Read-only quality accounting after a retained finite-gate failure.')
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.audit_missing_with_reuse)
    except Exception:
        save(args.run / 'status.json', dict(status='failed', phase='native_record_audit',
             error=traceback.format_exc(), new_fits=0, new_LLM_calls=0))
        raise
