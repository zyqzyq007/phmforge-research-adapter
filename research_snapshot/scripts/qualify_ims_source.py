"""Audit native IMS measurements and recording endpoints; never create RUL gold.

All records are retained in the census. Native terminal timestamps, README
claims, physical-bearing outcomes and any later target definition stay separate.
No fitting, scoring, temporal fault labels, or synthetic measurements occur.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

from native_archive_members import members
from prepare_fault_extensions import features


EXPECTED = {
    '1st_test': dict(records=2156, start='2003.10.22.12.06.24', end='2003.11.25.23.39.56',
                     channels=8, documented_terminal_defects={'3':'inner race', '4':'rolling element'}),
    '2nd_test': dict(records=984, start='2004.02.12.10.32.39', end='2004.02.19.06.22.39',
                     channels=4, documented_terminal_defects={'1':'outer race'}),
    'Set3_in_README': dict(records=4448, start='2004.03.04.09.27.46', end='2004.04.04.19.01.57',
                     channels=4, documented_terminal_defects={'3':'outer race'}),
}
TIME_FORMAT = '%Y.%m.%d.%H.%M.%S'


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_record(name, payload):
    parts = Path(name).parts
    groups = set(parts) & {'1st_test','2nd_test','3rd_test','4th_test'}
    if len(groups) != 1:
        raise ValueError('Ambiguous native acquisition folder: '+name)
    group = groups.pop()
    stamp = Path(name).name
    datetime.strptime(stamp, TIME_FORMAT)
    channels = 8 if group == '1st_test' else 4
    signal = np.loadtxt(io.BytesIO(payload), dtype=np.float64)
    if signal.shape != (20480, channels) or not np.isfinite(signal).all():
        raise ValueError('Native sample shape or finite-value check failed: '+name)
    values = features(signal).reshape(15, channels).T
    if values.shape != (channels, 15) or not np.isfinite(values).all():
        raise ValueError('Per-record features invalid: '+name)
    row = dict(native_member=name, acquisition=group, timestamp=stamp, channels=channels,
        samples_per_channel=20480, source_bytes=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        float64_sha256=hashlib.sha256(signal.astype('<f8').tobytes()).hexdigest(),
        constant_channels=(np.ptp(signal, axis=0)==0).tolist(),
        channel_rms=np.sqrt(np.mean(signal**2, axis=0)).tolist(),
        channel_max_abs=np.max(np.abs(signal), axis=0).tolist())
    return row, values


def main(root, workers):
    receipt = json.loads((root/'ims_native_parallel_manifest.json').read_text())
    source = root/receipt['file']
    assert receipt['zip_all_member_CRCS_verified'] and source.stat().st_size == receipt['bytes']
    assert sha256(source) == receipt['sha256']
    output = root/'qualified_ims'
    output.mkdir(exist_ok=False)
    (output/'metadata').mkdir()
    (output/'nested').mkdir()
    state_path = root/'ims_qualification_status.json'
    state = dict(status='running', pid=os.getpid(), started_utc=utc(), verified_records=0,
                 new_LLM_calls=0, training_fits=0, prediction_scores=0)
    save(state_path, state)
    rows, extracted_features, inventory, extremes = [], [], [], {}
    native_log = (root/'audit/IMS_NATIVE_RECORDS280.jsonl').open('w')
    try:
        native = output/'IMS.7z'
        with zipfile.ZipFile(source) as archive:
            selected = [x for x in archive.infolist() if not x.is_dir()]
            assert len(selected)==1 and selected[0].filename=='4. Bearings/IMS.7z'
            item = selected[0]
            with archive.open(item) as stream, native.open('xb') as target:
                count = 0
                while block := stream.read(1024**2):
                    count += len(block)
                    assert count <= item.file_size
                    target.write(block)
            assert count == item.file_size
        native_integrity = dict(path=str(native.relative_to(root)), bytes=count,
            sha256=sha256(native), source_member_CRC32=f'{item.CRC:08x}', source_member_CRC_verified=True)

        def descend(path, depth=0):
            if depth>2:
                raise ValueError('Unexpected archive nesting depth')
            for name, payload in members(path, maximum_member_bytes=2*1024**3):
                inventory.append(dict(archive=str(path.relative_to(root)), member=name, bytes=len(payload),
                                      sha256=hashlib.sha256(payload).hexdigest()))
                if Path(name).suffix.lower() in {'.7z','.zip','.rar'}:
                    child = output/'nested'/(hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+Path(name).name)
                    child.write_bytes(payload)
                    yield from descend(child, depth+1)
                else:
                    yield name, payload

        def consume(future):
            row, values = future.result()
            rows.append(row)
            extracted_features.append(values)
            native_log.write(json.dumps(row)+'\n')
            if len(rows)%200==0:
                native_log.flush()
                state.update(verified_records=len(rows), updated_utc=utc())
                save(state_path, state)
                print(json.dumps(dict(verified_records=len(rows))), flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            for name, payload in descend(native):
                filename = Path(name).name
                if not re.fullmatch(r'\d{4}(?:\.\d{2}){5}', filename):
                    if payload and ('readme' in filename.lower() or filename.lower().endswith('.pdf')):
                        safe = hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+filename
                        (output/'metadata'/safe).write_bytes(payload)
                    continue
                group_names = set(Path(name).parts) & {'1st_test','2nd_test','3rd_test','4th_test'}
                assert len(group_names)==1
                group = next(iter(group_names))
                if group not in extremes:
                    extremes[group] = {'first':(filename,payload),'last':(filename,payload)}
                if filename < extremes[group]['first'][0]: extremes[group]['first']=(filename,payload)
                if filename > extremes[group]['last'][0]: extremes[group]['last']=(filename,payload)
                pending.add(pool.submit(parse_record, name, payload))
                if len(pending)>=2*workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done: consume(future)
            for future in pending: consume(future)
        assert rows and len({x['native_member'] for x in rows})==len(rows)
        independent = []
        for group, endpoints in extremes.items():
            for kind, (stamp,payload) in endpoints.items():
                first = np.loadtxt(io.BytesIO(payload), dtype=np.float64)
                alternate = np.asarray([[float(x) for x in line.split()] for line in payload.decode().splitlines()])
                assert np.array_equal(first, alternate)
                independent.append(dict(acquisition=group, endpoint=kind, timestamp=stamp,
                                        compared_values=int(first.size), all_equal=True))

        groups = []
        for group in sorted({x['acquisition'] for x in rows}):
            take = sorted([i for i,x in enumerate(rows) if x['acquisition']==group], key=lambda i:rows[i]['timestamp'])
            stamps = [rows[i]['timestamp'] for i in take]
            assert len(stamps)==len(set(stamps))
            times = [datetime.strptime(t,TIME_FORMAT) for t in stamps]
            elapsed = np.asarray([(t-times[0]).total_seconds()/60 for t in times])
            gaps = np.diff(elapsed)
            assert (gaps>0).all()
            path = output/(group+'.npz')
            np.savez_compressed(path, per_channel_features=np.asarray([extracted_features[i] for i in take]),
                native_timestamps=np.asarray(stamps), calendar_clock_elapsed_minutes=elapsed,
                channel_rms=np.asarray([rows[i]['channel_rms'] for i in take]),
                channel_max_abs=np.asarray([rows[i]['channel_max_abs'] for i in take]))
            expected = EXPECTED[group] if group in EXPECTED else EXPECTED['Set3_in_README']
            groups.append(dict(native_folder=group, records=len(take), channels=rows[take[0]]['channels'],
                first_timestamp=stamps[0], last_timestamp=stamps[-1],
                calendar_clock_duration_minutes=float(elapsed[-1]),
                timestamp_gaps_minutes=dict(minimum=float(gaps.min()), maximum=float(gaps.max()),
                    counts={str(k):v for k,v in Counter(gaps.tolist()).items()}),
                README_candidate_correspondence=expected,
                README_record_count_equal=len(take)==expected['records'],
                README_first_timestamp_equal=stamps[0]==expected['start'],
                README_last_timestamp_equal=stamps[-1]==expected['end'],
                original_native_folder_name_preserved=True, features=str(path.relative_to(root)),
                features_sha256=sha256(path)))
        metadata = [dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha256(p))
                    for p in (output/'metadata').iterdir()]
        mirror_sha = sha256(root/'source/IMS_README_mirror.pdf')
        doc_match = any(x['sha256']==mirror_sha for x in metadata)
        report = dict(ended_utc=utc(), measured_values_verified=True, native_archive=native_integrity,
            complete_native_archive_read=True, native_records=len(rows),
            finite_measurements=sum(x['samples_per_channel']*x['channels'] for x in rows),
            independent_endpoint_parses=independent, acquisitions=groups, metadata=metadata,
            NASA_original_README_matches_mirror_bytes=doc_match,
            duplicate_complete_record_hash_counts={k:v for k,v in Counter(x['source_sha256'] for x in rows).items() if v>1},
            constant_channel_records=[x['native_member'] for x in rows if any(x['constant_channels'])],
            nominal_sampling_hz=20000, samples_per_native_record=20480,
            sample_count_divided_by_nominal_fs_seconds=1.024,
            README_rounds_snapshot_duration_to_one_second=True,
            calibration_unit='Not independently specified in the recovered README; no unit invented.',
            clock_scope='Acquisition filename clock, timezone and exact operating time across pauses unspecified.',
            RUL_gold_created=False, per_timestamp_fault_labels_created=False,
            PHMForge_original_score_qualified=False, Agent_task_admitted=False,
            next_gate='Resolve native count/end-time discrepancies and declare a defensible target before fitting. Other bearings cannot be assigned failure endpoints.',
            new_LLM_calls=0, training_fits=0, prediction_scores=0)
        save(root/'audit/IMS_ARCHIVE_INVENTORY280.json', inventory)
        save(root/'audit/IMS_MEASURED_SOURCE280.json', report)
        state.update(status='finished', verified_records=len(rows), measured_values_verified=True,
                     Agent_task_admitted=False)
        print(json.dumps({k:report[k] for k in ['native_records','finite_measurements','NASA_original_README_matches_mirror_bytes','Agent_task_admitted']}),flush=True)
    except BaseException as error:
        state.update(status='execution_error', error_type=type(error).__name__, error=str(error))
        save(root/'audit/IMS_ARCHIVE_INVENTORY280_partial.json', inventory)
        raise
    finally:
        native_log.close()
        state['ended_utc']=utc()
        save(state_path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    main(a.run.resolve(),a.workers)
