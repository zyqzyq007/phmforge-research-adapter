"""Bounded qualification of measured planetary crack stages, before Agent work.

The final allocation is never scored here. Source acquisition, native decoding,
grouped training/development checks and their failures share one auditable run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    staged.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while body := stream.read(8 * 1024 * 1024):
            h.update(body)
    return h.hexdigest()


def safe_env(root):
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ['API_KEY', 'TOKEN', 'PASSWORD', 'SECRET'])}
    env.update(TMPDIR=str(root/'runtime-temp'), TMP=str(root/'runtime-temp'),
               TEMP=str(root/'runtime-temp'), MPLCONFIGDIR=str(root/'mpl-cache'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def acquire(root):
    import requests
    protocol = json.loads((root/'protocol.json').read_text())
    directory = json.loads((root/'source/zip_index.json').read_text())
    selected = [x for x in directory if Path(x['path']).name in protocol['selected_members'][1:]]
    assert len(selected) == 2
    recovery_path = root/'audit/ACQUISITION_RECOVERY283.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    state = dict(status='acquiring', started_utc=utc(), pid=os.getpid(), workers=[],
                 recovery=recovery,
                 new_LLM_calls=0, training_fits=0, final_scores_opened=False)
    save(root/'audit/ACQUISITION283.json', state)
    started = time.monotonic()

    def header(item):
        begin = item['header_offset']
        end = begin + 511
        session = requests.Session()
        session.headers.update({'Accept-Encoding': 'identity', 'Range': f'bytes={begin}-{end}',
                                'If-Match': protocol['source_etag']})
        with session.get(protocol['source_url'], timeout=(12, 30), stream=True) as response:
            body = b''
            for chunk in response.iter_content(512):
                body += chunk
                if len(body) > 512:
                    raise ValueError('Header response exceeded exact range')
            assert response.status_code == 206
            assert response.headers['Content-Range'] == f'bytes {begin}-{end}/{protocol["source_bytes"]}'
            assert response.headers['ETag'] == protocol['source_etag']
            assert len(body) == 512
        fields = struct.unpack('<4s5H3I2H', body[:30])
        assert fields[0] == b'PK\x03\x04' and fields[3] == item['compression']
        assert fields[6] == int(item['CRC32'], 16)
        assert fields[7:9] == (item['compressed_bytes'], item['bytes'])
        assert body[30:30+fields[9]].decode('utf-8') == item['path']
        offset = begin + 30 + fields[9] + fields[10]
        name = Path(item['path']).stem.lower()
        (root/'source'/f'{name}_local_header.bin').write_bytes(body)
        return name, offset, item

    jobs = []
    try:
        if recovery:
            headers = []
            for item in selected:
                name = Path(item['path']).stem.lower()
                previous = json.loads((root/'audit'/f'{name}_COMMAND.json').read_text())
                assert previous['zip_member'] == item
                headers.append((name, previous['data_offset'], item))
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                headers = list(pool.map(header, selected))
        for name, offset, item in headers:
            acquisition_name = name+'_r2' if recovery else name
            workers = recovery['workers_per_file'] if recovery else protocol['workers_per_file']
            code = root/'code'/('acquire_range_source_resume.py' if recovery else 'acquire_range_source.py')
            argv = [sys.executable, '-B', str(code),
                    '--run', str(root), '--url', protocol['source_url'],
                    '--bytes', str(protocol['source_bytes']), '--etag', protocol['source_etag'],
                    '--name', acquisition_name, '--destination', f'inputs/{name}.deflate',
                    '--chunk-bytes', str(4*1024*1024), '--workers', str(workers),
                    '--route', 'proxy', '--format', 'raw', '--offset', str(offset),
                    '--length', str(item['compressed_bytes'])]
            if recovery:
                argv += ['--reuse-from',name]
            save(root/'audit'/f'{acquisition_name}_COMMAND.json', dict(argv=argv, zip_member=item, data_offset=offset,
                 code_sha256=sha(code)))
            log = (root/'logs'/f'{acquisition_name}_acquire.log').open('w')
            proc = subprocess.Popen(argv, env=safe_env(root), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            jobs.append((acquisition_name, proc, log))
        while any(proc.poll() is None for _, proc, _ in jobs):
            expired = time.time() > recovery['original_deadline_epoch'] if recovery else time.monotonic()-started > protocol['max_download_seconds']
            if expired:
                raise TimeoutError('Predefined source acquisition wall budget reached')
            state['workers'] = [dict(name=name, pid=proc.pid, exit_code=proc.poll()) for name, proc, _ in jobs]
            state['elapsed_seconds'] = time.monotonic()-started
            save(root/'audit/ACQUISITION283.json', state)
            time.sleep(5)
        assert all(proc.returncode == 0 for _, proc, _ in jobs), 'A bounded acquisition failed'
        if recovery:
            for name,_,_ in headers:
                destination = root/f'{name}_manifest.json'
                assert not destination.exists()
                shutil.copyfile(root/f'{name}_r2_manifest.json',destination)
            save(root/'audit/RESUMED_MANIFEST_BINDINGS283.json',dict(
                 aliases={name+'_manifest.json':name+'_r2_manifest.json' for name,_,_ in headers},
                 reason='Canonical preparation manifest points to completed recovery; original failed status/logs remain.'))
        state.update(status='complete', ended_utc=utc())
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), ended_utc=utc())
        for _, proc, _ in jobs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        raise
    finally:
        for _, _, log in jobs:
            log.close()
        state['workers'] = [dict(name=name, pid=proc.pid, exit_code=proc.poll()) for name, proc, _ in jobs]
        state['elapsed_seconds'] = time.monotonic()-started
        save(root/'audit/ACQUISITION283.json', state)


def native_audit(path,minimum_duration_seconds=929,expected_carrier_unit='mV'):
    import numpy as np
    from scipy.io import loadmat, whosmat
    from qualify_planetary_source import mat_index
    index = mat_index(path)
    independent = {name: (shape, kind) for name, shape, kind in whosmat(path)}
    headers=loadmat(path,variable_names=['File_Header','Channel_1_Header','Channel_2_Header'],simplify_cells=True)
    frequency_text=str(headers['File_Header']['SampleFrequency']).replace(',','.')
    declared_frequency=float(frequency_text)
    frequency_digits=len(frequency_text.split('.')[1]) if '.' in frequency_text else 0
    assert declared_frequency==38400.0 and headers['Channel_1_Header']['Unit']=='s'
    assert headers['Channel_2_Header']['SignalName']=='Acc_Carrier'
    carrier_unit=str(headers['Channel_2_Header']['Unit'])
    assert carrier_unit and (expected_carrier_unit is None or carrier_unit==expected_carrier_unit)
    rows = []
    for field in index:
        shape, kind = independent[field['name']]
        assert list(shape) == field['shape'] and kind == 'double'
        a = np.memmap(path, mode='r', dtype='<f8', offset=field['offset'], shape=(field['bytes']//8,))
        h = hashlib.sha256()
        lo, hi = float('inf'), -float('inf')
        for start in range(0, len(a), 1024*1024):
            block = a[start:start+1024*1024]
            assert np.isfinite(block).all()
            h.update(block.tobytes())
            lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
        row = dict(field, count=len(a), finite=True, minimum=lo, maximum=hi, raw_float64_sha256=h.hexdigest())
        if field['name'] == 'Channel_1_Data':
            assert len(a) > minimum_duration_seconds*38400
            origin=float(a[0]);period=float(a[1]-a[0])
            # Compare at the precision actually declared by the source header.
            # This authenticates the published clock representation, not the
            # physical hardware clock. Do not change or resample observations.
            assert period>0 and round(1/period,frequency_digits)==declared_frequency
            assert len(a)==int(headers['File_Header']['NumberOfSamplesPerChannel'])
            maximum_residual=0.0;maximum_nominal_difference=0.0
            for start in range(0, len(a), 1024*1024):
                ix = np.arange(start, min(start+1024*1024, len(a)))
                residual=float(np.max(np.abs(a[ix]-(origin+ix*period))))
                assert residual < 1e-9
                maximum_residual=max(maximum_residual,residual)
                maximum_nominal_difference=max(maximum_nominal_difference,float(np.max(np.abs(a[ix]-ix/38400))))
            row.update(nominal_sampling_hz=38400,stored_time_origin_seconds=origin,
                       stored_time_step_seconds=period,represented_axis_frequency_hz=1/period,
                       source_header_frequency=frequency_text,source_header_frequency_digits=frequency_digits,
                       source_header_frequency_precision_matched=True,
                       maximum_affine_clock_residual_seconds=maximum_residual,
                       maximum_difference_from_exact_nominal_clock_seconds=maximum_nominal_difference,
                       physical_clock_calibration_claimed=False)
        rows.append(row)
    carrier = loadmat(path, variable_names=['Channel_2_Data'])['Channel_2_Data'].ravel(order='F')
    expected = next(x for x in rows if x['name'] == 'Channel_2_Data')
    assert hashlib.sha256(carrier.astype('<f8', copy=False).tobytes()).hexdigest() == expected['raw_float64_sha256']
    return dict(native_path=str(path), native_sha256=sha(path), channels=rows,
                independent_carrier_values=len(carrier), independent_carrier_hash_equal=True,
                scipy_directory_equal=True, finite_values=sum(x['count'] for x in rows),
                native_metadata={k:headers[k] for k in ['File_Header','Channel_1_Header','Channel_2_Header']},
                carrier_measurement_unit=carrier_unit,calibrated_acceleration_conversion_performed=False)


def sequence_summary(path,audited,slots,window_samples=76800,windows_per_slot=68,verify_source_hash=True):
    """Summarize genuine pulse markers in documented sequence slots, without ML.

    Raw midpoint thresholds are descriptive only. Cadence is not calibrated RPM,
    load or a new fault label; a partial slot is rejected rather than padded.
    """
    import numpy as np
    from scipy.io import loadmat
    if verify_source_hash:assert sha(path)==audited['native_sha256']
    fields={v['name']:v for v in audited['channels']}
    time_field=fields['Channel_1_Data'];period=time_field['stored_time_step_seconds']
    frequency=float(audited['native_metadata']['File_Header']['SampleFrequency'].replace(',','.'))
    names=['Channel_'+str(i)+'_Header' for i in [4,5,6,7,8,9]]
    headers=loadmat(path,variable_names=names,simplify_cells=True)
    rows=[]
    for slot in slots:
        begin=int(slot['start_seconds']*frequency);end=begin+window_samples*windows_per_slot
        assert end<=time_field['count'] and end<=int(slot['author_window_end_seconds']*frequency)
        row=dict(slot=slot['id'],start_sample=begin,end_sample_exclusive=end,
                 sampled_duration_seconds=(end-begin)*period,window_samples=window_samples,windows=windows_per_slot,markers={})
        for channel in [4,5,6,7,8,9]:
            field=fields['Channel_'+str(channel)+'_Data'];header=headers['Channel_'+str(channel)+'_Header']
            data=np.memmap(path,mode='r',dtype='<f8',offset=field['offset'],shape=(field['count'],))[begin:end]
            result=dict(native_signal_name=str(header['SignalName']),native_unit=str(header['Unit']),
                        minimum=float(data.min()),maximum=float(data.max()),mean=float(data.mean()))
            if channel in [4,5,6]:
                threshold=(field['minimum']+field['maximum'])/2
                high=data>=threshold;edges=np.flatnonzero(high[1:]&~high[:-1])+1
                intervals=np.diff(edges)*period
                result.update(raw_threshold=threshold,rising_edges=len(edges),
                    pulse_cadence_hz=float((len(edges)-1)/((edges[-1]-edges[0])*period)) if len(edges)>1 else None,
                    pulse_interval_CV=float(intervals.std()/intervals.mean()) if len(intervals) else None)
            row['markers'][str(channel)]=result
        rows.append(row)
    return dict(native_sha256=audited['native_sha256'],slots=rows,
                threshold_rule='Midpoint of already verified full-record min/max;no label/model tuning.',
                interpretation='Descriptive source-marker summaries only;not calibrated RPM/load,independent runs,or fault-prediction results.',
                new_fits=0,new_LLM_calls=0,new_model_inference=0,new_Agent_cases=0)


def export_bound_source(root,source,plan_path):
    """Export the preregistered G2 windows after all native members close.

    This is source curation only. No model,score,imputation or fabricated fault
    is involved; stage aliases follow original files and cannot certify health.
    """
    import numpy as np
    from run_frozen_candidate_review import source_context
    plan=json.loads(plan_path.read_text())
    status=json.loads((source/'SOURCE_STATUS292.json').read_text())
    assert status['status']=='native_members_complete_waiting_condition_contract','All six native source members must finish before export'
    assert len(status['outcomes'])==6 and all(x.get('native_audit_passed') for x in status['outcomes'])
    assert not (root/'EXPORT_STARTED.json').exists(),'No implicit export retry or overwrite'
    index,receipt=source_context(source)
    frozen=source/'audit/qualification_precommit'
    assert sha(plan_path)==sha(frozen/'g2_ensemble_source_precommit.json')
    basis_path=frozen/'g2_ensemble_basis_plan.json'
    for relative in ['audit/qualification_precommit/g2_ensemble_source_precommit.json',
                     'audit/qualification_precommit/g2_ensemble_basis_plan.json','PROTOCOL292.json','SOURCE_STATUS292.json','MEMBER_OFFSETS292.json']:
        assert sha(source/relative)==index[relative]['sha256']
    assert sha(source/'PROTOCOL292.json')==plan['source_protocol_sha256']
    for name in ['audit','code','hidden','logs','runtime/tmp','runtime/mpl']:(root/name).mkdir(parents=True,exist_ok=True)
    save(root/'EXPORT_STARTED.json',dict(utc=utc(),pid=os.getpid(),source=str(source),source_manifest_sha256=receipt['manifest_sha256'],new_fits=0,new_LLM_calls=0))
    roles={'train':'train','development':'val','final':'test'}
    slots={x['id']:x for x in plan['nominal_slots']};width=plan['window_samples'];count=plan['windows_per_stage_slot']
    assert len(plan['stages'])==6 and width==76800 and count==68
    pieces={k:[] for k in roles};mapping=[];bindings=[];markers=[];source_stats=[]
    for stage in plan['stages']:
        name=Path(stage['source_file']).stem.lower();audit_relative='audit/'+name+'_NATIVE.json';native_relative='outputs/'+stage['source_file']
        for relative in [audit_relative,native_relative]:
            path=source/relative;expected=index[relative]
            assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
            bindings.append(dict(path=str(path),sha256=expected['sha256'],bytes=expected['bytes']))
        audited=json.loads((source/audit_relative).read_text());native=source/native_relative;details=audited['native']
        assert audited['integrity']['complete_member_CRC_verified'] and audited['integrity']['sha256']==details['native_sha256']==index[native_relative]['sha256']
        assert details['independent_carrier_hash_equal'] and details['scipy_directory_equal']
        fields={x['name']:x for x in details['channels']};assert len(fields)==9 and all(x['finite'] for x in fields.values())
        assert len({x['count'] for x in fields.values()})==1
        frequency=float(details['native_metadata']['File_Header']['SampleFrequency'].replace(',','.'))
        assert frequency==plan['nominal_sampling_hz']==38400
        assert details['carrier_measurement_unit']=='mV' and details['native_metadata']['Channel_2_Header']['SignalName']=='Acc_Carrier'
        assert all(fields['Channel_1_Data']['maximum']>=x['author_window_end_seconds'] for x in slots.values())
        summary=sequence_summary(native,details,plan['nominal_slots'],width,count,verify_source_hash=False)
        assert all(x['markers']['6']['rising_edges']>=2 for x in summary['slots'])
        markers.append(dict(stage=stage['label_alias'],summary=summary))
        source_stats.append(dict(path=str(native),bytes=native.stat().st_size,mtime_ns=native.stat().st_mtime_ns))
        field=fields['Channel_2_Data'];carrier=np.memmap(native,mode='r',dtype='<f8',offset=field['offset'],shape=(field['count'],))
        for role,identifiers in plan['split'].items():
            if role not in roles:continue
            for ident in identifiers:
                slot=slots[ident];begin=int(slot['start_seconds']*frequency);end=begin+width*count
                assert end<=field['count'] and end<=int(slot['author_window_end_seconds']*frequency)
                values=np.asarray(carrier[begin:end]).reshape(count,width).copy();assert np.isfinite(values).all()
                group='segment_'+hashlib.sha256((details['native_sha256']+ident).encode()).hexdigest()[:24]
                ids=[hashlib.sha256((details['native_sha256']+ident+':'+str(i)).encode()).hexdigest()[:32] for i in range(count)]
                rows=[]
                for i in range(count):
                    row=dict(sample_id=ids[i],label=stage['label'],role=role,condition_id=ident,group=group,
                        original_record=stage['source_file'],native_sha256=details['native_sha256'],begin_sample=begin+i*width,
                        end_sample_exclusive=begin+(i+1)*width,raw_float64_sha256=hashlib.sha256(values[i].astype('<f8',copy=False).tobytes()).hexdigest())
                    mapping.append(row);rows.append(row)
                pieces[role].append(dict(waveforms=values,rows=rows))
    assert len(mapping)==sum(plan['expected_rows'].values()) and len({x['sample_id'] for x in mapping})==len(mapping)
    rng=np.random.default_rng(plan['fitting_settings']['seed']);exports=[];identity={}
    for role,filename in roles.items():
        waves=np.concatenate([p['waveforms'] for p in pieces[role]]);rows=[r for p in pieces[role] for r in p['rows']]
        assert len(rows)==len(waves)==plan['expected_rows'][role]
        order=rng.permutation(len(rows));rows=[rows[i] for i in order]
        arrays=dict(waveforms=waves[order],labels=np.asarray([x['label'] for x in rows]),sample_ids=np.asarray([x['sample_id'] for x in rows]),
            condition_id=np.asarray([x['condition_id'] for x in rows]),groups=np.asarray([x['group'] for x in rows]),
            sampling_hz=np.asarray(frequency),dataset=np.asarray('PlanetaryG2'))
        path=root/'hidden'/(filename+'.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        with np.load(path,allow_pickle=False) as restored:
            restored_waves=restored['waveforms'];restored_ids=restored['sample_ids'];restored_labels=restored['labels']
            for i,row in enumerate(rows):
                assert str(restored_ids[i])==row['sample_id'] and int(restored_labels[i])==row['label']
                assert hashlib.sha256(restored_waves[i].astype('<f8',copy=False).tobytes()).hexdigest()==row['raw_float64_sha256']
        exports.append(dict(role=role,path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size,rows=len(rows)))
        identity[role]=dict(ids=set(arrays['sample_ids'].tolist()),groups=set(arrays['groups'].tolist()),conditions=set(arrays['condition_id'].tolist()))
    for a in identity:
        for b in identity:
            if a!=b:assert all(not identity[a][k]&identity[b][k] for k in ['ids','groups','conditions'])
    for file in source_stats:
        p=Path(file['path']);assert p.stat().st_size==file['bytes'] and p.stat().st_mtime_ns==file['mtime_ns']
    save(root/'FITTING_SETTINGS.json',plan['fitting_settings']);shutil.copy2(basis_path,root/'BASIS_PLAN.json')
    save(root/'public_dataset_metadata.json',dict(dataset='PlanetaryG2',task='Recorded-stage classification across the documented test sequence',
        classes={str(x['label']):x['label_alias'] for x in plan['stages']},nominal_sampling_hz=frequency,waveform_samples=width,
        signal='Acc_Carrier',stored_signal_unit='mV',train_shape=[plan['expected_rows']['train'],width],available_splits=['train'],
        physical_gears=1,source_record_shared_across_sequence_slots=True,
        source_kind='Measured gearbox vibration after external fatigue crack growth',
        interpretation='Stage aliases are not certified crack lengths or a healthy label. Sequence-slot IDs are not calibrated RPM or torque. Public-data pretraining contamination is not excluded.'))
    save(root/'audit/REAL_WINDOW_EXPORT.json',dict(source_manifest_sha256=receipt['manifest_sha256'],source_bindings=bindings,files=exports,mapping=mapping,
        represented_gears=1,stage_recordings=6,source_record_shared_across_slots=True,physical_crack_causality_certified=False,final_scores_opened=False))
    save(root/'audit/ACTUAL_SEQUENCE_CONTRACT.json',dict(role='Documented first five complete test-sequence slots;observed markers reported,not calibrated RPM/torque or perfectly stationary regimes.',records=markers))
    admission=dict(measured_source_verified=True,all_member_CRCs_verified=True,native_values_verified=True,operating_condition_contract_verified=True,
        export_integrity_verified=True,source_archive=str(source),source_manifest_sha256=receipt['manifest_sha256'],source_plan_sha256=sha(plan_path),
        final_scores_opened=False,new_fits=0,new_LLM_calls=0,new_Agent_cases=0,
        input_files=dict(train=next(x for x in exports if x['role']=='train'),development=next(x for x in exports if x['role']=='development'),
                         fitting_settings=dict(path='FITTING_SETTINGS.json',sha256=sha(root/'FITTING_SETTINGS.json'))),
        scope='Recorded-stage classification on one gear across a fixed documented sequence;no independent-device or natural-fleet claim.')
    for name in ['prepare_planetary_probe.py','qualify_planetary_source.py']:
        src=Path(__file__).with_name(name);dst=root/'code'/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    save(root/'SOURCE_ADMISSION.json',admission);print(json.dumps({k:v for k,v in admission.items() if k!='input_files'}))


def prepare(root):
    import numpy as np
    import secrets
    from qualify_planetary_source import decode
    protocol = json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'audit/ACQUISITION283.json').read_text())['status'] == 'complete'
    assert not (root/'audit/ALLOCATION283.json').exists(), 'No silent allocation overwrite'
    previous = Path(protocol['previous_source'])
    receipt = json.loads((previous/'ARCHIVE_VERIFIED.json').read_text())
    assert sha(previous/'SOURCE_MANIFEST.json') == receipt['manifest_sha256']
    manifest = {x['path']: x for x in json.loads((previous/'SOURCE_MANIFEST.json').read_text())['files']}
    p0 = 'outputs/g1p0/G1_P0_38400Hz.MAT'
    assert (previous/p0).stat().st_size == manifest[p0]['bytes']
    assert sha(previous/p0) == manifest[p0]['sha256']
    source_rows = []
    for name in protocol['selected_members']:
        if name == protocol['selected_members'][0]:
            path = previous/p0
            integrity = dict(reused_A279_complete_member_CRC=True, crc32='c17bf149',
                             archive_manifest_sha256=receipt['manifest_sha256'])
        else:
            stem = Path(name).stem.lower()
            entry = json.loads((root/'audit'/f'{stem}_COMMAND.json').read_text())
            downloaded = json.loads((root/f'{stem}_manifest.json').read_text())
            compressed = root/downloaded['file']
            assert compressed.stat().st_size == downloaded['bytes'] == entry['zip_member']['compressed_bytes']
            assert sha(compressed) == downloaded['sha256'] and downloaded['source_offset'] == entry['data_offset']
            path = root/'inputs'/name
            integrity = decode(compressed, path, entry['zip_member'])
        audited = native_audit(path)
        audited['integrity'] = integrity
        save(root/'audit'/(Path(name).stem+'_NATIVE.json'), audited)
        source_rows.append(audited)
    parts = {k: dict(waves=[], labels=[], ids=[], condition=[], segment=[]) for k in protocol['split_voltages']}
    mapping = []
    voltages = ['9.8', '9.6', '9.4', '9.2', '9.0', '8.8']
    for label, row in enumerate(source_rows):
        field = next(x for x in row['channels'] if x['name'] == 'Channel_2_Data')
        carrier = np.memmap(row['native_path'], mode='r', dtype='<f8', offset=field['offset'], shape=(field['count'],))
        for ci, voltage in enumerate(voltages):
            part = next(k for k,v in protocol['split_voltages'].items() if voltage in v)
            start = (42+150*ci)*38400
            end = start+protocol['windows_per_segment']*protocol['window_samples']
            assert end <= (179+150*ci)*38400 and end <= len(carrier)
            waves = np.asarray(carrier[start:end]).reshape(68, 76800).copy()
            identifiers = [secrets.token_hex(16) for _ in range(len(waves))]
            parts[part]['waves'].append(waves)
            parts[part]['labels'].extend([label]*len(waves))
            parts[part]['ids'].extend(identifiers)
            parts[part]['condition'].extend([voltage]*len(waves))
            segment_id='segment_'+secrets.token_hex(12)
            parts[part]['segment'].extend([segment_id]*len(waves))
            mapping.extend(dict(sample_id=identifiers[i], label=label, original_record=Path(row['native_path']).name,
                                segment_id=segment_id,
                                source_sha256=row['native_sha256'], condition=voltage, part=part,
                                begin_sample=start+i*76800, end_sample_exclusive=start+(i+1)*76800)
                           for i in range(68))
    rng = np.random.default_rng(protocol['seed'])
    files = []
    for part, values in parts.items():
        x = np.concatenate(values['waves'])
        order = rng.permutation(len(x))
        arrays = dict(waveforms=x[order], labels=np.asarray(values['labels'])[order],
                      sample_ids=np.asarray(values['ids'])[order], condition_id=np.asarray(values['condition'])[order],
                      groups=np.asarray(values['segment'])[order], dataset=np.asarray('PlanetaryG1'))
        path = root/'hidden'/(part+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        files.append(dict(part=part, path=str(path.relative_to(root)), sha256=sha(path), rows=len(x),
                          labels={str(c):int(np.sum(arrays['labels']==c)) for c in range(3)},
                          conditions=np.unique(arrays['condition_id']).tolist(), measured=True))
    assert len({x['sample_id'] for x in mapping}) == len(mapping) == 1224
    for a in parts:
        for b in parts:
            if a != b:
                assert not set(parts[a]['segment']) & set(parts[b]['segment'])
                assert not set(parts[a]['condition']) & set(parts[b]['condition'])
    save(root/'audit/ALLOCATION283.json', dict(source_records=source_rows, files=files, mapping=mapping,
         original_record_shared_across_conditions=True, physical_gears=1, stage_recordings=3,
         condition_disjoint=True, windows_nonoverlapping=True, final_scores_opened=False))
    save(root/'FITTING_SETTINGS.json', dict(seed=283, classes=[0,1,2], cv='leave_one_condition_out',
         cv_group_key='condition_id', expected_samples=76800))
    save(root/'audit/PREPARE_COMPLETE283.json', dict(utc=utc(), status='complete', files=files,
         source_values=sum(r['finite_values'] for r in source_rows), new_LLM_calls=0, real_fits=0,
         final_scores_opened=False, physical_gears=1, stage_recordings=3))


def independent_metrics(y, p):
    import numpy as np
    from controlled_feedback import metrics
    cm = np.zeros((3,3),dtype=np.int64)
    for truth, pred in zip(y,p):
        cm[int(truth),int(pred)] += 1
    denom = cm.sum(0)+cm.sum(1)
    f = np.divide(2*np.diag(cm), denom, out=np.zeros(3), where=denom!=0)
    value = dict(accuracy=float(cm.trace()/cm.sum()), macro_f1=float(f.mean()),
                 correct=int(cm.trace()), total=int(cm.sum()), confusion=cm.tolist())
    ref = metrics(y,p,[0,1,2])
    assert value['confusion'] == ref['confusion'] and value['correct'] == ref['correct']
    assert abs(value['macro_f1']-ref['macro_f1']) < 1e-12
    return value


def feasibility(root):
    import numpy as np
    assert (root/'audit/PREPARE_COMPLETE283.json').exists()
    assert not (root/'FEASIBILITY283.json').exists()
    recipes = [
        dict(features='stats15', model='lda', params=dict(shrinkage='auto')),
        dict(features='stats_fft64', model='svc', params=dict(C=10, gamma='scale')),
        dict(features='fft256', model='svc', params=dict(C=10, gamma='scale')),
        dict(features='stats15', model='extra', params=dict(max_depth=None, min_samples_leaf=1)),
        dict(features='stats_fft64', model='rf', params=dict(max_depth=None, min_samples_leaf=1)),
        dict(features='stats_fft64', model='logistic', params=dict(C=1)),
    ]
    save(root/'FEASIBILITY_RECIPES283.json', recipes)
    training = np.load(root/'hidden/train.npz', allow_pickle=False)
    development = np.load(root/'hidden/dev.npz', allow_pickle=False)
    initial = {p.name:sha(p) for p in [root/'hidden/train.npz',root/'hidden/dev.npz',root/'FITTING_SETTINGS.json']}
    save(root/'audit/FIT_INPUT_FREEZE283.json',initial)
    def run_one(pair):
        index, recipe = pair
        folder = root/'feasibility'/f'candidate_{index+1}'
        save(folder/'recipe.json', recipe)
        cmd = [sys.executable, '-B', str(root/'code/controlled_feedback.py'), 'fit',
               '--recipe', str(folder/'recipe.json'), '--train', str(root/'hidden/train.npz'),
               '--dev', str(root/'hidden/dev.npz'), '--output', str(folder),
               '--settings', str(root/'FITTING_SETTINGS.json')]
        save(folder/'command.json', cmd)
        state = dict(id=index+1, recipe=recipe, started_utc=utc(), status='running')
        save(folder/'status.json',state)
        try:
            with (folder/'stdout').open('w') as out, (folder/'stderr').open('w') as err:
                proc = subprocess.run(cmd, env=safe_env(root), stdout=out, stderr=err, timeout=180)
            state.update(status='complete' if proc.returncode==0 else 'fit_error', exit_code=proc.returncode)
        except subprocess.TimeoutExpired:
            state.update(status='fit_timeout')
        state['ended_utc'] = utc()
        events = [json.loads(x) for x in (folder/'fit_events.jsonl').read_text().splitlines()] if (folder/'fit_events.jsonl').exists() else []
        state['real_fits_started'] = sum(e['status']=='started' for e in events)
        state['real_fits_completed'] = sum(e['status']=='completed' for e in events)
        save(folder/'status.json', state)
        if state['status'] != 'complete':
            return state
        report = json.loads((folder/'fit_report.json').read_text())
        predictions = np.load(folder/'predictions.npz',allow_pickle=False)
        assert state['real_fits_completed'] == report['real_fits'] == 4 and report['cold_reload_equal']
        assert report['train_input_sha256'] == initial['train.npz'] and report['development_input_sha256'] == initial['dev.npz']
        checks = []
        for key, data, pred_key in [('train',training,'train'),('cv',training,'cv'),('development',development,'development')]:
            assert np.array_equal(predictions['train_ids' if key!='development' else 'development_ids'], data['sample_ids'])
            metric = independent_metrics(data['labels'], predictions[pred_key])
            assert metric['confusion'] == report[key]['confusion'] and metric['correct'] == report[key]['correct']
            assert abs(metric['accuracy']-report[key]['accuracy']) < 1e-12
            assert abs(metric['macro_f1']-report[key]['macro_f1']) < 1e-12
            groups = {str(g):independent_metrics(data['labels'][data['condition_id']==g], predictions[pred_key][data['condition_id']==g])
                      for g in np.unique(data['condition_id'])}
            checks.append(dict(split=key, recomputed=metric, per_condition=groups))
        folds = report['cv_folds']
        seen = []
        for fold in folds:
            assert not set(fold['fitting_conditions']) & set(fold['checking_conditions'])
            assert not set(fold['fitting_ids']) & set(fold['checking_ids'])
            seen += fold['checking_ids']
        assert len(seen)==len(training['labels']) and set(seen)==set(training['sample_ids'].tolist())
        save(folder/'INDEPENDENT_CHECKS.json', checks)
        return dict(state, train=report['train'], cv=report['cv'], development=report['development'],
                    checkpoint_sha256=report['checkpoint_sha256'], cold_reload_equal=True,
                    independent_metric_checks=3, grouped_CV_verified=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run_one, enumerate(recipes)))
    for name, expected in initial.items():
        path = root/name if name=='FITTING_SETTINGS.json' else root/'hidden'/name
        assert sha(path)==expected
    complete = [x for x in rows if x['status']=='complete']
    spread = max((x['development']['macro_f1'] for x in complete),default=0)-min((x['development']['macro_f1'] for x in complete),default=0)
    gate = dict(all_six_complete=len(complete)==6,
                has_cv_and_development_over_04333=any(x['cv']['macro_f1']>.4333 and x['development']['macro_f1']>.4333 for x in complete),
                development_macro_f1_span_at_least_005=spread>=.05,
                no_more_than_four_perfect_development=sum(x['development']['correct']==x['development']['total'] for x in complete)<=4)
    result = dict(utc=utc(), run_id=root.name, candidates=rows, admission_checks=gate, admission_passed=all(gate.values()),
                  development_macro_f1_span=spread, real_fits_completed=sum(x['real_fits_completed'] for x in rows),
                  real_fits_started=sum(x['real_fits_started'] for x in rows), new_LLM_calls=0,
                  final_scores_opened=False, new_Agent_failures=0,
                  interpretation='Train/development feasibility only; no Agent result or causal overfitting confirmation.')
    save(root/'FEASIBILITY283.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))


def continue_pipeline(root):
    protocol = json.loads((root/'protocol.json').read_text())
    deadline = datetime.fromisoformat(protocol['created_utc']).timestamp()+protocol['max_pipeline_seconds']
    state = dict(status='waiting_for_verified_acquisition', pid=os.getpid(), started_utc=utc(),
                 new_LLM_calls=0, final_scores_opened=False, real_fits_completed=0)
    process = None
    try:
        while True:
            assert time.time() < deadline, 'Overall predeclared wall budget reached'
            transport = json.loads((root/'audit/ACQUISITION283.json').read_text())
            if transport['status'] == 'complete':
                break
            if transport['status'] != 'acquiring':
                raise RuntimeError('Acquisition did not finish; no model admitted')
            os.kill(transport['pid'],0)
            state['last_acquisition_check_utc'] = utc()
            save(root/'audit/PIPELINE283.json',state)
            time.sleep(5)
        for stage in ['prepare','feasibility']:
            state.update(status=stage, stage_started_utc=utc())
            save(root/'audit/PIPELINE283.json',state)
            cmd = [sys.executable,'-B',str(Path(__file__).resolve()),stage,'--run',str(root)]
            save(root/'audit'/(stage+'_COMMAND283.json'),dict(argv=cmd,code_sha256=sha(Path(__file__))))
            with (root/'logs'/(stage+'.log')).open('w') as log:
                process = subprocess.Popen(cmd,env=safe_env(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['stage_pid'] = process.pid
                save(root/'audit/PIPELINE283.json',state)
                process.wait(timeout=max(1,deadline-time.time()))
            assert process.returncode == 0, stage+' failed; preserve error without automatic retry'
        result = json.loads((root/'FEASIBILITY283.json').read_text())
        state.update(status='complete',admission_passed=result['admission_passed'],
                     real_fits_completed=result['real_fits_completed'],ended_utc=utc())
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),ended_utc=utc())
        raise
    finally:
        save(root/'audit/PIPELINE283.json',state)


def adopt_acquisition(root):
    """Monitor existing range workers after a recorded source-only time revision."""
    recovery = json.loads((root/'audit/ACQUISITION_RECOVERY283.json').read_text())
    prior = json.loads((root/'audit/ACQUISITION283_BEFORE_BUDGET_REVISION.json').read_text())
    workers = prior['workers']
    state = dict(prior, pid=os.getpid(), started_utc=utc(), adopted_worker_pids=[x['pid'] for x in workers],
                 source_budget_revision='audit/SOURCE_BUDGET_REVISION283.json', status='acquiring')
    for row in workers:
        command = Path(f'/proc/{row["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert str(root) in command and row['name'] in command
    try:
        while True:
            results = [json.loads((root/(row['name']+'_status.json')).read_text()) for row in workers]
            if all(x['status']=='finished' for x in results):
                break
            # Independent source files keep their original allowance; a failed
            # member must not prematurely interrupt another healthy transfer.
            if all(x['status']!='running' for x in results):
                raise RuntimeError('An adopted bounded download failed; all source workers are terminal')
            if time.time() > recovery['original_deadline_epoch']:
                raise TimeoutError('Revised source-only transfer deadline reached')
            for row, result in zip(workers,results):
                if result['status']=='running':
                    os.kill(row['pid'],0)
            receipt_bytes=[]
            for row in workers:
                records=[json.loads(line) for line in (root/(row['name']+'_ranges.jsonl')).read_text().splitlines()]
                verified={x['index']:x['bytes'] for x in records if x['status']=='verified'}
                receipt_bytes.append(sum(verified.values()))
            state.update(last_check_utc=utc(), verified_bytes=sum(receipt_bytes))
            save(root/'audit/ACQUISITION283.json',state)
            time.sleep(5)
        aliases = {}
        for row in workers:
            name = row['name'];assert name.endswith('_r2')
            dest = root/(name[:-3]+'_manifest.json');assert not dest.exists()
            shutil.copyfile(root/(name+'_manifest.json'),dest)
            aliases[dest.name] = name+'_manifest.json'
        save(root/'audit/RESUMED_MANIFEST_BINDINGS283.json',dict(aliases=aliases,utc=utc()))
        state.update(status='complete',ended_utc=utc())
    except BaseException as error:
        for row in workers:
            path = root/(row['name']+'_status.json');current=json.loads(path.read_text())
            if current['status']=='running':
                try:
                    os.killpg(row['pid'],signal.SIGTERM)
                except ProcessLookupError:
                    pass
                current.update(status='stopped_source_deadline_or_error',stopped_utc=utc())
                save(path,current)
        state.update(status='failed',error_type=type(error).__name__,error=str(error),ended_utc=utc())
        raise
    finally:
        save(root/'audit/ACQUISITION283.json',state)


def recover_one_range(root):
    """One recorded recovery of a fixed range set; scientific budgets unchanged."""
    plan = json.loads((root/'audit/MISSING_RANGE_RECOVERY_PLAN283.json').read_text())
    state = dict(status='waiting_for_original_transfers_to_stop', started_utc=utc(), pid=os.getpid(),
                 source_deadline_epoch=plan['source_deadline_epoch'], new_LLM_calls=0,
                 new_final_scores=0, maximum_additional_range_requests=len(plan['allowed_missing_indices']))
    save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
    process = None
    try:
        while True:
            assert time.time() < plan['source_deadline_epoch'], 'Original revised source deadline reached'
            old = json.loads((root/(plan['old_name']+'_status.json')).read_text())
            other = json.loads((root/(plan['other_name']+'_status.json')).read_text())
            controllers_closed = all(not Path(f'/proc/{pid}').exists() or
                Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]=='Z'
                for pid in plan['original_monitor_pids'])
            if old['status'] != 'running' and other['status'] != 'running' and controllers_closed:
                break
            state['last_check_utc'] = utc();save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
            time.sleep(5)
        assert old['status']=='execution_error' and other['status']=='finished', (old['status'],other['status'])
        records = [json.loads(line) for line in (root/(plan['old_name']+'_ranges.jsonl')).read_text().splitlines()]
        good = {row['index'] for row in records if row['status']=='verified'}
        missing = sorted(set(range(old['total_chunks']))-good)
        assert missing == plan['allowed_missing_indices'], missing
        failures = [row for row in records if row['status']=='transport_error']
        assert len(failures) < 20
        assert not (root/'audit/PREPARE_COMPLETE283.json').exists() and not (root/'FEASIBILITY283.json').exists()
        for name in ['ACQUISITION283.json','PIPELINE283.json']:
            shutil.copy2(root/'audit'/name,root/'audit'/('BEFORE_ONE_RANGE_RECOVERY_'+name))
        state.update(status='fixed_original_range_recovery', original_failed_ranges=missing,
                     original_transport_failures=len(failures), reused_verified_ranges=len(good))
        save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
        command = plan['acquisition_command'] + ['--max-failures',str(20-len(failures))]
        assert command.count('--max-attempts')==1 and command[command.index('--max-attempts')+1]=='1'
        with (root/'logs/one_range_recovery.log').open('w') as log:
            process = subprocess.Popen(command,env=safe_env(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['range_pid']=process.pid;save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
            process.wait(timeout=min(120*len(missing),max(1,plan['source_deadline_epoch']-time.time())))
        assert process.returncode==0, 'Recorded range recovery failed; no automatic retry'
        result=json.loads((root/(plan['new_name']+'_status.json')).read_text())
        assert result['status']=='finished'
        aliases={}
        for original,qualified in plan['manifest_bindings'].items():
            target=root/original;assert not target.exists()
            shutil.copy2(root/qualified,target);aliases[original]=qualified
        save(root/'audit/RESUMED_MANIFEST_BINDINGS283.json',dict(utc=utc(),aliases=aliases,
             original_failed_attempt_preserved=True,additional_requests_at_most=len(missing)))
        save(root/'audit/ACQUISITION283.json',dict(status='complete',utc=utc(),pid=os.getpid(),
             completion='One bounded recovery of the frozen missing ranges; original failures retained.',
             verified_bytes=old['expected_bytes']+other['expected_bytes'],new_LLM_calls=0,training_fits=0,
             final_scores_opened=False,source_deadline_unchanged=True))
        state.update(status='source_recovered_continuing_original_pipeline', source_recovered_utc=utc())
        save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'continue','--run',str(root)]
        with (root/'logs/continued_after_one_range.log').open('w') as log:
            process=subprocess.Popen(cmd,env=safe_env(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['pipeline_pid']=process.pid;save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)
            process.wait()
        assert process.returncode==0, 'Original pipeline failed after source recovery; no automatic retry'
        state.update(status='complete',ended_utc=utc())
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        state.update(status='failed',error_type=type(error).__name__,error=str(error),ended_utc=utc())
        raise
    finally:
        save(root/'audit/MISSING_RANGE_RECOVERY283.json',state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['acquire','prepare','feasibility','continue','adopt','recover-one-range','export-bound-source'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-archive',type=Path);parser.add_argument('--source-plan',type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    assert root.parent == Path('/srv/yyf-zyq/PHM_20261010')
    os.environ.update({k: v for k, v in safe_env(root).items()
                       if k in ['TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR', 'OMP_NUM_THREADS',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
    if args.stage=='export-bound-source':
        assert args.source_archive and args.source_plan
        for name in ['runtime-temp','mpl-cache']:(root/name).mkdir(exist_ok=True,parents=True)
        export_bound_source(root,args.source_archive.resolve(),args.source_plan.resolve());return
    {'acquire':acquire, 'prepare':prepare, 'feasibility':feasibility,
     'continue':continue_pipeline,'adopt':adopt_acquisition,'recover-one-range':recover_one_range}[args.stage](root)


if __name__ == '__main__':
    main()
