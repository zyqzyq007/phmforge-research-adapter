"""Prepare complete DCASE machine tasks with original WAV bytes and blinded queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.request
import zipfile

from scipy.io import wavfile


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def machine_name(entry):
    name = entry['key']
    if name.startswith('eval_data_'):
        return name[len('eval_data_'):].rsplit('_', 1)[0]
    return name[len('dev_'):-len('.zip')]


def prepare_file(root, entry, timeout, verified_source=None, receipt_directory='source_files'):
    machine = machine_name(entry)
    inputs = root / 'tasks' / machine / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    cached = verified_source or root / 'clients-cache' / 'raw_zip' / entry['key']
    cached.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'audits' / receipt_directory / entry['key'] / 'status.json'
    receipt = {'source_record':entry['source_record'], 'source_file':entry['key'],
               'machine':machine, 'role':entry['role'], 'url':entry['links']['self'],
               'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
               'timeout_seconds':timeout, 'LLM_calls':0}
    save(receipt_path, receipt)
    start = time.monotonic()
    md5, digest, size = hashlib.md5(), hashlib.sha256(), 0
    members = []
    writers = {}
    try:
        response = cached.open('rb') if verified_source else urllib.request.urlopen(entry['links']['self'], timeout=35)
        out = io.BytesIO() if verified_source else cached.open('xb')
        with response, out:
            while True:
                assert time.monotonic() - start < timeout, 'Source acquisition deadline exceeded'
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                assert size <= entry['size']
                md5.update(block)
                digest.update(block)
                if not verified_source:
                    out.write(block)
                # Reject slow routes before retaining hundreds of MB of partial data.
                if not verified_source and size >= 16 * 1024 * 1024:
                    elapsed = time.monotonic() - start
                    assert entry['size'] * elapsed / size <= timeout, 'Measured throughput cannot meet file deadline'
        assert size == entry['size'] and 'md5:' + md5.hexdigest() == entry['checksum']
        receipt.update(raw_zip_bytes=size, raw_zip_sha256=digest.hexdigest(), author_md5_match=True)
        with zipfile.ZipFile(cached) as source:
            for info in source.infolist():
                assert not PurePosixPath(info.filename).is_absolute() and '..' not in PurePosixPath(info.filename).parts
                if info.is_dir():
                    continue
                blob = source.read(info)  # zip CRC independently checked by Python reader.
                raw_sha = hashlib.sha256(blob).hexdigest()
                record = {'source_file':entry['key'], 'source_member':info.filename,
                          'bytes':len(blob), 'sha256':raw_sha, 'crc32':info.CRC}
                if not info.filename.lower().endswith('.wav'):
                    target = root / 'sources' / 'original_metadata' / entry['key'] / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(blob)
                    record.update(role='metadata', retained_path=str(target.relative_to(root)))
                    members.append(record)
                    continue
                if entry['role'] == 'development':
                    role = 'train' if 'train' in PurePosixPath(info.filename).parts else 'test'
                else:
                    role = 'train' if entry['role'] == 'additional' else 'test'
                name = PurePosixPath(info.filename).name
                match = re.search(r'section_(\d+)', name)
                assert match and match.group(1) == '00'
                domain = 'source' if '_source_' in name else 'target' if '_target_' in name else None
                label = 0 if '_normal_' in name else 1 if '_anomaly_' in name else None
                if role == 'train':
                    assert label == 0 and domain in ('source', 'target')
                    output_name = name
                    output_zip = inputs / 'train.zip'
                else:
                    output_name = 'clip_' + hashlib.sha256((entry['checksum'] + info.filename).encode()).hexdigest()[:24] + '.wav'
                    output_zip = inputs / 'queries.zip'
                    if entry['role'] == 'development':
                        assert label in (0, 1) and domain is not None
                rate, data = wavfile.read(io.BytesIO(blob))
                assert data.ndim == 2 and data.shape[1] == 2
                assert rate > 0 and 9.0 <= data.shape[0] / rate <= 13.0
                if output_zip not in writers:
                    assert not output_zip.exists()
                    writers[output_zip] = zipfile.ZipFile(output_zip, 'x', zipfile.ZIP_DEFLATED, compresslevel=3)
                writers[output_zip].writestr(output_name, blob)
                record.update(role=role, input_member=output_name, input_zip=str(output_zip.relative_to(root)),
                              sample_rate=int(rate), frames=int(data.shape[0]), channels=2,
                              dtype=str(data.dtype), section='00', domain=domain, label=label)
                members.append(record)
        for writer in writers.values():
            writer.close()
        writers.clear()
        for output_zip in {root / m['input_zip'] for m in members if m['role'] != 'metadata'}:
            subset = [m for m in members if m.get('input_zip') == str(output_zip.relative_to(root))]
            with zipfile.ZipFile(output_zip) as output:
                assert len(output.namelist()) == len(subset)
                for member in subset:
                    assert hashlib.sha256(output.read(member['input_member'])).hexdigest() == member['sha256']
        receipt.update(status='finished', repack_byte_parity=True, members=len(members),
                       waveform_members=sum(m['role'] != 'metadata' for m in members))
        save(root / 'hidden' / 'source_members' / (entry['key'] + '.json'), members)
        cached.unlink()  # Identical original waveform bytes and original metadata have been retained.
        receipt['duplicate_raw_zip_cleanup'] = 'complete_after_all_members_verified'
    except Exception as error:
        for writer in writers.values():
            writer.close()
        receipt.update(status='execution_error', downloaded_bytes=size,
                       error=type(error).__name__ + ': ' + str(error)[:300],
                       partial_input_packs_ineligible=True)
    receipt.update(elapsed_seconds=time.monotonic() - start, ended_utc=datetime.now(timezone.utc).isoformat())
    save(receipt_path, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['url']}, ensure_ascii=False), flush=True)
    return receipt


def finish_inputs(root, selection, receipts):
    assert all(r['status'] == 'finished' for r in receipts)
    all_members = []
    for entry in selection['files']:
        all_members.extend(json.loads((root / 'hidden/source_members' / (entry['key'] + '.json')).read_text()))
    for machine in selection['task_types']:
        inputs = root / 'tasks' / machine / 'inputs'
        records = [m for m in all_members if m['role'] != 'metadata' and
                   Path(m['input_zip']).parts[1] == machine]
        train = [m for m in records if m['role'] == 'train']
        queries = [m for m in records if m['role'] == 'test']
        assert len(train) == 1000 and len(queries) == 200
        assert sum(m['domain'] == 'source' for m in train) == 990
        assert sum(m['domain'] == 'target' for m in train) == 10
        visible = []
        hidden = []
        for member in sorted(queries, key=lambda m:m['input_member']):
            name = PurePosixPath(member['source_member']).name
            label, domain = member['label'], member['domain']
            attributes = None
            if machine == 'fan':
                attribute_match = re.match(r'section_\d+_(?:source|target)_test_(?:normal|anomaly)_\d+_(.+)\.wav$', name)
                if attribute_match and attribute_match.group(1) != 'noAttributes':
                    attributes = attribute_match.group(1)
            if machine != 'fan':
                truth_dir = root / 'sources/evaluator'
                truth_name = 'ground_truth_' + machine + '_section_00_test.csv'
                def lookup(folder):
                    values = dict(line.split(',') for line in (truth_dir / folder / truth_name).read_text().splitlines())
                    return int(values[name])
                label = lookup('ground_truth_data')
                domain = ['source','target'][lookup('ground_truth_domain')]
            assert label in (0,1) and domain in ('source','target')
            case_id = member['input_member'][:-4]
            visible.append({'case_id':case_id,'file':member['input_member'],'section':'00',
                            'domain':member['domain'] if machine == 'fan' else None,
                            'attributes':attributes})
            hidden.append({'case_id':case_id,'source_member':member['source_member'],
                           'source_file':member['source_file'],'label':label,'domain':domain,
                           'sha256':member['sha256']})
        assert sum(m['label'] == 1 for m in hidden) == 100
        save(inputs / 'queries.json', visible)
        save(root / 'hidden' / (machine + '_gold.json'), hidden)
        save(inputs / 'data_dictionary.json', {'machine_type':machine,'section':'00',
             'benchmark':'DCASE2026 Task2, complete selected machine section, not author Agent benchmark',
             'source':'https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring',
             'channels':{'0':'Near microphone','1':'Far microphone; synchronized'},
             'audio':'Original two-channel WAV payloads, original sample rate/dtype, no waveform transform; whole10/12sec clip',
             'training':'1000 normal clips,990source and10target; training domain/available attribute fields in original training filename. No anomalous training sounds.',
             'queries':'200 clips; outcomes hidden. fan developer domain/attributes available; evaluation machine domains/attributes unknown. Opaque query names preserve the original split and waveform.',
             'boundary':'Fit models, scalers, feature selection and thresholds using allowed training data only. Process each test clip with the frozen model and threshold; other test clips cannot determine its label or threshold. No external data/model downloads.',
             'output':'One finite anomaly_score (larger means more anomalous) and decision0normal/1anomaly for every case_id; use one training-defined threshold per section regardless of test domain.',
             'evaluation':'AUC all, official source/target AUC (domain normal versus all anomalies), pAUC maxFPR0.1, and diagnostic FP/FN/precision/recall/F1. Report threshold and training validation limitations.',
             'limitations':'Real controlled damage/noise-room recordings, not actual industrial deployment or maintenance loss; public benchmark contamination possible.'})
    save(root / 'acoustic_input_receipt85.json', {'status':'finished','tasks':selection['task_types'],
          'complete_tasks':len(selection['task_types']),'train_clips':6000,'query_clips':1200,
          'all_original_waveform_bytes_retained':True,'all_source_author_md5_verified':True,
          'new_LLM_calls':0,'query_labels_outside_workspaces':True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--selection', default='source_selection85_v2.json')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--file-timeout', type=int, default=600)
    args = parser.parse_args()
    root = args.run.resolve()
    selection = json.loads((root / args.selection).read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(lambda entry:prepare_file(root,entry,args.file_timeout), selection['files']))
    save(root / 'source_acquisition_receipt85.json', receipts)
    if all(r['status'] == 'finished' for r in receipts):
        finish_inputs(root, selection, receipts)
    else:
        print(json.dumps({'status':'ineligible_incomplete_source','new_LLM_calls':0}), flush=True)
