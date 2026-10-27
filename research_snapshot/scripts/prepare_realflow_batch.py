"""Prepare complete NASA wear / Ottawa diagnosis flows from verified originals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
from scipy.io import loadmat


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_files(root):
    receipt = json.loads((root / 'ARCHIVE_VERIFIED.json').read_text())
    assert sha(root / 'SOURCE_MANIFEST.json') == receipt['manifest_sha256']
    return {r['path']: r for r in json.loads((root / 'SOURCE_MANIFEST.json').read_text())['files']}


def checked(root, inventory, relative):
    path = root / relative
    entry = inventory[relative]
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--milling', type=Path, required=True)
    parser.add_argument('--bearing', type=Path, required=True)
    args = parser.parse_args()
    run, milling, bearing = args.run.resolve(), args.milling.resolve(), args.bearing.resolve()
    mi, bi = archive_files(milling), archive_files(bearing)
    started = time.monotonic()
    all_files = []
    for p in sorted((bearing / 'ottawa_file_metadata').glob('*.json')):
        if p.name != 'manifest.json':
            checked(bearing, bi, str(p.relative_to(bearing)))
            all_files.extend(json.loads(p.read_text()))
    assert len(all_files) == 60
    selected = [f for f in all_files if f['filename'].split('-')[1] in ['B', 'C', 'D'] and f['filename'].split('-')[2] in ['2.mat', '3.mat']]
    assert len(selected) == 30
    save(run / 'source_selection83.json', {'selected': selected, 'rule': 'All B/C/D speed profiles, all five health states, both original trials2/3. Selection independent of predictions; A trial1 references reused.', 'source_metadata': 'Author version2 JSON, originally retrieved A11; every metadata file checked against immutable archive SHA. New downloaded raw bytes must equal author SHA.'})
    def fetch(item):
        path = run / 'raw_data' / item['filename']
        url = item['content_details']['download_url']
        result = {'filename': item['filename'], 'url': url, 'author_sha256': item['content_details']['sha256_hash'], 'expected_bytes': item['content_details']['size']}
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
                with urllib.request.urlopen(request, timeout=90) as response, (path.with_suffix('.part')).open('wb') as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                path.with_suffix('.part').replace(path)
                result.update(bytes=path.stat().st_size, sha256=sha(path), status='finished', attempts=attempt + 1)
                assert result['sha256'] == result['author_sha256'] and result['bytes'] == result['expected_bytes']
                return result
            except Exception as error:
                result.update(status='source_error', error_type=type(error).__name__, http_code=getattr(error, 'code', None), attempts=attempt + 1)
                if attempt < 2:
                    time.sleep(attempt + 1)
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloads = list(pool.map(fetch, selected))
    save(run / 'source_download83.json', downloads)
    assert all(x['status'] == 'finished' for x in downloads)
    raw_checks = []
    for p in sorted((run / 'raw_data').glob('*.mat')):
        mat = loadmat(p)
        values = [np.asarray(mat[k]).reshape(-1) for k in ['Channel_1', 'Channel_2']]
        assert all(len(v) == 2000000 and np.isfinite(v).all() for v in values)
        raw_checks.append({'filename': p.name, 'sha256': sha(p), 'channels': 2, 'samples_each': 2000000, 'finite': True})
    save(run / 'raw_qualification83.json', raw_checks)

    jobs, gold, inputs = [], [], []
    for material in [1, 2]:
        old = f'milling_material{material}_r1'
        name = f'wear_material{material}_newsession'
        work = run / 'tasks' / name
        work.mkdir()
        names = ['data_dictionary.json', 'features.csv', 'prediction_schema.json', 'query_inventory.csv', 'sensors.npz', 'task_budget.json', 'test.csv', 'train.csv']
        for n in names:
            shutil.copy2(checked(milling, mi, 'tasks/' + old + '/' + n), work / n)
        budget = json.loads((work / 'task_budget.json').read_text())
        # Preserve original fields and make the current total budget explicit.
        budget['wall_clock_seconds_total'] = 1800
        save(work / 'task_budget.json', budget)
        prompt = checked(milling, mi, 'tasks/' + old + '_prompt.txt').read_text().replace('1200 second', '1800 second')
        prompt = prompt.replace('Read task_budget.json', 'The current total budget is1800seconds. Read task_budget.json')
        (run / 'tasks' / (name + '_prompt.txt')).write_text(prompt)
        shutil.copy2(checked(milling, mi, 'tasks/' + old + '_schema.json'), run / 'tasks' / (name + '_schema.json'))
        save(work / 'readonly_inputs.json', names)
        case_id = 'milling_material' + str(material)
        jobs.append({'id': name, 'condition': 'fresh_wear_discovery_repeat', 'repeat': 1, 'workspace': 'tasks/' + name, 'prompt_file': 'tasks/' + name + '_prompt.txt', 'schema_file': 'tasks/' + name + '_schema.json', 'case_ids': [case_id]})
        gold.append({'job_id': name, 'family': 'milling', 'historical_scored_task': True, 'independent_confirmation': False})

    original_prompt = checked(bearing, bi, 'tasks/Ottawa_B_r1_prompt.txt').read_text()
    schema = json.loads(checked(bearing, bi, 'tasks/Ottawa_B_r1_schema.json').read_text())
    dictionary = json.loads(checked(bearing, bi, 'tasks/Ottawa_B_r1/data_dictionary.json').read_text())
    dictionary['source_limitations'] = 'Original trial1 A references reused; query records are previously unrun original trials2/3 in B/C/D. Labels and original label-encoded filenames hidden. A complete cell panel contains one recording per class; this composition prior is legal. Physical bearing identity remains unknown; public-pretraining contamination cannot be excluded.'
    states = {'H': 0, 'I': 1, 'O': 2, 'B': 3, 'C': 4}
    profiles = {'B': 'decreasing', 'C': 'increasing_then_decreasing', 'D': 'decreasing_then_increasing'}
    old_refs = json.loads(checked(bearing, bi, 'tasks/Ottawa_B_r1/references.json').read_text())
    for trial in [2, 3]:
        for profile in ['B', 'C', 'D']:
            name = f'Ottawa_{profile}_trial{trial}'
            work = run / 'tasks' / name
            (work / 'data').mkdir(parents=True)
            for r in old_refs:
                shutil.copy2(checked(bearing, bi, 'tasks/Ottawa_B_r1/' + r['file']), work / r['file'])
            queries = []
            for letter, code in states.items():
                original = f'{letter}-{profile}-{trial}.mat'
                rid = 'rec_' + hashlib.sha256(('PHM_bearing83:' + original).encode()).hexdigest()[:16]
                relative = 'data/' + rid + '.mat'
                shutil.copy2(run / 'raw_data' / original, work / relative)
                queries.append({'case_id': rid, 'file': relative, 'speed_profile': profiles[profile]})
                gold.append({'job_id': name, 'case_id': rid, 'original_filename': original, 'class_code': code, 'family': 'bearing', 'prior_agent_scored': False, 'independent_physical_device': 'unknown'})
            queries.sort(key=lambda row: row['case_id'])
            save(work / 'queries.json', queries)
            save(work / 'references.json', old_refs)
            save(work / 'data_dictionary.json', dictionary)
            save(work / 'readonly_inputs.json', ['data', 'queries.json', 'references.json', 'data_dictionary.json'])
            prompt = original_prompt.replace('Ottawa_B_r1', name).replace('1200 seconds', '1800 seconds')
            (run / 'tasks' / (name + '_prompt.txt')).write_text(prompt)
            save(run / 'tasks' / (name + '_schema.json'), schema)
            jobs.append({'id': name, 'condition': 'unseen_original_trial_complete_flow', 'repeat': 1, 'workspace': 'tasks/' + name, 'prompt_file': 'tasks/' + name + '_prompt.txt', 'schema_file': 'tasks/' + name + '_schema.json', 'case_ids': [name]})
    for job in jobs:
        work = run / job['workspace']
        (work / 'AGENTS.md').write_text('All temporary files and caches must stay in this workspace/.runtime; never use /tmp. Source inputs are read-only. Use the installed scientific Python packages and the supplied task budget. Do not use network access or other model APIs.\n')
        for p in sorted(work.rglob('*')):
            if p.is_file():
                inputs.append({'job_id': job['id'], 'file': str(p.relative_to(work)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    save(run / 'plan83.json', {'jobs': jobs})
    save(run / 'hidden/gold83.json', gold)
    save(run / 'input_manifest83.json', inputs)
    save(run / 'preparation_receipt83.json', {'tasks': len(jobs), 'historical_wear_tasks': 2, 'new_bearing_record_cohorts': 6, 'new_bearing_query_records': 30, 'new_device_independence': 'unknown', 'downloaded_author_sha_matches': len(downloads), 'input_files': len(inputs), 'elapsed_seconds': time.monotonic() - started, 'task_scope': 'AI-authored complete real-data workflows reused from A09/A11; NOT original public Agentbenchmark or official method replication'})
    print(json.dumps({'tasks': len(jobs), 'new_records': 30, 'source_hash_matches': len(downloads), 'input_files': len(inputs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
