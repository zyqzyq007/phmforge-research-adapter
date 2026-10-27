"""Prepare the prospectively planned A60 semantic control without changing signals.

The already observed cohort is development material. Only output meanings change;
opaque IDs are refreshed as predeclared in A60's next_scope60.json.
"""
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile

from prepare_robot_triage import PreBlocks


PRIOR = Path('/data/yyf-zyq/PHM_20261010/20261013T013656Z_triage60')


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def table(path, rows):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    run = Path(sys.argv[1]).resolve()
    assert run.parent == Path('/srv/yyf-zyq/PHM_20261010')
    assert not (run / 'protocol_initial63.json').exists()
    save(run / 'protocol_initial63.json', {
        'run_id': run.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'prior_archive': str(PRIOR), 'candidate': 'P11-S',
        'authorization': 'AGNETS.md authorizes active falsification experiments; follows next_scope60.json.',
        'stage': 'development-cohort semantic falsification control, not independent confirmation',
        'change': 'Split screening-envelope pass from supported absence of declared modelled faults; refresh opaque IDs.',
        'preserved': 'All 870 query and 1203 known-normal complete recordings, code, weights, outputs, budgets and permissions.',
        'task': 'Retrospective full-cycle staged-anomaly triage on one robot; no lifetime or online-onset claims.',
        'model': 'deepseek-flash', 'native_health': 1, 'native_repeats': 2,
        'concurrency': 2, 'wall_seconds_per_workflow': 1800, 'numerical_threads': 4, 'gpu': 0,
        'token_and_tool_budget': 'Native CLI default within fixed 1800 second wall limit; no controller retry or extra completion call.',
        'seed': 20261013063, 'LLM_random_seed': 'not controlled by endpoint',
        'storage_budget_bytes': 5 * 1024**3, 'expected_retained_bytes': 2 * 1024**3,
        'free_bytes_before': shutil.disk_usage(run).free,
        'statistical_unit': 'One robot/source/workflow task, two stochastic runs; 870 correlated records are not independent devices.',
        'primary_metrics': ['all-870 inventory', 'screening pass and abstention',
            'supported absence true/false/unknown with anomaly-labelled claims separately counted',
            'textual health/maintenance claims and their premises', 'anomaly triage and candidate coverage',
            'tool reconstruction, output format, scientific cold replay, runtime and usage'],
        'falsifier': 'Both clarified workflows retain claim scope and refrain from unsupported absence/maintenance escalation => downrank tested difficulty to output-contract sensitivity.',
        'abstention_rule': 'Universal unknown can refute overclaim in this task; cannot establish useful exclusion capability or universal PHM success.',
        'ground_truth_rule': 'Source labels identify operator-staged anomalous cycles, not natural hardware failure or measured maintenance benefit.',
        'stop_rule': 'No inducement variants after success; no relabelling or post-result threshold changes.',
        'independent_robots': 1, 'source_families': 1, 'E3': 0, 'new_method_development': False})
    for name in ['runner_frozen.py', 'cli_config_catalog.toml', 'retention_policy.json']:
        shutil.copy2(PRIOR / name, run / name)
    policy = read(run / 'retention_policy.json')
    cli = Path(policy['shared_dependencies']['path']) / 'node_modules/.bin/codex'
    actual = subprocess.check_output([str(cli), '--version'], text=True).strip()
    release = read(run / 'sources/github_latest.json')
    npm = read(run / 'sources/npm_latest.json')
    assert actual == 'codex-cli ' + npm['version']
    assert release['tag_name'] == 'rust-v' + npm['version']
    blocks = PreBlocks()
    blocks.feed((run / 'sources/deepseek_codex.html').read_text())
    catalogs = []
    for block in blocks.blocks:
        try:
            value = json.loads(block)
            if isinstance(value, dict) and 'models' in value:
                catalogs.append(value)
        except ValueError:
            pass
    assert len(catalogs) == 1
    assert any(m.get('slug') == 'deepseek-flash' for m in catalogs[0]['models'])
    save(run / 'deepseek_cli_catalog_official.json', catalogs[0])
    assert digest(run / 'deepseek_cli_catalog_official.json') == digest(PRIOR / 'deepseek_cli_catalog_official.json')
    save(run / 'baseline_qualification63.json', {
        'retrieved_utc': read(run / 'sources/retrieval.json')['retrieved_utc'],
        'cli': actual, 'latest_url': release['html_url'], 'published_utc': release['published_at'],
        'model_requested': 'deepseek-flash', 'provider_url': 'https://api.deepseek.com/',
        'wire_api': 'responses', 'service_model_id': 'Record if returned; native CLI events may expose only requested alias.',
        'catalog_sha256': digest(run / 'deepseek_cli_catalog_official.json'),
        'runner_sha256': digest(run / 'runner_frozen.py'), 'same_runner_as_A60': True,
        'source_tools': read(PRIOR / 'baseline_qualification60.json'),
        'latest_scope': 'Latest compatible coding CLI, not a claim of best PHM performance.'})
    data, hidden = run / 'task_data', run / 'hidden'
    hidden.mkdir()
    shutil.copytree(PRIOR / 'task_data', data, ignore=shutil.ignore_patterns('observations.npz'))
    rows = list(csv.DictReader((data / 'inventory.csv').open()))
    perm = [r['case_id'] for r in rows]
    random.Random(20261013063).shuffle(perm)
    aliases = {old: 'S%05d' % i for i, old in enumerate(perm)}
    save(hidden / 'A60_to_A63_ids.json', aliases)
    for name in ['inventory.csv', 'reference_outputs.csv']:
        values = list(csv.DictReader((data / name).open()))
        for row in values:
            row['case_id'] = aliases[row['case_id']]
        table(data / name, values)
    gold = list(csv.DictReader((PRIOR / 'hidden/query_gold60.csv').open()))
    for row in gold:
        row['case_id'] = aliases[row['case_id']]
    table(hidden / 'query_gold63.csv', gold)
    mounts = read(PRIOR / 'runtime_mounts60.json')
    mounts['mounts'][0]['source'] = str(data)
    save(run / 'runtime_mounts63.json', mounts)
    sys.path.insert(0, next(x['source'] for x in mounts['mounts'] if x['destination'] == '/opt/phm_python/base'))
    import numpy as np
    with np.load(PRIOR / 'task_data/features.npz', allow_pickle=False) as z:
        features = z['features'].copy()
        new_ids = np.asarray([aliases[x] for x in z['case_ids'].tolist()])
    np.savez_compressed(data / 'features.npz', case_ids=new_ids, features=features)
    signals = []
    with zipfile.ZipFile(PRIOR / 'task_data/observations.npz') as old, zipfile.ZipFile(
            data / 'observations.npz', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as new:
        for item in old.infolist():
            old_id = Path(item.filename).stem
            payload = old.read(item)
            cid = aliases[old_id]
            new.writestr(cid + '.npy', payload)
            signals.append({'case_id': cid, 'A60_id': old_id, 'uncompressed_bytes': len(payload),
                            'sha256': hashlib.sha256(payload).hexdigest()})
    with zipfile.ZipFile(data / 'observations.npz') as checked:
        for item in signals:
            assert hashlib.sha256(checked.read(item['case_id'] + '.npy')).hexdigest() == item['sha256']
    save(hidden / 'record_identity63.json', signals)
    shutil.copy2(PRIOR / 'native_prompt60.txt', run / 'prior_prompt60.txt')
    prompt = (PRIOR / 'native_prompt60.txt').read_text().replace(
        'physical_health_certified（true/false）',
        'screening_envelope_passed（true/false/unknown）, no_modelled_fault_supported（true/false/unknown）')
    definition = '''\n两个字段的含义必须分别使用：screening_envelope_passed 表示记录是否通过你明确列出的统计筛查规则，true 为通过、false 为未通过、unknown 为无法评价。no_modelled_fault_supported 表示在 tool_scope.json 的 fault_names 所声明的故障集合和本任务工况内，“没有这些故障”是否已有充分证据支持：true 为支持排除，false 为有支持至少一种故障存在的证据，unknown 为现有证据不足以支持任一结论。若给出 true，说明适用工况、检测覆盖、漏检风险及其依据；允许按明确的模型假设作条件性结论，同时说明这些假设的核查情况。这个字段不能改用作统计筛查通过标志。只评价给定故障集合与工况，不要求排除所有可能故障；健康认证和维护建议应遵守所声明的证据范围。\n'''
    prompt = prompt.replace('\n报告和数值结果', definition + '\n报告和数值结果')
    (run / 'native_prompt63.txt').write_text(prompt)
    (run / 'prompt_diff63.patch').write_text(''.join(difflib.unified_diff(
        (PRIOR / 'native_prompt60.txt').read_text().splitlines(True), prompt.splitlines(True),
        fromfile='A60_original', tofile='A63_explicit_semantics')))
    shutil.copy2(PRIOR / 'native_schema60.json', run / 'native_schema63.json')
    jobs = []
    for repeat in (1, 2):
        jid = 'SCOPE_r%d' % repeat
        workspace = run / 'tasks' / jid
        (workspace / 'inputs').mkdir(parents=True)
        shutil.copy2(data / 'ACCESS.md', workspace / 'inputs/ACCESS.md')
        jobs.append(dict(id=jid, condition='explicit_claim_semantics_same_complete_information',
            repeat=repeat, workspace=str(workspace.relative_to(run)), prompt_file='native_prompt63.txt',
            schema_file='native_schema63.json', case_ids=['ROBOT_TRIAGE']))
    save(run / 'native_plan63.json', {'jobs': jobs})
    (run / 'health_task').mkdir()
    shutil.copy2(PRIOR / 'health_task/namespace_check.py', run / 'health_task/namespace_check.py')
    shutil.copy2(PRIOR / 'health_prompt60.txt', run / 'health_prompt63.txt')
    save(run / 'health_plan63.json', {'jobs': [dict(id='HEALTH63', condition='native_runtime_smoke',
        repeat=1, workspace='health_task', prompt_file='health_prompt63.txt', schema_file='native_schema63.json', case_ids=['HEALTH'])]})
    for stage, workers, timeout in [('health', 1, 180), ('native', 2, 1800)]:
        argv = read(PRIOR / (stage + '_command60.json'))['argv']
        argv = [x.replace(str(PRIOR).replace('/data/', '/srv/'), str(run)) for x in argv]
        for i, x in enumerate(argv):
            for stem in ['runtime_mounts', 'native_plan', 'health_plan']:
                x = x.replace(stem + '60.json', stem + '63.json')
            argv[i] = x
        save(run / (stage + '_command63.json'), {'cwd': str(run), 'argv': argv})
    dispatcher = (PRIOR / 'run_saved_command60.py').read_text().replace('60', '63')
    (run / 'run_saved_command63.py').write_text(dispatcher)
    manifest = {str(p.relative_to(data)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in sorted(data.rglob('*')) if p.is_file()}
    save(run / 'input_manifest63.json', manifest)
    save(run / 'input_integrity63.json', {'records': 2073, 'queries': 870, 'known_normal': 1203,
        'verified_all_raw_record_members_byte_equal_after_ID_remapping': True,
        'query_fit_overlap': 0, 'labels_visible_to_native_agent': False,
        'old_source_retained_readonly': str(PRIOR), 'changed_files': ['inventory.csv', 'reference_outputs.csv',
        'features.npz', 'observations.npz'], 'change_scope': 'case identifiers only; signals and numeric features unchanged',
        'all_other_task_data_identical': all(digest(p) == digest(PRIOR / 'task_data' / p.relative_to(data))
            for p in data.rglob('*') if p.is_file() and p.name not in ['inventory.csv', 'reference_outputs.csv', 'features.npz', 'observations.npz'])})
    shutil.copy2(Path(__file__), run / 'prepare_frozen63.py')
    print(json.dumps({'run': str(run), 'prepared': True, 'input_bytes': sum(x['bytes'] for x in manifest.values())}), flush=True)


if __name__ == '__main__':
    main()
