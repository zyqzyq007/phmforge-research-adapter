"""Freeze and falsify an existing program-ranking claim on unused measured contexts.

This is a fixed-weight inference audit, not a new Agent comparison or model fit.
Source acquisition and integrity qualification use the existing source runner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_free_program_comparison import independent_score
from program_revision_runtime import Runtime, load, score
from run_frozen_candidate_review import source_context

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')


def read(path):
    return json.loads(Path(path).read_text())


def freeze(root, spec):
    cfg = read(spec)
    tag = cfg['tag']
    assert str(tag).isdigit()
    source_protocol = root / ('PROTOCOL' + tag + '.json')
    source = read(source_protocol)
    assert not (root / ('SOURCE_PROGRESS' + tag + '.json')).exists()
    archive = Path(cfg['model_archive'])
    inventory, receipt = source_context(archive)
    bindings = {}
    for name, relative in cfg['models'].items():
        path = archive / relative
        report = read(path / 'REPORT.json')
        assert sha(path / 'REPORT.json') == inventory[relative + '/REPORT.json']['sha256']
        for filename, digest in report['artifacts'].items():
            assert sha(path / filename) == digest == inventory[relative + '/' + filename]['sha256']
        bindings[name] = dict(relative_path=relative, report_sha256=sha(path / 'REPORT.json'),
                              artifacts=report['artifacts'])
    prior = set()
    for entry in source['prior_bindings'] + [dict(archive=source['source_archive'])]:
        old = Path(entry['archive'])
        files, _ = source_context(old)
        for relative, item in files.items():
            if relative.startswith('workspace_snapshot/'):
                continue
            if ('IDENTITIES' in relative and relative.endswith('.json')):
                assert sha(old / relative) == item['sha256']
                for row in read(old / relative):
                    prior.add(row.get('raw_sha256', row.get('source_sha256')))
            elif relative.startswith('records/') and relative.endswith('/fetch/RECEIPT.json'):
                assert sha(old / relative) == item['sha256']
                row = read(old / relative)
                if row.get('complete_record_CRC_verified') and row.get('passed'):
                    prior.add(row['sha256'])
    prior.discard(None)
    assert prior
    for name in ['frozen_program_context_audit.py', 'program_revision_runtime.py',
                 'evaluate_free_program_comparison.py']:
        shutil.copy2(PROJECT / 'scripts' / name, root / 'code' / name)
    cfg.update(frozen_utc=utc(), source_protocol_sha256=sha(source_protocol),
               model_archive_manifest_sha256=receipt['manifest_sha256'], model_bindings=bindings,
               prior_raw_sha256=sorted(prior), source_role_count=len(source['roles_Hz']),
               code_hashes={p.name: sha(p) for p in (root / 'code').glob('*.py')},
               source_not_yet_acquired=True, new_fit_budget=0, new_LLM_budget=0,
               physical_source_clusters=1, independent_confirmation_scenarios=0,
               not_method_confirmation=True)
    save(root / 'FROZEN_CONTEXT_AUDIT.json', cfg)
    save(PROJECT / 'results' / source['result_key'] / 'AUDIT_PROTOCOL.json', cfg)
    print(json.dumps(dict(status='audit_frozen', models=list(bindings),
                         prior_raw_records=len(prior), source_roles=source['roles_Hz'])))


def evaluate(root):
    cfg = read(root / 'FROZEN_CONTEXT_AUDIT.json')
    tag = cfg['tag']
    assert not (root / 'CONTEXT_RESULTS.json').exists(), 'Do not repeat completed evaluation'
    assert not (root / 'CONTEXT_EVALUATION_STARTED.json').exists(), 'Inspect partial run before recovery'
    for name, digest in cfg['code_hashes'].items():
        assert sha(root / 'code' / name) == digest
    protocol = read(root / ('PROTOCOL' + tag + '.json'))
    assert sha(root / ('PROTOCOL' + tag + '.json')) == cfg['source_protocol_sha256']
    qualification = read(root / ('QUALIFICATION' + tag + '.json'))
    assert qualification['passed']
    ids = read(root / 'qualification' / ('IDENTITIES' + tag + '.json'))
    current = [x['raw_sha256'] for x in ids]
    assert len(current) == len(set(current)) == protocol['expected_records']
    assert not set(current) & set(cfg['prior_raw_sha256']), 'Prior/reserved raw-content overlap'
    archive = Path(cfg['model_archive'])
    _, receipt = source_context(archive)
    assert receipt['manifest_sha256'] == cfg['model_archive_manifest_sha256']
    models = {}
    for name, binding in cfg['model_bindings'].items():
        path = archive / binding['relative_path']
        assert sha(path / 'REPORT.json') == binding['report_sha256']
        for relative, digest in binding['artifacts'].items():
            assert sha(path / relative) == digest
        models[name] = path
    for name, item in qualification['files'].items():
        assert sha(root / 'qualification' / name) == item['sha256']
    assert cfg['max_prediction_processes'] == 2 * len(models) * len(ids)
    save(root / 'ENGINE_CONFIG.json', dict(input_root=str(root / 'qualification'),
         common_tools=str(archive / 'common_tools'), fit_seconds=0, predict_seconds=45, program_seconds=300))
    started = time.monotonic()
    deadline = started + cfg['total_evaluation_seconds']
    save(root / 'CONTEXT_EVALUATION_STARTED.json', dict(utc=utc(), protocol_sha256=sha(root / 'FROZEN_CONTEXT_AUDIT.json'),
         source_qualification_sha256=sha(root / ('QUALIFICATION' + tag + '.json')), all_models_fixed_before_capture=True,
         prior_raw_overlap=0, new_fits=0, new_LLM_calls=0))
    save(root / 'CONTEXT_HANDLE.json', dict(pid=os.getpid(), status='running', phase='fixed_model_prediction'))
    data = {role: load(root / 'qualification' / (role + '.npz')) for role in protocol['roles_Hz']}

    def replay(name, role):
        output = root / 'predictions' / role / name
        output.mkdir(parents=True)
        engine = Runtime(root)
        engine.deadline = min(deadline, time.monotonic() + 300)
        try:
            path = models[name]
            a, audits = engine.predict(output / 'forward', path / 'source', path / 'full_fit/artifacts', data[role])
            b, audits2 = engine.predict(output / 'cold_repeat', path / 'source', path / 'full_fit/artifacts', data[role])
            assert np.array_equal(a, b)
            assert all(x['completed_estimator_fits'] == x['optimizer_steps'] == 0 for x in audits + audits2)
            result = score(data[role], a)
            check = independent_score(data[role], a)
            assert result['confusion'] == check['confusion'] and result['record_error'] == check['record_error']
            result.update(status='complete', prediction_processes=len(audits + audits2), cold_repeat_identical=True,
                          independent_votes_verified=True)
        except Exception as error:
            result = dict(status='evaluation_error', error_type=type(error).__name__, error=str(error))
        result.update(program=name, role=role)
        save(output / 'RESULT.json', result)
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=cfg['max_parallel_predictions']) as pool:
        for future in as_completed([pool.submit(replay, name, role) for name in models for role in data]):
            rows.append(future.result())
            save(root / 'CONTEXT_PROGRESS.json', rows)
    complete = all(row['status'] == 'complete' for row in rows)
    summary = None
    if complete:
        lookup = {(r['program'], r['role']): r['record_error'] for r in rows}
        compare = [{ 'role': role,
                     'full_feature_error': lookup['full_features', role],
                     'selected_repair_error': lookup['selected_repair', role],
                     'difference_full_minus_repair': lookup['full_features', role] - lookup['selected_repair', role]}
                   for role in data]
        strict_wins = sum(r['difference_full_minus_repair'] < 0 for r in compare)
        strict_losses = sum(r['difference_full_minus_repair'] > 0 for r in compare)
        average = {name: float(np.mean([lookup[name, role] for role in data])) for name in models}
        summary = dict(per_context=compare, mean_record_errors=average, full_feature_wins=strict_wins,
                       ties=len(compare) - strict_wins - strict_losses, full_feature_losses=strict_losses,
                       prespecified_candidate_advantage_reproduced=(strict_wins >= 3 and average['full_features'] < average['selected_repair']),
                       rival_repair_wins_majority=(strict_losses >= 3 and average['selected_repair'] < average['full_features']),
                       rank_reversal_in_new_contexts=(strict_wins > 0 and strict_losses > 0))
    result = dict(utc=utc(), status='complete' if complete else 'incomplete_evaluation', rows=rows, summary=summary,
                  new_fits=0, new_LLM_calls=0, prediction_processes=sum(r.get('prediction_processes', 0) for r in rows),
                  physical_source_clusters=1, independent_confirmation_scenarios=0, not_method_comparison=True,
                  seconds=time.monotonic() - started, prior_raw_overlap=0)
    save(root / 'CONTEXT_RESULTS.json', result)
    save(PROJECT / 'results' / protocol['result_key'] / 'CONTEXT_RESULTS.json', result)
    save(root / 'CONTEXT_HANDLE.json', dict(status='terminal', utc=utc(), result_status=result['status']))
    print(json.dumps(dict(status=result['status'], summary=summary, prediction_processes=result['prediction_processes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'evaluate'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        assert args.spec
        freeze(args.run.resolve(), args.spec)
    else:
        evaluate(args.run.resolve())
