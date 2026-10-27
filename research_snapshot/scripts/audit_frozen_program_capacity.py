"""A345: all original real Agent programs on opened A344 data, no new fit/LLM.

Posthoc capacity diagnosis only. Reuses three certified predictions, evaluates
every other complete original program, and never selects a new method winner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_free_program_comparison import independent_score
from program_revision_runtime import Runtime, load, subset, score
from run_frozen_candidate_review import source_context

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
MODEL_SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261024T211000Z_r342l2')
DATA_SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261024T233500Z_query344')


def read(p):
    return json.loads(Path(p).read_text())


def freeze(root):
    assert not root.exists()
    mi, mr = source_context(MODEL_SOURCE); di, dr = source_context(DATA_SOURCE)
    root.mkdir()
    for name in ['code', 'audit', 'outputs', 'runtime/tmp', 'runtime/mpl']:
        (root / name).mkdir(parents=True)
    catalog = dict(common_p00='reference/p00')
    for subject in sorted((MODEL_SOURCE / 'subjects').iterdir()):
        file = subject / 'broker/STATE.json'
        assert sha(file) == mi[str(file.relative_to(MODEL_SOURCE))]['sha256']
        for ident, value in read(file)['programs'].items():
            if ident == 'p00':
                continue
            relative = 'subjects/' + subject.name + '/programs/' + ident
            assert read(MODEL_SOURCE / relative / 'REPORT.json')['status'] == 'complete'
            catalog[subject.name + '__' + ident] = relative
    assert len(catalog) == 14
    cached = {'common_p00': 'p00', 'controlled_edits_2__p01': 'p01', 'controlled_edits_2__p04': 'p04'}
    bound = {}
    for name, relative in catalog.items():
        p = MODEL_SOURCE / relative; report = read(p / 'REPORT.json')
        assert sha(p / 'REPORT.json') == mi[relative + '/REPORT.json']['sha256']
        for filename, digest in report['artifacts'].items():
            assert sha(p / filename) == digest == mi[relative + '/' + filename]['sha256']
        bound[name] = dict(relative=relative, report_sha256=sha(p / 'REPORT.json'), artifacts=report['artifacts'])
    data = DATA_SOURCE / 'qualification/final.npz'
    assert sha(data) == di['qualification/final.npz']['sha256']
    cache_hash = {}
    for name, ident in cached.items():
        relative = 'evaluation/bank/' + ident + '/PREDICTIONS.npz'
        assert sha(DATA_SOURCE / relative) == di[relative]['sha256']
        cache_hash[name] = dict(relative=relative, sha256=sha(DATA_SOURCE / relative))
    proto = dict(iteration='A345', frozen_utc=utc(), purpose='Diagnose original three-bank and action-space limitations on already-opened data;not new method/case confirmation.',
        model_archive=str(MODEL_SOURCE), model_manifest_sha256=mr['manifest_sha256'], models=bound,
        data_archive=str(DATA_SOURCE), data_manifest_sha256=dr['manifest_sha256'], data_relative='qualification/final.npz', data_sha256=sha(data),
        cached=cache_hash, all_original_complete_programs_included=True, outcome_filtering=False,
        hypothesis='Actual nonlinear/other original Agent programs may lower the restricted three-program error floor.',
        rival='All14existing full checkpoints remain unable to correct the difficult acquisitions;new generation/training or information is needed.',
        maximum_new_prediction_processes=264, reused_certified_program_predictions=3, max_parallel_jobs=8,
        total_seconds=900, process_seconds=45, maximum_disk_GiB=4, new_fits=0, new_LLM_calls=0,
        final_performance_previously_opened=True, independent_confirmation_scenarios=0,
        failure_policy='Preserve every program prediction error;any failed new program makes complete-capacity claim incomplete. No repair/refit/retry.',
        metrics=['Everyprogram whole-record loss', 'Bestglobalprogram descriptive oracle', 'Per-record hindsight oracle', 'All-models-wrong acquisitions'])
    save(root / 'PROTOCOL345.json', proto)
    for name in ['audit_frozen_program_capacity.py', 'program_revision_runtime.py', 'evaluate_free_program_comparison.py',
                 'controlled_feedback.py', 'run_frozen_candidate_review.py', 'prepare_fault_extensions.py',
                 'acquire_forge_pdmb_sources244.py']:
        shutil.copy2(PROJECT / 'scripts' / name, root / 'code' / name)
    save(root / 'CODE_FREEZE345.json', {p.name: sha(p) for p in (root / 'code').glob('*.py')})
    save(root / 'ENGINE_CONFIG.json', dict(input_root=str(DATA_SOURCE / 'qualification'), common_tools=str(MODEL_SOURCE / 'common_tools'),
         fit_seconds=0, predict_seconds=45, program_seconds=900))
    shutil.copy2(DATA_SOURCE / 'retention_policy.json', root / 'retention_policy.json')
    save(root / 'OWNERSHIP.json', dict(run=str(root), archive='/data/yyf-zyq/PHM_20261010/' + root.name, created_utc=utc(),
         authorized_goal='Continue real PHM program-update experiments;capacity diagnosis,not restricted-study success.'))
    save(PROJECT / 'results/capacity345/PROTOCOL345.json', proto)
    print(json.dumps(dict(status='capacity_protocol_frozen', programs=14, new_predict_process_budget=264, new_fits=0, new_LLM_calls=0)))


def run(root):
    assert not (root / 'RESULT345.json').exists() and not (root / 'STARTED345.json').exists()
    cfg = read(root / 'PROTOCOL345.json')
    for filename, digest in read(root / 'CODE_FREEZE345.json').items():
        assert sha(root / 'code' / filename) == digest
    data_path = DATA_SOURCE / cfg['data_relative']; assert sha(data_path) == cfg['data_sha256']
    data = load(data_path); conditions = sorted(set(data['condition_id'].tolist()))
    started = time.monotonic(); deadline = started + cfg['total_seconds']
    save(root / 'STARTED345.json', dict(utc=utc(), protocol_sha256=sha(root / 'PROTOCOL345.json'), posthoc_opened_data=True))
    vectors = {}; results = {}
    for name, item in cfg['cached'].items():
        file = DATA_SOURCE / item['relative']; assert sha(file) == item['sha256']
        with np.load(file, allow_pickle=False) as z:
            assert np.array_equal(z['sample_ids'], data['sample_ids']); vectors[name] = z['prediction']
        results[name] = dict(status='complete', reused_prediction=True, **score(data, vectors[name]))
    new_names = sorted(set(cfg['models']) - set(cfg['cached'])); pieces = {}; failures = []
    def one(name, condition):
        binding = cfg['models'][name]; model = MODEL_SOURCE / binding['relative']
        for filename, digest in binding['artifacts'].items():
            assert sha(model / filename) == digest
        mask = data['condition_id'] == condition; part = subset(data, mask)
        output = root / 'outputs' / name / str(int(condition)); output.mkdir(parents=True)
        engine = Runtime(root); engine.deadline = deadline
        try:
            a, aa = engine.predict(output / 'forward', model / 'source', model / 'full_fit/artifacts', part)
            b, ab = engine.predict(output / 'cold_repeat', model / 'source', model / 'full_fit/artifacts', part)
            assert np.array_equal(a, b) and all(x['completed_estimator_fits'] == x['optimizer_steps'] == 0 for x in aa + ab)
            s = score(part, a); independent = independent_score(part, a)
            assert s['record_error'] == independent['record_error'] and s['confusion'] == independent['confusion']
            return dict(name=name, condition=int(condition), status='complete', mask=mask, prediction=a, process_count=len(aa + ab))
        except Exception as error:
            return dict(name=name, condition=int(condition), status='prediction_error', error_type=type(error).__name__, error=str(error))
    progress = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(one, name, condition) for name in new_names for condition in conditions]):
            item = f.result(); progress.append({k: v for k, v in item.items() if k not in ['mask', 'prediction']})
            if item['status'] == 'complete':
                pieces.setdefault(item['name'], np.full(len(data['labels']), -1, dtype=int))[item['mask']] = item['prediction']
            else:
                failures.append(item)
            save(root / 'PROGRESS345.json', progress)
    for name in new_names:
        if name in pieces and np.all(pieces[name] >= 0):
            vectors[name] = pieces[name]; results[name] = dict(status='complete', reused_prediction=False, **score(data, pieces[name]))
            np.savez_compressed(root / 'outputs' / name / 'PREDICTIONS.npz', sample_ids=data['sample_ids'], prediction=pieces[name])
        else:
            results[name] = dict(status='prediction_error', errors=[x for x in failures if x['name'] == name])
    record_maps = {name: {r['group']: r for r in value['records']} for name, value in results.items() if value['status'] == 'complete'}
    hard = [group for group in record_maps['common_p00'] if not any(r[group]['actual'] == r[group]['predicted'] for r in record_maps.values())]
    best = min(r['record_error'] for r in results.values() if r['status'] == 'complete')
    result = dict(iteration='A345', utc=utc(), status='complete' if not failures else 'incomplete_prediction_failures', programs=results,
        known_best_global_error=best, known_best_global_programs=[p for p, v in results.items() if v.get('record_error') == best],
        known_record_oracle_error=len(hard) / len(record_maps['common_p00']), all_models_wrong_groups=hard,
        fully_evaluated_programs=len(vectors), source_programs=14, new_prediction_processes=sum(x.get('process_count', 0) for x in progress),
        new_fits=0, new_LLM_calls=0, seconds=time.monotonic() - started, not_method_improvement=True,
        original_R5_did_not_supply_extra11programs_to_actors=True, independent_confirmation_scenarios=0)
    save(root / 'RESULT345.json', result); save(PROJECT / 'results/capacity345/RESULT345.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'programs'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['freeze', 'run']); p.add_argument('--run', required=True, type=Path)
    args = p.parse_args(); globals()[args.action](args.run.resolve())
