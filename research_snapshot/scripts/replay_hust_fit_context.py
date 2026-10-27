"""Replay original full/CV artifacts on identical legal queries; no new training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save, sha, utc
from program_revision_runtime import Runtime, load
from screen_score_blinding import restore


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    cfg = read(root/'PROTOCOL456.json')
    assert not (root/'PREPARED456.json').exists()
    groups = defaultdict(list)
    for resource in cfg['resource_bindings']:
        groups[resource['archive']].append(resource)
    def one_archive(item):
        path, resources = item
        archive = Path(path)
        assert sha(archive/'SOURCE_MANIFEST.json') == cfg['archives'][path]['manifest_sha256']
        index = {r['path']: r for r in read(archive/'SOURCE_MANIFEST.json')['files']}
        missing = {r['member'] for r in resources if not (archive/r['member']).is_file()}
        cache = root/'clients-cache'/archive.name
        if missing:
            restored = restore(archive, cache, lambda p: p in missing)
            assert {r['member'] for r in restored} == missing
        records = []
        for resource in resources:
            member = resource['member']
            source = (cache if member in missing else archive)/member
            assert sha(source) == resource['sha256'] == index[member]['sha256']
            dest = root/resource['target']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            assert sha(dest) == resource['sha256']
            records.append(resource)
        return records
    bindings = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for records in pool.map(one_archive, groups.items()):
            bindings.extend(records)
    data = load(root/'inputs/observations.npz')
    assert set(data) == {'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
    assert len(data['sample_ids']) == 8 and set(data['cycle']) == {500}
    (root/'queries').mkdir()
    for i in range(8):
        np.savez_compressed(root/'queries'/f'{i}.npz', **{k: v[i:i+1] for k,v in data.items()})
    tasks = []
    for program in cfg['programs']:
        gold = dict(zip(program['unit_ids'], program['expected_full_predictions']))
        assert set(gold) == set(data['groups'])
        for context in cfg['contexts']:
            for i in range(8):
                tasks.append(dict(id=program['name']+'__'+context+'__'+str(i), program=program['name'],
                    context=context, query_index=i, unit=str(data['groups'][i]), sample_id=str(data['sample_ids'][i]),
                    source=program['source'], artifacts=program['contexts'][context], input=f'queries/{i}.npz',
                    expected_full=gold[str(data['groups'][i])]))
    assert len(tasks) == cfg['planned_prediction_processes'] == 416
    save(root/'TASKS456.json', tasks)
    save(root/'ENGINE_CONFIG.json', dict(input_root=str(root/'inputs'), common_tools=str(root/'code'),
        fit_seconds=0, predict_seconds=cfg['predict_seconds']))
    save(root/'BOUND_RESOURCES456.json', bindings)
    frozen = {r['target']: r['sha256'] for r in bindings}
    for name in ['PROTOCOL456.json','TASKS456.json','ENGINE_CONFIG.json']:
        frozen[name] = sha(root/name)
    for directory in ['code','queries']:
        frozen.update({str(p.relative_to(root)): sha(p) for p in (root/directory).rglob('*') if p.is_file()})
    save(root/'LAUNCH_FREEZE456.json', frozen)
    save(root/'PREPARED456.json', dict(utc=utc(), passed=True, programs=13, contexts=4, queries=8,
        prediction_processes=416, materialized_bytes=sum(r['bytes'] for r in bindings),
        truth_restored=False, new_fits=0, resource_files=len(bindings)))
    print(json.dumps(read(root/'PREPARED456.json')))


def run(root):
    cfg = read(root/'PROTOCOL456.json')
    assert read(root/'PREPARED456.json')['passed']
    assert not (root/'HANDLE456.json').exists(), 'Do not restart a previous invocation'
    for relative, digest in read(root/'LAUNCH_FREEZE456.json').items():
        assert sha(root/relative) == digest
    tasks = read(root/'TASKS456.json')
    start = time.monotonic()
    save(root/'HANDLE456.json', dict(pid=os.getpid(), utc=utc(), state='running'))
    completed = []
    def one(task):
        output = root/'predictions'/task['id']
        try:
            runtime = Runtime(root)
            runtime.deadline = start + cfg['total_prediction_seconds']
            process, audit = runtime.command(root/task['source'], root/task['artifacts'], root/task['input'], output, 'predict')
            assert audit['completed_estimator_fits'] == 0 and audit['optimizer_steps'] == 0
            value = read(output/'predictions.json')
            assert set(value) == {'rul'} and len(value['rul']) == 1
            p = value['rul'][0]
            assert isinstance(p, (float,int)) and np.isfinite(p) and p >= 0
            if task['context'] == 'full':
                assert np.isclose(p, task['expected_full'], rtol=1e-7, atol=1e-5), 'Original full prediction mismatch'
            result = dict(**task, status='complete', prediction=float(p), seconds=process['seconds'],
                          prediction_sha256=sha(output/'predictions.json'))
        except Exception as error:
            result = dict(**task, status='error', error_type=type(error).__name__, error=str(error))
        save(output/'JOB_RESULT456.json', result)
        return result
    def batch(work, phase):
        with ThreadPoolExecutor(max_workers=cfg['prediction_workers']) as pool:
            for future in as_completed([pool.submit(one, t) for t in work]):
                completed.append(future.result())
                save(root/'PROGRESS456.json', dict(utc=utc(), phase=phase, total=416, finished=len(completed),
                    errors=sum(r['status'] != 'complete' for r in completed), elapsed=time.monotonic()-start))
    anchors = [t for t in tasks if t['context'] == 'full' and t['query_index'] == 0]
    batch(anchors, 'full_prediction_qualification')
    anchors_passed = all(r['status'] == 'complete' for r in completed)
    save(root/'QUALIFICATION456.json', dict(utc=utc(), passed=anchors_passed, rows=list(completed)))
    if anchors_passed:
        done = {r['id'] for r in completed}
        batch([t for t in tasks if t['id'] not in done], 'same_query_context_replay')
    passed = len(completed) == 416 and all(r['status'] == 'complete' for r in completed)
    save(root/'PREDICTIONS_SEALED456.json', dict(utc=utc(), passed=passed, rows=completed,
        attempted_jobs=len(completed), process_records=len(list((root/'predictions').glob('*/PROCESS.json'))),
        seconds=time.monotonic()-start, outcomes_restored=False, new_LLM_calls=0, new_Agent_stages=0))
    save(root/'HANDLE456.json', dict(pid=os.getpid(), utc=utc(), state='terminal', passed=passed))
    save(root/'TERMINAL456.json', dict(utc=utc(), passed=passed, attempted_jobs=len(completed),
        expected_jobs=416, errors=sum(r['status'] != 'complete' for r in completed)))
    print(json.dumps(read(root/'TERMINAL456.json')))
    assert passed, 'Incomplete context replay; retain all failed/untested tasks'


def evaluate(root):
    cfg = read(root/'PROTOCOL456.json')
    sealed = read(root/'PREDICTIONS_SEALED456.json')
    assert sealed['passed'] and read(root/'HANDLE456.json')['state'] == 'terminal'
    archive = Path('/data/yyf-zyq/PHM_20261010/20261025T104200Z_hust367')
    assert sha(archive/'SOURCE_MANIFEST.json') == cfg['archives'][str(archive)]['manifest_sha256']
    member = 'private_inputs/discovery_truth.npz'
    digest = next(r['sha256'] for r in read(archive/'SOURCE_MANIFEST.json')['files'] if r['path'] == member)
    assert sha(archive/member) == digest
    target = root/'score/truth.npz';target.parent.mkdir()
    shutil.copyfile(archive/member, target)
    save(root/'TRUTH_BINDING456.json', dict(archive=str(archive), member=member, sha256=digest,
        restored_utc=utc(), prediction_seal_sha256=sha(root/'PREDICTIONS_SEALED456.json')))
    truth_data = load(target)
    truth = dict(zip(truth_data['sample_ids'].astype(str), truth_data['y']))
    table = {(r['program'], r['context'], r['query_index']): r for r in sealed['rows']}
    values, metrics = {}, {}
    for program in cfg['programs']:
        name = program['name']
        matrix = np.array([[table[name, context, i]['prediction'] for i in range(8)] for context in cfg['contexts']])
        targets = np.array([truth[table[name, 'full', i]['sample_id']] for i in range(8)])
        errors = abs(matrix-targets[None,:])
        fold_mean_errors = errors[:3].mean(axis=0)
        values[name] = dict(predictions=matrix.tolist(), errors=errors.tolist(), truth=targets.tolist(),
                            fold_mean_errors=fold_mean_errors.tolist())
        full_drift = matrix[3]-matrix[:3].mean(axis=0)
        metrics[name] = dict(MAE_by_context=dict(zip(cfg['contexts'], errors.mean(axis=1).tolist())),
            mean_fold_loss=float(fold_mean_errors.mean()), full_prediction_outside_fold_range=int(((matrix[3]<matrix[:3].min(axis=0)-1e-7)|(matrix[3]>matrix[:3].max(axis=0)+1e-7)).sum()),
            mean_absolute_full_vs_fold_prediction_drift=float(abs(full_drift).mean()),
            mean_signed_full_vs_fold_prediction_drift=float(full_drift.mean()))
    comparisons = []
    p00 = np.asarray(values['p00']['errors'])
    for program in cfg['programs']:
        if program['initial']:
            continue
        error = np.asarray(values[program['name']]['errors'])
        differences = error-p00
        fold_margin = differences[:3].mean(axis=0)
        full_margin = differences[3]
        comparisons.append(dict(program=program['name'], origin_id=program['origin_id'],
            margins_by_context=dict(zip(cfg['contexts'], differences.mean(axis=1).tolist())),
            mean_fold_margin=float(fold_margin.mean()), full_margin=float(full_margin.mean()),
            fold_good_full_bad_units=int(((fold_margin<-1e-8)&(full_margin>1e-8)).sum()),
            fold_bad_full_good_units=int(((fold_margin>1e-8)&(full_margin<-1e-8)).sum()),
            fold_and_full_bad_units=int(((fold_margin>1e-8)&(full_margin>1e-8)).sum()),
            fold_and_full_good_units=int(((fold_margin<-1e-8)&(full_margin<-1e-8)).sum()),
            fold_mean_error_differences=fold_margin.tolist(), full_error_differences=full_margin.tolist()))
    result = dict(iteration='A456', utc=utc(), metrics=metrics, values=values, native_comparisons=comparisons,
        context_effect_scope='Same8openedqueries,sourceunchanged,originalfold/fullfittedartifactcontexts;notseparatecausalidentificationofsamplecountversusidentityoroptimization.',
        new_PHM_fits=0, new_cold_predictions=len(sealed['rows']), full_reproduction_checks=104,
        new_fold_context_predictions=312, new_LLM_calls=0, new_Agent_stages=0, independent_confirmation=0,
        prediction_seconds=sealed['seconds'], goal_achieved=False, new_method_superiority=False)
    save(root/'RESULT456.json', result)
    print(json.dumps(dict(metrics=metrics, native_comparisons=comparisons, seconds=sealed['seconds'])))


def audit(root):
    cfg = read(root/'PROTOCOL456.json'); result = read(root/'RESULT456.json')
    sealed = read(root/'PREDICTIONS_SEALED456.json'); checks = 0
    for name,digest in read(root/'LAUNCH_FREEZE456.json').items():
        assert sha(root/name) == digest; checks += 1
    truths = load(root/'score/truth.npz'); original_inputs = load(root/'inputs/observations.npz')
    assert np.array_equal(truths['sample_ids'], original_inputs['sample_ids']); checks += 1
    raw = defaultdict(dict); fits = 0; steps = 0
    for row in sealed['rows']:
        directory = root/'predictions'/row['id']
        process = read(directory/'PROCESS.json'); training = read(directory/'training_audit.json')
        assert process['returncode'] == 0 and process['mode'] == 'predict'; checks += 1
        fits += training['completed_estimator_fits']; steps += training['optimizer_steps']
        assert not training['estimator_fit_events'] and training['optimizer_steps'] == 0; checks += 1
        assert '--unshare-net' in process['command'] and '/input.npz' in process['command']; checks += 1
        assert sha(directory/'predictions.json') == row['prediction_sha256']; checks += 1
        inp = load(root/row['input']); i = row['query_index']
        assert set(inp) == set(original_inputs) and 'y' not in inp and 'cap' not in inp; checks += 1
        for k in inp:
            assert np.array_equal(inp[k], original_inputs[k][i:i+1]); checks += 1
        pred = read(directory/'predictions.json')['rul'][0]
        assert pred == row['prediction']; checks += 1
        raw[row['program']][row['context'],i] = pred
    assert fits == steps == 0 and len(sealed['rows']) == 416; checks += 1
    for program in cfg['programs']:
        name=program['name']; data=result['values'][name]
        for c,context in enumerate(cfg['contexts']):
            preds=[raw[name][context,i] for i in range(8)]
            errors=[abs(p-float(y)) for p,y in zip(preds,truths['y'])]
            assert preds==data['predictions'][c];checks+=1
            assert np.allclose(errors,data['errors'][c]);checks+=1
            assert abs(math.fsum(errors)/8-result['metrics'][name]['MAE_by_context'][context])<1e-8;checks+=1
    for row in result['native_comparisons']:
        for context in cfg['contexts']:
            delta=math.fsum(abs(raw[row['program']][context,i]-float(truths['y'][i]))-abs(raw['p00'][context,i]-float(truths['y'][i])) for i in range(8))/8
            assert abs(delta-row['margins_by_context'][context])<1e-8;checks+=1
        assert abs(math.fsum(row['margins_by_context'][c] for c in cfg['contexts'][:3])/3-row['mean_fold_margin'])<1e-8;checks+=1
    save(root/'AUDIT456.json',dict(utc=utc(),passed=True,checks=checks,actual_cold_processes=416,
        actual_estimator_fit_calls=fits,actual_optimizer_steps=steps,independent_confirmation=0))
    print(json.dumps(read(root/'AUDIT456.json')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=['prepare','run','evaluate','audit'])
    args=parser.parse_args();globals()[args.stage](args.run.resolve())
