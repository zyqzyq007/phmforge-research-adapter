"""Frozen-weight component controls for a completed HUST RUL program.

This is a posthoc mechanism diagnostic, not a new Agent or trained candidate.
Only observation-only single queries and existing fitted artifacts are mounted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save, sha, utc
from hust_program_runtime import OBSERVABLE
from program_revision_runtime import Runtime, load


WORKER = r'''
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from candidate_01.program import featurize,predict_models

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['predict']);p.add_argument('--input',type=Path)
p.add_argument('--model-dir',type=Path);p.add_argument('--output',type=Path)
a=p.parse_args()
assert not Path('/data').exists() and not Path('/srv').exists()
assert not Path('/home/jupyter-yyf-zyq/.bashrc').exists()
with np.load(a.input,allow_pickle=False) as z:
    assert set(z.files)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
    assert len(z['age'])==1
    x=np.asarray(z['x']);nn=np.asarray(z['nn'])
    age=np.asarray(z['age'],dtype=np.float64);rates=np.asarray(z['rates'],dtype=np.float64)
b=joblib.load(a.model_dir/'stacked_guard.joblib')
c=predict_models(b['models'],featurize(x,nn,age,rates))
assert c.shape==(1,7)
s=np.asarray(b['stack'].predict(c),dtype=np.float64).reshape(-1)
med=np.median(c,axis=1);cons=.8*med+.2*c.mean(axis=1)
if b['choice']=='stack':life=.75*s+.25*cons
elif b['choice']=='median':life=.40*s+.60*med
else:life=.55*s+.45*cons
h=np.maximum(np.asarray(b['direct_hgb'].predict(x),dtype=np.float64).reshape(-1),0)
e=np.maximum(np.asarray(b['direct_et'].predict(x),dtype=np.float64).reshape(-1),0)
def mix(lifetime):
    return np.clip(.90*np.maximum(lifetime.reshape(-1)-age.reshape(-1),0)+.08*h+.02*e,0,2800)
variants={'original':mix(life),'consensus':mix(cons),'median':mix(med),
          'stack_only':mix(s),'without_teachers':np.clip(np.maximum(life-age,0),0,2800)}
names=['ridge','pls','svr','extra_trees','random_forest','hgb_squared','hgb_absolute']
for i,name in enumerate(names):variants['base_'+name]=mix(c[:,i])
assert all(np.isfinite(v).all() and len(v)==1 for v in variants.values())
payload=dict(variants={k:v.tolist() for k,v in variants.items()},base_lifetimes=c.tolist(),
    stack=s.tolist(),median=med.tolist(),consensus=cons.tolist(),lifetime=life.tolist(),
    hgb=h.tolist(),extra_trees=e.tolist(),age=age.tolist(),choice=b['choice'],
    coefficients=np.asarray(b['stack'].coef_).tolist(),intercept=float(b['stack'].intercept_),
    internal_scores=b['internal_scores'],train_rows=b['train_rows'],train_cells=b['train_cells'],
    query_targets_visible=False,source_data_visible=False)
a.output.with_name('COMPONENTS.json').write_text(json.dumps(payload,indent=2)+'\n')
a.output.write_text(json.dumps({'rul':variants['original'].tolist()})+'\n')
'''


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    cfg = read(root/'PROTOCOL446.json')
    source = Path(cfg['source_archive'])
    assert sha(source/'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    index = {x['path']: x for x in read(source/'SOURCE_MANIFEST.json')['files']}
    needed = ['JOBS366.json', 'private_inputs/training_queries_private.npz',
              'public_data/development_observations.npz', 'private_inputs/discovery_truth.npz']
    bindings = {p: index[p]['sha256'] for p in needed}
    for path, digest in bindings.items():
        assert sha(source/path) == digest
    for path, digest in read(root/'REUSED_BINDINGS446.json').items():
        assert sha(root/path) == digest
    save(root/'SOURCE_BINDINGS446.json', bindings)
    (root/'code').mkdir()
    for name in ['decompose_hust_saved_program.py', 'program_revision_runtime.py',
                 'hust_program_runtime.py', 'program_training_entry.py', 'controlled_feedback.py']:
        shutil.copyfile(Path(__file__).with_name(name), root/'code'/name)
    shutil.copytree(root/'reused/source', root/'worker')
    (root/'worker/program.py').write_text(WORKER)
    save(root/'ENGINE_CONFIG.json', dict(input_root=str(source/'private_inputs'),
        common_tools=str(root/'code'), fit_seconds=0, predict_seconds=cfg['predict_seconds']))
    jobs = read(source/'JOBS366.json')
    queries = load(source/'private_inputs/training_queries_private.npz')
    development = load(source/'public_data/development_observations.npz')
    assert len(queries['age']) == 47 and len(development['age']) == 8
    assert set(development) == set(OBSERVABLE)
    (root/'queries').mkdir()
    tasks = []
    for context in ['fold_0', 'fold_1', 'fold_2', 'full']:
        data = development if context == 'full' else queries
        mask = np.ones(8, dtype=bool) if context == 'full' else np.isin(data['groups'], jobs[context]['heldout_cells'])
        if context != 'full':
            assert not set(jobs[context]['fitting_cells']) & set(data['groups'][mask])
        for i in np.flatnonzero(mask):
            ident = context+'_'+str(i)
            obs = {k: data[k][i:i+1].copy() for k in OBSERVABLE}
            path = root/'queries'/(ident+'.npz')
            np.savez_compressed(path, **obs)
            tasks.append(dict(id=ident, context=context, sample_id=str(data['sample_ids'][i]),
                              group=str(data['groups'][i]), input=str(path.relative_to(root))))
    assert len(tasks) == 55 and len({t['sample_id'] for t in tasks}) == 55
    save(root/'TASKS446.json', tasks)
    retained = ['PROTOCOL446.json', 'REUSED_BINDINGS446.json', 'SOURCE_BINDINGS446.json',
                'ENGINE_CONFIG.json', 'TASKS446.json']
    retained += [str(p.relative_to(root)) for folder in ['code', 'worker', 'queries']
                 for p in (root/folder).rglob('*') if p.is_file()]
    save(root/'LAUNCH_FREEZE446.json', {p: sha(root/p) for p in retained})


def run(root):
    cfg = read(root/'PROTOCOL446.json')
    tasks = read(root/'TASKS446.json')
    runtime = Runtime(root)
    for path, digest in read(root/'LAUNCH_FREEZE446.json').items():
        assert sha(root/path) == digest
    started = time.monotonic()
    def one(task):
        context = 'full_fit' if task['context'] == 'full' else task['context']+'/fit'
        artifacts = root/'reused'/context/'artifacts'
        output = root/'predictions'/task['id']
        process, audit = runtime.command(root/'worker', artifacts, root/task['input'], output, 'predict')
        row = read(output/'COMPONENTS.json')
        assert list(row['variants']) == cfg['variants']
        assert not audit['estimator_fit_events'] and audit['optimizer_steps'] == 0
        return dict(**task, seconds=process['seconds'], components_sha256=sha(output/'COMPONENTS.json'))
    with ThreadPoolExecutor(max_workers=cfg['prediction_workers']) as pool:
        results = list(pool.map(one, tasks))
    save(root/'PREDICTIONS_SEALED446.json', dict(utc=utc(), rows=results,
        actual_seconds=time.monotonic()-started, cold_processes=len(results),
        new_fits=0, new_LLM_calls=0, new_Agent_stages=0, reserved22_opened=False))


def evaluate(root):
    cfg = read(root/'PROTOCOL446.json')
    sealed = read(root/'PREDICTIONS_SEALED446.json')
    by_context = {}
    for context in ['fold_0', 'fold_1', 'fold_2', 'full']:
        path = root/'reused'/('development' if context == 'full' else context+'/check')/'PREDICTIONS.npz'
        old = load(path)
        by_context[context] = dict(zip(old['sample_ids'].tolist(), old['prediction'].tolist()))
    rows = []
    for task in sealed['rows']:
        path = root/'predictions'/task['id']/'COMPONENTS.json'
        assert sha(path) == task['components_sha256']
        row = read(path)
        np.testing.assert_allclose(row['variants']['original'][0],
            by_context[task['context']][task['sample_id']], **cfg['original_tolerance'])
        rows.append(dict(**task, **row))
    # Connect scoring labels only after all cold predictions and exact replay checks.
    source = Path(cfg['source_archive'])
    truth = {}
    for name in ['training_queries_private', 'discovery_truth']:
        data = load(source/'private_inputs'/(name+'.npz'))
        truth.update(dict(zip(data['sample_ids'].tolist(), data['y'].tolist())))
    metrics = {}
    for variant in cfg['variants']:
        values = {}
        for split, take in [('CV', lambda r: r['context'] != 'full'), ('exploration', lambda r: r['context'] == 'full')]:
            subset = [r for r in rows if take(r)]
            errors = np.asarray([abs(r['variants'][variant][0]-truth[r['sample_id']]) for r in subset])
            original = np.asarray([abs(r['variants']['original'][0]-truth[r['sample_id']]) for r in subset])
            values[split] = dict(MAE=float(errors.mean()), per_cell_error=errors.tolist(),
                per_cell_ids=[r['sample_id'] for r in subset], delta_vs_original=float((errors-original).mean()),
                harm=float(np.maximum(errors-original, 0).mean()), gain=float(np.maximum(original-errors, 0).mean()),
                wins=int((errors < original-1e-8).sum()), ties=int((np.abs(errors-original) <= 1e-8).sum()),
                losses=int((errors > original+1e-8).sum()))
        metrics[variant] = values
    cv_choice = min(cfg['variants'], key=lambda v: (metrics[v]['CV']['MAE'], v != 'original'))
    details = {context: {k: next(r for r in rows if r['context'] == context)[k]
                        for k in ['choice', 'coefficients', 'intercept', 'internal_scores', 'train_rows', 'train_cells']}
               for context in by_context}
    result = dict(utc=utc(), original_reproduced=55, metrics=metrics, fitted_bundle_details=details,
        fixed_CV_choice=cv_choice, fixed_CV_exploration_MAE=metrics[cv_choice]['exploration']['MAE'],
        new_PHM_fits=0, new_LLM_calls=0, new_Agent_stages=0, independent_confirmation=0,
        scope='Posthoc source-specific frozen-weight ablations; 8 already-opened discovery cells.',
        new_method_superiority=False, reserved22_opened=False)
    save(root/'RESULT446.json', result)
    print(json.dumps(dict(seconds=sealed['actual_seconds'], CV_choice=cv_choice,
        metrics={v: {s: d['MAE'] for s,d in m.items()} for v,m in metrics.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--stage', choices=['prepare', 'run', 'evaluate'], required=True)
    args = parser.parse_args()
    globals()[args.stage](args.run.resolve())
