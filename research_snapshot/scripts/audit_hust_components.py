"""Independent scalar reconstruction of the frozen HUST component diagnostic."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from controlled_feedback import save, sha, utc


def audit(root):
    read = lambda p: json.loads(p.read_text())
    cfg = read(root/'PROTOCOL446.json')
    result = read(root/'RESULT446.json')
    sealed = read(root/'PREDICTIONS_SEALED446.json')
    checks = 0
    def check(condition, label):
        nonlocal checks
        assert condition, label
        checks += 1
    def close(actual, expected, label):
        check(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6), label)
    for name in ['LAUNCH_FREEZE446.json', 'REUSED_BINDINGS446.json']:
        for path, digest in read(root/name).items():
            check(sha(root/path) == digest, path)
    source = Path(cfg['source_archive'])
    for path, digest in read(root/'SOURCE_BINDINGS446.json').items():
        check(sha(source/path) == digest, path)
    gold = {}
    for name in ['training_queries_private', 'discovery_truth']:
        with np.load(source/'private_inputs'/(name+'.npz'), allow_pickle=False) as z:
            gold.update(zip(z['sample_ids'].tolist(), z['y'].tolist()))
    old = {}
    for context in ['fold_0', 'fold_1', 'fold_2', 'full']:
        path = root/'reused'/('development' if context == 'full' else context+'/check')/'PREDICTIONS.npz'
        with np.load(path, allow_pickle=False) as z:
            old[context] = dict(zip(z['sample_ids'].tolist(), z['prediction'].tolist()))
    reconstructed = {'CV': [], 'exploration': []}
    names = ['ridge', 'pls', 'svr', 'extra_trees', 'random_forest', 'hgb_squared', 'hgb_absolute']
    for task in sealed['rows']:
        directory = root/'predictions'/task['id']
        check(sha(directory/'COMPONENTS.json') == task['components_sha256'], 'sealed output')
        data = read(directory/'COMPONENTS.json')
        cost = read(directory/'training_audit.json')
        check(cost['status'] == 'completed' and not cost['estimator_fit_events']
              and cost['optimizer_steps'] == 0, 'no training')
        check(read(directory/'PROCESS.json')['returncode'] == 0, 'process success')
        with np.load(root/task['input'], allow_pickle=False) as z:
            check(set(z.files) == {'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}, 'observable fields')
            check(len(z['age']) == 1 and str(z['sample_ids'][0]) == task['sample_id'], 'single query')
        base = data['base_lifetimes'][0]
        stack = math.fsum(a*b for a,b in zip(data['coefficients'],base))+data['intercept']
        close(data['stack'][0], stack, 'saved ridge equation')
        med = median(base)
        cons = .8*med+.2*mean(base)
        if data['choice'] == 'stack': local = .75*stack+.25*cons
        elif data['choice'] == 'median': local = .40*stack+.60*med
        else: local = .55*stack+.45*cons
        age, h, e = data['age'][0], data['hgb'][0], data['extra_trees'][0]
        clip = lambda x: min(2800., max(0., x))
        blend = lambda x: clip(.90*max(0., x-age)+.08*h+.02*e)
        variants = dict(original=blend(local), consensus=blend(cons), median=blend(med),
                        stack_only=blend(stack), without_teachers=clip(local-age))
        variants.update({'base_'+n: blend(v) for n,v in zip(names,base)})
        for name,prediction in variants.items():
            close(data['variants'][name][0], prediction, name)
        close(variants['original'], old[task['context']][task['sample_id']], 'original cold replay')
        split = 'exploration' if task['context'] == 'full' else 'CV'
        reconstructed[split].append((task['sample_id'], {n: abs(v-gold[task['sample_id']]) for n,v in variants.items()}))
    check(len(reconstructed['CV']) == 47 and len(reconstructed['exploration']) == 8, 'complete denominator')
    for variant in cfg['variants']:
        for split, rows in reconstructed.items():
            metric = result['metrics'][variant][split]
            check(metric['per_cell_ids'] == [i for i,_ in rows], 'ordering')
            errors = [r[variant] for _,r in rows]
            changes = [r[variant]-r['original'] for _,r in rows]
            close(metric['MAE'], mean(errors), 'MAE')
            close(metric['delta_vs_original'], mean(changes), 'delta')
            close(metric['harm'], mean(max(d,0) for d in changes), 'harm')
            close(metric['gain'], mean(max(-d,0) for d in changes), 'gain')
            check(metric['wins'] == sum(d < -1e-8 for d in changes), 'wins')
            check(metric['losses'] == sum(d > 1e-8 for d in changes), 'losses')
            for a,b in zip(metric['per_cell_error'],errors): close(a,b,'per cell error')
    chosen = min(cfg['variants'], key=lambda n: (mean(r[n] for _,r in reconstructed['CV']), n != 'original'))
    check(chosen == result['fixed_CV_choice'], 'frozen CV choice')
    report = dict(passed=True, utc=utc(), checks=checks, actual_cold_processes=55,
                  original_predictions_reproduced=55, new_fits=0, new_LLM_calls=0,
                  confirmation_opened=False, scope='Scalar formula, artifact, information and full-denominator audit; no new independent experiment.')
    save(root/'AUDIT446.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    audit(p.parse_args().run.resolve())
