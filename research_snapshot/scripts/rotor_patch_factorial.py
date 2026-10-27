"""Frozen feature-set x tree-recipe contrast for a real recorded Agent revision.

This is development mechanism qualification, not a new Agent experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from snu_order_control import read, save, sha, utc, packet
from rotor_original_features import spatial_features, spectral_features


def source_file(cfg, name):
    path = Path(cfg['source']) / cfg['files'][name]['path']
    assert sha(path) == cfg['files'][name]['sha256'], name
    return path


def feature_matrix(data):
    raw = data['features'].astype(float)
    spectral = np.concatenate([spectral_features(data['waveforms'][i:i+256])
                               for i in range(0, len(raw), 256)])
    full = np.concatenate([raw, spatial_features(raw), spectral], axis=1)
    assert full.shape[1] == 1070 and spectral.shape[1] == 770
    assert np.isfinite(full).all()
    return {'all': full, 'spectral': spectral}


def metrics(y, pred, groups):
    record_errors = []
    for group in sorted(set(groups)):
        mask = groups == group
        assert len(set(y[mask])) == 1
        record_errors.append(float(np.mean(pred[mask] != y[mask])))
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=range(5), average='macro')),
                record_mean_error=float(np.mean(record_errors)), record_errors=record_errors)


def prepare(root, cfg):
    started = time.monotonic()
    sizes = {}
    identities = {}
    for split in ['train', 'val']:
        data = packet(source_file(cfg, split))
        matrices = feature_matrix(data)
        np.savez(root / 'legal' / (split + '.npz'), **matrices,
                 y=data['labels'], groups=data['groups'], sample_ids=data['sample_ids'],
                 loads=data['operating_load_Nm'])
        sizes[split] = dict(rows=len(data['labels']), groups=len(set(data['groups'])))
        identities[split] = set(data['groups'])
    assert not identities['train'] & identities['val']
    save(root / 'PREPARED424.json', dict(utc=utc(), seconds=time.monotonic()-started,
        sizes=sizes, legal_hashes={s: sha(root/'legal'/(s+'.npz')) for s in sizes},
        final_inputs_accessed=False))


def worker(root, cfg, job):
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '/external/' in path or '/hidden/' in path or path.endswith('/test.npz'):
                raise PermissionError('No external observations or labels during fit/selection')
    sys.addaudithook(guard)
    train, val = packet(root/'legal/train.npz'), packet(root/'legal/val.npz')
    info = cfg['jobs'][job]
    X, V = train[info['features']], val[info['features']]
    probabilities = np.zeros((len(X), 5))
    reports = []
    for fold, load in enumerate([*cfg['loads'], None]):
        fit = np.ones(len(X), dtype=bool) if load is None else train['loads'] != load
        held = ~fit
        prefix = root/'models'/f'{job}_{fold}'
        save(str(prefix)+'_STARTED.json', dict(utc=utc(), job=job, fold=fold,
             train_rows=int(fit.sum()), held_load=load, completed=False))
        start = time.monotonic()
        estimator = ExtraTreesClassifier(**cfg['tree_recipes'][info['recipe']])
        estimator.fit(X[fit], train['y'][fit])
        assert np.array_equal(estimator.classes_, np.arange(5))
        output = estimator.predict_proba(V if load is None else X[held])
        if load is None:
            np.save(root/'predictions'/(job+'_val.npy'), output)
        else:
            probabilities[held] = output
            assert not set(train['groups'][fit]) & set(train['groups'][held])
        path = Path(str(prefix)+'.joblib')
        joblib.dump(estimator, path, compress=0)
        report = dict(utc=utc(), job=job, fold=fold, held_load=load,
            train_rows=int(fit.sum()), features=X.shape[1], classifier_fits=1,
            actual_trees=len(estimator.estimators_),
            actual_nodes=sum(e.tree_.node_count for e in estimator.estimators_),
            model_sha256=sha(path), seconds=time.monotonic()-start)
        save(str(prefix)+'_REPORT.json', report)
        reports.append(report)
        print(f'{job} fold={fold} trees={report["actual_trees"]} seconds={report["seconds"]:.2f}', flush=True)
    np.save(root/'predictions'/(job+'_oof.npy'), probabilities)
    val_prob = np.load(root/'predictions'/(job+'_val.npy'), allow_pickle=False)
    save(root/'reports'/(job+'.json'), dict(utc=utc(), folds=reports,
         oof=metrics(train['y'], probabilities.argmax(1), train['groups']),
         development=metrics(val['y'], val_prob.argmax(1), val['groups']),
         final_accessed=False))


def seal(root, cfg):
    train, val = packet(root/'legal/train.npz'), packet(root/'legal/val.npz')
    bank = packet(source_file(cfg, 'old_probability_bank'))
    assert np.array_equal(bank['y'], train['y']) and np.array_equal(bank['vy'], val['y'])
    assert np.array_equal(bank['loads'], train['loads'])
    names = bank['names'].tolist()
    reproduction = {}
    for job, old in [('all_old', 'et_all_mf005'), ('spectral_new', 'et_spectral_leaf2')]:
        for split, field in [('oof', 'oof'), ('val', 'dev')]:
            original = bank[field][:, names.index(old), :]
            actual = np.load(root/'predictions'/f'{job}_{split}.npy')
            delta = float(np.max(abs(original-actual)))
            assert delta < 1e-12 and np.array_equal(original.argmax(1), actual.argmax(1))
            reproduction[job+'_'+split] = delta
    reports = {j: read(root/'reports'/(j+'.json')) for j in cfg['jobs']}
    blend = {}
    for split, field, truth in [('oof', 'oof', train), ('val', 'dev', val)]:
        value = .9*np.load(root/'predictions'/('spectral_new_'+split+'.npy'))
        value += .1*bank[field][:, names.index('logreg_raw'), :]
        np.save(root/'predictions'/('review_blend_'+split+'.npy'), value)
        blend['oof' if split == 'oof' else 'development'] = metrics(truth['y'], value.argmax(1), truth['groups'])
    reports['review_blend'] = blend
    order = [*cfg['jobs'], 'review_blend']
    by_cv = min(order, key=lambda j: (-reports[j]['oof']['macro_f1'], order.index(j)))
    by_dev = min(order, key=lambda j: (-reports[j]['development']['macro_f1'], order.index(j)))
    save(root/'CHOICES424.json', dict(utc=utc(), ordinary_CV=by_cv, ordinary_dev=by_dev,
        partial_patch='all_new', all_legal=reports, historical_probability_checks=reproduction,
        model_hashes={p.name: sha(p) for p in (root/'models').glob('*.joblib')},
        final_predictions_started=False))


def evaluate(root, cfg):
    choice = read(root/'CHOICES424.json')
    # All choices and training have ended before this read.
    data = packet(source_file(cfg, 'test'))
    assert 'labels' not in data and 'groups' not in data
    X = feature_matrix(data)
    np.savez(root/'external/features.npz', **X, sample_ids=data['sample_ids'])
    predictions = {}
    for job, info in cfg['jobs'].items():
        path = root/'models'/(job+'_5.joblib')
        assert sha(path) == choice['model_hashes'][path.name]
        model = joblib.load(path)
        predictions[job] = model.predict_proba(X[info['features']])
        np.save(root/'external'/(job+'_probability.npy'), predictions[job])
    logistic = joblib.load(source_file(cfg, 'logistic_model'))
    logistic_prob = logistic.predict_proba(data['features'].astype(float))
    predictions['review_blend'] = .9*predictions['spectral_new']+.1*logistic_prob
    np.save(root/'external/review_blend_probability.npy', predictions['review_blend'])
    save(root/'EXTERNAL_PREDICTIONS_SEALED424.json', dict(utc=utc(),
        hashes={j:sha(root/'external'/(j+'_probability.npy')) for j in predictions}))
    gold = {e['sample_id']:e for e in read(source_file(cfg, 'gold'))}
    y = np.array([gold[str(s)]['label'] for s in data['sample_ids']])
    groups = np.array([gold[str(s)]['group'] for s in data['sample_ids']])
    np.savez(root/'external/truth.npz', y=y, groups=groups, sample_ids=data['sample_ids'], loads=data['operating_load_Nm'])
    results = {j:metrics(y, prob.argmax(1), groups) for j,prob in predictions.items()}
    reproduction = {}
    for job, source in [('all_old','old_prediction'), ('spectral_new','spectral_prediction'), ('review_blend','review_prediction')]:
        old = {e['sample_id']:e['predicted_label'] for e in read(source_file(cfg, source))['predictions']}
        assert set(old) == set(data['sample_ids'].astype(str))
        expect = np.array([old[str(s)] for s in data['sample_ids']])
        assert np.array_equal(expect, predictions[job].argmax(1))
        reproduction[job] = len(expect)
    loss = {j:np.array(v['record_errors']) for j,v in results.items()}
    base, oracle = loss['all_old'], np.min(np.stack(list(loss.values())), axis=0)
    opportunity = base-oracle
    for j, errors in loss.items():
        valid = opportunity > 1e-12
        results[j].update(selection_loss=float((errors-oracle).mean()),
            harmful_update=float(np.maximum(errors-base,0).mean()),
            gain_capture=float(np.mean(np.maximum(base[valid]-errors[valid],0)/opportunity[valid])) if valid.any() else None,
            gain_available_records=int(valid.sum()))
    patch, before, review = results['all_new'], results['all_old'], results['review_blend']
    gate = (patch['record_mean_error'] < before['record_mean_error']-1e-12
        and patch['harmful_update'] <= review['harmful_update']+1e-12
        and patch['gain_capture'] is not None and patch['gain_capture'] >= review['gain_capture']-1e-12)
    feature_effect = {p:results['spectral_'+p]['record_mean_error']-results['all_'+p]['record_mean_error'] for p in ['old','new']}
    parameter_effect = {f:results[f+'_new']['record_mean_error']-results[f+'_old']['record_mean_error'] for f in ['all','spectral']}
    save(root/'RESULT424.json', dict(utc=utc(), final=results,
        feature_deletion_effect_at_each_recipe=feature_effect,
        parameter_change_effect_at_each_features=parameter_effect,
        interaction=feature_effect['new']-feature_effect['old'],
        historical_final_exact_matches=reproduction, primary_qualification_gate=gate,
        ordinary_CV=choice['ordinary_CV'], ordinary_dev=choice['ordinary_dev'],
        new_classifier_fits=24, existing_logistic_reused=True,
        new_Agent_stages=0,new_LLM_calls=0,new_confirmation_scenarios=0,
        source_clusters=1, final_previously_opened=True, formal_confirmation_opened=False))
    print({j:round(v['accuracy'],6) for j,v in results.items()}, flush=True)


def run(root, cfg):
    started = time.monotonic()
    def invoke(phase, job=None):
        label = phase + ('_'+job if job else '')
        command = [sys.executable, '-B', str(Path(__file__)), phase, '--run', str(root)]
        if job: command += ['--job', job]
        begin = time.monotonic()
        save(root/'logs'/(label+'_STARTED.json'), dict(utc=utc(), command=command))
        try:
            with (root/'logs'/(label+'.stdout')).open('wb') as out, (root/'logs'/(label+'.stderr')).open('wb') as err:
                p = subprocess.run(command, stdout=out, stderr=err, timeout=cfg['budget'][phase+'_seconds'])
            status = dict(exit_code=p.returncode, status='complete' if p.returncode==0 else 'error')
        except subprocess.TimeoutExpired:
            status = dict(exit_code=None,status='timeout')
        status.update(utc=utc(),seconds=time.monotonic()-begin)
        save(root/'logs'/(label+'_TERMINAL.json'),status)
        print(label,status,flush=True)
        assert status['exit_code']==0, label
    try:
        invoke('prepare')
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(invoke,'worker',j) for j in cfg['jobs']]
            for future in as_completed(futures): future.result()
        invoke('seal');invoke('evaluate')
        status=dict(exit_code=0,scientific_terminal=True,audit_pending=True)
    except Exception as exc:
        status=dict(exit_code=1,error=repr(exc),scientific_terminal=True)
    save(root/'TERMINAL424.json',dict(utc=utc(),seconds=time.monotonic()-started,**status))
    if status['exit_code']:raise SystemExit(status['exit_code'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','worker','seal','evaluate','run']);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job');args=parser.parse_args()
    root=args.run.resolve();cfg=read(root/'PROTOCOL424.json');assert sha(Path(__file__))==cfg['code_sha256']
    for name, expected in cfg['support_code_sha256'].items():assert sha(Path(__file__).with_name(name))==expected
    if args.phase=='worker':worker(root,cfg,args.job)
    else:globals()[args.phase](root,cfg)
