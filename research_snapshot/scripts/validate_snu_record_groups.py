"""Ordinary record-held-out control on immutable real SNU candidate recipes.

No Agent method; no target data are used to fit or select these recipes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import joblib
import numpy as np

from snu_order_control import metric, packet, read, save, sha, utc


def factory(kind):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == 'et':
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                    random_state=322, n_jobs=2)
    estimator = (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                 if kind == 'lda' else MLPClassifier(hidden_layer_sizes=(64, 32),
                                                     max_iter=300, random_state=322))
    return make_pipeline(StandardScaler(), estimator)


def worker(root, cfg, job):
    # A mechanical access check in addition to the explicit train-only paths.
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '/external/' in path or '_final.' in path or '/RUN_CONTENTS.' in path:
                raise PermissionError('Target outcome access forbidden in CV worker')
    sys.addaudithook(guard)
    rep, kind = job.rsplit('__', 1)
    truth = packet(root/'legal/fit_labels.npz')
    X = packet(root/'legal'/(rep+'_fit.npz'))['X']
    predictions = np.full(len(X), -1, dtype=int)
    reports = []
    for fold, held in enumerate(cfg['groups']):
        train = truth['source'] != held
        test = ~train
        prefix = root/'models'/f'{job}__fold{fold}'
        assert train.sum() == 216 and test.sum() == 36
        assert not set(truth['source'][train]) & set(truth['source'][test])
        save(str(prefix)+'_STARTED.json', dict(utc=utc(), train_rows=int(train.sum()),
             held_out=held, job=job, fold=fold))
        started = time.monotonic()
        model = factory(kind)
        model.fit(X[train], truth['y'][train])
        estimator = model if kind == 'et' else model[-1]
        path = Path(str(prefix)+'.joblib')
        joblib.dump(model, path)
        answer = model.predict(X[test])
        predictions[test] = answer
        np.save(str(prefix)+'_prediction.npy', answer)
        evidence = dict(classifier_fits=1, scaler_fits=int(kind != 'et'),
                        classes=estimator.classes_.tolist(), train_rows=int(train.sum()),
                        test_rows=int(test.sum()), features=int(X.shape[1]),
                        healthy_class_supported=0 in estimator.classes_)
        if kind == 'mlp':
            evidence.update(epochs=int(estimator.n_iter_), samples_seen=int(estimator.t_),
                            loss_curve=list(map(float, estimator.loss_curve_)))
        elif kind == 'et':
            evidence.update(actual_trees=len(estimator.estimators_),
                            nodes=[int(tree.tree_.node_count) for tree in estimator.estimators_])
        else:
            evidence.update(coefficient_norm=float(np.linalg.norm(estimator.coef_)))
        report = dict(utc=utc(), held_out=held, fold=fold, job=job,
                      training=evidence, model_sha256=sha(path),
                      prediction_sha256=sha(str(prefix)+'_prediction.npy'),
                      seconds=time.monotonic()-started)
        save(str(prefix)+'_REPORT.json', report)
        reports.append(report)
    assert np.all(predictions >= 0)
    np.save(root/'predictions'/(job+'_oof.npy'), predictions)
    save(root/'reports'/(job+'.json'), dict(utc=utc(), job=job, folds=reports,
         oof=metric(truth, predictions), source_class_support=cfg['class_group_counts'],
         class_zero_without_independent_training_group=True, final_data_read=False))


def seal(root, cfg):
    reports = {job: read(root/'reports'/(job+'.json')) for job in cfg['jobs']}
    selected = min(cfg['jobs'], key=lambda job: (-reports[job]['oof']['macro_f1'],
                                               job != cfg['incumbent'], job))
    save(root/'CHOICES422.json', dict(utc=utc(), primary=selected,
         criterion='all-class macro-F1 on all seven record-held-out folds',
         oof={job: report['oof'] for job, report in reports.items()},
         report_hashes={job: sha(root/'reports'/(job+'.json')) for job in cfg['jobs']},
         final_cache_restored=bool(list((root/'external').glob('*')))))
    assert not read(root/'CHOICES422.json')['final_cache_restored']


def archived_bytes(cfg, relative):
    entry = next(e for e in read(Path(cfg['source421'])/'SOURCE_MANIFEST.json')['files']
                 if e['path'] == relative)
    with tarfile.open(Path(cfg['source421'])/'RUN_CONTENTS.tar.gz', 'r:gz') as archive:
        handle = archive.extractfile(relative)
        assert handle is not None
        value = handle.read()
    assert len(value) == entry['bytes'] and hashlib.sha256(value).hexdigest() == entry['sha256']
    return value


def evaluate(root, cfg):
    choice = read(root/'CHOICES422.json')
    external = root/'external'
    target = external/'final_labels.npz'
    target.write_bytes(archived_bytes(cfg, 'private/final_labels.npz'))
    truth = packet(target)
    # Read each old source once, and bind every restored prediction to its manifest.
    source421 = Path(cfg['source421'])
    inventory = {e['path']: e for e in read(source421/'SOURCE_MANIFEST.json')['files']}
    requested = {'predictions/'+j+'_final.npy': j for j in cfg['jobs']
                 if not j.startswith('coarse__')}
    with tarfile.open(source421/'RUN_CONTENTS.tar.gz', 'r:gz') as archive:
        for member in archive:
            if member.name not in requested:
                continue
            value = archive.extractfile(member).read()
            assert hashlib.sha256(value).hexdigest() == inventory[member.name]['sha256']
            (external/(requested[member.name]+'_final.npy')).write_bytes(value)
    source322 = Path(cfg['source322'])
    old_manifest = {e['path']: e for e in read(source322/'SOURCE_MANIFEST.json')['files']}
    values = {}
    for job in cfg['jobs']:
        if job.startswith('coarse__'):
            rel = job.rsplit('__', 1)[1]+'_final_prediction.npy'
            value = (source322/rel).read_bytes()
            assert hashlib.sha256(value).hexdigest() == old_manifest[rel]['sha256']
            (external/(job+'_final.npy')).write_bytes(value)
        values[job] = metric(truth, np.load(external/(job+'_final.npy'), allow_pickle=False))
    selected, incumbent = values[choice['primary']], values[cfg['incumbent']]
    oracle = max(v['macro_f1'] for v in values.values())
    gate = (selected['macro_f1'] > incumbent['macro_f1'] + 1e-12 and
            selected['record_accuracy'] >= incumbent['record_accuracy'])
    result = dict(utc=utc(), selected=choice['primary'], selected_metrics=selected,
        incumbent=cfg['incumbent'], incumbent_metrics=incumbent,
        all_final=values, primary_qualification_passed=gate,
        selection_loss=oracle-selected['macro_f1'],
        harmful_update_loss=max(0., incumbent['macro_f1']-selected['macro_f1']),
        improvement_opportunity=oracle-incumbent['macro_f1'],
        improvement_capture=max(0., selected['macro_f1']-incumbent['macro_f1']) /
                            (oracle-incumbent['macro_f1']),
        source_clusters=1, independent_confirmation=0,
        new_classifier_fits=105, new_scaler_fits=70, new_neural_models=35,
        new_Agent_calls=0, new_LLM_calls=0, new_final_model_predictions=0,
        target_cache_only_read_after_seal=True, formal_confirmation_opened=False)
    save(root/'RESULT422.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_final']}), flush=True)


def audit(root, cfg):
    from audit_snu_order_control import forward, measures
    started = time.monotonic()
    result, choice = read(root/'RESULT422.json'), read(root/'CHOICES422.json')
    truth = packet(root/'legal/fit_labels.npz')
    checks = rows = fitted = scalers = 0
    neural = []
    for job in cfg['jobs']:
        assert sha(root/'reports'/(job+'.json')) == choice['report_hashes'][job]
        rep, kind = job.rsplit('__', 1)
        X = packet(root/'legal'/(rep+'_fit.npz'))['X']
        predictions = np.full(len(X), -1, dtype=int)
        for fold, held in enumerate(cfg['groups']):
            train, test = truth['source'] != held, truth['source'] == held
            prefix = root/'models'/f'{job}__fold{fold}'
            report = read(str(prefix)+'_REPORT.json')
            assert sha(str(prefix)+'.joblib') == report['model_sha256']
            assert report['held_out'] == held and report['training']['train_rows'] == int(train.sum())
            model = joblib.load(str(prefix)+'.joblib')
            estimator = model if kind == 'et' else model[-1]
            assert set(estimator.classes_) == set(truth['y'][train])
            if kind != 'et':
                np.testing.assert_allclose(model[0].mean_, X[train].mean(axis=0), rtol=1e-11, atol=1e-11)
                assert model[0].n_samples_seen_ == int(train.sum())
                scalers += 1
            if kind == 'mlp':
                assert estimator.n_iter_ > 1 and estimator.loss_curve_[-1] < estimator.loss_curve_[0]
                neural.append(dict(job=job, fold=fold, epochs=int(estimator.n_iter_),
                                   first_loss=estimator.loss_curve_[0], last_loss=estimator.loss_curve_[-1]))
            if kind == 'et':
                assert len(estimator.estimators_) == 300
                assert all(tree.tree_.node_count > 1 for tree in estimator.estimators_)
            # The historical audit covered three-class models. This explicitly
            # handles the binary healthy-held-out fold without model.predict.
            if kind != 'et' and len(estimator.classes_) == 2:
                z = (X[test]-model[0].mean_)/model[0].scale_
                if kind == 'lda':
                    z = z@estimator.coef_.T+estimator.intercept_
                else:
                    for layer, (weights, bias) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
                        z = z@weights+bias
                        if layer < len(estimator.coefs_)-1:
                            z = np.maximum(z, 0)
                rebuilt = estimator.classes_[(z.ravel() > 0).astype(int)]
            else:
                rebuilt = forward(model, X[test], kind)
            saved = np.load(str(prefix)+'_prediction.npy', allow_pickle=False)
            np.testing.assert_array_equal(rebuilt, saved)
            assert sha(str(prefix)+'_prediction.npy') == report['prediction_sha256']
            predictions[test] = rebuilt
            rows += len(rebuilt); fitted += 1; checks += 5
        np.testing.assert_array_equal(predictions, np.load(root/'predictions'/(job+'_oof.npy')))
        metrics = measures(truth['y'], predictions, truth['source'])
        for key, val in metrics.items():
            np.testing.assert_allclose(val, choice['oof'][job][key], rtol=0, atol=1e-12)
            checks += 1
        target = packet(root/'external/final_labels.npz')
        pred = np.load(root/'external'/(job+'_final.npy'), allow_pickle=False)
        for key, val in measures(target['y'], pred, target['source']).items():
            np.testing.assert_allclose(val, result['all_final'][job][key], rtol=0, atol=1e-12)
            checks += 1
    winner = min(cfg['jobs'], key=lambda j: (-choice['oof'][j]['macro_f1'], j != cfg['incumbent'], j))
    assert winner == choice['primary'] == result['selected']
    assert fitted == 105 and scalers == 70 and len(neural) == 35
    a, b = result['all_final'][winner], result['all_final'][cfg['incumbent']]
    assert result['primary_qualification_passed'] == (a['macro_f1'] > b['macro_f1']+1e-12 and a['record_accuracy'] >= b['record_accuracy'])
    healthy = truth['y'] == 0
    assert all(np.all(np.load(root/'predictions'/(j+'_oof.npy'))[healthy] != 0) for j in cfg['jobs'])
    audit = dict(utc=utc(), passed=True, seconds=time.monotonic()-started,
        independent_parameter_rows=rows, numeric_and_manifest_checks=checks,
        actual_classifier_fits=fitted, scaler_fits=scalers, neural_models=neural,
        missing_healthy_class_folds=15, new_audit_model_fits=0,
        final_result_sha256=sha(root/'RESULT422.json'))
    save(root/'INDEPENDENT_AUDIT422.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'neural_models'}), flush=True)


def stage(root, cfg, phase, job=None):
    label = phase+('_'+job if job else '')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), phase, '--run', str(root)]
    if job: command += ['--job', job]
    started = time.monotonic()
    with (root/'logs'/(label+'.stdout')).open('w') as out, (root/'logs'/(label+'.stderr')).open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        save(root/'logs'/(label+'_HANDLE.json'), dict(utc=utc(), pid=child.pid, command=command))
        try:
            code = child.wait(timeout=cfg['timeouts'][phase])
        except subprocess.TimeoutExpired:
            import os, signal
            os.killpg(child.pid, signal.SIGKILL); child.wait(); code = 124
    result = dict(utc=utc(), phase=phase, job=job, exit_code=code, seconds=time.monotonic()-started)
    save(root/'logs'/(label+'_TERMINAL.json'), result)
    return result


def run(root, cfg):
    started = time.monotonic()
    save(root/'STARTED422.json', dict(utc=utc(), controller_pid=__import__('os').getpid()))
    with ThreadPoolExecutor(max_workers=cfg['parallel']) as pool:
        futures = [pool.submit(stage, root, cfg, 'worker', j) for j in cfg['jobs']]
        history = [future.result() for future in futures]
    save(root/'PROGRESS422.json', dict(utc=utc(), history=history))
    code = 0 if all(v['exit_code'] == 0 for v in history) else 1
    print(json.dumps(dict(phase='workers', failures=sum(v['exit_code'] != 0 for v in history), seconds=time.monotonic()-started)), flush=True)
    if code == 0:
        for phase in ['seal', 'evaluate', 'audit']:
            record = stage(root, cfg, phase); history.append(record)
            save(root/'PROGRESS422.json', dict(utc=utc(), history=history))
            print(json.dumps(record), flush=True)
            if record['exit_code']:
                code = record['exit_code']; break
    save(root/'TERMINAL422.json', dict(utc=utc(), exit_code=code, seconds=time.monotonic()-started))
    raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['worker', 'seal', 'evaluate', 'audit', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job')
    args = parser.parse_args(); root = args.run.resolve(); cfg = read(root/'PROTOCOL422.json')
    assert sha(Path(__file__)) == cfg['code_sha256']
    if args.phase == 'worker': worker(root, cfg, args.job)
    else: globals()[args.phase](root, cfg)
