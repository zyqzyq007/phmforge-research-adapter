"""Real, bounded classifier fitting with privately timed development feedback.

The broker never accepts code, paths, labels, arbitrary parameters or final data.
Each accepted recipe consumes one configured fitting opportunity. Both conditions
receive identical train-only CV; only development-feedback timing differs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import time
import uuid


SPACE = {
    'features': ['stats15', 'fft64', 'fft256', 'raw120', 'stats_fft64'],
    'models': {
        'rf': {'max_depth': [None, 8, 16], 'min_samples_leaf': [1, 3, 8]},
        'extra': {'max_depth': [None, 8, 16], 'min_samples_leaf': [1, 3, 8]},
        'svc': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.001, 0.01, 0.1]},
        'lda': {'shrinkage': ['auto', 0.1, 0.5]},
        'knn': {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance']},
        'logistic': {'C': [0.1, 1, 10, 100]},
    },
    'budget': 6,
    'numerical_seed': 282,
    'fitting_rule': 'Train-only 3-fold stratified CV and one full-training fit per accepted recipe. No development refit.',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.new')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def remaining_budget(run):
    """Expose only task-local numeric usage; credentials and paths stay private."""
    run = Path(run)
    path = run/'MODEL_GATEWAY_CONFIG.json'
    if not path.exists():
        return None
    limits = json.loads(path.read_text())
    usage_path = run/'model_gateway'/limits['routes'][0]['job']/'status.json'
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
    result = {key: dict(used=usage.get(key, 0), limit=limits[cap],
                       remaining=max(0, limits[cap]-usage.get(key, 0)))
              for key, cap in [('requests', 'max_requests'),
                               ('input_tokens', 'max_input_tokens'),
                               ('output_tokens', 'max_output_tokens')]}
    clock = run/'EPISODE_BUDGET.json'
    if clock.exists():
        result['wall_seconds_remaining'] = max(0, json.loads(clock.read_text())['deadline_epoch']-time.time())
    return result


def validate(recipe):
    assert isinstance(recipe, dict) and set(recipe) == {'features', 'model', 'params'}
    assert recipe['features'] in SPACE['features'] and recipe['model'] in SPACE['models']
    rules = SPACE['models'][recipe['model']]
    assert isinstance(recipe['params'], dict) and set(recipe['params']) == set(rules)
    assert all(recipe['params'][key] in values for key, values in rules.items())
    return recipe


def transform(waves, kind):
    import numpy as np
    from prepare_fault_extensions import features
    x = np.asarray(waves, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] >= 512 and x.shape[1] % 120 == 0 and np.isfinite(x).all()
    if kind in ['stats15', 'stats_fft64']:
        stats = np.asarray([features(row[:, None]) for row in x])
        if kind == 'stats15': return stats
    if kind == 'raw120': return x.reshape(len(x), 120, -1).mean(axis=-1)
    n = 256 if kind == 'fft256' else 64
    # Relative array-coordinate bands, not an asserted physical frequency axis.
    amplitude = np.abs(np.fft.rfft(x-x.mean(axis=1, keepdims=True), axis=1))
    bins = np.array_split(amplitude[:, 1:], n, axis=1)
    spectrum = np.log1p(np.stack([np.mean(b*b, axis=1) for b in bins], axis=1))
    return np.concatenate([stats, spectrum], axis=1) if kind == 'stats_fft64' else spectrum


def estimator(recipe, seed=282):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    p = recipe['params']
    name = recipe['model']
    if name == 'rf': model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=2, **p)
    elif name == 'extra': model = ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=2, **p)
    elif name == 'svc': model = SVC(**p)
    elif name == 'lda': model = LinearDiscriminantAnalysis(solver='lsqr', **p)
    elif name == 'knn': model = KNeighborsClassifier(**p)
    else: model = LogisticRegression(max_iter=1500, random_state=seed, **p)
    return make_pipeline(StandardScaler(), model)


def metrics(y, p, classes=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    classes = list(range(9)) if classes is None else list(classes)
    assert set(np.asarray(y).tolist()) <= set(classes) and set(np.asarray(p).tolist()) <= set(classes)
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, labels=classes, average='macro', zero_division=0)),
                correct=int(np.sum(np.asarray(y) == p)), total=len(y), confusion=confusion_matrix(y, p, labels=classes).tolist())


def fit(recipe_file, train_file, dev_file, output, settings_file=None):
    import joblib
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
    started = time.monotonic()
    path = Path(output)
    path.mkdir(exist_ok=True, parents=True)
    def fit_model(model, xx, yy, role):
        with (path/'fit_events.jsonl').open('a') as event:
            event.write(json.dumps(dict(utc=utc(), role=role, status='started', rows=len(yy)))+'\n')
        model.fit(xx, yy)
        with (path/'fit_events.jsonl').open('a') as event:
            event.write(json.dumps(dict(utc=utc(), role=role, status='completed', rows=len(yy)))+'\n')
    recipe = validate(json.loads(Path(recipe_file).read_text()))
    train = np.load(train_file, allow_pickle=False)
    dev = np.load(dev_file, allow_pickle=False)
    settings = (json.loads(Path(settings_file).read_text()) if settings_file else
                dict(seed=282, classes=list(range(9)), cv='stratified', expected_samples=3600))
    assert train['waveforms'].shape[1] == dev['waveforms'].shape[1] == settings['expected_samples']
    classes, seed = settings['classes'], settings['seed']
    x = transform(train['waveforms'], recipe['features'])
    dx = transform(dev['waveforms'], recipe['features'])
    y = train['labels']
    cv = np.zeros_like(y)
    save_probabilities=bool(settings.get('save_probabilities',False))
    probability_cv=np.zeros((len(y),len(classes)),dtype=np.float64) if save_probabilities else None
    fold_rows = []
    if settings['cv'] == 'leave_one_condition_out':
        group = train[settings['cv_group_key']]
        assert len(np.unique(group)) == 3
        folds = LeaveOneGroupOut().split(x, y, group)
    else:
        assert settings['cv'] == 'stratified'
        folds = StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y)
    for n, (fitting, checking) in enumerate(folds):
        assert set(y[fitting].tolist()) == set(classes)
        model = estimator(recipe, seed=seed)
        fit_model(model, x[fitting], y[fitting], 'cv_'+str(n))
        cv[checking] = model.predict(x[checking])
        if save_probabilities:
            assert list(model.classes_)==list(classes) and hasattr(model,'predict_proba')
            probability_cv[checking]=model.predict_proba(x[checking])
        row = dict(fold=n, fitting_ids=train['sample_ids'][fitting].tolist(), checking_ids=train['sample_ids'][checking].tolist())
        if settings['cv'] == 'leave_one_condition_out':
            row.update(fitting_conditions=np.unique(group[fitting]).tolist(), checking_conditions=np.unique(group[checking]).tolist())
            assert not set(row['fitting_conditions']) & set(row['checking_conditions'])
        fold_rows.append(row)
    assert len(fold_rows) == 3
    model = estimator(recipe, seed=seed)
    fit_model(model, x, y, 'full_training')
    joblib.dump(dict(recipe=recipe, model=model, fitting_settings=settings), path/'checkpoint.joblib')
    pred = model.predict(x)
    dpred = model.predict(dx)
    cold = joblib.load(path/'checkpoint.joblib')['model'].predict(dx)
    assert np.array_equal(dpred, cold)
    np.savez_compressed(path/'predictions.npz', train=pred, cv=cv, development=dpred,
                        train_ids=train['sample_ids'], development_ids=dev['sample_ids'])
    if save_probabilities:
        probability_train=model.predict_proba(x);probability_dev=model.predict_proba(dx)
        probability_cold=joblib.load(path/'checkpoint.joblib')['model'].predict_proba(dx)
        assert np.allclose(probability_dev,probability_cold,rtol=0,atol=1e-12)
        for values in [probability_train,probability_cv,probability_dev]:
            assert np.isfinite(values).all() and values.min()>=0 and np.allclose(values.sum(axis=1),1,atol=1e-8,rtol=0)
        np.savez_compressed(path/'probabilities.npz',train=probability_train,cv=probability_cv,development=probability_dev,
                            train_ids=train['sample_ids'],development_ids=dev['sample_ids'],classes=np.asarray(classes))
    report = dict(recipe=recipe, train=metrics(y, pred, classes), cv=metrics(y, cv, classes), development=metrics(dev['labels'], dpred, classes),
                  real_fits=4, cv_folds=fold_rows, fitting_rows=len(y), elapsed_seconds=time.monotonic()-started,
                  fitting_settings=settings, settings_sha256=sha(settings_file) if settings_file else None,
                  cold_reload_equal=True, checkpoint_sha256=sha(path/'checkpoint.joblib'),
                  train_input_sha256=sha(train_file), development_input_sha256=sha(dev_file), final_input_read=False,
                  fitting_scope='Training observations only; scalers refit inside every CV fold.')
    # Keep genuinely computed row-level evidence. The broker publishes each file
    # only when that validation split is permitted by the frozen timing policy.
    for part, data, predictions in [('train', train, pred), ('cv', train, cv), ('development', dev, dpred)]:
        rows = []
        for i, value in enumerate(predictions):
            row = dict(sample_id=str(data['sample_ids'][i]), actual=int(data['labels'][i]), predicted=int(value))
            for key in ['condition_id', 'groups']:
                if key in data.files: row[key] = str(data[key][i])
            rows.append(row)
        by_condition = {}
        if 'condition_id' in data.files:
            for condition in np.unique(data['condition_id']):
                mask = data['condition_id'] == condition
                by_condition[str(condition)] = metrics(data['labels'][mask], predictions[mask], classes)
        save(path/'evidence'/(part+'.json'), dict(split=part, classes=classes, rows=rows,
             metrics=report[part], by_condition=by_condition, cv_folds=fold_rows if part == 'cv' else None))
    save(path/'fit_report.json', report)
    print(json.dumps({'real_fits': 4, 'cold_reload_equal': True, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


class Broker:
    def __init__(self, run):
        self.run = Path(run)
        self.config = json.loads((self.run/'BROKER_CONFIG.json').read_text())
        self.budget = int(self.config.get('candidate_budget', SPACE['budget']))
        assert 1 <= self.budget <= 12
        self.state = dict(started_utc=utc(), condition=self.config['condition'], trials=[], final_choice=None)
        assert not (self.run/'broker/STATE.json').exists(), 'No implicit broker restart'
        save(self.run/'broker/STATE.json', self.state)

    def public(self, row):
        value = {k: row[k] for k in ['id', 'recipe', 'status']}
        if row['status'] == 'complete':
            report = json.loads((self.run/'broker'/row['id']/'fit_report.json').read_text())
            value.update(train=report['train'], train_cv=report['cv'], real_fits=report['real_fits'])
            released = self.config['condition'] == 'immediate' or len(self.state['trials']) == self.budget
            if released:
                value['development'] = report['development']
            else: value['development'] = 'withheld_until_all_recipes_committed'
            if self.config.get('publish_validation_evidence', False):
                value['validation_evidence'] = {}
                for part in ['train', 'cv'] + (['development'] if released else []):
                    source = self.run/'broker'/row['id']/'evidence'/(part+'.json')
                    target = self.run/'feedback/evidence'/row['id']/(part+'.json')
                    if not target.exists():
                        save(target, json.loads(source.read_text()))
                    assert sha(source) == sha(target)
                    value['validation_evidence'][part] = '/feedback/evidence/'+row['id']+'/'+part+'.json'
        return value

    def dispatch(self, request):
        action = request.get('action')
        if action == 'status':
            assert set(request) == {'action'}
        elif action == 'fit':
            assert set(request) == {'action', 'recipe'} and self.state['final_choice'] is None
            recipe = validate(request['recipe'])
            assert len(self.state['trials']) < self.budget, 'Configured fitting opportunities exhausted'
            assert recipe not in [x['recipe'] for x in self.state['trials']], 'Duplicate recipe; choose a different candidate'
            ident = 'candidate_'+str(len(self.state['trials'])+1)
            folder = self.run/'broker'/ident
            row = dict(id=ident, recipe=recipe, status='running', started_utc=utc())
            self.state['trials'].append(row)
            save(self.run/'broker/STATE.json', self.state)
            save(folder/'recipe.json', recipe)
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'fit', '--recipe', str(folder/'recipe.json'),
                       '--train', self.config['train'], '--dev', self.config['development'], '--output', str(folder)]
            if self.config.get('fitting_settings'):
                command += ['--settings', self.config['fitting_settings']]
            save(folder/'command.json', command)
            try:
                with (folder/'stdout').open('wb') as out, (folder/'stderr').open('wb') as err:
                    result = subprocess.run(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL, timeout=180)
                row['status'] = 'complete' if result.returncode == 0 else 'fit_error'
                row['returncode'] = result.returncode
            except subprocess.TimeoutExpired:
                row['status'] = 'fit_timeout'
            row['ended_utc'] = utc()
        elif action == 'select':
            assert set(request) == {'action', 'candidate_id', 'reason'}
            assert len(self.state['trials']) == self.budget, 'Complete all allowed candidate requests before selecting'
            assert self.state['final_choice'] is None, 'Selection is sealed; no resubmission'
            row = next(x for x in self.state['trials'] if x['id'] == request['candidate_id'])
            assert row['status'] == 'complete' and isinstance(request['reason'], str) and request['reason'].strip()
            self.state['final_choice'] = dict(candidate_id=row['id'], reason=request['reason'], sealed_utc=utc(),
                checkpoint_sha256=sha(self.run/'broker'/row['id']/'checkpoint.joblib'))
        else: raise ValueError('Allowed actions:status,fit,select')
        save(self.run/'broker/STATE.json', self.state)
        return dict(condition=self.config['condition'], trials=[self.public(x) for x in self.state['trials']],
                    remaining=self.budget-len(self.state['trials']), final_choice=self.state['final_choice'])


def serve(run,broker_class=Broker):
    broker = broker_class(run)
    # File requests are compatible with the CLI's command-network sandbox.
    # Request files are writable only in this subject's /app; responses are
    # published outside /app and mounted read-only. No network rule is relaxed.
    root=Path(run)
    queues=[root/p for p in broker.config.get('request_directories',[broker.config['request_directory']])]
    for requests in queues: requests.mkdir(parents=True,exist_ok=True)
    responses=root/'feedback/responses';responses.mkdir(exist_ok=True)
    save(root/'BROKER_READY.json',dict(pid=os.getpid(),transport='scoped_file_queue',utc=utc()))
    while True:
        for path in sorted(p for requests in queues for p in requests.glob('*.json')):
            ident=path.stem
            if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):continue
            output=responses/(ident+'.json')
            if output.exists():continue
            started=utc();raw=b''
            try:
                assert not path.is_symlink() and path.stat().st_size<=65536
                raw=path.read_bytes()
                request = json.loads(raw)
                response = dict(ok=True, result=broker.dispatch(request))
            except Exception as exc:
                response = dict(ok=False, error=type(exc).__name__+': '+str(exc))
            with (root/'broker/ACCESS.jsonl').open('a') as log:
                log.write(json.dumps(dict(request_id=ident,source_queue=str(path.parent.relative_to(root)),started_utc=started, ended_utc=utc(), request=raw.decode(errors='replace'), response=response))+'\n')
            save(output,response)
        time.sleep(.05)


def client(request):
    queue=Path('/app/.runtime/feedback_requests');queue.mkdir(parents=True,exist_ok=True)
    ident=uuid.uuid4().hex
    save(queue/(ident+'.json'),request)
    response=Path('/feedback/responses')/(ident+'.json')
    deadline=time.monotonic()+195
    while time.monotonic()<deadline:
        if response.exists():return json.loads(response.read_text())
        time.sleep(.05)
    raise TimeoutError('The scoped fitting service did not respond within 195 seconds')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fit', 'serve', 'client'])
    p.add_argument('--run', type=Path)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--train', type=Path)
    p.add_argument('--dev', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--settings', type=Path)
    p.add_argument('--request')
    a = p.parse_args()
    if a.action == 'fit': fit(a.recipe, a.train, a.dev, a.output, a.settings)
    elif a.action == 'serve': serve(a.run)
    else: print(json.dumps(client(json.loads(a.request)), indent=2))


if __name__ == '__main__': main()
