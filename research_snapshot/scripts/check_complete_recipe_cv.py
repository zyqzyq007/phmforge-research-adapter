"""One fixed-parameter, train-only check of two historical HUST recipes.

This is exploratory missing-evidence measurement, not a new method comparison.
Source data, preprocessing, parameters and all three held families are frozen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261022T023131Z_validation274')
PROBE = SOURCE / 'audit/early_closed_review/d_choice_probe'
WORKSPACE = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = '/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu'
DEPS = '/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps'
RECIPES = {'before': 'prior_family57', 'after': 'chosen_loadfamily42'}


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def prepare(root):
    root.mkdir(exist_ok=False)
    for d in ['code', 'app', 'workers', 'runtime/tmp', 'runtime/mpl']: (root/d).mkdir(parents=True, exist_ok=True)
    inputs = {
        'train.npz': SOURCE/'d/public_data/s274_hust_geometry_r2_initial/datasets/HUST/train.npz',
        'train_order_v2.npy': PROBE/'app/.runtime/train_order_v2.npy',
        'PROBE.json': PROBE/'output/PROBE.json',
    }
    for side, name in RECIPES.items():
        inputs[side+'.joblib'] = PROBE/'output'/f'{name}.joblib'
        inputs[side+'.json'] = PROBE/'output'/f'train_{name}.json'
    manifest = json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    assert sha(SOURCE/'SOURCE_MANIFEST.json') == 'ed5a5e4eef7ecff596f99c91d2d6c5de83c0fa9f351994f9f6f0e0118fdc357f'
    inventory = {r['path']: r for r in manifest['files']}
    bindings = []
    for path in list(inputs.values()) + [PROBE/'app'/s for s in ['hust_features.py', 'train_model.py', 'probe_choices.py']]:
        entry = inventory[str(path.relative_to(SOURCE))]
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        bindings.append(dict(source_path=str(path), **entry))
    for name in ['hust_features.py', 'train_model.py', 'probe_choices.py']: shutil.copy2(PROBE/'app'/name, root/'app'/name)
    shutil.copy2(__file__, root/'code'/Path(__file__).name)
    plan = dict(iteration='A390', frozen_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Measure missing complete-recipe CV; no candidate or parameter tuning',
        historical_outcomes_already_known=True, independent_confirmation=False,
        supersedes_old_no_refit_scope_only_for='One original-parameter missing-evidence check authorized by the ongoing method goal; no seeds, search, new candidates or final predictions',
        inputs={k:str(v) for k,v in inputs.items()}, bindings=bindings,
        recipes=RECIPES, held_families=['6204','6205','6206'],
        before=dict(normalization='family only; original float32 aggregation', training='all non-held families'),
        after=dict(normalization='family x load; original implementation', training='non-held intersection of 6205 and 6206'),
        invariant='StandardScaler + LDA(solver=lsqr, shrinkage=auto), original record grouping and original labels',
        test_population='All 57 training records / 11030 windows; each held exactly once',
        validation_boundary='Preprocessing groups are whole families or family x load; recomputed independently on each fitted and held batch and checked against the cached full-batch feature rows',
        primary='Pooled window classification error, paired on identical records',
        secondary=['Record error', 'per-family errors', 'fixed-seven-label macro-F1'],
        frozen_fixed_rule='Choose after only if pooled window CV error does not increase and existing development error strictly improves; otherwise before. No threshold search.',
        interpretation='If before wins, ordinary exact CV supplies a solution for this case; if after wins, training CV still does not identify this known unseen-family harm. Neither outcome establishes Agent-specific difficulty or novelty.',
        budget=dict(classifier_fits=6, scaler_fits=6, full_checkpoint_replays=2, fold_prediction_calls=6, LLM_calls=0, final_test_predictions=0, workers=2, seconds_per_worker=300, numeric_threads_per_worker=1, disk_estimate_MB=160, GPU_hours=0),
        no_hidden_data_mount=True, no_synthetic_data=True,
        code_sha256=sha(root/'code'/Path(__file__).name))
    save(root/'PROTOCOL390.json', plan)
    save(root/'FREEZE390.json', dict(protocol_sha256=sha(root/'PROTOCOL390.json'), code_sha256=plan['code_sha256']))
    save(root/'retention_policy.json', dict(excluded_rebuildable_directory='clients-cache', shared_dependencies=dict(path=str(Path(DEPS).parent), usage='6 original-parameter LDA fits; train-only cached measured features and immutable sources')))
    print(str(root), flush=True)


def worker(side):
    import joblib
    import numpy as np
    from sklearn.metrics import f1_score
    sys.path.insert(0, '/app')
    import hust_features as hf
    from probe_choices import earlier_features
    from train_model import fit_pipeline
    class Data(dict):
        @property
        def files(self): return list(self)
    with np.load('/inputs/train.npz', allow_pickle=False) as z:
        observation = Data({k:z[k] for k in ['native_record_mean','native_record_scale','shaft_frequency_hz','bearing_family','nominal_load_watts']})
        y = z['labels'].astype(int); ids = z['sample_ids'].astype(str)
    raw = np.load('/inputs/train_order_v2.npy', allow_pickle=False)
    assert len(y) == 11030 and raw.shape == (11030, 918)
    def features(obs, cached):
        hf.extract_window_features = lambda _: cached
        current, inv, families, loads = hf.build_record_features(obs)
        result = current if side == 'after' else earlier_features(cached, inv, families)
        return result, inv, families, loads
    x, inv, family, load = features(observation, raw)
    labels = np.array([np.unique(y[inv==i]).item() for i in range(len(x))], int)
    counts = np.bincount(inv)
    assert len(x) == 57 and set(family) == {'6204','6205','6206'}
    selected = np.isin(family, ['6205','6206']) if side == 'after' else np.ones(57, bool)
    original = joblib.load('/inputs/'+side+'.joblib')['pipeline']
    scaler = original.named_steps['standardscaler']
    mean_difference = float(np.max(np.abs(scaler.mean_ - x[selected].mean(0))))
    assert int(scaler.n_samples_seen_) == int(selected.sum()) and mean_difference < 1e-12
    replay = original.predict(x).astype(int)
    saved = json.loads(Path('/inputs/'+side+'.json').read_text())['predictions']
    assert {str(i):int(p) for i,p in zip(ids, replay[inv], strict=True)} == {r['sample_id']:r['predicted_label'] for r in saved}
    lineage=[]; rows=[]; started=time.monotonic()
    for held in ['6204','6205','6206']:
        fit = selected & (family != held); test = family == held
        differences=[]
        for mask in [fit,test]:
            window_mask = mask[inv]
            subset = Data({k:v[window_mask] for k,v in observation.items()})
            rebuilt, _, _, _ = features(subset, raw[window_mask])
            difference = float(np.max(np.abs(rebuilt-x[mask])))
            assert difference == 0.0
            differences.append(difference)
        model = fit_pipeline(x[fit], labels[fit])
        pred = model.predict(x[test]).astype(int)
        s = model.named_steps['standardscaler']; lda = model.named_steps['lineardiscriminantanalysis']
        direct = lda.classes_[np.argmax(((x[test]-s.mean_)/s.scale_) @ lda.coef_.T + lda.intercept_, axis=1)]
        assert np.array_equal(direct, pred)
        joblib.dump(model, '/output/'+held+'.joblib', compress=3)
        lineage.append(dict(held=held, fit_record_indices=np.flatnonzero(fit).tolist(), held_record_indices=np.flatnonzero(test).tolist(), fit_families=sorted(set(family[fit])), fitting_records=int(fit.sum()), fitting_windows=int(counts[fit].sum()), scaler_n_samples=int(s.n_samples_seen_), scaler_mean_max_abs=float(np.max(np.abs(s.mean_-x[fit].mean(0)))), normalization_rebuild_differences=differences, coefficient_predict_agreement=True))
        for i,p in zip(np.flatnonzero(test), pred, strict=True):
            rows.append(dict(record_index=int(i), signature='|'.join(str(observation[k][np.flatnonzero(inv==i)[0]]) for k in observation), family=str(family[i]), load=int(load[i]), windows=int(counts[i]), label=int(labels[i]), prediction=int(p)))
        save('/output/PARTIAL.json',dict(rows=rows,lineage=lineage))
        print(json.dumps(dict(side=side,held=held,fitting_records=int(fit.sum()),held_correct=int(np.sum(pred==labels[test])))),flush=True)
    save('/output/EVIDENCE.json',dict(side=side,rows=rows,lineage=lineage,checkpoint_replay_matches=True,original_scaler_mean_max_abs=mean_difference,classifier_fits=3,scaler_fits=3,pipeline_predictions=4,coefficient_checks=3,seconds=time.monotonic()-started,final_data_mounted=False,LLM_calls=0))


def run(root):
    assert not (root/'HANDLE390.json').exists()
    plan=json.loads((root/'PROTOCOL390.json').read_text());freeze=json.loads((root/'FREEZE390.json').read_text())
    assert sha(root/'PROTOCOL390.json')==freeze['protocol_sha256'] and sha(root/'code'/Path(__file__).name)==freeze['code_sha256']
    save(root/'HANDLE390.json',dict(pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat()))
    started=time.monotonic()
    def job(side):
        out=root/'workers'/side
        for d in ['runtime/tmp','runtime/mpl']: (out/d).mkdir(parents=True,exist_ok=False)
        cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']: cmd += ['--ro-bind',p,p]
        cmd += ['--ro-bind',PYTHON,'/opt/python','--ro-bind',DEPS,'/opt/deps','--ro-bind',str(root/'app'),'/app','--ro-bind',str(root/'code'),'/code','--dir','/inputs']
        for k,p in plan['inputs'].items():cmd+=['--ro-bind',p,'/inputs/'+k]
        cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/app','--remount-ro','/','/opt/python/bin/python3.12','-B','/code/'+Path(__file__).name,'worker','--side',side]
        save(out/'COMMAND.json',cmd)
        env=dict(os.environ,PYTHONPATH='/opt/deps',TMPDIR='/output/runtime/tmp',TMP='/output/runtime/tmp',TEMP='/output/runtime/tmp',MPLCONFIGDIR='/output/runtime/mpl',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'stdout').open('w') as stdout,(out/'stderr').open('w') as stderr:
            try:code=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=300).returncode
            except subprocess.TimeoutExpired:code=124
        save(out/'TERMINAL.json',dict(exit_code=code));return dict(side=side,exit_code=code)
    states=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,side) for side in RECIPES]):
            states.append(f.result());save(root/'PROGRESS390.json',states);print(json.dumps(states[-1]),flush=True)
    ok=all(s['exit_code']==0 for s in states)
    if ok:
        evidence={s:json.loads((root/'workers'/s/'EVIDENCE.json').read_text()) for s in RECIPES}
        save(root/'EVIDENCE390.json',evidence);save(root/'EVIDENCE_SEAL390.json',dict(sha256=sha(root/'EVIDENCE390.json'),sealed_utc=datetime.now(timezone.utc).isoformat()))
    save(root/'TERMINAL390.json',dict(exit_code=0 if ok else 1,states=states,seconds=time.monotonic()-started))
    if not ok:raise RuntimeError('Failure retained; no automatic science retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--run',type=Path);p.add_argument('--side',choices=['before','after']);args=p.parse_args()
    if args.action=='worker':worker(args.side)
    elif args.action=='prepare':prepare(args.run)
    else:run(args.run)
