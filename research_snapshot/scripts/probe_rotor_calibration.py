"""Bounded A305 program replay on previously opened A325 records, not confirmation.

Reuses source-qualified observations, original feature code, registered candidate
driver, and immutable fitted checkpoints. No LLM calls or new supervised fitting.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import numpy as np

from controlled_feedback import save, sha, utc
from prepare_fault_extensions import features
from run_full_benchmark_campaign import clean_env


def read(path):
    return json.loads(Path(path).read_text())


def config(root):
    freeze = read(root / 'FREEZE417.json')
    for path, digest in freeze.items():
        assert sha(Path(path)) == digest, path
    return read(root / 'PROTOCOL417.json')


def prepare(root, cfg):
    start = time.monotonic()
    batches = []
    source = Path(cfg['data_archive'])
    assert sha(Path(__file__).with_name('prepare_fault_extensions.py')) == cfg['feature_sha256']
    for acquisition in cfg['acquisitions']:
        for load in cfg['loads']:
            tag = f'a{acquisition}_l{int(load*10)}'
            records = [r for r in cfg['source_records']
                       if r['acquisition'] == acquisition and r['load_nm'] == load]
            assert len(records) == 5 and {r['broken_bars'] for r in records} == set(range(5))
            values, gold, lineage = [], [], []
            for record in records:
                source_path = source / record['export_path']
                assert sha(source_path) == record['export_sha256']
                with np.load(source_path, allow_pickle=False) as data:
                    signal = data['signals']
                    assert int(data['broken_bars']) == record['broken_bars']
                    assert int(data['acquisition']) == acquisition
                    assert float(data['load_nm']) == load and int(data['sampling_hz']) == 7600
                    assert signal.ndim == 2 and signal.shape[1] == 5 and np.isfinite(signal).all()
                    group = secrets.token_hex(12)
                    for offset in range(0, len(signal)-2048+1, 2048):
                        wave = signal[offset:offset+2048]
                        identity = secrets.token_hex(16)
                        values.append(dict(features=features(wave), waveforms=wave.T.copy(),
                                           sample_ids=identity, sampling_hz=7600, operating_load_Nm=load))
                        gold.append(dict(sample_id=identity, group=group, label=record['broken_bars']))
                        lineage.append(dict(sample_id=identity, source=record['export_path'],
                                            source_sha256=record['export_sha256'], start=offset))
            order = np.random.default_rng(cfg['seed']+acquisition+int(load*10)).permutation(len(values))
            payload = {k: np.asarray([values[i][k] for i in order]) for k in values[0]}
            payload['dataset'] = np.asarray('ROTORBROKENBAR')
            path = root / 'observations' / tag / 'datasets/ROTORBROKENBAR/test.npz'
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, **payload)
            with np.load(path, allow_pickle=False) as saved:
                assert set(saved.files) == set(payload)
                assert all(np.array_equal(saved[k], v) for k, v in payload.items())
                assert 'labels' not in saved and 'groups' not in saved
            save(root/'private'/tag/'gold.json', gold)
            save(root/'private'/tag/'lineage.json', lineage)
            batches.append(dict(tag=tag, acquisition=acquisition, load=load, windows=len(gold),
                                records=5, input_sha256=sha(path), gold_sha256=sha(root/'private'/tag/'gold.json')))
    save(root/'PREPARED417.json', dict(utc=utc(), seconds=time.monotonic()-start,
        records=30, windows=sum(x['windows'] for x in batches), batches=batches,
        prior_exposure='A325 opened acquisitions7-10. This is a development replay, not unseen data.'))


def run_one(root, cfg, batch, cohort, repeat):
    started = time.monotonic()
    dest = root / 'replays' / batch['tag'] / cohort / str(repeat)
    if started >= cfg['_deadline']:
        status = dict(batch=batch['tag'], cohort=cohort, repeat=repeat,
                      status='not_run_budget', exit_code=None, seconds=0)
        save(dest/'STATUS.json', status)
        return status
    out = dest / 'output'
    (out/'tmp').mkdir(parents=True)
    (out/'mpl').mkdir()
    archive = Path(cfg['program_archive'])
    command = read(archive/f'registered_replays/test/{cohort}/COMMAND.json')
    old = cfg['historical_live_prefix']
    command = [str(archive)+s[len(old):] if s.startswith(old+'/') else s for s in command]
    for i, value in enumerate(command):
        if value not in ['--bind', '--ro-bind']:
            continue
        destination = command[i+2]
        if destination == '/data':
            command[i+1] = str(root/'observations'/batch['tag'])
        elif destination == '/app':
            command[i] = '--ro-bind'
        elif destination == '/output':
            command[i+1] = str(out)
    mount_at = command.index('--remount-ro')
    command[mount_at:mount_at] = ['--ro-bind', str(root/'code'), '/driver']
    command[command.index('/app/_registered_components.py')] = '/driver/guard417.py'
    save(dest/'COMMAND.json', command)
    env = clean_env(root)
    env.update(PYTHONPATH='/opt/phm_python/extra', PYTHONDONTWRITEBYTECODE='1',
               PATH='/opt/python312/bin:/usr/bin:/bin', TMPDIR='/output/tmp', TMP='/output/tmp',
               TEMP='/output/tmp', MPLCONFIGDIR='/output/mpl', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    status = dict(batch=batch['tag'], cohort=cohort, repeat=repeat, start_utc=utc())
    try:
        with (dest/'stdout').open('wb') as stdout, (dest/'stderr').open('wb') as stderr:
            proc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stdout, stderr=stderr,
                                  timeout=min(cfg['seconds_per_process'],cfg['_deadline']-time.monotonic()))
        status.update(exit_code=proc.returncode, status='complete' if proc.returncode == 0 else 'error')
    except subprocess.TimeoutExpired:
        status.update(exit_code=None, status='timeout')
    status['seconds'] = time.monotonic()-started
    if status['status'] == 'complete':
        assert read(out/'FIT_GUARD417.json')['observed_fit_calls'] == 0
        assert read(out/'VARIANTS.json')['variants'] == cfg['variants'][cohort]
    save(dest/'STATUS.json', status)
    return status


def execute(root, cfg):
    start = time.monotonic()
    cfg = dict(cfg, _deadline=start+cfg['execution_wall_seconds'])
    batches = read(root/'PREPARED417.json')['batches']
    tasks = [(b, c, r) for b in batches for c in cfg['cohorts'] for r in range(2)]
    rows = [run_one(root, cfg, *tasks[0])]
    assert rows[0]['status'] == 'complete', 'Stop at the first interface qualification failure'
    with ThreadPoolExecutor(max_workers=cfg['parallel']) as pool:
        futures = [pool.submit(run_one, root, cfg, *task) for task in tasks[1:]]
        for future in as_completed(futures):
            rows.append(future.result())
            save(root/'LIVE417.json', dict(utc=utc(), finished=len(rows), total=len(tasks),
                statuses=dict(Counter(r['status'] for r in rows)), seconds=time.monotonic()-start))
    elapsed = time.monotonic()-start
    save(root/'EXECUTION417.json', dict(rows=rows, seconds=elapsed, terminal=True,
        within_budget=elapsed <= cfg['execution_wall_seconds'], actual_processes=len(rows),
        new_LLM_calls=0, new_supervised_fits=0))
    assert elapsed <= cfg['execution_wall_seconds']
    assert all(row['status']=='complete' for row in rows)


def predictions(root, batch, cohort, variant, repeat=0):
    rows = read(root/'replays'/batch/cohort/str(repeat)/'output'/(variant+'.json'))['predictions']
    result = {r['sample_id']: int(r['predicted_label']) for r in rows}
    assert len(result) == len(rows) and set(result.values()) <= set(range(5))
    return result


def record_error(pred, gold):
    records = defaultdict(list)
    assert set(pred) == {r['sample_id'] for r in gold}
    for row in gold:
        records[row['group']].append(int(pred[row['sample_id']] != row['label']))
    return sum((Fraction(sum(v),len(v)) for v in records.values()), Fraction())/len(records)


def select(root, cfg):
    choices = []
    for calibration_acq, held_acq in [(7,8),(8,7)]:
        for family in 'abcd':
            for rule, loads in [('same_load_validation',[3.0]), ('broader_load_validation',[3.0,3.5])]:
                risk = {}
                for cohort in [family, family+'_review']:
                    errors = []
                    for load in loads:
                        tag = f'a{calibration_acq}_l{int(load*10)}'
                        gold = read(root/'private'/tag/'gold.json')
                        errors.append(record_error(predictions(root,tag,cohort,'complete'),gold))
                    risk[cohort] = sum(errors,Fraction())/len(errors)
                selected = family+'_review' if risk[family+'_review'] < risk[family] else family
                choices.append(dict(calibration_acquisition=calibration_acq, held_acquisition=held_acq,
                    family=family, rule=rule, selected=selected,
                    risks={k:str(v) for k,v in risk.items()}, held_load=4.0))
    save(root/'CHOICES417.json', dict(utc=utc(), choices=choices, tie='initial',
        current_selector_read_4Nm_labels=False, prior_research_exposure=True,
        protocol_sha256=sha(root/'PROTOCOL417.json')))


def evaluate(root, cfg):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    rows = []
    repeat_checks = 0
    batches = read(root/'PREPARED417.json')['batches']
    for batch in batches:
        tag = batch['tag']
        gold = read(root/'private'/tag/'gold.json')
        for cohort in cfg['cohorts']:
            for variant in cfg['variants'][cohort]:
                pred = predictions(root,tag,cohort,variant['name'])
                assert pred == predictions(root,tag,cohort,variant['name'],1)
                repeat_checks += 1
                truth = [r['label'] for r in gold]
                predicted = [pred[r['sample_id']] for r in gold]
                groups = defaultdict(list)
                for row in gold:
                    groups[row['group']].append((row['label'],pred[row['sample_id']]))
                detail = []
                for group, values in sorted(groups.items()):
                    counts = Counter(p for _,p in values)
                    majority = min(counts,key=lambda k:(-counts[k],k))
                    detail.append(dict(group=group, true_label=values[0][0], majority_label=majority,
                        errors=sum(y!=p for y,p in values), windows=len(values)))
                rows.append(dict(batch=tag, acquisition=batch['acquisition'], load=batch['load'],
                    cohort=cohort, **variant, record_mean_error=str(record_error(pred,gold)),
                    accuracy=float(accuracy_score(truth,predicted)),
                    macro_f1=float(f1_score(truth,predicted,labels=range(5),average='macro',zero_division=0)),
                    confusion=confusion_matrix(truth,predicted,labels=range(5)).tolist(),
                    record_majority_accuracy=sum(x['true_label']==x['majority_label'] for x in detail)/len(detail),
                    records=detail))
    scores = {(r['batch'],r['cohort'],r['name']):Fraction(r['record_mean_error']) for r in rows}
    paired = []
    for choice in read(root/'CHOICES417.json')['choices']:
        tag = f"a{choice['held_acquisition']}_l40"
        f = choice['family']
        initial, reviewed = (scores[tag,c,'complete'] for c in [f,f+'_review'])
        selected = scores[tag,choice['selected'],'complete']
        oracle = min(initial,reviewed)
        paired.append(dict(**choice, initial_error=float(initial), ordinary_review_error=float(reviewed),
            selected_error=float(selected), selection_loss=float(selected-oracle),
            harmful_update=float(max(selected-initial,Fraction())),
            improvement=float(max(initial-selected,Fraction())),
            available_improvement=float(initial-oracle)))
    save(root/'RESULT417.json', dict(utc=utc(), rows=rows, pair_choices=paired,
        repeated_predictions_equal=repeat_checks, prediction_artifacts=repeat_checks*2,
        new_independent_sources=0, new_native_Agent_sessions=0, qualification_only=True,
        formal_confirmation_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare','run','select','score'], required=True)
    args = parser.parse_args()
    cfg = config(args.run)
    {'prepare':prepare, 'run':execute, 'select':select, 'score':evaluate}[args.mode](args.run,cfg)


if __name__ == '__main__':
    main()
