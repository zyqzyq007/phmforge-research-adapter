"""Close A305 with registered cold controls, joint sealing and external scoring."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import audit, files, namespace, terminal
from run_full_benchmark_campaign import clean_env
from summarize_selection_expansion import metrics, contrast

COHORTS = ['a', 'b', 'c', 'd', 'a_review', 'b_review', 'c_review', 'd_review']


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    """Read every registered result, preserving negative and positive contrasts."""
    rows, controls, pairs, bindings = [], [], [], {}
    predictions, golds = {}, {}
    for cohort in COHORTS:
        run = root/cohort
        job, = terminal(run)
        score_path = run/'CODING_SCORE.json'
        score, = read(score_path)['rows']
        assert score['formal_submission'] and score['final']['valid'] and score['reference']['valid']
        public = {x['split']: x for x in read(run/'PUBLIC_SPLIT_SCORES.json')['rows']}
        gold = read(run/'hidden/ROTORBROKENBAR/labels.json')
        pred = read(run/'evaluation'/job['id']/'output/predictions.json')['predictions']
        predictions[cohort], golds[cohort] = pred, gold
        indexed = {x['sample_id']: x['predicted_label'] for x in pred}
        with np.load(job['evaluation_test'], allow_pickle=False) as z:
            loads = dict(zip(z['sample_ids'].astype(str), z['operating_load_Nm'].tolist()))
        m = metrics(pred, gold)
        cm = np.asarray(m['confusion_matrix']); tp = cm.diagonal()
        manual = float(np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(len(tp)),
                                 where=cm.sum(0)+cm.sum(1)>0).mean())
        assert abs(manual-score['final']['metrics']['macro_f1']) < 1e-12
        by_load = []
        for load in sorted(set(loads.values())):
            g = [x for x in gold if loads[x['sample_id']] == load]
            p = [dict(sample_id=x['sample_id'], predicted_label=indexed[x['sample_id']]) for x in g]
            by_load.append(dict(load_Nm=load, n=len(g), metrics=metrics(p,g)))
        majority = []
        for group in sorted({x['group'] for x in gold}):
            g = [x for x in gold if x['group'] == group]
            labels, counts = np.unique([indexed[x['sample_id']] for x in g], return_counts=True)
            majority.append(dict(group=group, actual=g[0]['label'],
                predicted=int(labels[counts.argmax()]), tied=int(sum(counts==counts.max()))>1))
        assert not any(x['tied'] for x in majority), 'Report ties explicitly instead of resolving silently'
        row = dict(cohort=cohort, block='block34' if cohort[0] in 'ab' else 'block56',
            arm='ordinary' if cohort[0] in 'ac' else 'full_program',
            stage='review' if cohort.endswith('_review') else 'initial',
            n=len(gold), records=len(majority), train_accuracy=public['train']['metrics']['accuracy'],
            development_accuracy=public['val']['metrics']['accuracy'],
            development_macro_f1=public['val']['metrics']['macro_f1'],
            final_accuracy=m['accuracy'], final_macro_f1=m['macro_f1'],
            correct=int(tp.sum()), reference_accuracy=score['reference']['metrics']['accuracy'],
            record_majority_correct=sum(x['actual']==x['predicted'] for x in majority),
            metrics=m, by_load=by_load, record_majority=majority,
            independent_metric_checks_agree=True)
        rows.append(row)
        for split in ['val','test']:
            path=root/'registered_replays'/split/cohort/'RESULT.json'
            result=read(path)
            assert result['exit_code']==0 and result['complete_matches_separate_cold_run']
            bindings[str(path.relative_to(root))]=sha256(path)
        for variant in read(root/'registered_replays/test'/cohort/'RESULT.json')['rows']:
            path=root/'registered_replays/test'/cohort/'output'/(variant['name']+'.json')
            assert sha256(path)==variant['prediction_sha256']
            delta=contrast(pred,read(path)['predictions'],gold)
            d=delta['delta']
            controls.append(dict(cohort=cohort,**variant,compared_to_submission=delta,
                material_threshold_met=((d['accuracy']>=.05-1e-12 and d['macro_f1']>=-1e-12)
                    or (d['macro_f1']>=.05-1e-12 and d['accuracy']>=-1e-12))))
        bindings[str(score_path.relative_to(root))]=sha256(score_path)
    for cohort in 'abcd':
        assert golds[cohort] == golds[cohort+'_review']
        pairs.append(dict(cohort=cohort, **contrast(predictions[cohort],predictions[cohort+'_review'],golds[cohort])))
    selectors=[]
    for cohort,block in [('a','block34'),('c','block56')]:
        case=next(x for x in read(root/'CAMPAIGN_CONFIG.json')['cases'] if x['cohort']==cohort)
        dev=read(root/'controls'/case['case']/'FROZEN_SELECTION.json')['selected']
        cv=read(root/'bank_selection'/block/'TRAINING_CV.json')['selected']
        selectors.append(dict(block=block,training_CV_selected=cv,development_selected=dev))
    ws=root/'a_review/workspaces/s305_block34_ordinary_review'
    old=np.load(ws/'.runtime/review_et_oof.npz',allow_pickle=False)
    new=np.load(ws/'.runtime/final_oof_probabilities.npz',allow_pickle=False)
    assert all(np.array_equal(old[k],new[k]) for k in ['y','groups','loads'])
    cv_rows=[]
    from sklearn.metrics import accuracy_score, f1_score
    for load in [None]+np.unique(old['loads']).tolist():
        mask=np.ones(len(old['y']),bool) if load is None else old['loads']==load
        cv_rows.append(dict(heldout_load=load, n=int(mask.sum()),
            initial_accuracy=float(accuracy_score(old['y'][mask],old['pred'][mask])),
            initial_macro_f1=float(f1_score(old['y'][mask],old['pred'][mask],average='macro')),
            review_accuracy=float(accuracy_score(new['y'][mask],new['prediction'][mask])),
            review_macro_f1=float(f1_score(new['y'][mask],new['prediction'][mask],average='macro'))))
    result=dict(utc=datetime.now(timezone.utc).isoformat(), task='pdm_fault_004',
        official_original_PHMForge_score=False, measured_source=True, independent_motor_claim=False,
        sessions=8, original_task_directions=1, physical_record_blocks=2,
        total_final_unique_records=40, total_final_unique_windows=2941,
        rows=rows,registered_controls=controls,review_changes=pairs,fixed_selectors=selectors,
        a_recorded_OOF_recalculation=cv_rows, OOF_fold_weights_independently_replayed=False,
        new_analysis_LLM_calls=0,new_analysis_supervised_fits=0,source_hashes=bindings,
        causal_repeated_feedback_overfitting_confirmed=False,
        hard_Agent_mechanism_not_inferred_automatically=True)
    save(root/'NUMERICAL_RESULTS305.json',result)
    project=Path(read(root/'CAMPAIGN_CONFIG.json')['project'])/'results/rotor305'
    save(project/'NUMERICAL_RESULTS305.json',result)
    columns=['cohort','block','arm','stage','n','records','train_accuracy','development_accuracy',
             'development_macro_f1','final_accuracy','final_macro_f1','correct','reference_accuracy','record_majority_correct']
    for target in [root/'STAGE_RESULTS305.csv',project/'STAGE_RESULTS305.csv']:
        with target.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=[{k:r[k] for k in columns} for r in rows],
                         review_changes=[{k:r[k] for k in ['cohort','delta','fixed','broken']}for r in pairs],
                         fixed_selectors=selectors, recorded_OOF=cv_rows)),flush=True)


def replay(root, cohort, split):
    run = root/cohort
    job, = terminal(run)
    ws = run/job['workspace']
    assert files(ws) == read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]['files']
    dest = root/'registered_replays'/split/cohort
    dest.mkdir(parents=True, exist_ok=False)
    app = dest/'app'
    shutil.copytree(ws, app, symlinks=True)
    shutil.copy2(root/'audit/rotor_registered_components.py', app/'_registered_components.py')
    out = dest/'output'
    out.mkdir()
    mount_path = (run/'public_split_evaluation'/job['id']/'val/mounts.json'
                  if split == 'val' else Path(job['evaluation_mounts']))
    mounts = read(mount_path)
    if cohort in ['a', 'c']:
        case = next(c for c in read(root/'CAMPAIGN_CONFIG.json')['cases'] if c['cohort'] == cohort)
        mounts['mounts'].append(dict(source=str(root/'controls'/case['case']), destination='/controls'))
    if cohort == 'a':
        mounts['mounts'].append(dict(source=str(root/'audit/a_matched_recipe'), destination='/matched'))
    save(dest/'mounts.json', mounts)
    command = namespace(run, {**job, 'evaluation_mounts': str(dest/'mounts.json')}, app, out, [
        '/opt/python312/bin/python3.12', '-B', '/app/_registered_components.py',
        '--cohort', cohort, '--input', '/data/datasets/ROTORBROKENBAR/'+split+'.npz',
        '--output-dir', '/output'])
    save(dest/'COMMAND.json', command)
    env = clean_env(run)
    env.update(PYTHONPATH='/opt/phm_python/extra', PYTHONDONTWRITEBYTECODE='1',
        PATH='/opt/python312/bin:/usr/bin:/bin', TMPDIR='/app/.runtime/tmp',
        TMP='/app/.runtime/tmp', TEMP='/app/.runtime/tmp', MPLCONFIGDIR='/app/.runtime/mpl',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (dest/'stdout').open('wb') as stdout, (dest/'stderr').open('wb') as stderr:
        code = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                              stdout=stdout, stderr=stderr, timeout=180).returncode
    result = dict(cohort=cohort, split=split, exit_code=code, new_supervised_fits=0)
    save(dest/'STATUS.json', result)
    assert code == 0, str(dest/'stderr')
    complete = read(out/'complete.json')['predictions']
    original = (run/'public_split_evaluation'/job['id']/'val/output/predictions.json'
                if split == 'val' else run/'evaluation'/job['id']/'output/predictions.json')
    assert {x['sample_id']: x['predicted_label'] for x in complete} == {
        x['sample_id']: x['predicted_label'] for x in read(original)['predictions']}
    if split == 'val':
        source = run/'public_data'/job['id']/'datasets/ROTORBROKENBAR/val.npz'
        with np.load(source, allow_pickle=False) as z:
            gold = [dict(sample_id=str(i), label=y.item(), group=str(g))
                    for i, y, g in zip(z['sample_ids'], z['labels'], z['groups'])]
    else:
        gold = read(run/'hidden/ROTORBROKENBAR/labels.json')
    result.update(complete_matches_separate_cold_run=True, rows=[])
    for variant in read(out/'VARIANTS.json')['variants']:
        path = out/(variant['name']+'.json')
        result['rows'].append(dict(**variant, prediction_sha256=sha256(path),
                                  metrics=metrics(read(path)['predictions'], gold)))
    assert files(ws) == read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]['files']
    save(dest/'RESULT.json', result)
    print(json.dumps(dict(cohort=cohort, split=split, variants=len(result['rows']), exit_code=code)), flush=True)
    return result


def main(root, mode):
    registration = root/'audit/REGISTERED_CONTROLS305.json'
    seal_path = root/'audit/ALL_CHOICES_SEALED305.json'
    if mode == 'public':
        assert not list(root.glob('*/CODING_SCORE.json'))
        assert not registration.exists()
        for cohort in COHORTS:
            audit(root/cohort, reuse_existing=True)
            assert (root/cohort/'PUBLIC_SPLIT_SCORES.json').exists()
        driver = root/'audit/rotor_registered_components.py'
        assert driver.exists()
        binding = {str(p.relative_to(root)): sha256(p) for cohort in COHORTS
                   for p in (root/cohort/'CODING_CHOICE_AUDIT.json', root/cohort/'OBSERVABLE_REVIEW_COMPLETE.json')}
        for p in (root/'controls').rglob('*'):
            if p.is_file(): binding[str(p.relative_to(root))] = sha256(p)
        for p in [driver, root/'audit/finalize_rotor_replicates.py',
                  root/'audit/a_matched_recipe/matched_et_spectral_leaf2.joblib',
                  root/'audit/A_DISCARDED_RECIPE_REGISTRATION305.json',
                  root/'audit/a_matched_recipe/MATCHED_RECIPE_RECEIPT305.json']:
            binding[str(p.relative_to(root))] = sha256(p)
        save(registration, dict(utc=datetime.now(timezone.utc).isoformat(), cohorts=COHORTS,
            frozen_files=binding, final_scores_opened=False, new_supervised_fits=0,
            scope='all8programs;12fixedbank;actual saved components;one previously registered matched recipe',
            roles_separate=True, no_postscore_search=True))
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda c: replay(root, c, 'val'), COHORTS))
        save(root/'audit/REGISTERED_PUBLIC_CHECKS305.json', dict(rows=rows, final_data_accessed=False))
        for cohort in COHORTS: audit(root/cohort, reuse_existing=True)
        for path, digest in binding.items(): assert sha256(root/path) == digest
        save(seal_path, dict(utc=datetime.now(timezone.utc).isoformat(), all8_terminal=True,
            registration_sha256=sha256(registration), public_checks_sha256=sha256(root/'audit/REGISTERED_PUBLIC_CHECKS305.json'),
            final_scores_opened=False, all_original_submissions_unchanged=True,
            native_agent_sessions_to_rerun=0, independent_devices_claimed=False))
        print('ALL8_CHOICES_AND_CONTROLS_SEALED', flush=True)
    else:
        assert read(seal_path)['all8_terminal']
        assert sha256(registration) == read(seal_path)['registration_sha256']
        for path, digest in read(registration)['frozen_files'].items(): assert sha256(root/path) == digest
        def score_one(cohort):
            run = root/cohort
            assert not (run/'CODING_SCORE.json').exists()
            with (root/'logs'/('final_'+cohort+'.stdout')).open('wb') as out, (root/'logs'/('final_'+cohort+'.stderr')).open('wb') as err:
                code = subprocess.run([sys.executable, '-B', str(root/'code/evaluate_forge_coding.py'),
                    '--run', str(run), '--mode', 'score'], stdout=out, stderr=err, timeout=600).returncode
            print(json.dumps(dict(cohort=cohort, external_score_exit=code)), flush=True)
            assert code == 0
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(score_one, COHORTS))
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda c: replay(root, c, 'test'), COHORTS))
        save(root/'REGISTERED_EXTERNAL_CONTROLS305.json', dict(rows=rows, new_supervised_fits=0,
            new_LLM_calls=0, all_registered_controls_reported=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=['public', 'score', 'summarize'], required=True)
    args = p.parse_args()
    if args.mode == 'summarize': summarize(args.run.resolve())
    else: main(args.run.resolve(), args.mode)
