"""Run the prespecified HUST evidence contrast, with a pre-Agent equality gate."""
import argparse
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from acquire_forge_pdmb_sources244 import save, sha256
from run_validation_transfer import campaign, prepare, execute_wave, state, utc


def evidence_gate(root):
    config=campaign(root)
    source=root/'common/restored/HUST/data'
    train=dict(np.load(source/'train.npz',allow_pickle=False))
    identities={str(i):n for n,i in enumerate(train['sample_ids'])}
    fold_checks=[]
    for arm in ['record','geometry']:
        cache=root/'bank_selection'/arm
        cv=json.loads((cache/'TRAINING_CV.json').read_text())
        field=cv['cv_group_field']
        for record in cv['rows']:
            if not record['valid']:
                raise AssertionError('Invalid predeclared CV candidate; inspect before Agent execution')
            p=np.empty_like(train['labels']);assigned=np.zeros(len(p),int)
            for fold in cv['folds']:
                fit=np.asarray([identities[i] for i in fold['fit_ids']])
                held=np.asarray([identities[i] for i in fold['heldout_ids']])
                assert not set(train[field][fit])&set(train[field][held])
                assert not set(train['groups'][fit])&set(train['groups'][held])
                path=cache/'cv_models'/(record['name']+'_fold'+str(fold['fold'])+'.joblib')
                model=joblib.load(path)
                if hasattr(model,'named_steps') and 'standardscaler' in model.named_steps:
                    assert np.allclose(model.named_steps['standardscaler'].mean_,
                        train['features'][fit].astype(float).mean(axis=0),rtol=1e-10,atol=1e-10)
                p[held]=model.predict(train['features'][held].astype(float));assigned[held]+=1
            assert np.all(assigned==1)
            saved={r['sample_id']:r['predicted_label'] for r in
                   json.loads((cache/(record['name']+'_OOF.json')).read_text())['predictions']}
            assert np.array_equal(p,np.asarray([saved[str(i)] for i in train['sample_ids']]))
            acc=float(accuracy_score(train['labels'],p))
            f1=float(f1_score(train['labels'],p,labels=list(range(7)),average='macro',zero_division=0))
            assert abs(acc-record['accuracy'])<1e-12 and abs(f1-record['macro_f1'])<1e-12
            fold_checks.append(dict(arm=arm,model=record['name'],folds=3,
                saved_weight_prediction_reproduction=True,normalization_train_fold_only=True,
                oof_accuracy=acc,oof_macro_f1=f1))
    views=[]
    for name in config['cohorts']:
        run=root/name
        job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
        assert job['test_inputs_visible_during_development'] is False
        ws=run/job['workspace']
        mounts=json.loads(Path(job['mounts']).read_text())['mounts']
        ref=Path(next(m['source'] for m in mounts if m['destination']=='/reference'))
        data=Path(next(m['source'] for m in mounts if m['destination']=='/data'))
        bank=json.loads((ref/'candidates/manifest.json').read_text())
        assert bank['starting_model']=='svc_c1'
        assert not any(p.name=='test.npz' for p in data.rglob('*'))
        # Remove only the intended evidence manipulation before equality checking.
        for key in ['training_cv_selected','training_cv_selection_rule','cv_group_field','training_cv_scope']:
            bank.pop(key)
        for row in bank['models']:row.pop('training_group_cv')
        views.append(dict(prompt=sha256(Path(job['prompt_file'])),solution=sha256(ws/'solution.py'),
            inputs={str(p.relative_to(data)):sha256(p) for p in data.rglob('*') if p.is_file()},
            common_reference={str(p.relative_to(ref)):sha256(p) for p in ref.rglob('*') if p.is_file()
                              and p.name not in ['manifest.json','training_cv_folds.json']},
            bank_except_intended_evidence=bank))
    assert all(v==views[0] for v in views), 'An unintended between-arm difference exists'
    result=dict(utc=utc(),passed=True,physical_split_count=1,views_identical_except_training_cv=True,
        common_starting_model='svc_c1',unique_fold_models_replayed=36,fold_checks=fold_checks,
        repeated_cache_counted_once=True,no_final_observations_or_labels_used=True,
        protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'))
    save(root/'audit/EVIDENCE_CONTRAST_GATE274.json',result)
    project=Path(__file__).resolve().parents[1]
    save(project/config['result_directory']/'EVIDENCE_CONTRAST_GATE274.json',result)
    print(json.dumps(dict(evidence_gate_passed=True,fold_models_replayed=36)),flush=True)


def main(root):
    config=campaign(root)
    assert not (root/'CAMPAIGN_STATUS.json').exists(), 'No automatic restart'
    state(root,status='preparing_initial',supervisor_pid=os.getpid(),planned_model_sessions=8,
        task_count=1,automatic_retries=False,hidden_scores_read=False)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    first=[root/name for name in config['cohorts']]
    second=[root/(r.name+'_review') for r in first]
    try:
        for run in first:prepare(root,run,'classification','initial')
        evidence_gate(root)
        execute_wave(root,'initial',first)
        for run,prior in zip(second,first):prepare(root,run,'classification','review',prior)
        execute_wave(root,'review',second)
        state(root,status='all_sessions_terminal_waiting_choice_review_and_external_scores',
            completed_model_sessions=8,no_final_scores_opened=True,
            next_action='Register actual choices/candidates before external scoring.')
    except BaseException as error:
        state(root,status='controller_error_inspect_before_resume',error_type=type(error).__name__,
            error=str(error),automatic_retries=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
