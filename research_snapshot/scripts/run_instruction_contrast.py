"""Run one prespecified wave differing only in an explicit validation instruction.

Existing native CLI, MCP, isolation, budget and scoring gates are reused.
Final scoring is deliberately not part of this controller.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from acquire_forge_pdmb_sources244 import save, sha256
from run_validation_transfer import campaign, prepare, execute_wave, state, freeze, utc


def equality_gate(root):
    config=campaign(root)
    views=[]
    for case in config['cases']:
        run=root/case['cohort']
        job,=json.loads((run/'PLAN187.json').read_text())['jobs']
        assert job['test_inputs_visible_during_development'] is False
        mounts={m['destination']:Path(m['source']) for m in json.loads(Path(job['mounts']).read_text())['mounts']}
        assert not any(mounts['/data'].rglob('test.npz'))
        assert not any(mounts['/data'].rglob('labels.json'))
        prompt=Path(job['prompt_file']).read_text()
        suffix=config['arm_prompt_addenda'][case['arm']]
        assert prompt.endswith(suffix)
        views.append(dict(prompt_without_arm=prompt.removesuffix(suffix),
            solution_sha256=sha256(run/job['workspace']/'solution.py'),
            inputs={str(p.relative_to(mounts['/data'])):sha256(p) for p in mounts['/data'].rglob('*') if p.is_file()},
            references={str(p.relative_to(mounts['/reference'])):sha256(p) for p in mounts['/reference'].rglob('*') if p.is_file()}))
    assert all(x==views[0] for x in views), 'Unexpected between-arm difference'
    source=root/'common/restored/ROTORBROKENBAR/data'
    with np.load(source/'train.npz',allow_pickle=False) as f: train=dict(f)
    classes=json.loads((source/'dataset.json').read_text())['class_names']
    index={str(v):i for i,v in enumerate(train['sample_ids'])}
    cache=root/'bank_selection'/config['candidate_bank']['cache_key']
    cv=json.loads((cache/'TRAINING_CV.json').read_text())
    checks=[]
    for candidate in cv['rows']:
        assert candidate['valid'], 'Invalid fixed CV candidate'
        pred=np.empty_like(train['labels']);assigned=np.zeros(len(pred),int)
        for fold in cv['folds']:
            fit=np.asarray([index[x] for x in fold['fit_ids']])
            held=np.asarray([index[x] for x in fold['heldout_ids']])
            for group in ['groups',cv['cv_group_field']]:
                assert not set(train[group][fit])&set(train[group][held])
            assert set(train['labels'][fit])==set(classes)
            model=joblib.load(cache/'cv_models'/(candidate['name']+'_fold'+str(fold['fold'])+'.joblib'))
            if hasattr(model,'named_steps') and 'standardscaler' in model.named_steps:
                assert np.allclose(model.named_steps['standardscaler'].mean_,
                    train['features'][fit].astype(float).mean(0),rtol=1e-10,atol=1e-10)
            pred[held]=model.predict(train['features'][held].astype(float));assigned[held]+=1
        assert np.all(assigned==1)
        saved={x['sample_id']:x['predicted_label'] for x in json.loads((cache/(candidate['name']+'_OOF.json')).read_text())['predictions']}
        assert np.array_equal(pred,np.asarray([saved[str(x)] for x in train['sample_ids']]))
        acc=float(accuracy_score(train['labels'],pred))
        f1=float(f1_score(train['labels'],pred,labels=classes,average='macro',zero_division=0))
        assert abs(acc-candidate['accuracy'])<1e-12 and abs(f1-candidate['macro_f1'])<1e-12
        checks.append(dict(candidate=candidate['name'],folds=len(cv['folds']),accuracy=acc,macro_f1=f1))
    result=dict(utc=utc(),passed=True,identical_except_arm_instruction=True,
        all_training_fold_classes_present=True,fold_models_replayed=sum(x['folds'] for x in checks),
        checks=checks,final_observations_or_answers_used=False,protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'))
    tag=config['iteration'].removeprefix('A')
    save(root/'audit'/('CONTRAST_GATE'+tag+'.json'),result)
    save(Path(__file__).resolve().parents[1]/config['result_directory']/('CONTRAST_GATE'+tag+'.json'),result)


def run(root):
    config=campaign(root)
    assert not (root/'CAMPAIGN_STATUS.json').exists(), 'No automatic restart'
    assert json.loads((root/('PREPARATION'+config['iteration'].removeprefix('A')+'.json')).read_text())['passed']
    state(root,status='preparing_instruction_contrast',supervisor_pid=os.getpid(),
        planned_model_sessions=config['planned_sessions'],task_count=1,automatic_retries=False,hidden_scores_read=False)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    cohorts=[]
    try:
        for case in config['cases']:
            cohort=root/case['cohort'];prepare(root,cohort,'classification','initial')
            job,=json.loads((cohort/'PLAN187.json').read_text())['jobs']
            prompt=Path(job['prompt_file']);prompt.write_text(prompt.read_text()+config['arm_prompt_addenda'][case['arm']])
            freeze(cohort);cohorts.append(cohort)
        equality_gate(root)
        execute_wave(root,'instruction_contrast',cohorts)
        state(root,status='all_sessions_terminal_waiting_choice_audit_and_scores',
            completed_model_sessions=len(cohorts),no_final_scores_opened=True,
            next_action='Register actual candidates and component contrasts before external scoring; no additional fits after final scores.')
    except BaseException as error:
        state(root,status='controller_error_inspect_before_resume',error_type=type(error).__name__,
            error=str(error),automatic_retries=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run.resolve())
