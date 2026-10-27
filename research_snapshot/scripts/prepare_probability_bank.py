"""Fit a frozen real basis bank only after measured-source/export admission.

No final observations, labels or scores are accepted by this stage. Fitting,
features and sandboxed cold inference reuse the established numerical code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

import numpy as np

from controlled_feedback import save,sha,utc,validate
from evaluate_controlled_feedback import cold,independent
from prepare_forge_coding_baseline import PYTHON,RUNTIME
from run_full_benchmark_campaign import clean_env


def prepare(root):
    # Require actual source/export gates before making a fitting directory.
    admission_path=root/'SOURCE_ADMISSION.json';admission=json.loads(admission_path.read_text())
    for key in ['measured_source_verified','all_member_CRCs_verified','native_values_verified',
                'operating_condition_contract_verified','export_integrity_verified']:
        assert admission.get(key) is True,'Source gate incomplete: '+key
    assert admission.get('final_scores_opened') is False
    config=json.loads((root/'BASIS_PLAN.json').read_text())
    assert not (root/'BASIS_FITTING_STARTED.json').exists(),'No implicit retry or new seed'
    assert len(config['models'])==6 and len({x['name'] for x in config['models']})==6
    assert config['parallel_fits']==2 and config['fit_timeout_seconds']==180
    inputs={}
    for key in ['train','development','fitting_settings']:
        bound=admission['input_files'][key];path=root/bound['path']
        assert path.resolve().is_relative_to(root) and sha(path)==bound['sha256']
        inputs[key]=path
    settings=json.loads(inputs['fitting_settings'].read_text())
    assert settings['save_probabilities'] is True and settings['cv']=='leave_one_condition_out'
    training=np.load(inputs['train'],allow_pickle=False);development=np.load(inputs['development'],allow_pickle=False)
    classes=settings['classes'];group_key=settings['cv_group_key']
    assert len(np.unique(training[group_key]))==3
    assert not set(training[group_key].tolist())&set(development[group_key].tolist())
    assert not set(training['groups'].tolist())&set(development['groups'].tolist())
    assert not set(training['sample_ids'].tolist())&set(development['sample_ids'].tolist())
    for data in [training,development]:assert len(data['sample_ids'])==len(set(data['sample_ids'].tolist()))
    assert training['waveforms'].shape[1]==development['waveforms'].shape[1]==settings['expected_samples']
    assert set(training['labels'].tolist())==set(development['labels'].tolist())==set(classes)
    for model in config['models']:
        validate(model['recipe']);assert model['recipe']['model']!='svc','The existing SVC has no probability head'
    for name in ['audit','code','shared_tools','logs','runtime/tmp','runtime/mpl']:(root/name).mkdir(exist_ok=True,parents=True)
    source=Path(__file__).parent
    for name in ['controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(source/name,root/'shared_tools'/name)
    for name in ['prepare_probability_bank.py','controlled_feedback.py','evaluate_controlled_feedback.py']:
        if (source/name).resolve()!=(root/'code'/name).resolve():shutil.copy2(source/name,root/'code'/name)
    initial=dict(admission_sha256=sha(admission_path),plan_sha256=sha(root/'BASIS_PLAN.json'),
                 input_files=admission['input_files'],code=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in [*sorted((root/'shared_tools').glob('*.py')),*sorted((root/'code').glob('*.py'))]])
    save(root/'audit/BASIS_INPUT_FREEZE.json',initial)
    save(root/'BASIS_FITTING_STARTED.json',dict(utc=utc(),pid=os.getpid(),maximum_wrapper_fit_calls=24,new_LLM_calls=0,final_data_read=False))
    env=clean_env(root);env.update(PYTHONPATH=str(RUNTIME/'deps')+':'+str(root/'shared_tools'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def fit_one(item):
        folder=root/'basis'/item['name'];folder.mkdir(parents=True,exist_ok=False)
        save(folder/'recipe.json',item['recipe'])
        command=[str(PYTHON/'bin/python3.12'),'-B',str(root/'shared_tools/controlled_feedback.py'),'fit',
                 '--recipe',str(folder/'recipe.json'),'--train',str(inputs['train']),'--dev',str(inputs['development']),
                 '--settings',str(inputs['fitting_settings']),'--output',str(folder)]
        save(folder/'command.json',command)
        status=dict(name=item['name'],started_utc=utc(),status='running',recipe=item['recipe'])
        try:
            with (folder/'stdout').open('wb') as out,(folder/'stderr').open('wb') as err:
                process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
                status['pid']=process.pid;save(folder/'status.json',status)
                try:code=process.wait(timeout=config['fit_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            status.update(status='complete' if code==0 else 'fit_error',exit_code=code)
        except Exception as exc:status.update(status='fit_error',error_type=type(exc).__name__)
        events=[json.loads(x) for x in (folder/'fit_events.jsonl').read_text().splitlines()] if (folder/'fit_events.jsonl').exists() else []
        status.update(ended_utc=utc(),fit_calls_started=sum(x['status']=='started' for x in events),fit_calls_completed=sum(x['status']=='completed' for x in events))
        save(folder/'status.json',status);return status
    with ThreadPoolExecutor(max_workers=2) as executor:outcomes=list(executor.map(fit_one,config['models']))
    save(root/'audit/BASIS_FIT_OUTCOMES.json',outcomes)
    assert all(x['status']=='complete' and x['fit_calls_completed']==x['fit_calls_started']==4 for x in outcomes),'Preserve all failed fits;no automatic retry'
    # Actual development observations only, with labels absent from cold workers.
    observations=root/'evaluation_inputs/probability_development';observations.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(observations/'observations.npz',waveforms=development['waveforms'],sample_ids=development['sample_ids'])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={item['name']:executor.submit(cold,root,'basis_bank',item['name'],'probability_development',
                 checkpoint=root/'basis'/item['name']/'checkpoint.joblib',probability_output=True) for item in config['models']}
        cold_outputs={name:future.result() for name,future in futures.items()}
    fields={part:[] for part in ['cv','development']};checks=[];models=[]
    for item in config['models']:
        folder=root/'basis'/item['name'];report=json.loads((folder/'fit_report.json').read_text())
        saved=np.load(folder/'probabilities.npz',allow_pickle=False)
        prediction=np.load(folder/'predictions.npz',allow_pickle=False)
        assert report['checkpoint_sha256']==sha(folder/'checkpoint.joblib')
        assert report['train_input_sha256']==admission['input_files']['train']['sha256']
        assert report['development_input_sha256']==admission['input_files']['development']['sha256']
        assert saved['classes'].tolist()==classes and report['cold_reload_equal']
        assert np.array_equal(saved['train_ids'],training['sample_ids']) and np.array_equal(saved['development_ids'],development['sample_ids'])
        seen=[]
        for fold in report['cv_folds']:
            assert not set(fold['fitting_conditions'])&set(fold['checking_conditions'])
            assert set(fold['fitting_ids'])==set(training['sample_ids'].tolist())-set(fold['checking_ids'])
            seen+=fold['checking_ids']
        assert len(seen)==len(set(seen))==len(training['labels']) and set(seen)==set(training['sample_ids'].tolist())
        actual=cold_outputs[item['name']]
        assert actual['classes']==classes and actual['sample_ids']==development['sample_ids'].tolist()
        assert np.allclose(actual['probabilities'],saved['development'],atol=1e-12,rtol=0)
        for part,data in [('train',training),('cv',training),('development',development)]:
            predicted=np.asarray(classes)[saved[part].argmax(axis=1)]
            assert np.array_equal(predicted,prediction[part])
            metric=independent(data['labels'],predicted,classes)
            assert metric['confusion']==report[part]['confusion']
            assert abs(metric['macro_f1']-report[part]['macro_f1'])<1e-12
            checks.append(dict(model=item['name'],split=part,metrics=metric,passed=True))
            if part!='train':fields[part].append(saved[part])
        models.append(dict(item,directory=str(folder.relative_to(root)),checkpoint_sha256=sha(folder/'checkpoint.joblib'),
                           fit_report_sha256=sha(folder/'fit_report.json'),cold_process_development_probabilities_equal=True))
    arrays=dict(classes=np.asarray(classes),basis_names=np.asarray([x['name'] for x in config['models']]))
    for part,data in [('cv',training),('development',development)]:
        arrays[part]=np.stack(fields[part],axis=1)
        for suffix,key in [('labels','labels'),('ids','sample_ids'),('conditions',group_key),('groups','groups')]:arrays[part+'_'+suffix]=data[key]
    bank=root/'bank/probabilities.npz';bank.parent.mkdir(exist_ok=False)
    with bank.open('xb') as stream:np.savez_compressed(stream,**arrays)
    assert sha(admission_path)==initial['admission_sha256'] and sha(root/'BASIS_PLAN.json')==initial['plan_sha256']
    for row in [*initial['input_files'].values(),*initial['code']]:assert sha(root/row['path'])==row['sha256']
    save(root/'audit/BASIS_INDEPENDENT_CHECKS.json',checks)
    result=dict(passed=True,utc=utc(),models=models,wrapper_fit_calls_started=24,wrapper_fit_calls_completed=24,
                independent_metric_checks=len(checks),actual_cold_processes=6,bank=str(bank.relative_to(root)),bank_sha256=sha(bank),
                final_observations_or_labels_read=False,new_final_scores=0,new_LLM_calls=0,new_Agent_cases=0,
                scope='Real training-only basis bank;not an Agent outcome or overfitting evidence.')
    save(root/'BASIS_QUALIFICATION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='models'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();prepare(args.run.resolve())
