"""Seal all feedback-timing choices, then cold-score actual saved estimators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from controlled_feedback import save, sha, utc, metrics
from prepare_forge_coding_baseline import PYTHON, RUNTIME
from run_full_benchmark_campaign import clean_env

PROJECT=Path(__file__).resolve().parents[1]

PREDICT=r'''
import json,sys
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,'/tools')
from controlled_feedback import transform
model=joblib.load('/model/checkpoint.joblib')
d=np.load('/input/observations.npz',allow_pickle=False)
assert set(d.files)=={'waveforms','sample_ids'}
pred=model['model'].predict(transform(d['waveforms'],model['recipe']['features']))
classes=model.get('fitting_settings',{}).get('classes',list(range(9)))
assert len(pred)==len(d['sample_ids']) and set(pred.tolist())<=set(classes)
Path('/output/predictions.json').write_text(json.dumps({'predictions':[{'sample_id':str(i),'predicted_label':int(y)} for i,y in zip(d['sample_ids'],pred)]},indent=2)+'\n')
'''

PREDICT_PROBABILITIES=r'''
import json,sys
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,'/tools')
from controlled_feedback import transform
saved=joblib.load('/model/checkpoint.joblib')
data=np.load('/input/observations.npz',allow_pickle=False)
assert set(data.files)=={'waveforms','sample_ids'}
model=saved['model']
probabilities=model.predict_proba(transform(data['waveforms'],saved['recipe']['features']))
assert probabilities.shape==(len(data['sample_ids']),len(model.classes_))
assert np.isfinite(probabilities).all() and probabilities.min()>=0
assert np.allclose(probabilities.sum(axis=1),1,atol=1e-8,rtol=0)
Path('/output/probabilities.json').write_text(json.dumps(dict(classes=model.classes_.tolist(),sample_ids=data['sample_ids'].tolist(),probabilities=probabilities.tolist()))+'\n')
'''


def independent(y,p,classes=None):
    classes=list(range(9)) if classes is None else list(classes)
    lookup={int(c):i for i,c in enumerate(classes)}
    cm=np.zeros((len(classes),len(classes)),dtype=np.int64)
    for actual,pred in zip(y,p):cm[lookup[int(actual)],lookup[int(pred)]]+=1
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(len(classes)),where=den!=0)
    value=dict(accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(f.mean()),correct=int(np.trace(cm)),total=int(cm.sum()),confusion=cm.tolist())
    reference=metrics(y,p,classes)
    assert value['confusion']==reference['confusion'] and value['correct']==reference['correct']
    assert abs(value['accuracy']-reference['accuracy'])<1e-12 and abs(value['macro_f1']-reference['macro_f1'])<1e-12
    return value


def seal(root):
    assert not (root/'CHOICES_SEALED282.json').exists()
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());rows=[];frozen=[];policy=[]
    bindings=root/'audit/SOURCE_BINDINGS.json'
    if bindings.exists():
        for bound in json.loads(bindings.read_text())['files']:
            assert sha(root/bound['path'])==bound['sha256']
            frozen.append(dict(path=bound['path'],sha256=bound['sha256']))
        for p in [root/'CAMPAIGN_CONFIG.json',*sorted((root/'shared_tools').glob('*.py'))]:
            frozen.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
    for case in config['cases']:
        run=root/case['id'];job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
        status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        assert status['status']!='running'
        broker=json.loads((run/'broker/STATE.json').read_text())
        choice=broker.get('final_choice');ws=run/job['workspace']
        delivered=(ws/'selection.json').exists() and (ws/'report.md').exists() and choice is not None
        if delivered:
            copy=json.loads((ws/'selection.json').read_text())
            # The final deployment identity is always the private sealed selection.
            submitted=copy.get('candidate_id',copy.get('result',{}).get('final_choice',{}).get('candidate_id'))
            if submitted is None and copy.get('final_choice'):submitted=copy['final_choice'].get('candidate_id')
            assert submitted==choice['candidate_id'],(case['id'],copy)
            assert choice['checkpoint_sha256']==sha(run/'broker'/choice['candidate_id']/'checkpoint.joblib')
        for trial in broker['trials']:
            assert trial['status']!='running'
            folder=run/'broker'/trial['id']
            for p in folder.rglob('*'):
                if p.is_file():frozen.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
        for line in (run/'broker/ACCESS.jsonl').read_text().splitlines():
            event=json.loads(line);response=event['response']
            if not response['ok']:continue
            result=response['result'];n=len(result['trials'])
            for trial in result['trials']:
                if trial['status']!='complete':continue
                visible=isinstance(trial.get('development'),dict)
                required=case['condition']=='immediate' or n==config['candidate_fits_per_case']
                assert visible==required,('feedback_policy_violation',case['id'],n)
                if config.get('complete_validation_evidence'):
                    paths=trial.get('validation_evidence',{})
                    assert set(paths)==({'train','cv','development'} if required else {'train','cv'})
                    for part, public_path in paths.items():
                        local=run/'feedback'/Path(public_path).relative_to('/feedback')
                        source=run/'broker'/trial['id']/'evidence'/(part+'.json')
                        assert sha(local)==sha(source)
            policy.append(dict(case=case['id'],trial_count=n,feedback_policy_passed=True))
        for p in [run/'broker/STATE.json',run/'broker/ACCESS.jsonl',ws/'selection.json',ws/'report.md']:
            if p.exists():frozen.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
        rows.append(dict(case,job=job['id'],status=status['status'],delivered=delivered,choice=choice,
            trials=[{k:t[k] for k in ['id','recipe','status']} for t in broker['trials']]))
    result=dict(utc=utc(),cases=rows,files=frozen,feedback_policy_checks=policy,final_scores_opened=False,
        developer_protocol='No final scores until every case is terminal and every actual candidate is sealed.')
    save(root/'CHOICES_SEALED282.json',result)
    save(PROJECT/config.get('result_directory','results/feedback282')/'CHOICES_SEALED282.json',{k:v for k,v in result.items() if k!='files'})
    return result


def cold(root,case,trial,part,checkpoint=None,probability_output=False):
    folder=root/'evaluation282'/case/trial/part;folder.mkdir(parents=True,exist_ok=False)
    command=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():command+=['--ro-bind',p,p]
    for src,dst in [(PYTHON,'/opt/python312'),(RUNTIME/'deps','/opt/phm_python/extra'),(root/'shared_tools','/tools'),
        (root/'evaluation_inputs'/part,'/input')]:command+=['--ro-bind',str(src),dst]
    model_path=Path(checkpoint) if checkpoint else root/case/'broker'/trial/'checkpoint.joblib'
    command+=['--dir','/model','--ro-bind',str(model_path),'/model/checkpoint.joblib']
    (folder/'tmp').mkdir();(folder/'mpl').mkdir()
    command+=['--bind',str(folder),'/output','--proc','/proc','--dev','/dev','--chdir','/output','--remount-ro','/',
              '/opt/python312/bin/python3.12','-B','-c',PREDICT_PROBABILITIES if probability_output else PREDICT]
    save(folder/'command.json',command)
    env=clean_env(root);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/mpl',
                                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=180)
    (folder/'stdout').write_text(result.stdout);(folder/'stderr').write_text(result.stderr)
    assert result.returncode==0,(case,trial,part,result.stderr[-1500:])
    if probability_output:return json.loads((folder/'probabilities.json').read_text())
    return json.loads((folder/'predictions.json').read_text())['predictions']


def evaluate(root):
    assert not (root/'NUMERICAL_RESULTS282.json').exists()
    sealed=json.loads((root/'CHOICES_SEALED282.json').read_text())
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    classes=config.get('classes',list(range(9)))
    assert all(sha(root/x['path'])==x['sha256'] for x in sealed['files'])
    for case in sealed['cases']:
        run=root/case['id'];assert not any(p.name.endswith('.sock') for p in (run/'feedback').iterdir()),'Close broker sockets before evaluation'
    inputs={part:dict(np.load(root/'hidden'/(part+'.npz'),allow_pickle=False)) for part in ['train','val','test']}
    for part in ['val','test']:
        p=root/'evaluation_inputs'/part;p.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(p/'observations.npz',waveforms=inputs[part]['waveforms'],sample_ids=inputs[part]['sample_ids'])
    jobs=[(r['id'],t['id'],part) for r in sealed['cases'] for t in r['trials'] if t['status']=='complete' for part in ['val','test']]
    predictions={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[(job,pool.submit(cold,root,*job)) for job in jobs]
        for job,f in futures:predictions[job]=f.result()
    rows=[];checks=[]
    for case in sealed['cases']:
        candidates=[]
        for trial in case['trials']:
            if trial['status']!='complete':continue
            folder=root/case['id']/'broker'/trial['id'];report=json.loads((folder/'fit_report.json').read_text())
            saved=np.load(folder/'predictions.npz',allow_pickle=False)
            computed={}
            for part in ['val','test']:
                pp=predictions[(case['id'],trial['id'],part)];index={r['sample_id']:r['predicted_label'] for r in pp}
                assert len(index)==len(pp)==len(inputs[part]['sample_ids']) and set(index)==set(inputs[part]['sample_ids'].tolist())
                pred=np.asarray([index[str(i)] for i in inputs[part]['sample_ids']]);computed[part]=independent(inputs[part]['labels'],pred,classes)
                if 'condition_id' in inputs[part]:
                    computed[part]['by_condition']={str(c):independent(inputs[part]['labels'][inputs[part]['condition_id']==c],
                        pred[inputs[part]['condition_id']==c],classes) for c in np.unique(inputs[part]['condition_id'])}
                if part=='val':assert np.array_equal(pred,saved['development'])
            for part,field in [('train','train'),('cv','cv'),('development','val')]:
                actual=independent(inputs['val' if part=='development' else 'train']['labels'],saved[part],classes)
                assert actual['confusion']==report[part]['confusion'] and abs(actual['macro_f1']-report[part]['macro_f1'])<1e-12
                checks.append(dict(case=case['id'],candidate=trial['id'],split=part,passed=True))
            candidates.append(dict(id=trial['id'],recipe=trial['recipe'],train=report['train'],cv=report['cv'],development=computed['val'],final=computed['test']))
        chosen=next((x for x in candidates if case['choice'] and x['id']==case['choice']['candidate_id']),None)
        best=max(candidates,key=lambda x:(x['final']['macro_f1'],x['final']['accuracy'])) if candidates else None
        fixed_selectors={}
        for split in ['cv','development']:
            # Tie-breaking uses the earliest actual candidate, not final scores.
            selected=max(candidates,key=lambda x:(x[split]['macro_f1'],x[split]['accuracy'],-int(x['id'].split('_')[-1]))) if candidates else None
            fixed_selectors[split]=dict(candidate_id=selected['id'],final=selected['final']) if selected else None
        regret=(best['final']['macro_f1']-chosen['final']['macro_f1']) if chosen else None
        major=bool(chosen and regret>=0.05 and best['final']['accuracy']>=chosen['final']['accuracy'])
        rows.append(dict(case=case['id'],condition=case['condition'],replicate=case['replicate'],status=case['status'],delivered=case['delivered'],
            selection=case['choice'],chosen=chosen,candidates=candidates,posthoc_available_candidate_best=best['id'] if best else None,
            macro_f1_regret=regret,major_available_candidate_loss=major,fixed_selectors_on_same_actual_pool=fixed_selectors))
    groups={}
    for condition in ['immediate','delayed']:
        ready=[x for x in rows if x['condition']==condition and x['delivered'] and x['chosen']]
        groups[condition]=dict(completed=len(ready),planned=sum(c['condition']==condition for c in config['cases']),
            mean_final_macro_f1=float(np.mean([r['chosen']['final']['macro_f1'] for r in ready])) if ready else None,
            mean_final_accuracy=float(np.mean([r['chosen']['final']['accuracy'] for r in ready])) if ready else None,
            mean_development_macro_f1=float(np.mean([r['chosen']['development']['macro_f1'] for r in ready])) if ready else None,
            mean_gap=float(np.mean([r['chosen']['development']['macro_f1']-r['chosen']['final']['macro_f1'] for r in ready])) if ready else None)
    effects={}
    if all(x['completed']==x['planned'] for x in groups.values()):
        effects={k:groups['immediate'][k]-groups['delayed'][k] for k in ['mean_final_macro_f1','mean_final_accuracy','mean_gap']}
    result=dict(utc=utc(),cases=rows,groups=groups,immediate_minus_delayed=effects,independent_metric_checks=checks,
        cold_processes=len(jobs),real_formal_fits=sum(4*len(r['candidates']) for r in rows),
        qualification_accounting=config.get('qualification_accounting',dict(fresh_recovery_fits=4,earlier_fits_reused=4,total_distinct=8)),
        infrastructure_attempts=config.get('infrastructure_attempts',6),infrastructure_model_requests=config.get('infrastructure_model_requests',210),
        logical_cases=len(config['cases']),total_native_attempts=len(config['cases'])+config.get('infrastructure_attempts',6),
        final_scores_opened=True,goal_status='active',mechanism_confirmed=False,
        limitations=config.get('limitations',['one released collection,unknown acquisition overlap','three stochastic trajectories per condition','restricted recipe-search probe','public-data pretraining contamination not excluded']))
    assert all(sha(root/x['path'])==x['sha256'] for x in sealed['files'])
    save(root/'NUMERICAL_RESULTS282.json',result);save(PROJECT/config.get('result_directory','results/feedback282')/'NUMERICAL_RESULTS282.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(groups=groups,immediate_minus_delayed=effects,cold_processes=len(jobs),real_formal_fits=result['real_formal_fits']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--mode',required=True,choices=['seal','evaluate'])
    a=p.parse_args();(seal if a.mode=='seal' else evaluate)(a.run.resolve())
