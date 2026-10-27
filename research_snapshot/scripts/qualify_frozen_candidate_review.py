"""Inference-only development replay for the six frozen candidate reviews."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time

from acquire_forge_pdmb_sources244 import save, sha256
from run_full_benchmark_campaign import clean_env
from run_frozen_candidate_review import source_context
from run_validation_transfer import state, utc

PROBE = r'''
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'/tools')
from verified_phmforge import VerifiedForge
name=sys.argv[1]
work=Path('/app/.runtime/qualification281');work.mkdir(exist_ok=False)
dev=np.load('/data/datasets/'+name+'/val.npz',allow_pickle=False)
catalog=json.loads(Path('/app/candidates.json').read_text())['candidates']
checks={'final_observations_absent':not Path('/data/datasets/'+name+'/test.npz').exists(),
        'host_credentials_absent':not Path('/home/jupyter-yyf-zyq/.bashrc').exists()}
readonly=json.loads(Path('/app/readonly_inputs.json').read_text())
for relative in readonly:
 try:
  with open(Path('/app')/relative,'r+b'):pass
  checks['readonly_'+relative]=False
 except OSError:checks['readonly_'+relative]=True
historical=Path('/app/historical_validation/INDEX.json')
if historical.exists():
 for item in json.loads(historical.read_text())['files']:
  path=historical.parent/item['path']
  checks['historical_readable_'+item['path']]=hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  if path.suffix=='.npz':
   arrays=np.load(path,allow_pickle=False)
   checks['historical_finite_'+item['path']]=all(np.isfinite(arrays[k]).all() for k in arrays.files)
api=VerifiedForge('/data',work/'mcp')
for tool in ['train_fault_classifier','train_rul_model']:
 result=api.call(tool,{'dataset_name':name})
 checks['denied_'+tool]=result.get('status')=='unavailable' and result.get('training_performed') is False
checks['no_training_outputs']=not list((work/'mcp').iterdir())
inputs=[]
if name=='XJTU':
 for i in range(len(dev['sample_ids'])):
  p=work/('query_'+str(i)+'.npz')
  start,end=map(int,dev['history_offsets'][i:i+2])
  payload={k:(dev[k] if k=='dataset' else dev[k][i:i+1]) for k in dev.files if k not in ['labels','groups'] and not k.startswith('history_')}
  for k in ['history_features','history_elapsed_minutes','history_channel_rms','history_channel_max_abs']:payload[k]=dev[k][start:end]
  payload['history_offsets']=np.asarray([0,end-start])
  np.savez_compressed(p,**payload)
  inputs.append(p)
else:
 p=work/'development.npz';np.savez_compressed(p,**{k:dev[k] for k in dev.files if k not in ['labels','groups']});inputs.append(p)
rows=[]
for candidate in catalog:
 predictions=[];returncodes=[]
 for i,p in enumerate(inputs):
  output=work/(candidate['id']+'_'+str(i)+'.json')
  command=[sys.executable,'-B','/app/review_candidates.py','--candidate',candidate['id'],'--input',str(p),'--output',str(output)]
  result=subprocess.run(command,capture_output=True,text=True,timeout=180)
  returncodes.append(result.returncode)
  print(json.dumps(dict(candidate=candidate['id'],query=i,returncode=result.returncode,stderr=result.stderr[-2000:])),flush=True)
  if result.returncode==0:predictions+=json.loads(output.read_text())['predictions']
 field='predicted_rul' if name=='XJTU' else 'predicted_label'
 indexed={str(r['sample_id']):r[field] for r in predictions}
 valid=all(x==0 for x in returncodes) and len(predictions)==len(indexed)==len(dev['sample_ids']) and set(indexed)==set(dev['sample_ids'].tolist())
 metric={}
 if valid:
  values=[indexed[str(x)] for x in dev['sample_ids']]
  if name=='XJTU':
   valid=bool(np.isfinite(values).all() and (np.asarray(values)>=0).all())
   metric={'mae_minutes':float(np.mean(np.abs(np.asarray(values)-dev['labels'])))}
  else:
   valid=set(values)<=set(dev['labels'].tolist())
   metric={'accuracy':float(accuracy_score(dev['labels'],values)),'macro_f1':float(f1_score(dev['labels'],values,average='macro'))}
 rows.append(dict(candidate=candidate['id'],passed=bool(valid),returncodes=returncodes,predictions=predictions,development=metric))
 Path('/audit/live.json').write_text(json.dumps({'finished':len(rows),'total':len(catalog),'last':candidate['id']}))
result=dict(passed=all(checks.values()) and all(x['passed'] for x in rows),checks=checks,candidates=rows,new_fits=0,final_inputs_mounted=False)
Path('/audit/replay.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
raise SystemExit(0 if result['passed'] else 1)
'''


def qualify_case(root, case, attempt=1):
    started=time.monotonic();run=root/case['id']
    job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
    ws=run/job['workspace'];out=run/('audit/fixed_candidate_replay281'+('' if attempt==1 else '_attempt'+str(attempt)))
    out.mkdir(exist_ok=False)
    frozen=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())
    assert all(sha256(Path(x['path']))==x['sha256'] for x in frozen['files'])
    command=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
    for path in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(path).exists():command+=['--ro-bind',path,path]
    for mount in json.loads(Path(job['mounts']).read_text())['mounts']:
        command+=['--ro-bind',mount['source'],mount['destination']]
    command+=['--bind',str(ws),'/app','--bind',str(out),'/audit']
    for relative in json.loads((ws/'readonly_inputs.json').read_text()):
        command+=['--ro-bind',str(ws/relative),'/app/'+relative]
    command+=['--proc','/proc','--dev','/dev','--chdir','/app','--remount-ro','/',
              '/opt/python312/bin/python3.12','-B','-c',PROBE,case['dataset']]
    save(out/'command.json',command)
    env=clean_env(root);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',PATH='/opt/python312/bin:/usr/bin:/bin',
        TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (out/'stdout').open('wb') as stdout,(out/'stderr').open('wb') as stderr:
        try:code=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=1200).returncode
        except subprocess.TimeoutExpired:code='timeout'
    result=json.loads((out/'replay.json').read_text()) if (out/'replay.json').exists() else {'passed':False,'candidates':[]}
    result.update(exit_code=code,elapsed_seconds=time.monotonic()-started,case=case['id'])
    source=Path(case['source_archive']);manifest,receipt=source_context(source)
    comparisons=[]
    for row in result['candidates']:
        if not row['passed']:continue
        ident=row['candidate'];paths=[]
        if case['family']=='rul':
            variant='agent_'+case['cohort'] if ident=='original' else ident
            prefix='evaluation278/'+variant+'/development/'
            paths=[p for p in manifest if p.startswith(prefix) and p.endswith('/predictions.json')]
        elif ident not in ['original','reference_mlp'] and not ident.startswith('bank_'):
            p='candidate_replays/'+case['source_case']+'/public_validation/output/predictions/'+ident+'.json'
            if p in manifest:paths=[p]
        elif ident=='original':
            candidates=[p for p in manifest if p.startswith(case['cohort']+'/workspaces/') and '/.runtime/' not in p and p.endswith('/development_predictions.json')]
            if candidates:paths=[candidates[0]]
        if paths:
            old=[]
            for p in paths:
                assert sha256(source/p)==manifest[p]['sha256']
                old+=json.loads((source/p).read_text())['predictions']
            field='predicted_rul' if case['family']=='rul' else 'predicted_label'
            before={str(x['sample_id']):x[field] for x in old}
            after={str(x['sample_id']):x[field] for x in row['predictions']}
            same=set(before)==set(after)
            maximum=max((abs(float(before[k])-float(after[k])) for k in before),default=0) if same else None
            comparisons.append(dict(candidate=ident,paths=paths,passed=same and maximum<1e-5,max_absolute_difference=maximum))
        if ident.startswith('bank_'):
            bank=json.loads((run/'references/candidates/manifest.json').read_text())
            old=next(x for x in bank['models'] if 'bank_'+x['name']==ident)
            expected=old['development']['pooled_mae_minutes'] if case['family']=='rul' else old['development_accuracy']
            actual=row['development']['mae_minutes' if case['family']=='rul' else 'accuracy']
            comparisons.append(dict(candidate=ident,kind='frozen_bank_development_metric',passed=abs(expected-actual)<1e-5,expected=expected,actual=actual))
    result['archived_development_comparisons']=comparisons
    result['frozen_files_unchanged']=all(sha256(Path(x['path']))==x['sha256'] for x in frozen['files'])
    result['passed']=result['passed'] and code==0 and result['frozen_files_unchanged'] and all(x['passed'] for x in comparisons)
    save(run/'FIXED_CANDIDATE_REPLAY281.json',result)
    # Preserve qualification predictions outside the Agent-visible workspace.
    generated=ws/'.runtime/qualification281'
    if generated.exists():shutil.move(str(generated),out/'working')
    return dict(case=case['id'],passed=result['passed'],candidates=len(result['candidates']),comparisons=len(comparisons),exit_code=code)


def main(root, selected=None, attempt=1):
    assert not (root/'MODEL_EXECUTION_STARTED281.json').exists()
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    parallel=min(6,len(config['cases']))
    candidate_count=sum(len(json.loads((root/c['id']/j['workspace']/'candidates.json').read_text())['candidates'])
        for c in config['cases'] for j in json.loads((root/c['id']/'PLAN187.json').read_text())['jobs'])
    save(root/'audit'/('REPLAY_QUALIFICATION_BUDGET281_attempt'+str(attempt)+'.json'),dict(utc=utc(),max_parallel=parallel,timeout_per_case_seconds=1200,
        max_candidate_ids=candidate_count,new_fits=0,new_model_calls=0,development_only=True))
    state(root,status='checking_frozen_candidates_on_development',model_sessions_started=0,new_fits=0)
    cases=[c for c in config['cases'] if selected is None or c['id'] in selected]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures=[pool.submit(qualify_case,root,c,attempt) for c in cases]
        rows=[]
        for c,future in zip(cases,futures):
            try:rows.append(future.result())
            except Exception as error:rows.append(dict(case=c['id'],passed=False,error=repr(error)))
    if selected:
        previous=json.loads((root/'CANDIDATE_REPLAY_GATE281.json').read_text())
        save(root/'audit'/('CANDIDATE_REPLAY_GATE281_before_attempt'+str(attempt)+'.json'),previous)
        rows += [r for r in previous['cases'] if r['case'] not in selected]
    expected={x['id'] for x in config['cases']}
    result=dict(passed=len(rows)==len(expected) and {x['case'] for x in rows}==expected and all(x['passed'] for x in rows),
                cases=rows,utc=utc(),new_fits=0,model_calls=0)
    save(root/'CANDIDATE_REPLAY_GATE281.json',result)
    save(Path(__file__).resolve().parents[1]/config.get('result_directory','results/review281')/'CANDIDATE_REPLAY_GATE281.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    state(root,status='candidate_replay_passed' if result['passed'] else 'candidate_replay_failed_inspect',candidate_replay=result)
    assert result['passed'],rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',nargs='+');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();main(a.run.resolve(),a.cases,a.attempt)
