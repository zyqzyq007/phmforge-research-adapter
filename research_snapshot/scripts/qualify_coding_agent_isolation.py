"""Probe the actual native code sandbox before admitting a coding cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env, save


PROBE = r'''
import json, os, select, socket, subprocess, sys
from pathlib import Path
import numpy as np
import torch, sklearn
sys.path.insert(0,'/tools')
name=sys.argv[1]
if name in ['LDM_RTF','IMS_ENDPOINT']:from real_gap_rul_model import infer
elif name=='FEMTO':from real_bearing_rul_model import infer
elif name=='XJTU':from real_xjtu_rul_model import infer
elif name.startswith('CMAPSS_'):from real_rul_model import infer
else:from real_fault_model import infer
checks={}
checks['working_directory']=str(Path.cwd())=='/app'
checks['private_host_paths_absent']=not any(Path(p).exists() for p in [
 '/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/AGNETS.md',
 '/home/jupyter-yyf-zyq/.bashrc','/home/jupyter-yyf-zyq/.codex/auth.json',
 '/home/jupyter-yyf-zyq/.codex/.env',
 '/data/yyf-zyq/PHM_20261010','/srv/yyf-zyq/PHM_20261010/_shared',sys.argv[3]])
checks['credential_names_absent']=not any(any(x in k.upper() for x in
 ['API_KEY','ACCESS_TOKEN','PASSWORD','SECRET']) for k in os.environ)
checks['proc_parent_credentials_absent']=not any(x in Path('/proc/1/environ').read_bytes()
 for x in [b'DEEPSEEK_API_KEY=',b'OPENAI_API_KEY='])
checks['managed_temporary_directory']=os.environ['TMPDIR'].startswith('/app/.runtime/')
s=None
try:
 s=socket.socket();s.settimeout(1)
 s.connect(('1.1.1.1',443));checks['external_network_blocked']=False
except OSError: checks['external_network_blocked']=True
finally:
 if s is not None:s.close()
data=Path('/data/datasets')/name
info=json.loads((data/'dataset.json').read_text())
sealed=len(sys.argv)>5 and sys.argv[5]=='sealed'
if sealed:
 checks['test_observations_absent']=not (data/'test.npz').exists()
 allowed={'train.npz','val.npz','dataset.json'}
 if name in ['XJTU','LDM_RTF','IMS_ENDPOINT']:allowed.add('training_config.json')
 if name=='FEMTO':
  allowed.update({'training_config.json','trajectories'})
  with np.load(data/'train.npz',allow_pickle=False) as fitting:
   learning_units={int(v) for v in fitting['unit_ids']}
  expected={f'unit{v:03}.npz' for v in learning_units}
  checks['only_learning_trajectories']=set(p.name for p in (data/'trajectories').iterdir())==expected
 if name.startswith('CMAPSS_'):allowed.update({'training_config.json','train.txt'})
 if name=='ENGINE_QA_HEALTH':allowed.update({'supervision.json','sequences.npz'})
 if name=='ENGINE_QA_TREND':allowed.update({'cycle_prefixes.npz','tuple_members.json'})
 if name=='MENDELEY':
  allowed.add('records')
  authorized=set()
  for part in ['train','val']:
   with np.load(data/(part+'.npz'),allow_pickle=False) as records:
    authorized.update(str(v) for v in records['record_files'])
  actual=set(str(p.relative_to(data)) for p in (data/'records').rglob('*') if p.is_file())
  checks['only_train_validation_raw_records']=actual==authorized and all(not Path(v).is_absolute() and '..' not in Path(v).parts for v in actual)
  checks['raw_records_have_no_targets']=all(not set(np.load(data/v,allow_pickle=False).files)&{'labels','label','origin_labels','fault_type','actual_rul'} for v in actual)
 checks['authoring_data_allowlist']=set(p.name for p in data.iterdir())==allowed
 checks['test_labels_absent']=checks['test_observations_absent']
 checks['external_evaluation_data_unmounted']=not (Path(sys.argv[3]).parent/'evaluation_data').exists()
else:
 test=np.load(data/'test.npz',allow_pickle=False)
 checks['test_labels_absent']=not any(k in test for k in ['labels','origin_labels','groups','actual_rul','rul_percentage'])
checks['one_dataset_only']=len(list(Path('/data/datasets').iterdir()))==1
out=Path('/app/artifacts/preflight');out.mkdir(exist_ok=True)
dev=np.load(data/'val.npz',allow_pickle=False)
np.savez_compressed(out/'dev.npz',**{k:dev[k] for k in dev.files if k not in ['labels','origin_labels','groups']})
prediction=infer('/reference/model.pt',out/'dev.npz',out/'dev_predictions.json')
rows=prediction['predictions']
if name in ['FEMTO','XJTU','LDM_RTF','IMS_ENDPOINT'] or name.startswith('CMAPSS_'):
 checks['cold_reference_inference']=len(rows)==len(dev['sample_ids']) and all(np.isfinite(r['predicted_rul']) and r['predicted_rul']>=0 for r in rows)
 if name=='FEMTO':metric={'refitted_development_MAE_seconds':float(np.mean(np.abs(np.asarray([r['predicted_rul'] for r in rows])-dev['labels']))),'scope':'final reference includes development learning bearings; not held-out performance'}
 elif name=='XJTU':metric={'development_MAE_minutes':float(np.mean(np.abs(np.asarray([r['predicted_rul'] for r in rows])-dev['labels']))),'scope':'train-only fit; three held-out development bearings'}
 elif name in ['LDM_RTF','IMS_ENDPOINT']:metric={'development_MAE_acquisitions':float(np.mean(np.abs(np.asarray([r['predicted_rul'] for r in rows])-dev['labels']))),'scope':info['validation_scope']}
 else:metric={'development_MAE_cycles':float(np.mean(np.abs(np.asarray([r['predicted_rul'] for r in rows])-dev['labels']))),'scope':'held-out training engines; no final RUL labels'}
else:
 checks['cold_reference_inference']=len(rows)==len(dev['sample_ids']) and all(r['predicted_label'] in info['class_names'] for r in rows)
 metric={'accuracy':sum(r['predicted_label']==v for r,v in zip(rows,dev['labels'].tolist()))/len(rows)}
audit=json.loads(Path('/reference/training_audit.json').read_text())
checks['real_reference_updates']=audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0
bank=Path('/reference/candidates/manifest.json')
if bank.exists():
 metadata=json.loads(bank.read_text())
 expected=next(x for x in metadata['models'] if x['name']==metadata['starting_model'])
 backend='real_xjtu_rul_model.py' if name=='XJTU' else 'frozen_classifier.py'
 p=subprocess.run(['/opt/python312/bin/python3.12','-B','/tools/'+backend,'--model','/reference/candidates/'+expected['checkpoint'],'--input',str(out/'dev.npz'),'--output',str(out/'starting_predictions.json')],capture_output=True,text=True,timeout=float(sys.argv[4]))
 start_rows=json.loads((out/'starting_predictions.json').read_text())['predictions'] if p.returncode==0 else []
 field='predicted_rul' if name=='XJTU' else 'predicted_label'
 indexed={r['sample_id']:r[field] for r in start_rows}
 checks['cold_starting_candidate_identity']=len(indexed)==len(start_rows)==len(dev['sample_ids']) and set(indexed)==set(dev['sample_ids'].tolist())
 if name=='XJTU':
  actual=float(np.mean([abs(indexed[str(i)]-y) for i,y in zip(dev['sample_ids'],dev['labels'].tolist())]))
  expected_metric=expected['development']['pooled_mae_minutes']
 else:
  actual=sum(indexed.get(str(i))==y for i,y in zip(dev['sample_ids'],dev['labels'].tolist()))/len(dev['sample_ids'])
  expected_metric=expected['development_accuracy']
 checks['cold_starting_candidate_metrics']=p.returncode==0 and abs(actual-expected_metric)<1e-9
def mcp_probe():
 with (out/'mcp.stderr').open('w') as err:
  p=subprocess.Popen(['/opt/python312/bin/python3.12','-B',
   '/tools/verified_benchmark_worker.py','--family','forge','--data','/data',
   '--output',str(out),'--transport','mcp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
   stderr=err,text=True,bufsize=1)
  def request(value):
   p.stdin.write(json.dumps(value)+'\n');p.stdin.flush()
   if not select.select([p.stdout],[],[],float(sys.argv[4]))[0]:
    raise TimeoutError('No MCP response; child exit='+str(p.poll()))
   row=p.stdout.readline()
   if not row:raise RuntimeError('MCP closed; child exit='+str(p.poll()))
   return json.loads(row)['result']
  try:
   request({'jsonrpc':'2.0','id':1,'method':'initialize','params':{
    'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'gate','version':'1'}}})
   p.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'})+'\n');p.stdin.flush()
   catalog=request({'jsonrpc':'2.0','id':2,'method':'tools/list'})
   checks['actual_mcp_catalog']=len(catalog['tools'])==22
   result=request({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{
    'name':'load_dataset','arguments':{'dataset_name':name,'split':'val'}}})
   loaded=json.loads(result['content'][0]['text'])
   checks['actual_mcp_data']=not result.get('isError',False) and Path(loaded['data_path']).is_file()
   result=request({'jsonrpc':'2.0','id':4,'method':'tools/call','params':{
    'name':'load_ground_truth','arguments':{'dataset_name':name,'file':'test'}}})
   denied=json.loads(result['content'][0]['text'])
   checks['actual_mcp_final_truth_denied']=not result.get('isError',False) and denied['status']=='unavailable'
   if sealed:
    result=request({'jsonrpc':'2.0','id':5,'method':'tools/call','params':{
     'name':'load_dataset','arguments':{'dataset_name':name,'split':'test'}}})
    blocked=json.loads(result['content'][0]['text'])
    checks['actual_mcp_final_observations_denied']=not result.get('isError',False) and blocked['status']=='unavailable' and blocked.get('data_path') is None
  finally:
   p.terminate();p.wait(timeout=5)
try:
 if sys.argv[2]=='mcp':mcp_probe()
except Exception as error:
 checks['mcp_error']=False
 print(type(error).__name__+': '+str(error),file=sys.stderr,flush=True)
 print((out/'mcp.stderr').read_text(),file=sys.stderr,flush=True)
print(json.dumps({'checks':checks,'passed':all(checks.values()),'development_metric':metric,
 'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__}))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--jobs',nargs='+')
    p.add_argument('--roles',nargs='+',choices=['code','mcp'],default=['code','mcp'])
    args=p.parse_args();run=args.run.resolve()
    plan=json.loads((run/'PLAN187.json').read_text())['jobs']
    protocol=json.loads((run/'PROTOCOL187.json').read_text())
    def probe_role(job,role):
        folder=run/'audit/isolation'/f"{job['id']}_attempt{args.attempt}_{role}";folder.mkdir(parents=True)
        ns=SimpleNamespace(cli=Path(protocol['cli']),model_gateway_socket=Path(job['model_gateway_socket']),
            runtime_mounts_file=Path(job['mounts']),credential_env_file=None,
            isolation_config=run/'cli_config187.toml',model_catalog=run/'model_catalog187.json',
            permission_profile='iot-offline',workspace_alias='/app')
        inner=['/opt/python312/bin/python3.12','-B','-c',PROBE,job['dataset'],role,str(run/'hidden'),
               str(protocol.get('mcp_tool_timeout_seconds',180)),
               'sealed' if protocol.get('test_inputs_visible',True) is False else 'visible']
        if role=='code':
            inner=[str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--']+inner
        command=isolate_filesystem(inner,ns,run/job['workspace'],folder,None)
        env=clean_env(run)
        env.update(PYTHONPATH='/opt/phm_python/extra',TMPDIR='/app/.runtime/tmp',
                   TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
                   MPLCONFIGDIR='/app/.runtime/mpl',PYTHONDONTWRITEBYTECODE='1',
                   PATH='/opt/python312/bin:/usr/bin:/bin')
        result=subprocess.run(command,env=env,capture_output=True,text=True,
            timeout=protocol.get('mcp_tool_timeout_seconds',180)+120)
        (folder/'stdout.txt').write_text(result.stdout);(folder/'stderr.txt').write_text(result.stderr)
        save(folder/'command.json',command)
        try: value=json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError,IndexError): value={'passed':False,'error':'No valid probe response'}
        value.update(job=job['id'],probe_role=role,exit_code=result.returncode)
        value['passed']=value['passed'] and result.returncode==0
        save(folder/'result.json',value)
        # These researcher-generated development checks are never Agent outputs.
        shutil.rmtree(run/job['workspace']/'artifacts/preflight',ignore_errors=True)
        return value
    def probe(job):return [probe_role(job,role) for role in args.roles]
    selected=[job for job in plan if args.jobs is None or job['id'] in args.jobs]
    assert selected and (args.jobs is None or {j['id'] for j in selected}==set(args.jobs))
    previous=[]
    if len(selected)!=len(plan) or set(args.roles)!={'code','mcp'}:
        previous=json.loads((run/'AGENT_ISOLATION_GATE251.json').read_text())['results']
    with ThreadPoolExecutor(max_workers=3) as pool:
        new_results=[r for rows in pool.map(probe,selected) for r in rows]
    merged={(r['job'],r['probe_role']):r for r in previous}
    merged.update({(r['job'],r['probe_role']):r for r in new_results})
    assert set(merged)=={(j['id'],role) for j in plan for role in ['code','mcp']}
    results=[merged[(j['id'],role)] for j in plan for role in ['code','mcp']]
    gate={'passed':all(r['passed'] for r in results),'results':results,
          'no_model_calls':True,'scope':'actual native code sandbox and actual MCP; not final test scoring'}
    save(run/'AGENT_ISOLATION_GATE251.json',gate)
    print(json.dumps(gate,ensure_ascii=False))
    raise SystemExit(0 if gate['passed'] else 1)


if __name__=='__main__':main()
