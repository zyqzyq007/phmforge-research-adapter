"""One explicit, remaining-budget recovery after a verified provider timeout.

No score-conditioned retry or candidate selection. Preserve the first attempt,
restore the exact initial starter and immutable inputs, retain the old failures.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from acquire_forge_pdmb_sources244 import save, sha256
from run_validation_transfer import execute_wave, freeze, metadata_gate, state


def read(path):
    return json.loads(Path(path).read_text())


def build(root, previous):
    old=previous/'n';oldjob=next(j for j in read(old/'PLAN187.json')['jobs'] if j['task_id']=='pdm_cost_benefit_001')
    prior_status=read(old/'primary'/oldjob['id']/oldjob['id']/'status.json')
    assert prior_status['status']=='execution_error' and prior_status['exit_code']==1
    receipts=sorted((old/'model_gateway'/oldjob['id']).glob('*_receipt.json'))
    last=read(receipts[-1]);assert last.get('error_type')=='URLError' and 'timed out' in last['error']
    assert not (old/'CODING_SCORE.json').exists() and not (previous/'EXTERNAL_SCORE307.json').exists()
    run=root/'n';ident='recovery308_pdm_cost_benefit_001'
    for f in ['audit','logs','code','plans','runtime/tmp','runtime/mpl','clients-cache']:(run/f).mkdir(parents=True,exist_ok=True)
    for f in ['audit','logs','code','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    timeout=int(3600-prior_status['elapsed_seconds']);requests=180-len(receipts)
    assert timeout>0 and requests>0
    protocol=read(old/'PROTOCOL187.json');protocol.update(run_id=run.name,subject_timeout_seconds=timeout,
        max_model_requests_per_subject=requests,max_new_model_sessions=1,local_max_concurrency=1,max_concurrency=5,
        purpose='One source-identical, initial-starter, remaining-budget transport recovery; original failed attempt retained.',
        source_recovery=str(old/'primary'/oldjob['id']/oldjob['id']),automatic_retries=False)
    for f in ['cli_config187.toml','model_catalog187.json','TASK_QUALIFICATION251.json','PUBLIC_METADATA_GATE267.json']:
        shutil.copy2(old/f,run/f)
    shutil.copy2(old/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
    shutil.copytree(old/'public_tools',run/'public_tools')
    for folder in ['public_data','references','evaluation_data']:
        shutil.copytree(old/folder/oldjob['id'],run/folder/ident)
    shutil.copytree(old/'hidden',run/'hidden')
    ws=run/'workspaces'/ident
    for f in ['.runtime/tmp','.runtime/mpl','.runtime/xdg','artifacts']:(ws/f).mkdir(parents=True,exist_ok=True)
    initial='''import argparse,sys
from pathlib import Path
sys.path.insert(0,'/tools')
from real_rul_model import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();infer(Path('/reference/model.pt'),a.input,a.output)
'''
    (ws/'solution.py').write_text(initial)
    original_files={v['path']:v['sha256'] for v in read(old/'INITIAL_FREEZE187_FIFO_FILES.json')['files']}
    assert sha256(ws/'solution.py')==original_files[str(old/oldjob['workspace']/'solution.py')]
    for f in ['AGENTS.md','TASK.txt']:
        source=old/oldjob['workspace']/f;assert sha256(source)==original_files[str(source)]
        shutil.copy2(source,ws/f)
    prompt=run/'plans'/(ident+'.txt')
    prompt.write_text(Path(oldjob['prompt_file']).read_text()+f'\nInfrastructure recovery notice: the previous attempt ended after an upstream connection timeout without a completed submission or final-score feedback. This is a single fresh-context recovery from the exact original initial starter and source inputs. The enforced remaining budget is {timeout}seconds and {requests}model requests; no extra original-task budget is added.\n')
    job=dict(oldjob,id=ident,workspace=str(ws.relative_to(run)),prompt_file=str(prompt),
        model_gateway_socket=str(run/'ipc/s0.sock'))
    for key in ['mounts','evaluation_mounts']:
        values=read(oldjob[key]);new=run/'plans'/(ident+'_'+key+'.json')
        for m in values['mounts']:
            src=Path(m['source'])
            if src.is_relative_to(old):m['source']=str(run/src.relative_to(old)).replace(oldjob['id'],ident)
        save(new,values);job[key]=str(new)
    job['evaluation_test']=str(run/'evaluation_data'/ident/'datasets/CMAPSS_FD001/test.npz')
    native={k:job[k] for k in ['id','family','task_id','condition','repeat','prompt_file','answer_mode']};native['workspace']=str(ws)
    plan=run/'plans'/(ident+'.json');save(plan,{'jobs':[native]});job['plan']=str(plan)
    save(run/'PLAN187.json',{'jobs':[job]});save(run/'PROTOCOL187.json',protocol)
    gateway=read(old/'MODEL_GATEWAY_CONFIG.json');gateway.update(routes=[dict(job=ident,socket=job['model_gateway_socket'])],max_requests=requests)
    save(run/'MODEL_GATEWAY_CONFIG.json',gateway)
    save(run/'audit/AUTHORIZATION251.json',dict(user_reply='把PHMforge能真实实验的全做了；允许 NASA 官方仿真数据，禁止占位返回和伪训练',
        scope='One necessary bounded transport recovery within the original unused budget; no source, model, split or final-score change.',max_new_primary=1))
    save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_isolation'))
    metadata_gate(run);freeze(run)
    save(root/'CAMPAIGN_CONFIG.json',dict(iteration='A308',result_directory='results/recovery308',planned_sessions=1,
        purpose=protocol['purpose'],source_attempt=str(previous),original_task_count=1,not_an_independent_replicate=True,
        session_budget={k:protocol[k] for k in ['subject_timeout_seconds','max_model_requests_per_subject','max_concurrency']}))
    save(root/'RECOVERY_BASIS308.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        original_status=prior_status,transport_receipt=last,original_completed_request_count=len(receipts),
        no_final_scores_opened=True,exact_initial_starter_restored=True,remaining_seconds=timeout,
        remaining_requests=requests,new_source_or_scientific_replication=False,reason='Provider URLError connection timed out; not an Agent diagnostic failure.'))
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Read-only hash-bound runtime retained.')))
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    state(root,status='one_remaining_budget_transport_recovery_prepared',planned_model_sessions=1,task_count=1,final_scoring_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--phase',choices=['build','run'],required=True);a=p.parse_args();root=a.run.resolve()
    if a.phase=='build':build(root,a.previous.resolve())
    else:execute_wave(root,'transport_recovery',[root/'n'])
