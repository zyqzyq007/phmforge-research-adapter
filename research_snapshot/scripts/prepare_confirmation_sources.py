"""A338: source-only qualification of previously reserved real acquisitions.

No model fitting, Agent calls, candidate ranking, or task-performance scoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib

from controlled_feedback import save,sha,utc

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T152400Z_comfaulda332')
PRIOR=Path('/data/yyf-zyq/PHM_20261010/20261024T155100Z_validation333')
OUT=PROJECT/'results/confirmsource338'
ROLES=[dict(train=[14,40],development=[20],independent_check=[50],final=[26,34]),
       dict(train=[14,26],development=[20],independent_check=[32],final=[38,44]),
       dict(train=[32,44],development=[38],independent_check=[50],final=[20,26])]


def read(path):return json.loads(Path(path).read_text())


def rotation(row):
    try:value=float(Path(row['path']).stem)
    except ValueError:return None
    return value if math.isfinite(value) else None


def freeze(root):
    assert not root.exists();root.mkdir(parents=True)
    for name in ['source','code','logs','records','cases','runtime/tmp','runtime/mpl']:(root/name).mkdir(parents=True)
    for archive,relative in [(SOURCE,'source/outer_zip_index.json'),(PRIOR,'CONFIRMATION_RESERVATION333.json')]:
        receipt=read(archive/'ARCHIVE_VERIFIED.json');manifest=archive/receipt['manifest_filename'];assert sha(manifest)==receipt['manifest_sha256']
        entry=next(x for x in read(manifest)['files'] if x['path']==relative);assert sha(archive/relative)==entry['sha256']
    prior=read(PRIOR/'CONFIRMATION_RESERVATION333.json');shutil.copy2(PRIOR/'CONFIRMATION_RESERVATION333.json',root/'source/RESERVATION333.json')
    shutil.copy2(SOURCE/'source/outer_zip_index.json',root/'source/outer_zip_index.json')
    members=[];cases=[]
    for j,panel in enumerate(prior['panels'],1):
        role=ROLES[j-1];assert sorted(role['train']+role['development']+role['independent_check'])==sorted(panel['training_target_Hz']) and role['final']==panel['final_target_Hz']
        for label,(name,member) in enumerate(zip(['normal','unbalance','horizontal','vertical'],panel['source_members'])):
            members.append(dict(class_name='c'+str(j)+'_'+name,path=member,label=label,case='c'+str(j),semantic_class=name))
        cases.append(dict(id='c'+str(j),members=panel['source_members'],roles_Hz=role,physical_rig='same COMFAULDA rig',planned_records=24))
    url='https://zenodo.org/api/records/8208454/files/COMFAULDA.zip/content'
    save(root/'source/zip_config.json',dict(url=url,bytes=22995655942,network_route='direct',range_chunk_bytes=524288))
    save(root/'PROTOCOL332.json',dict(pilot_outer_members=members,per_nested_metadata_download_budget_bytes=1024**2,per_raw_download_budget_bytes=32*1024**2,per_raw_decoded_budget_bytes=40*1024**2))
    protocol=dict(iteration='A338',frozen_utc=utc(),goal='program_update_iteration_goal330',phase='source qualification only',
        reservation_sha256=sha(PRIOR/'CONFIRMATION_RESERVATION333.json'),source_url=url,outer_bytes=22995655942,cases=cases,
        source_max_parallel=12,index_max_parallel=6,index_seconds=120,record_seconds=1200,record_download_budget_bytes=32*1024**2,
        raw_decoded_cap_bytes=40*1024**2,total_source_wall_seconds=3600,total_admitted_network_cap_bytes=2450*1024**2,max_disk_GiB=12,
        window_points=25600,sensor_columns_zero_based=[2,3,4,5,6,7],raw_data='author-acquired controlled laboratory faults,not simulated waveforms',
        selection='Nearest numeric CSV stem within1Hz per pre-reserved nominal condition;lexical tie;all4classes;no score-based exclusions.',
        total_planned_records=72,new_LLM_budget=0,new_fit_budget=0,new_prediction_budget=0,
        future_confirmation_method_frozen=False,full_confirmation_panel_frozen=False,independent_source_clusters=1,
        independence='Physical source files and role assignments must be disjoint;three severity cohorts remain correlated on one rig,not three independent machines.',
        no_method_tuning_on_these_data=True,no_task_scores_before_method_and_full_comparison_freeze=True,
        failure_policy='One bounded attempt per indexed member and record;preserve incomplete cohorts,do not replace based on outcomes.')
    save(root/'PROTOCOL338.json',protocol)
    for name in ['prepare_confirmation_sources.py','comfaulda_source.py','fetch_zip_members.py','prepare_comfaulda_study.py','qualify_program_source.py','controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    save(root/'CODE_FREEZE338.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    shutil.copy2(SOURCE/'retention_policy.json',root/'retention_policy.json')
    OUT.mkdir(exist_ok=True);save(OUT/'PROTOCOL338.json',protocol)


def capture(root):
    cfg=read(root/'PROTOCOL338.json');started=time.monotonic();deadline=started+cfg['total_source_wall_seconds'];members=read(root/'PROTOCOL332.json')['pilot_outer_members']
    env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),MPLCONFIGDIR=str(root/'runtime/mpl'))
    handle=dict(pid=os.getpid(),status='running',phase='source_index',new_LLM_calls=0,new_fits=0,new_predictions=0);save(root/'HANDLE338.json',handle)
    def execute(command,log,limit):
        if time.monotonic()>=deadline:
            row=dict(returncode=None,error='global_source_budget_exhausted');save(log,row);return row
        try:
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=max(1,min(limit,deadline-time.monotonic())))
            row=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired:row=dict(returncode=None,error='bounded_source_timeout')
        save(log,row);return row
    def index_one(member):
        folder=root/'records'/member['class_name']/'index';receipt=folder/'RECEIPT.json'
        if receipt.exists() and read(receipt).get('passed') and (folder/'nested_index.json').exists():
            prior=read(root/'logs'/('index_'+member['class_name']+'.json'));assert prior['returncode']==0
            return dict(prior,reused_completed_index=True)
        return execute([sys.executable,'-B',str(root/'code/comfaulda_source.py'),'index','--run',str(root),'--class-name',member['class_name']],root/'logs'/('index_'+member['class_name']+'.json'),cfg['index_seconds'])
    index_rows=[]
    with ThreadPoolExecutor(max_workers=cfg['index_max_parallel']) as pool:
        futures={pool.submit(index_one,m):m for m in members}
        for f in as_completed(futures):index_rows.append(dict(member=futures[f]['class_name'],**f.result()));save(root/'INDEX_PROGRESS338.json',index_rows)
    if not all(x['returncode']==0 for x in index_rows):
        handle.update(status='terminal_index_failure');save(root/'HANDLE338.json',handle);return
    jobs=[]
    for case in cfg['cases']:
        child=root/'cases'/case['id']
        for name in ['records','logs','source']:(child/name).mkdir(parents=True,exist_ok=True)
        configs=[];selected=[]
        for m in [x for x in members if x['case']==case['id']]:
            source=root/'records'/m['class_name']/'index';receipt=read(source/'RECEIPT.json');assert receipt['passed']
            index=read(source/'nested_index.json');shutil.copy2(source/'nested_index.json',child/'source'/(m['semantic_class']+'_index.json'))
            save(child/'source'/(m['semantic_class']+'_unusable_filename_metadata.json'),[x for x in index if x['path'].lower().endswith('.csv') and rotation(x) is None])
            configs.append(dict(class_name=m['semantic_class'],label=m['label'],outer_member=m['path'],nested_data_offset=receipt['nested_data_offset'],index_file='source/'+m['semantic_class']+'_index.json',index_sha256=sha(source/'nested_index.json')))
            for role,conditions in case['roles_Hz'].items():
                for target in conditions:
                    candidates=[x for x in index if x['path'].lower().endswith('.csv') and rotation(x) is not None and abs(rotation(x)-target)<=1]
                    assert candidates,(case['id'],m['class_name'],target)
                    item=min(candidates,key=lambda x:(abs(float(Path(x['path']).stem)-target),x['path']))
                    selected.append(dict(unit_id=m['semantic_class']+'_'+str(target),class_name=m['semantic_class'],label=m['label'],role=role,target_condition_Hz=target,
                        recorded_rotation_Hz=float(Path(item['path']).stem),outer_member=m['path'],nested_data_offset=receipt['nested_data_offset'],reuse_qualified_pilot=False,**item))
        assert len(selected)==24 and len({(x['outer_member'],x['path']) for x in selected})==24
        protocol=dict(cfg,configs=configs,selected_records=selected,source_archive=str(SOURCE),range_chunk_bytes=cfg.get('range_chunk_bytes',524288),
            per_new_record_network_cap_bytes=cfg['record_download_budget_bytes'],per_new_record_wall_seconds=cfg['record_seconds'])
        save(child/'PROTOCOL338.json',protocol);shutil.copy2(child/'PROTOCOL338.json',child/'PROTOCOL333.json')
        jobs.extend((child,row) for row in selected)
    selected_records=[dict(case=c.name,**r) for c,r in jobs]
    if (root/'SOURCE_SELECTION338.json').exists():
        assert read(root/'SOURCE_SELECTION338.json')['records']==selected_records,'Recovery must preserve every originally selected physical record'
        save(root/'RECORD_SELECTION_PARITY338.json',dict(utc=utc(),passed=True,records=len(selected_records),original_selection_sha256=sha(root/'SOURCE_SELECTION338.json')))
    else:save(root/'SOURCE_SELECTION338.json',dict(frozen_utc=utc(),records=selected_records,raw_record_retrieval_started=False))
    handle['phase']='raw_source_capture';save(root/'HANDLE338.json',handle);rows=[]
    def fetch_one(child,row):
        receipt=child/'records'/row['unit_id']/'RECEIPT.json'
        if receipt.exists() and read(receipt).get('passed'):
            saved=read(receipt);raw=Path(saved['raw_path']).read_bytes()
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==saved['sha256'] and f'{zlib.crc32(raw):08x}'==row['CRC32']
            return dict(returncode=0,reused_completed_recovery_preflight=True,stdout='',stderr='')
        return execute([sys.executable,'-B',str(root/'code/prepare_comfaulda_study.py'),'--run',str(child),'--unit',row['unit_id']],child/'logs'/(row['unit_id']+'.json'),cfg['record_seconds'])
    with ThreadPoolExecutor(max_workers=cfg['source_max_parallel']) as pool:
        futures={pool.submit(fetch_one,c,r):(c,r) for c,r in jobs}
        for f in as_completed(futures):
            c,row=futures[f];rows.append(dict(case=c.name,unit=row['unit_id'],**f.result()));save(root/'SOURCE_PROGRESS338.json',rows)
    qualifications=[]
    for case in cfg['cases']:
        child=root/'cases'/case['id'];rr=[x for x in rows if x['case']==case['id']];passed=sum(x['returncode']==0 for x in rr)
        save(child/'HANDLE338.json',dict(status='terminal',passed=passed,completed=len(rr)))
        if passed==24:
            result=execute([sys.executable,'-B',str(root/'code/qualify_program_source.py'),'--run',str(child),'--protocol','PROTOCOL338.json','--handle','HANDLE338.json','--tag','338'],child/'logs/qualification.json',600)
            qualifications.append(dict(case=case['id'],passed=result['returncode']==0))
        else:qualifications.append(dict(case=case['id'],passed=False,source_records_completed=passed))
    handle.update(status='terminal',phase='source_qualification',source_records_passed=sum(x['returncode']==0 for x in rows),qualifications=qualifications,seconds=time.monotonic()-started)
    save(root/'HANDLE338.json',handle);save(OUT/'current_run.json',dict(handle,run=str(root)))
    print(json.dumps(handle))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','capture']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run)
