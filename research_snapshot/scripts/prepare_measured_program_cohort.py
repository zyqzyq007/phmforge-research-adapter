"""One small, prespecified, measured development cohort;no performance access."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone

from controlled_feedback import save,sha,utc
from run_frozen_candidate_review import source_context

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T152400Z_comfaulda332')
OUT=PROJECT/'results/programsource341'
ROLES=dict(train=[18,28],development=[38],independent_check=[54],final=[44])
HISTORY=[(SOURCE,'PROTOCOL332.json'),
 (Path('/data/yyf-zyq/PHM_20261010/20261024T155100Z_validation333'),'PROTOCOL333.json'),
 (Path('/data/yyf-zyq/PHM_20261010/20261024T172000Z_programsource336'),'PROTOCOL336.json'),
 (Path('/data/yyf-zyq/PHM_20261010/20261024T191000Z_source338r1'),'SOURCE_SELECTION338.json')]


def read(path):return json.loads(Path(path).read_text())


def prepare(root,spec_path=None):
    spec=read(spec_path) if spec_path else {}
    tag=str(spec.get('tag','341'));assert tag.isdigit()
    roles=spec.get('roles_Hz',ROLES)
    assert roles and all(isinstance(v,list) and v for v in roles.values())
    targets=[x for values in roles.values() for x in values]
    assert len(targets)==len(set(targets)) and all(isinstance(x,int) and x>0 for x in targets)
    out=PROJECT/'results'/spec.get('result_key','programsource341')
    assert out.parent==PROJECT/'results'
    history=HISTORY+[(Path(x['archive']),x['file']) for x in spec.get('additional_history',[])]
    root.mkdir(exist_ok=False)
    for name in ['code','logs','audit','records','source','runtime/tmp','runtime/mpl']:(root/name).mkdir(parents=True)
    inventory,receipt=source_context(SOURCE)
    prior=read(SOURCE/'PROTOCOL332.json');assert sha(SOURCE/'PROTOCOL332.json')==inventory['PROTOCOL332.json']['sha256']
    used=set();used_crc_sizes=set();bindings=[]
    for archive,relative in history[1:]:
        items,entry=source_context(archive);assert sha(archive/relative)==items[relative]['sha256'];payload=read(archive/relative)
        history_rows=payload.get('selected_records',payload.get('records',[]))
        used.update((r['outer_member'],r['path']) for r in history_rows)
        used_crc_sizes.update((r['CRC32'],int(r['bytes'])) for r in history_rows if 'CRC32' in r and 'bytes' in r)
        bindings.append(dict(archive=str(archive),manifest_sha256=entry['manifest_sha256'],file=relative))
    configs=[];selected=[]
    for member in prior['pilot_outer_members']:
        name=member['class_name']
        if name not in ['normal','unbalance','horizontal','vertical']:continue
        label=['normal','unbalance','horizontal','vertical'].index(name)
        prefix='records/'+name+'/'
        for relative in [prefix+'index/nested_index.json',prefix+'fetch/RECEIPT.json']:
            assert sha(SOURCE/relative)==inventory[relative]['sha256']
        index=read(SOURCE/(prefix+'index/nested_index.json'));old=read(SOURCE/(prefix+'fetch/RECEIPT.json'))
        configs.append(dict(class_name=name,label=label,outer_member=member['path'],nested_data_offset=old['nested_data_offset']))
        for role,conditions in roles.items():
            for target in conditions:
                candidates=[]
                for row in index:
                    try:rotation=float(Path(row['path']).stem)
                    except ValueError:continue
                    if row['path'].lower().endswith('.csv') and math.isfinite(rotation) and abs(rotation-target)<=1:
                        candidates.append(row)
                assert candidates,(name,target)
                row=min(candidates,key=lambda r:(abs(float(Path(r['path']).stem)-target),r['path']))
                assert (member['path'],row['path']) not in used,(name,target,'previously used or reserved')
                assert (row['CRC32'],int(row['bytes'])) not in used_crc_sizes,(name,target,'known raw-content metadata alias;different filename is insufficient')
                selected.append(dict(unit_id=name+'_'+str(target),class_name=name,label=label,role=role,target_condition_Hz=target,
                    recorded_rotation_Hz=float(Path(row['path']).stem),outer_member=member['path'],nested_data_offset=old['nested_data_offset'],
                    reuse_qualified_pilot=False,**row))
    expected=4*len(targets)
    assert len(selected)==expected and len({(r['outer_member'],r['path']) for r in selected})==expected
    # Include the three compound pilot records in the disjointness ledger too.
    for name in ['unbalance_horizontal','unbalance_vertical','vertical_horizontal']:
        p=SOURCE/'records'/name/'fetch/RECEIPT.json'
        if p.exists():
            assert sha(p)==inventory[str(p.relative_to(SOURCE))]['sha256']
            assert not any(r['path']==read(p)['original_member'] and r['class_name']==name for r in selected)
    url='https://zenodo.org/api/records/8208454/files/COMFAULDA.zip/content'
    cfg=dict(iteration='A'+tag,tag=tag,result_key=out.name,frozen_utc=utc(),source_archive=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],
        source_url=url,outer_bytes=22995655942,configs=configs,selected_records=selected,roles_Hz=roles,
        qualified_roles=list(roles),expected_records=expected,
        selection='Nearest numeric filename within1Hz,lexical tie,all4physical-label classes;exclude old/reserved names and known CRC/size aliases;no task-score based changes. Full raw SHA admission still required.',
        window_points=25600,sensor_columns_zero_based=[2,3,4,5,6,7],range_chunk_bytes=1048576,
        per_new_record_network_cap_bytes=32*1024**2,raw_decoded_cap_bytes=40*1024**2,
        source_max_parallel=2,global_request_interval_seconds=2,per_new_record_wall_seconds=1200,total_wall_seconds=2400,
        total_admitted_network_cap_bytes=640*1024**2,maximum_disk_GiB=3,
        new_LLM_budget=0,new_fit_budget=0,new_model_prediction_budget=0,physical_source_clusters=1,
        record_selection_disjoint_from_all_prior_and_reserved=True,prior_bindings=bindings,
        purpose=spec.get('purpose','New development cohort for controlled complete-program edits and common build/choice budget qualification;not independent-machine confirmation.'),
        future_method_not_yet_frozen=True,source_only_qualification=True,
        transport='Globally rate-limited1MiB requests using prior source-qualified transport;honor Retry-After,2attempts;preserve failures,no role substitution.')
    allowed={'per_new_record_network_cap_bytes','total_admitted_network_cap_bytes','maximum_disk_GiB','total_wall_seconds','range_chunk_bytes'}
    overrides=spec.get('source_budget_overrides',{});assert set(overrides)<=allowed
    cfg.update(overrides)
    assert all(isinstance(v,(int,float)) and v>0 for v in overrides.values())
    cfg['future_method_not_yet_frozen']=spec.get('future_method_not_yet_frozen',True)
    cfg['frozen_method_file']=spec.get('frozen_method_file','FROZEN_CONTEXT_AUDIT.json')
    save(root/('PROTOCOL'+tag+'.json'),cfg);save(root/'PROTOCOL333.json',cfg)
    save(root/'SOURCE_CONFIG.json',dict(tag=tag,protocol='PROTOCOL'+tag+'.json',result_key=out.name))
    save(root/('source/EMPTY_CACHE'+tag+'.json'),dict(url=url,outer_bytes=cfg['outer_bytes'],ranges={}))
    names=['prepare_measured_program_cohort','prepare_comfaulda_study','verified_range_recovery','fetch_zip_members','qualify_program_source',
           'controlled_feedback','run_frozen_candidate_review','run_full_benchmark_campaign','prepare_fault_extensions','acquire_forge_pdmb_sources244']
    for name in names:shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    save(root/('CODE_FREEZE'+tag+'.json'),{p.name:sha(p) for p in (root/'code').glob('*.py')})
    shutil.copy2(SOURCE/'retention_policy.json',root/'retention_policy.json')
    save(root/'OWNERSHIP.json',dict(run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name,user_request='围绕错误替换有效PHM程序持续迭代真实实验',created_utc=utc()))
    save(out/('PROTOCOL'+tag+'.json'),cfg);save(out/'current_run.json',dict(run=str(root),status='prepared',planned_records=expected,new_fits=0))
    print(json.dumps(dict(run=str(root),planned=expected,prior_or_reserved_overlaps=0)))


def capture(root):
    meta=read(root/'SOURCE_CONFIG.json') if (root/'SOURCE_CONFIG.json').exists() else dict(tag='341',protocol='PROTOCOL341.json',result_key='programsource341')
    tag=meta['tag'];out=PROJECT/'results'/meta['result_key']
    for name,digest in read(root/('CODE_FREEZE'+tag+'.json')).items():assert sha(root/'code'/name)==digest
    cfg=read(root/meta['protocol']);expected=len(cfg['selected_records']);started=time.monotonic()
    os.environ.update(TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),MPLCONFIGDIR=str(root/'runtime/mpl'),
        PHM_RANGE_RATE_FILE=str(root/'runtime/rate.json'),PHM_RANGE_INTERVAL_SECONDS='2',PHM_RANGE_CACHE_INDEX=str(root/('source/EMPTY_CACHE'+tag+'.json')))
    save(root/('HANDLE'+tag+'.json'),dict(pid=os.getpid(),status='running',phase='real_source_capture',new_model_calls=0))
    def fetch(row):
        try:
            remaining=cfg['total_wall_seconds']-(time.monotonic()-started);assert remaining>0
            command=[sys.executable,'-B','-c',
                'import sys;from pathlib import Path;import prepare_comfaulda_study as p;from verified_range_recovery import RecoveringRangeReader;p.RangeReader=RecoveringRangeReader;p.fetch(Path(sys.argv[1]),sys.argv[2])',
                str(root),row['unit_id']]
            env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['KEY','TOKEN','SECRET','PASSWORD'])}
            env['PYTHONPATH']=str(root/'code')+':'+env.get('PYTHONPATH','')
            process=subprocess.run(command,env=env,capture_output=True,text=True,timeout=min(remaining,cfg['per_new_record_wall_seconds']))
            save(root/'logs'/(row['unit_id']+'.json'),dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
            assert process.returncode==0,process.stderr[-1600:]
            receipt=read(root/'records'/row['unit_id']/'RECEIPT.json')
            return dict(unit=row['unit_id'],passed=receipt['passed'],sha256=receipt['sha256'])
        except Exception as error:return dict(unit=row['unit_id'],passed=False,error_type=type(error).__name__,error=str(error))
    results=[]
    with ThreadPoolExecutor(max_workers=cfg['source_max_parallel']) as pool:
        for f in as_completed([pool.submit(fetch,row) for row in cfg['selected_records']]):
            results.append(f.result());save(root/('SOURCE_PROGRESS'+tag+'.json'),results)
    passed=sum(r['passed'] for r in results);save(root/('HANDLE'+tag+'.json'),dict(pid=os.getpid(),status='terminal',passed=passed,planned=expected,new_LLM_calls=0,new_fits=0,new_predictions=0))
    qualification=None
    if passed==expected:
        from qualify_program_source import main
        main(root,meta['protocol'],'HANDLE'+tag+'.json',tag);qualification=read(root/('QUALIFICATION'+tag+'.json'))
    result=dict(run=str(root),status='qualified' if qualification else 'terminal_source_failure',passed=passed,planned=expected,
        seconds=time.monotonic()-started,qualification=qualification,new_LLM_calls=0,new_fits=0,new_predictions=0)
    save(root/('RESULT'+tag+'.json'),result);save(out/'current_run.json',result)
    print(json.dumps(dict(status=result['status'],passed=passed,seconds=result['seconds'])))


def recover(root):
    """One exact-byte retry of failed source ranges within the original budgets.

    Preserve the frozen source/helper code, all successful records and original
    failures. No prediction is admitted until every original record qualifies.
    """
    meta=read(root/'SOURCE_CONFIG.json');tag=meta['tag'];out=PROJECT/'results'/meta['result_key']
    cfg=read(root/meta['protocol']);old=read(root/('RESULT'+tag+'.json'))
    assert old['status']=='terminal_source_failure'
    assert not (root/'SOURCE_RECOVERY_ADMISSION.json').exists(),'One recovery only'
    assert not (root/'CONTEXT_EVALUATION_STARTED.json').exists()
    for name,digest in read(root/('CODE_FREEZE'+tag+'.json')).items():assert sha(root/'code'/name)==digest
    failed=[r for r in cfg['selected_records'] if not read(root/'records'/r['unit_id']/'RECEIPT.json')['passed']]
    assert failed
    # Retrying only transient HTTP5xx refusals;corrupt/mismatched206bytes fail closed.
    for row in failed:
        requests=read(root/'records'/row['unit_id']/'RECEIPT.json')['requests']
        last=requests[-1];assert last.get('status_code') in [500,502,503,504] and last.get('attempt')==1
    deadline=datetime.fromisoformat(read(root/'OWNERSHIP.json')['created_utc']).timestamp()+cfg['total_wall_seconds']
    assert time.time()<deadline
    keep=root/'attempts/source_initial';keep.mkdir(parents=True)
    for filename in ['HANDLE'+tag+'.json','SOURCE_PROGRESS'+tag+'.json','RESULT'+tag+'.json']:
        shutil.copy2(root/filename,keep/filename)
    seeds={};prior_bytes={};reused=0
    for row in failed:
        folder=root/'records'/row['unit_id'];receipt=read(folder/'RECEIPT.json')
        prior_bytes[row['unit_id']]=receipt['received_bytes']
        shutil.move(str(folder),str(keep/row['unit_id']))
        for item in receipt['requests']:
            if item.get('status')!='saved':continue
            p=keep/row['unit_id']/'ranges'/item['file'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
            start,end=map(int,item['range'][6:].split('-'));assert end-start+1==item['bytes']
            assert str(start) not in seeds
            seeds[str(start)]=dict(path=str(p),bytes=item['bytes'],sha256=item['sha256']);reused+=item['bytes']
    seed=root/'source/RECOVERY_SEED_INDEX.json';save(seed,dict(url=cfg['source_url'],outer_bytes=cfg['outer_bytes'],ranges=seeds))
    driver=root/'audit/SOURCE_RECOVERY_DRIVER.py';shutil.copy2(Path(__file__),driver)
    save(root/'SOURCE_RECOVERY_ADMISSION.json',dict(utc=utc(),failed_units=[r['unit_id'] for r in failed],
        successful_records_not_repeated=True,physical_selection_unchanged=True,model_weights_unchanged=True,
        original_protocol_sha256=sha(root/meta['protocol']),original_frozen_context_sha256=sha(root/cfg.get('frozen_method_file','FROZEN_CONTEXT_AUDIT.json')),
        driver_sha256=sha(driver),reused_partial_bytes=reused,prior_record_network_bytes=prior_bytes,
        original_global_deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        recovery_attempts=1,max_upstream_attempts_per_recovered_range=1,new_LLM_calls=0,new_fits=0,new_predictions=0))
    os.environ.update(TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),
        PHM_RANGE_RATE_FILE=str(root/'runtime/rate.json'),PHM_RANGE_INTERVAL_SECONDS='2',PHM_RANGE_CACHE_INDEX=str(seed))
    save(root/('HANDLE'+tag+'.json'),dict(pid=os.getpid(),status='running',phase='single_exact_byte_source_recovery'))
    snippet='''import sys,os
from pathlib import Path
import prepare_comfaulda_study as p
from verified_range_recovery import RecoveringRangeReader
class OnePass(RecoveringRangeReader):
 def __init__(self,*args,**kwargs):
  args=list(args);args[3]-=int(os.environ['PHM_PRIOR_RECORD_BYTES']);kwargs['max_request_attempts']=1
  super().__init__(*args,**kwargs)
p.RangeReader=OnePass
p.fetch(Path(sys.argv[1]),sys.argv[2])
'''
    save(root/'audit/SOURCE_RECOVERY_CHILD_CODE.json',dict(code=snippet))
    def retry(row):
        unit=row['unit_id'];env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['KEY','TOKEN','SECRET','PASSWORD'])}
        env['PYTHONPATH']=str(root/'code')+':'+env.get('PYTHONPATH','');env['PHM_PRIOR_RECORD_BYTES']=str(prior_bytes[unit])
        try:
            result=subprocess.run([sys.executable,'-B','-c',snippet,str(root),unit],env=env,capture_output=True,text=True,
                timeout=max(1,min(deadline-time.time(),cfg['per_new_record_wall_seconds']-old['seconds'])))
            save(root/'logs'/(unit+'_recovery.json'),dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            receipt=read(root/'records'/unit/'RECEIPT.json')
            return dict(unit=unit,passed=receipt['passed'],new_network_bytes=receipt['received_bytes'])
        except Exception as error:
            save(root/'logs'/(unit+'_recovery_exception.json'),dict(error_type=type(error).__name__,error=str(error)))
            if not (root/'records'/unit/'RECEIPT.json').exists():
                save(root/'records'/unit/'RECEIPT.json',dict(passed=False,received_bytes=cfg['per_new_record_network_cap_bytes']-prior_bytes[unit],
                    received_bytes_is_conservative_cap=True,error_type=type(error).__name__,error=str(error)))
            return dict(unit=unit,passed=False,error_type=type(error).__name__,error=str(error))
    progress=[]
    with ThreadPoolExecutor(max_workers=cfg['source_max_parallel']) as pool:
        for future in as_completed([pool.submit(retry,row) for row in failed]):
            progress.append(future.result());save(root/'SOURCE_RECOVERY_PROGRESS.json',progress)
    receipts=[read(root/'records'/row['unit_id']/'RECEIPT.json') for row in cfg['selected_records']]
    passed=sum(r['passed'] for r in receipts);new_bytes=sum(r['received_bytes'] for r in receipts)+sum(prior_bytes.values())
    assert new_bytes<=cfg['total_admitted_network_cap_bytes'] and time.time()<=deadline
    save(root/('HANDLE'+tag+'.json'),dict(status='terminal',passed=passed,planned=len(receipts),new_LLM_calls=0,new_fits=0))
    qualification=None
    if passed==len(receipts):
        from qualify_program_source import main
        main(root,meta['protocol'],'HANDLE'+tag+'.json',tag);qualification=read(root/('QUALIFICATION'+tag+'.json'))
    result=dict(status='qualified' if qualification else 'terminal_source_failure',passed=passed,planned=len(receipts),
        qualification=qualification,physical_selection_unchanged=True,new_LLM_calls=0,new_fits=0,new_predictions=0,
        initial_and_recovery_network_bytes=new_bytes,reused_partial_bytes=reused,original_failures_preserved=True)
    save(root/'SOURCE_RECOVERY_RESULT.json',result);save(out/'current_run.json',dict(result,run=str(root)))
    print(json.dumps({k:v for k,v in result.items() if k!='qualification'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','capture','recover']);p.add_argument('--run',required=True,type=Path);p.add_argument('--spec',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.run.resolve(),a.spec)
    else:globals()[a.action](a.run.resolve())
