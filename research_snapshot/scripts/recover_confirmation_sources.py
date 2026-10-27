"""One bounded, rate-limited recovery of A338's unchanged physical records."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from controlled_feedback import save,sha,utc

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
INITIAL=Path('/data/yyf-zyq/PHM_20261010/20261024T183500Z_confirmsource338')


def read(path):return json.loads(Path(path).read_text())


def environment(root):
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['KEY','TOKEN','SECRET','PASSWORD'])}
    env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),MPLCONFIGDIR=str(root/'runtime/mpl'),
        PHM_RANGE_RATE_FILE=str(root/'runtime/range_rate.json'),PHM_RANGE_INTERVAL_SECONDS='2',PHM_RANGE_CACHE_INDEX=str(root/'source/SEED_RANGE_INDEX338.json'))
    return env


def prepare(root):
    assert not root.exists();receipt=read(INITIAL/'ARCHIVE_VERIFIED.json');manifest=INITIAL/receipt['manifest_filename'];assert sha(manifest)==receipt['manifest_sha256']
    inventory={x['path']:x for x in read(manifest)['files']}
    for relative in ['PROTOCOL338.json','SOURCE_SELECTION338.json','audit/ONE_MIB_TRANSPORT_PROBE338.json']:
        assert sha(INITIAL/relative)==inventory[relative]['sha256']
    assert read(INITIAL/'audit/ONE_MIB_TRANSPORT_PROBE338.json')['passed']
    root.mkdir(parents=True)
    for name in ['code','audit','logs','records','cases','runtime/tmp','runtime/mpl']:(root/name).mkdir(parents=True)
    shutil.copytree(INITIAL/'source',root/'source')
    for name in ['PROTOCOL332.json','SOURCE_SELECTION338.json','retention_policy.json']:shutil.copy2(INITIAL/name,root/name)
    for folder in (INITIAL/'records').iterdir():shutil.copytree(folder,root/'records'/folder.name)
    for p in (INITIAL/'logs').glob('index_*.json'):shutil.copy2(p,root/'logs'/p.name)
    cfg=read(INITIAL/'PROTOCOL338.json');cfg.update(iteration='A338-R1',frozen_utc=utc(),recovery_of=str(INITIAL),source_max_parallel=4,range_chunk_bytes=1048576,
        total_source_wall_seconds=5400,global_request_interval_seconds=2,HTTP429_policy='Honor full Retry-After and reset across all workers;two bounded request attempts.',
        recovery_attempt_limit=1,raw_selection_and_role_assignments_unchanged=True,new_LLM_budget=0,new_fit_budget=0,new_prediction_budget=0,
        rate_policy_source='https://developers.zenodo.org/#rate-limiting',actual_endpoint_limit_observed=133,
        cache_integrity='Every reused range must match the archived manifest,exact start/length and saved SHA;complete records must pass author CRC and decoded length.')
    save(root/'PROTOCOL338.json',cfg);save(root/'audit/INITIAL_ARCHIVE_BINDING338.json',receipt)
    ranges={};seed_bytes=0
    for p in INITIAL.glob('cases/*/records/*/RECEIPT.json'):
        for record in read(p)['requests']:
            if record.get('status')!='saved':continue
            f=p.parent/'ranges'/record['file'];relative=str(f.relative_to(INITIAL));assert sha(f)==record['sha256']==inventory[relative]['sha256'] and f.stat().st_size==record['bytes']
            start,end=map(int,record['range'][6:].split('-'));assert end-start+1==record['bytes']
            entry=dict(path=str(f),bytes=record['bytes'],sha256=record['sha256'])
            if str(start) in ranges:assert ranges[str(start)]==entry
            else:ranges[str(start)]=entry;seed_bytes+=record['bytes']
    save(root/'source/SEED_RANGE_INDEX338.json',dict(url=cfg['source_url'],outer_bytes=cfg['outer_bytes'],archive_manifest_sha256=receipt['manifest_sha256'],bytes=seed_bytes,ranges=ranges))
    for case in cfg['cases']:
        child=root/'cases'/case['id'];(child/'logs').mkdir(parents=True);(child/'records').mkdir();shutil.copytree(INITIAL/'cases'/case['id']/'source',child/'source')
        protocol=read(INITIAL/'cases'/case['id']/'PROTOCOL338.json');protocol.update(range_chunk_bytes=1048576,source_max_parallel=4)
        save(child/'PROTOCOL338.json',protocol);shutil.copy2(child/'PROTOCOL338.json',child/'PROTOCOL333.json')
    for name in ['recover_confirmation_sources.py','prepare_confirmation_sources.py','verified_range_recovery.py','comfaulda_source.py','fetch_zip_members.py','prepare_comfaulda_study.py','qualify_program_source.py','controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    transport=root/'code/prepare_comfaulda_study.py';text=transport.read_text();assert text.count('from fetch_zip_members import RangeReader')==1
    transport.write_text(text.replace('from fetch_zip_members import RangeReader','from verified_range_recovery import RecoveringRangeReader as RangeReader'))
    save(root/'CODE_FREEZE338.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    save(PROJECT/'results/confirmsource338/RECOVERY_PROTOCOL338.json',cfg)
    print(json.dumps(dict(run=str(root),verified_reusable_range_bytes=seed_bytes,source_records=72,new_model_calls=0)))


def smoke(root):
    assert not (root/'PREFLIGHT338.json').exists();row=read(root/'SOURCE_SELECTION338.json')['records'][0];case=root/'cases'/row['case']
    command=[sys.executable,'-B',str(root/'code/prepare_comfaulda_study.py'),'--run',str(case),'--unit',row['unit_id']]
    result=subprocess.run(command,env=environment(root),capture_output=True,text=True,timeout=300)
    save(root/'logs/PREFLIGHT338.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,command=command))
    assert result.returncode==0,result.stderr[-2400:]
    receipt=read(case/'records'/row['unit_id']/'RECEIPT.json');assert receipt['passed'] and receipt['CSV_CRC_verified']
    save(root/'PREFLIGHT338.json',dict(passed=True,utc=utc(),case=case.name,unit=row['unit_id'],complete_author_CRC_verified=True,
        raw_sha256=receipt['sha256'],new_network_bytes=receipt['received_bytes'],reused_verified_bytes=sum(x.get('bytes',0) for x in receipt['requests'] if x.get('status')=='seed_verified'),
        new_LLM_calls=0,new_fits=0,new_predictions=0))
    print(json.dumps(read(root/'PREFLIGHT338.json')))


def capture(root):
    assert read(root/'PREFLIGHT338.json')['passed']
    for name,digest in read(root/'CODE_FREEZE338.json').items():assert sha(root/'code'/name)==digest
    os.environ.update(environment(root))
    from prepare_confirmation_sources import capture as bounded_capture
    bounded_capture(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','capture']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run)
