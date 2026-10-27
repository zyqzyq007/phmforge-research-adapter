"""Wait for the finite A321 batch, evaluate once, archive under the shared lock."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from acquire_forge_pdmb_sources244 import save,sha256
from evaluate_forge_gap_recovery import seal,evaluate

PROJECT=Path(__file__).resolve().parents[1]


def main(root):
    public=PROJECT/'results/gap321';deadline=time.monotonic()+4200
    while True:
        status=json.loads((root/'CAMPAIGN_STATUS.json').read_text())
        if status['status']=='gap_recovery_terminal_frozen':
            # The producer sets the status before closing its model gateways.
            ready=all((root/c/'MODEL_GATEWAY_STOP').exists() for c in ['l1','l2','i1','i2','i3'])
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('Finite batch finalizer deadline; inspect live tasks, no automatic rerun.')
        time.sleep(5)
    if not (root/'CHOICES_SEALED321.json').exists():seal(root)
    if not (root/'EXTERNAL_RESULTS321.json').exists():evaluate(root)
    result=json.loads((root/'EXTERNAL_RESULTS321.json').read_text())
    simple=[]
    for v in result['variants']:
        if v['id'].startswith('agent_'):
            controls=[w for w in result['variants'] if w['dataset']==v['dataset'] and w['id']=='fixed_hgb100']
            simple.append(dict(task_id=v['task_id'],cohort=v['cohort'],dataset=v['dataset'],status=v['numerical_scoring'],
                agent_mae=v.get('metrics',{}).get('mean_bearing_mae'),starting_mae=controls[0].get('metrics',{}).get('mean_bearing_mae') if controls else None,
                physical_risk_or_full_original_certified=False))
    save(public/'NUMERICAL_SUMMARY321.json',dict(rows=simple,new_confirmed_Agent_hard_problems=0,
        scope='Supplementary source/proxy adaptations. Read traces and competing explanations before mechanistic claims.'))
    # Preserve all scientific files; materialize researcher-created aliases so
    # archive paths remain usable after the live directory has been removed.
    changes=[]
    for c in ['l1','l2','i1','i2','i3']:
        run=root/c
        for name in ['public_tools','references']:
            link=run/name
            if link.is_symlink():
                target=link.resolve();before={str(f.relative_to(target)):sha256(f) for f in target.rglob('*') if f.is_file()}
                link.unlink();shutil.copytree(target,link)
                assert before=={str(f.relative_to(link)):sha256(f) for f in link.rglob('*') if f.is_file()}
                changes.append(dict(path=str(link.relative_to(root)),original_target=str(target),byte_identity_verified=True))
        cache=run/'clients-cache'
        if cache.exists():
            dst=root/'clients-cache'/('closed_'+c);dst.parent.mkdir(exist_ok=True);shutil.move(str(cache),str(dst))
    save(root/'audit/ARCHIVE_MATERIALIZATION321.json',dict(aliases=changes,numerical_submissions_modified=False,
        rebuildable_closed_cli_caches_relocated_under_predeclared_cache_exclusion=True))
    unexpected=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_symlink() and 'clients-cache' not in p.parts]
    assert not unexpected,'Unexpected symlinks: retain live evidence for review: '+repr(unexpected)
    destination=Path('/data/yyf-zyq/PHM_20261010')/root.name
    status=json.loads((public/'current_run.json').read_text());status.update(status='results_ready_archiving',expected_archive=str(destination))
    save(public/'current_run.json',status)
    proc=subprocess.run([sys.executable,'-B',str(PROJECT/'scripts/archive_run_locked.py'),'--run',str(root),
        '--workspace',str(PROJECT),'--destination',str(destination)],capture_output=True,text=True)
    save(public/'ARCHIVE_COMMAND321.json',dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    assert proc.returncode==0,'Archive failed; source preserved'
    receipt=json.loads((destination/'ARCHIVE_VERIFIED.json').read_text());save(public/'ARCHIVE_VERIFIED.json',receipt)
    with (PROJECT/'logs/background_research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index_path=PROJECT/'results/archive_index.json';index=json.loads(index_path.read_text())
        # Existing index is an object with its archive records in `archives`.
        records=index['archives'] if isinstance(index,dict) else index
        assert not any(v.get('run_id')==root.name or v.get('archive_path')==str(destination) for v in records)
        records.append(receipt);save(index_path,index)
    status.update(run=str(destination),status='five_adapted_tasks_scored_and_archived',archive=receipt,actual_running_jobs=0,
        new_model_sessions=5,controller_pids=[],cohorts=[],stage_progress=[])
    save(public/'current_run.json',status)
    print(json.dumps(dict(status=status['status'],summary=simple,archive=str(destination)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:main(a.run.resolve())
    except BaseException as e:
        save(PROJECT/'results/gap321/FINALIZER_ERROR321.json',dict(utc=datetime.now(timezone.utc).isoformat(),error_type=type(e).__name__,message=str(e),automatic_retry=False))
        raise
