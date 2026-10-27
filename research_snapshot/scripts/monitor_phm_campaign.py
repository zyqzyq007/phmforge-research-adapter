"""Observe native PHM campaign processes and real request counters; never score."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def process_state(pid):
    if not pid:
        return None
    try:
        raw=Path('/proc',str(pid),'stat').read_text()
        return raw.rsplit(')',1)[1].split()[0]
    except FileNotFoundError:
        return None


def observe(root):
    config=read(root/'CAMPAIGN_CONFIG.json')
    names=([*config['cohorts'],*(c+'_review' for c in config['cohorts'])]
           if 'cohorts' in config else [c['id'] for c in config['cases']])
    iteration=config.get('iteration') or 'A'+re.search(r'(\d+)$',config['result_directory']).group(1)
    rows=[]
    for name in names:
        run=root/name
        for job in read(run/'PLAN187.json').get('jobs',[]):
            status=read(run/'primary'/job['id']/job['id']/'status.json')
            gateway=read(run/'model_gateway'/job['id']/'status.json')
            state=process_state(status.get('cli_pid'))
            rows.append(dict(cohort=name,job=job['id'],status=status.get('status','not_started'),
                cli_pid=status.get('cli_pid'),actual_process_state=state,
                process_live=state is not None and state!='Z',
                requests=gateway.get('requests',0),inflight=gateway.get('inflight',False),
                returned_model_ids=gateway.get('returned_model_ids',[])))
    campaign=read(root/'CAMPAIGN_STATUS.json')
    return dict(utc=datetime.now(timezone.utc).isoformat(),iteration=iteration,
        phase=campaign.get('status'),planned_native_sessions=config['planned_sessions'],
        native_started=sum(r['cli_pid'] is not None for r in rows),
        running=sum(r['process_live'] for r in rows),requests=sum(r['requests'] for r in rows),
        final_scores_opened=(root/('NUMERICAL_RESULTS'+iteration.removeprefix('A')+'.json')).exists() or
            any((root/c/'CODING_SCORE.json').exists() for c in names),rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();root=args.run.resolve()
    result=observe(root)
    config=read(root/'CAMPAIGN_CONFIG.json')
    output=Path(__file__).resolve().parents[1]/config['result_directory']/('LIVE_PROGRESS'+result['iteration'].removeprefix('A')+'.json')
    temporary=output.with_suffix('.new')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
