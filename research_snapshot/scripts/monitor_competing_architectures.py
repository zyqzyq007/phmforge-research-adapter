"""Read-only experiment census, with a compact project-side progress artifact."""
import argparse
from collections import Counter
import json
from pathlib import Path
from datetime import datetime,timezone


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def monitor(root,output):
    config=read(root/'CAMPAIGN_CONFIG.json')
    rows=[]
    for p in sorted((root/'episodes').iterdir()):
        cost=read(p/'model_gateway'/p.name/'status.json',{})
        state=read(p/'broker/STATE.json',{})
        result=read(p/'EPISODE_RESULT.json')
        receipts=[read(f) for f in (p/'model_gateway').glob('*/*_receipt.json')]
        stage_files=list((p/'primary').glob('*/status.json'))
        stages=[read(f) for f in stage_files]
        rows.append(dict(id=p.name,phase=read(p/'PHASE.json',{}).get('role'),
            requests=cost.get('requests',0),inflight=cost.get('inflight_count',0),
            input_tokens=cost.get('input_tokens',0),output_tokens=cost.get('output_tokens',0),
            provider_statuses=dict(Counter(str(x.get('upstream_status')) for x in receipts)),
            returned_model_ids=cost.get('returned_model_ids',[]),checks=list(state.get('checks',{})),
            finished_stages=sum(x.get('status')=='finished' for x in stages),
            terminal=result is not None,completed=result.get('completed') if result else None,
            selected=result.get('selected') if result else None,fallback=result.get('fallback') if result else None))
    payload=dict(updated_utc=datetime.now(timezone.utc).isoformat(),run=str(root),
        all_choices_sealed=(root/'ALL_CHOICES_SEALED.json').exists(),
        terminal=sum(r['terminal'] for r in rows),completed=sum(r['completed'] is True for r in rows),
        planned=config['episodes'],total_requests=sum(r['requests'] for r in rows),rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix('.new');pending.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');pending.replace(output)
    print(json.dumps(payload,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();monitor(a.run,a.output)
