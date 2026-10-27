"""Post-run evidence audit, including failed/cancelled subjects and service timing."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re

from controlled_feedback import save, sha, utc


def audit(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());suffix=config['iteration'][1:]
    preparation=json.loads((root/('PREPARATION'+suffix+'.json')).read_text())
    assert all(sha(root/r['path'])==r['sha256'] for r in preparation['copied'])
    assert all(sha(root/'code'/name)==digest for name,digest in preparation['code_hashes'].items())
    cases=[];timings=[]
    for folder in sorted((root/'episodes').iterdir()):
        result=json.loads((folder/'EPISODE_RESULT.json').read_text())
        counters=Counter();rejected=[];missing=[];fit_text=[];commands=0
        receipts=[json.loads(p.read_text()) for p in (folder/'model_gateway').glob('*/*_receipt.json')]
        missing=[r['sequence'] for r in receipts if not {'input_tokens','output_tokens'}<=r.get('usage',{}).keys()]
        for f in sorted((folder/'model_gateway').glob('*/*_response.sse')):
            terminal=None
            for line in f.read_text().splitlines():
                if not line.startswith('data: '):continue
                try:event=json.loads(line[6:])
                except ValueError:continue
                item=event.get('item',{})
                if event.get('type')=='response.output_item.done' and item.get('name') in ['spawn_agent','followup_task','interrupt_agent','list_agents','send_message','wait_agent']:
                    counters[item['name']]+=1
                if event.get('type')=='response.completed':terminal=event.get('response',{})
            receipt_path=f.with_name(f.name.replace('_response.sse','_receipt.json'))
            receipt=json.loads(receipt_path.read_text())
            if receipt.get('blocked_response_calls'):rejected.append(receipt['sequence'])
            if terminal and terminal.get('created_at') and terminal.get('completed_at'):
                start=datetime.fromisoformat(receipt['started_utc']).timestamp()
                timings.append(dict(episode=folder.name,sequence=receipt['sequence'],elapsed_seconds=receipt['elapsed_seconds'],
                    before_provider_created_seconds=terminal['created_at']-start,
                    provider_created_to_completed_seconds=terminal['completed_at']-terminal['created_at'],
                    output_tokens=receipt.get('usage',{}).get('output_tokens'),client_disconnected=receipt.get('client_disconnected',False),
                    source=str(receipt_path.relative_to(root))))
        pattern=re.compile(r'\.(?:fit|fit_transform|partial_fit)\s*\(|loss\.backward\s*\(|optimizer\.step\s*\(')
        for f in (folder/'primary').glob('*/events.jsonl'):
            for line in f.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                item=event.get('item',{})
                if event.get('type')=='item.completed' and item.get('type')=='command_execution':
                    commands+=1
                    if pattern.search(item.get('command','')):
                        fit_text.append(dict(source=str(f.relative_to(root)),command=item['command']))
        seen=set()
        for app in [folder/'app',*list((folder/'branches').glob('*/app')),folder/'source_history']:
            for f in app.rglob('*.py'):
                digest=sha(f)
                if digest in seen:continue
                seen.add(digest)
                if pattern.search(f.read_text(errors='replace')):
                    fit_text.append(dict(source=str(f.relative_to(root)),kind='generated_python_source'))
        upstream=[]
        for f in (folder/'model_gateway').glob('*/*_request.json'):
            data=json.loads(f.read_text())
            if any(t.get('name')=='collaboration' for t in data.get('tools',[])):upstream.append(str(f.relative_to(root)))
        if config.get('enforce_external_roles'):
            assert not upstream and not counters and not rejected
        if result['arm']=='A':assert not counters
        state=json.loads((folder/'model_gateway'/folder.name/'status.json').read_text())
        broker=json.loads((folder/'broker/STATE.json').read_text())
        planned={c for p in broker['plans'] for c in p['candidate_ids']}
        missing_checks=sorted(planned-set(broker['checks']))
        assert not state['active'] and not state['inflight']
        cases.append(dict(id=folder.name,arm=result['arm'],completed=result['completed'],fallback=result['fallback'],
            selected=result['selected'],model_requests=state['requests'],no_model_call_subject=state['requests']==0,
            native_delegation_calls=dict(counters),upstream_catalog_included_collaboration=bool(upstream),
            prohibited_calls_blocked=rejected,missing_usage_sequences=missing,
            planned_checks_not_executed=missing_checks,
            inspected_shell_commands=commands,fit_text_requiring_manual_review=fit_text,
            stage_statuses=[dict(role=s.get('role'),branch=s.get('branch_id'),status=s['status'],seconds=s['elapsed_seconds']) for s in result['stages']]))
    assert all(not c['fit_text_requiring_manual_review'] for c in cases),'Review possible extra training before claiming zero new fits'
    payload=dict(iteration=config['iteration'],utc=utc(),unchanged_copied_asset_files=len(preparation['copied']),
        frozen_code_hashes_passed=True,no_extra_fit_calls_observed=True,
        caveat='Command/source audit is not universal proof of absence; deployed checkpoints are independently hash-bound and unchanged.',
        final_data_unmounted_in_all_native_gates=True,cases=cases,request_timings=timings,
        timing_caveat='Provider timestamps are service-reported; the pre-created interval includes transport/service handling and is not a direct queue measurement.')
    save(root/'audit/EXPERIMENT_AUDIT.json',payload)
    out=Path(config['project'])/config['result_directory'];save(out/'EXPERIMENT_AUDIT.json',payload)
    print(json.dumps(dict(iteration=config['iteration'],cases=len(cases),native_calls={c['id']:c['native_delegation_calls'] for c in cases if c['native_delegation_calls']},
        missing_usage=sum(len(c['missing_usage_sequences']) for c in cases),new_fit_text_hits=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
