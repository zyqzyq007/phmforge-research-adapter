"""Bounded, equal-budget selection replay on immutable real PHM programs.

The model has no file/network tools. Historical final losses are read only by
the scoring stage after every planned choice has a terminal, sealed record.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import tarfile
import threading
import time
from pathlib import Path

from isolated_model_gateway import gateway
from mlestar_phm_adapter import post_socket


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read(path):
    return json.loads(path.read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def restore(archive, destination, predicate):
    manifest = read(archive/'SOURCE_MANIFEST.json')['files']
    wanted = {r['path']:r for r in manifest if predicate(r['path'])}
    got = []
    with tarfile.open(archive/'RUN_CONTENTS.tar.gz', 'r|gz') as stream:
        for member in stream:
            name = member.name.removeprefix('./')
            if name not in wanted:
                continue
            raw = stream.extractfile(member).read()
            assert sha(raw) == wanted[name]['sha256'] and len(raw) == wanted[name]['bytes']
            path = destination/name
            assert '..' not in Path(name).parts
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            got.append(dict(archive=str(archive), member=name, sha256=sha(raw)))
            if len(got) == len(wanted):
                break
    assert len(got) == len(wanted)
    return got


AUDIT = '''Assess each complete PHM program against the stated task. For each candidate, identify at most two concrete assumptions needed for generalization; cite a code expression and the relevant validation unit or query-time rule. Distinguish a demonstrated violation, a supported assumption, and an untested assumption. Do not assume that uncertainty means the candidate is wrong, that a complex model is wrong, or that the starting program is better. Do not invent unseen operating conditions, scores, data, or tests. Do not solve the task anew. Return a concise JSON object with candidate assessments and open questions; no program selection yet.'''
ORDINARY = '''Review the complete candidate PHM programs and available validation results to prepare a final choice that generalizes to the task's unseen queries. Use your best judgement about code, numerical evidence, improvements and regressions. No final labels or scores are available. Do not invent tests or observations. Return a concise JSON review; do not make the final selection yet.'''
SELECT = '''Select exactly one of the available complete program IDs for the task, using only the supplied code, actual validation evidence and your preceding review. No candidate is guaranteed optimal. A low observed score is evidence, not certification; an unsupported concern alone does not prove failure. Consider both improvement and regression, and retain useful improvements. Do not generate a new program or request hidden evaluation information. Return only JSON: {"selected_id":"EXACT_ID", "reason":"brief concrete rationale", "remaining_uncertainty":"brief"}.'''


def validation_design(value, key=''):
    """Keep actual validation identities, units and structure; defer outcomes."""
    visible = {'fold','cell','engine','condition_hz','group','windows','count','scope',
               'query_cycle','cycle','fraction','observation_time','available_points','program_id'}
    if isinstance(value, dict):
        return {k:validation_design(v,k) for k,v in value.items()}
    if isinstance(value, list):
        return [validation_design(v,key) for v in value]
    return value if key in visible or value is None else '[performance withheld until selection]'


def prepare(run):
    cfg = read(run/'PROTOCOL451.json')
    source = Path(cfg['corpus_archive'])
    assert sha((source/'SOURCE_MANIFEST.json').read_bytes()) == cfg['corpus_manifest_sha256']
    bindings = restore(source, run/'corpus', lambda p:
        p in ['analysis/LEGAL_EVIDENCE450.json', 'analysis/EPISODES450.json'] or
        p.startswith('inputs/') and ('/source/' in p or p.endswith('/COMPOSITION.json')))
    # Original common task instructions, independent of arm-specific workflow.
    tasks = {}
    for batch, item in cfg['task_sources'].items():
        destination = run/'original_tasks'/batch
        bindings += restore(Path(item['archive']), destination, lambda p:p==item['member'])
        tasks[item['family']] = read(destination/item['member'])['ordinary']
    legal = read(run/'corpus/analysis/LEGAL_EVIDENCE450.json')
    episodes = read(run/'corpus/analysis/EPISODES450.json')
    normal = [e for e in episodes if e['completed']]
    assert len(normal) == 9 and len(episodes) == 16
    records, jobs = [], []
    allowed = ['status','cv','development','fit_entry_executions','confirmed_estimator_fits',
               'actual_optimizer_steps','nonzero_sampled_optimizer_steps','training_only',
               'whole_condition_split','whole_cell_split','identity_invariance_passed',
               'independent_check_used','final_data_read','development_execution_passed']
    for ix, episode in enumerate(normal):
        alias = f'Q{ix+1:02d}'
        rows = [r for r in legal if r['episode']==episode['id']]
        random.Random(451+ix).shuffle(rows)
        code, evidence = [], []
        for row in rows:
            files = {}
            for relative in row['code_files']:
                path = run/'corpus'/relative
                text = path.read_text()
                if path.suffix == '.py':
                    tree = ast.parse(text)
                    # Exclude source comments/docstrings from all arms equally.
                    for node in ast.walk(tree):
                        if isinstance(getattr(node,'body',None), list):
                            node.body = [x for x in node.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))] or [ast.Pass()]
                    text = ast.unparse(tree)
                elif path.name == 'COMPOSITION.json':
                    c = json.loads(text)
                    text = json.dumps(dict(rule=c['rule'],tie_anchor=c['tie_anchor'],members=list(c['members'])))
                files[path.name] = text
            code.append(dict(id=row['program_id'],initial=row['initial'],files=files))
            evidence.append(dict(id=row['program_id'],report={k:v for k,v in row['legal_report'].items() if k in allowed},
                                 additional_actual_check=row['additional_legal_check']))
        task = tasks[episode['source_family']]
        task += '\nThis is a frozen-candidate SELECT-ONLY replay. Original tools, fitting, file paths and earlier budgets above describe how these artifacts were produced; they are unavailable here. All complete candidates below are already genuinely trained or compose trained members. Select one, with exactly two model responses total. No current query inputs, final labels, original choices, historical final scores or other episodes are provided.'
        full = dict(task=task,candidates=code,validation=evidence)
        deferred = []
        for item in evidence:
            report = {k:(validation_design(v) if k in ['cv','development'] else v) for k,v in item['report'].items()}
            deferred.append(dict(id=item['id'],report=report,additional_actual_check=validation_design(item['additional_actual_check'])))
        code_only = dict(task=task,candidates=code,validation=deferred,
                         validation_results='Validation design and training provenance are present; performance values are deferred until final selection.')
        save(run/'visible'/alias/'full.json',full)
        save(run/'visible'/alias/'code_only.json',code_only)
        for arm in cfg['arms']:
            name = alias+'_'+arm
            jobs.append(dict(id=name,pool=alias,arm=arm))
        records.append(dict(pool=alias,episode=episode['id'],source=episode['source_family'],original_arm=episode['arm'],
                            candidate_ids=[r['program_id'] for r in rows],original_selected=episode['selected'],
                            original_native_completed=True,member_ids=[r['id'] for r in rows]))
    save(run/'PRIVATE_POOL_MAPPING451.json',records)
    save(run/'JOBS451.json',jobs)
    save(run/'PREPARATION451.json',dict(bindings=bindings,normal_primary_pools=9,incomplete_not_relabelled=7,
        all_native_episodes_retained=episodes,files_sha256={str(p.relative_to(run)):sha(p.read_bytes()) for p in (run/'visible').rglob('*.json')},
        new_fits=0,new_LLM_calls=0,task_rendering='Original ordinary task; comments/docstrings removed from displayed code equally in all arms; original bytes retained.'))
    save(run/'PROMPTS451.json',dict(audit=AUDIT,ordinary=ORDINARY,select=SELECT))
    print(json.dumps(dict(prepared=True,pools=len(records),jobs=len(jobs),max_model_calls=2*len(jobs))))


def output_text(response):
    return '\n'.join(b['text'] for item in response.get('output',[]) if item.get('type')=='message'
                     for b in item.get('content',[]) if b.get('type')=='output_text')


def json_output(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(text)


def run_jobs(run):
    cfg=read(run/'PROTOCOL451.json');jobs=read(run/'JOBS451.json')
    assert not (run/'ALL_CHOICES_SEALED451.json').exists()
    assert not (run/'historical_outcomes').exists()
    routes=[dict(job=j['id'],socket=str(run/'ipc'/(j['id']+'.sock'))) for j in jobs]
    save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=routes,base_url='https://api.deepseek.com/',model='deepseek-flash',
        max_requests=2,max_input_tokens=cfg['per_job_input_tokens'],max_output_tokens=2*cfg['per_request_output'],max_output_per_request=cfg['per_request_output'],
        max_inflight=1,buffer_verified_responses=True,denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent']))
    gt=threading.Thread(target=gateway,args=(run,));gt.start()
    started=time.monotonic()
    while not (run/'MODEL_GATEWAY_READY.json').exists():
        if not gt.is_alive() or time.monotonic()-started>30:
            raise RuntimeError('Gateway not ready')
        time.sleep(.1)
    prompts=read(run/'PROMPTS451.json')
    def execute(job):
        folder=run/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();socket=run/'ipc'/(job['id']+'.sock')
        try:
            assert time.monotonic()-started < cfg['wall_seconds'], 'Frozen wall budget exhausted before job'
            full=read(run/'visible'/job['pool']/'full.json')
            first=read(run/'visible'/job['pool']/('code_only.json' if job['arm']=='delayed_scores' else 'full.json'))
            instruction=prompts['ordinary' if job['arm']=='ordinary_review' else 'audit']
            response=post_socket(socket,dict(model='deepseek-flash',instructions=instruction,input=json.dumps(first),tools=[],stream=False,max_output_tokens=cfg['per_request_output']))
            save(folder/'review_response.json',response);review=output_text(response)
            assert review.strip(), 'Missing review'
            second=dict(material=full,previous_review=review)
            assert time.monotonic()-started < cfg['wall_seconds'], 'Frozen wall budget exhausted before final selection'
            response=post_socket(socket,dict(model='deepseek-flash',instructions=prompts['select'],input=json.dumps(second),tools=[],stream=False,max_output_tokens=cfg['per_request_output']))
            save(folder/'selection_response.json',response)
            choice=json_output(output_text(response))
            assert choice['selected_id'] in [c['id'] for c in full['candidates']]
            result=dict(**job,status='completed',choice=choice,elapsed_seconds=time.monotonic()-began)
        except Exception as error:
            result=dict(**job,status='incomplete',error_type=type(error).__name__,error=str(error)[:800],elapsed_seconds=time.monotonic()-began)
        save(folder/'TERMINAL451.json',result)
        print(json.dumps(dict(job=job['id'],status=result['status'],seconds=round(result['elapsed_seconds'],2))),flush=True)
        return result
    results=[]
    try:
        # Every arm on the largest candidate pool qualifies output budget.
        preflight=[j for j in jobs if j['pool']==cfg['preflight_pool']]
        remaining=[j for j in jobs if j['pool']!=cfg['preflight_pool']]
        assert len(preflight)==len(cfg['arms'])
        with ThreadPoolExecutor(max_workers=len(preflight)) as pool:
            results.extend(pool.map(execute,preflight))
        if all(r['status']=='completed' for r in results):
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                futures=[pool.submit(execute,j) for j in remaining]
                for f in as_completed(futures):
                    results.append(f.result())
        else:
            for j in remaining:
                value=dict(**j,status='not_started_preflight_failed');save(run/'jobs'/j['id']/'TERMINAL451.json',value);results.append(value)
    finally:
        (run/'MODEL_GATEWAY_STOP').touch();gt.join(timeout=640)
        assert not gt.is_alive()
    assert len(results)==len(jobs)
    save(run/'ALL_CHOICES_SEALED451.json',dict(utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        jobs=results,all_jobs_terminal=True,hashes={str(p.relative_to(run)):sha(p.read_bytes()) for p in (run/'jobs').rglob('TERMINAL451.json')}))


def score(run):
    seal=read(run/'ALL_CHOICES_SEALED451.json');cfg=read(run/'PROTOCOL451.json')
    assert seal['all_jobs_terminal'] and len(seal['jobs'])==27
    for path,h in seal['hashes'].items():
        assert sha((run/path).read_bytes())==h
    restore(Path(cfg['corpus_archive']),run/'historical_outcomes',lambda p:p=='analysis/HISTORICAL_OUTCOMES450.json')
    historical=read(run/'historical_outcomes/analysis/HISTORICAL_OUTCOMES450.json')
    mappings=read(run/'PRIVATE_POOL_MAPPING451.json');rows=[]
    for p in mappings:
        outcomes={r['program_id']:r['final_loss'] for r in historical if r['episode']==p['episode']}
        full=read(run/'visible'/p['pool']/'full.json')
        base=outcomes['p00'];lo=min(outcomes.values());scale=max(outcomes.values())-lo
        methods={}
        for j in seal['jobs']:
            if j['pool']==p['pool']:
                methods[j['arm']]=j.get('choice',{}).get('selected_id')
        # Deterministic strong validation control; same printed evidence, no final-score selection.
        metrics={}
        for item in full['validation']:
            rep=item['report'];cv=rep.get('cv')
            if cv:
                errors=[c['MAE'] for f in cv for c in f.get('cells',[])]
                val=sum(errors)/len(errors) if errors else sum(f['record_error'] for f in cv)/len(cv)
            else:
                val=rep['development']['MAE']
            metrics[item['id']]=val
        methods['validation_best']=min(metrics,key=lambda k:(metrics[k],k!='p00',k))
        methods['keep_p00']='p00';methods['historical_native']=p['original_selected']
        scored={}
        for method,choice in methods.items():
            if choice is None:
                scored[method]=dict(choice=None,status='incomplete');continue
            loss=outcomes[choice]
            scored[method]=dict(choice=choice,status='completed',loss=loss,regret=loss-lo,
                normalized_regret=(loss-lo)/scale if scale else 0,
                harmful_update=max(0,loss-base)/scale if scale else 0,
                useful_improvement=max(0,base-loss)/scale if scale else 0)
        rows.append(dict(**p,metric='record_error' if p['source']=='COMFAULDA' else 'MAE_cycles',oracle=lo,anchor=base,methods=scored))
    sources=[]
    for source in sorted({r['source'] for r in rows}):
        rs=[r for r in rows if r['source']==source];means={}
        for method in list(rows[0]['methods']):
            values=[r['methods'][method] for r in rs]
            means[method]=dict(completed=sum(v['status']=='completed' for v in values),total=len(values))
            if all(v['status']=='completed' for v in values):
                means[method].update({k:sum(v[k] for v in values)/len(values) for k in ['loss','normalized_regret','harmful_update','useful_improvement']})
        sources.append(dict(source=source,pools=len(rs),means=means))
    receipts=[read(p) for p in (run/'model_gateway').rglob('*_receipt.json')]
    states=[read(p) for p in (run/'model_gateway').rglob('status.json')]
    cost=dict(requests=sum(s['requests'] for s in states),receipts=len(receipts),input_tokens=sum(s['input_tokens'] for s in states),output_tokens=sum(s['output_tokens'] for s in states),returned_models=sorted({m for s in states for m in s['returned_model_ids']}),all_terminal=all(not s['inflight'] for s in states),new_PHM_fits=0,new_native_Agent_stages=0)
    save(run/'RESULT451.json',dict(rows=rows,sources=sources,cost=cost,independent_confirmation=0,exploratory_only=True))
    print(json.dumps(dict(sources=sources,cost=cost)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','run','score'],required=True)
    args=ap.parse_args();root=args.run.resolve();assert Path('/tmp') not in root.parents
    {'prepare':prepare,'run':run_jobs,'score':score}[args.stage](root)
