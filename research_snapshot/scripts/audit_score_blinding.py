"""Recount sealed selection choices, information equality and primary criteria."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def audit(run):
    cfg=read(run/'PROTOCOL451.json');result=read(run/'RESULT451.json')
    seal=read(run/'ALL_CHOICES_SEALED451.json');protocol_prompts=read(run/'PROMPTS451.json')
    historical=read(run/'historical_outcomes/analysis/HISTORICAL_OUTCOMES450.json')
    checks=[]
    def ck(name,ok):
        checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    ck('27terminal_jobs', len(seal['jobs'])==27 and seal['all_jobs_terminal'])
    ck('9pools3sources',len(result['rows'])==9 and len(result['sources'])==3)
    ck('all_model_calls_accounted',result['cost']['requests']==result['cost']['receipts'])
    ck('one_requested_returned_model',result['cost']['returned_models']==['deepseek-flash'])
    for job in seal['jobs']:
        pool=job['pool'];folder=run/'model_gateway'/job['id']
        full=read(run/'visible'/pool/'full.json')
        deferred=read(run/'visible'/pool/'code_only.json')
        ck(job['id']+'_same_code',full['candidates']==deferred['candidates'])
        ck(job['id']+'_same_task',full['task']==deferred['task'])
        files=sorted(folder.glob('*_request.json'))
        ck(job['id']+'_two_call_cap',len(files)<=2)
        for n,p in enumerate(files,1):
            payload=read(p);receipt=read(folder/(p.name.replace('_request','_receipt')))
            ck(job['id']+f'_{n}_receipt_SHA',hashlib.sha256(p.read_bytes()).hexdigest()==receipt['request_sha256'])
            ck(job['id']+f'_{n}_no_tools',payload.get('tools')==[])
            ck(job['id']+f'_{n}_budget',payload['max_output_tokens']==cfg['per_request_output'])
            ck(job['id']+f'_{n}_model',payload['model']=='deepseek-flash')
            data=json.loads(payload['input'])
            if n==1:
                ck(job['id']+'_first_material',data==(deferred if job['arm']=='delayed_scores' else full))
                instruction=protocol_prompts['ordinary' if job['arm']=='ordinary_review' else 'audit']
                ck(job['id']+'_first_instruction',payload['instructions']==instruction)
            else:
                ck(job['id']+'_full_final_material',data['material']==full)
                ck(job['id']+'_same_final_instruction',payload['instructions']==protocol_prompts['select'])
    for row in result['rows']:
        losses={r['program_id']:r['final_loss'] for r in historical if r['episode']==row['episode']}
        low=min(losses.values());width=max(losses.values())-low
        ck(row['pool']+'_true_oracle',math.isclose(row['oracle'],low,abs_tol=1e-12))
        for name,value in row['methods'].items():
            if value['status']!='completed':continue
            score=losses[value['choice']];ck(row['pool']+'_'+name+'_loss',score==value['loss'])
            values=dict(regret=score-low,normalized_regret=(score-low)/width if width else 0,
                        harmful_update=max(0,score-losses['p00'])/width if width else 0,
                        useful_improvement=max(0,losses['p00']-score)/width if width else 0)
            for k,v in values.items():ck(row['pool']+'_'+name+'_'+k,math.isclose(value[k],v,abs_tol=1e-12))
    complete=all(j['status']=='completed' for j in seal['jobs'])
    aggregate={};comparison={};passed=False
    eligible=[name for name in result['rows'][0]['methods']
              if all(r['methods'][name]['status']=='completed' for r in result['rows'])]
    if 'delayed_scores' in eligible:
        for name in eligible:
            means=[]
            for source in sorted({r['source'] for r in result['rows']}):
                rs=[r['methods'][name] for r in result['rows'] if r['source']==source]
                vals={k:sum(r[k] for r in rs)/len(rs) for k in ['loss','normalized_regret','harmful_update','useful_improvement']}
                declared=next(s['means'][name] for s in result['sources'] if s['source']==source)
                for k,v in vals.items():ck(name+'_'+source+'_'+k,math.isclose(v,declared[k],abs_tol=1e-12))
                means.append(vals)
            aggregate[name]={k:sum(v[k] for v in means)/3 for k in ['normalized_regret','harmful_update','useful_improvement']}
        method=aggregate['delayed_scores']
        for control in ['ordinary_review','unblinded_audit','validation_best','historical_native']:
            if control not in eligible:
                continue
            deltas=[s['means']['delayed_scores']['loss']-s['means'][control]['loss'] for s in result['sources']]
            comparison[control]=dict(wins=sum(d < -1e-10 for d in deltas),ties=sum(abs(d)<=1e-10 for d in deltas),losses=sum(d>1e-10 for d in deltas))
        passed=(complete and comparison['ordinary_review']['wins']>=2 and
                all(method['normalized_regret'] < aggregate[b]['normalized_regret']-1e-10 for b in ['ordinary_review','unblinded_audit','validation_best']) and
                all(method['harmful_update']<=aggregate[b]['harmful_update']+1e-10 and
                    method['useful_improvement']>=aggregate[b]['useful_improvement']-1e-10 for b in ['ordinary_review','unblinded_audit']))
    report=dict(passed_audit=True,checks=checks,count=len(checks),all27_completed=complete,
                source_equal_aggregate=aggregate,source_wins_ties_losses=comparison,primary_gate_passed=passed,
                novel_method_confirmed=False,independent_confirmation=0,
                next_action='Proceed only to fresh fully controlled study; no confirmatory claim.' if passed else
                            'Stop this report-delay policy; no prompt, cutoff or source-subset tuning on these results.')
    (run/'INDEPENDENT_AUDIT451.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);audit(ap.parse_args().run.resolve())
