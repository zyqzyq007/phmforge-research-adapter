"""External numeric evaluation after ordinary and gated choices are sealed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from controlled_feedback import save,sha,utc


def read(path):return json.loads(Path(path).read_text())


def summarize(rows):
    # Replicates, then counterfactual incumbents, then related holdout cases.
    cases=defaultdict(lambda:defaultdict(list))
    for row in rows:cases[row['case']][row['incumbent']].append(row)
    keys=['loss','selection_regret','harm','gain','opportunity','normalized_regret']
    values={k:statistics.mean(statistics.mean(statistics.mean(v[k] for v in rr) for rr in old.values()) for old in cases.values()) for k in keys}
    values['gain_capture']=values['gain']/values['opportunity'] if values['opportunity']>1e-12 else None
    values.update(decisions=len(rows),harmful_updates=sum(r['harm']>1e-12 for r in rows),
                  cases=len(cases),independent_source_studies=1,confirmation_scenarios=0)
    return values


def evaluate(root):
    cfg=read(root/'CAMPAIGN_CONFIG.json');study=Path(cfg['source_study'])
    if not study.exists():
        relocation=read(root/'SOURCE_RELOCATION334.json')
        assert relocation['from']==str(study)
        study=Path(relocation['to'])
        receipt=read(study/'ARCHIVE_VERIFIED.json')
        assert sha(study/receipt['manifest_filename'])==receipt['manifest_sha256']==relocation['manifest_sha256']
        assert sha(study/'PROTOCOL333.json')==cfg['source_protocol_sha256']
    seal=read(root/'ALL_METHOD_CHOICES_SEALED334.json')
    assert seal['source_final_score_join_started'] is False
    assert seal['code_sha256']==sha(root/'code/validation_reliability_policy.py')
    save(root/'audit/EXTERNAL_JOIN_STARTED334.json',dict(utc=utc(),choices_sha256=sha(root/'ALL_METHOD_CHOICES_SEALED334.json')))
    rows=[];costs=[];source_bindings=[];checks=[]
    for child in sorted((root/'case_runs').iterdir()):
        original=read(child/'CAMPAIGN_CONFIG.json')['case_source_id']
        path=study/'audit'/(original+'_outer_scores.json');scores=read(path)
        source_bindings.append(dict(path=str(path),sha256=sha(path)))
        for name,score in scores.items():
            truth=score['record_truth'];pred=score['record_prediction'];n=len(truth)
            assert n==len(pred)==len(score['records'])==4
            confusion=[[sum(a==i and b==j for a,b in zip(truth,pred)) for j in range(4)] for i in range(4)]
            loss=sum(a!=b for a,b in zip(truth,pred))/n
            f1=[]
            for i in range(4):
                denominator=sum(confusion[i])+sum(row[i] for row in confusion)
                f1.append(2*confusion[i][i]/denominator if denominator else 0.)
            assert confusion==score['confusion'] and abs(loss-score['loss'])<1e-12
            assert abs(statistics.mean(f1)-score['macro_F1'])<1e-12
            checks.append(dict(case=child.name,candidate=name,numerical_metrics_match=True))
        lo=min(s['loss'] for s in scores.values());hi=max(s['loss'] for s in scores.values());scale=hi-lo
        def append(method,incumbent,selected,episode,valid=True,fallback=False):
            old=scores[incumbent]['loss'];new=scores[selected]['loss']
            rows.append(dict(case=child.name,method=method,incumbent=incumbent,selected=selected,episode=episode,
                valid=valid,fallback=fallback,loss=new,selection_regret=new-lo,
                normalized_regret=(new-lo)/scale if scale>1e-12 else 0.,
                harm=max(0,new-old),gain=max(0,old-new),opportunity=old-lo))
        for episode in [e for e in seal['episodes'] if e['case']==child.name]:
            if episode['completed']:
                for decision in episode['choices']:
                    append('ordinary',decision['incumbent'],decision['ordinary'],episode['episode'])
                    for name,selected in decision['variants'].items():append(name,decision['incumbent'],selected,episode['episode'])
            else:
                for old in scores:
                    append('ordinary_fallback',old,old,episode['episode'],valid=False,fallback=True)
                    append('calibrated_gate_fallback',old,old,episode['episode'],valid=False,fallback=True)
            run=child/'episodes'/episode['episode'];p=run/'model_gateway'/episode['episode']/'status.json'
            status=read(p) if p.exists() else {}
            receipts=[read(x) for x in (run/'model_gateway').glob('*/*_receipt.json')]
            costs.append(dict(case=child.name,episode=episode['episode'],completed=episode['completed'],
                model_requests=status.get('requests',0),input_tokens=status.get('input_tokens',0),output_tokens=status.get('output_tokens',0),
                returned_model_ids=status.get('returned_model_ids',[]),
                missing_success_usage=sum(r.get('upstream_status')==200 and not r.get('usage') for r in receipts),
                upstream_statuses=[r.get('upstream_status') for r in receipts]))
        for decision in read(child/'FIXED_CHOICES.json')['choices']:
            for method,selected in decision['selected'].items():append('fixed_'+method,decision['incumbent'],selected,'fixed')
    methods=sorted(set(r['method'] for r in rows if r['valid']))
    panels={}
    for panel in ['original_anchor','all_counterfactual_incumbents']:
        subset=[r for r in rows if r['valid'] and (panel!='original_anchor' or r['incumbent']==cfg['main_anchor'].split(';')[0])]
        panels[panel]={m:summarize([r for r in subset if r['method']==m]) for m in methods if any(r['method']==m for r in subset)}
    comparisons=[]
    for case in sorted(set(r['case'] for r in rows)):
        rr=[r for r in rows if r['valid'] and r['case']==case and r['incumbent']==cfg['main_anchor'].split(';')[0]]
        for method in ['calibrated_gate','record_gate','condition_gate']:
            ordinary=[r['loss'] for r in rr if r['method']=='ordinary'];variant=[r['loss'] for r in rr if r['method']==method]
            if ordinary and variant:
                a,b=statistics.mean(ordinary),statistics.mean(variant)
                comparisons.append(dict(case=case,method=method,ordinary_loss=a,method_loss=b,
                    result='win' if b<a-1e-12 else 'loss' if b>a+1e-12 else 'tie',
                    not_independent_confirmation_case=True))
    result=dict(utc=utc(),scope='Exploratoryacquired-data,nestedLOOselectionstudywithfreshnativeAgents;onecorrelatedsource,noconfirmatorymajorityclaim.',
        panels=panels,rows=rows,related_case_comparisons=comparisons,costs=costs,
        planned_agent_sessions=8,completed_agent_sessions=sum(c['completed'] for c in costs),
        new_model_requests=sum(c['model_requests'] for c in costs),
        new_input_tokens=sum(c['input_tokens'] for c in costs),new_output_tokens=sum(c['output_tokens'] for c in costs),
        gate_reuses_same_actual_Agent_proposals=True,additional_gate_model_requests=0,
        common_source_check_compute_identical=True,new_PHM_fits=0,source_independent_replications=1,
        independent_confirmation_scenarios=0,goal_achieved=False,
        independent_metric_checks=checks,source_bindings=source_bindings)
    save(root/'RESULTS334.json',result)
    out=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/results/native334');save(out/'RESULTS334.json',result)
    print(json.dumps(dict(panels=panels,costs=costs,related_case_comparisons=comparisons)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();evaluate(a.run)
