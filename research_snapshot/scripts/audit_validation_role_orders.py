"""Characterize strong fixed validation priorities; no method tuning or execution."""
import argparse
import itertools
import json
from pathlib import Path
import time
from program_evidence_corpus import load_pools,read,sha,mean

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
root=parser.parse_args().run.resolve();cfg=read(root/'PROTOCOL353.json')
for name,digest in cfg['code_bindings'].items():assert sha(root/name)==digest
assert not (root/'STARTED353.json').exists()
(root/'STARTED353.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_LLM_calls=0,new_fits=0,new_predictions=0))+'\n')
pools,qualification=load_pools(cfg);rows=[]
orders=list(itertools.permutations(['check','CV','development']));rules=['keep','CV','full_validation','equal_CV_and_full_validation']+['>'.join(v) for v in orders]
for pool in pools:
    members=pool['members'];oracle=min(p['final'] for p in members.values());starting=members['p00']['final'];opportunity=starting-oracle
    selections={}
    for rule in rules:
        if rule=='keep':selected='p00'
        elif '>' in rule:
            order=rule.split('>')
            if any(v['check'] is None for v in members.values()):selections[rule]=dict(status='missing_complete_pool_check');continue
            selected=min(members,key=lambda i:tuple(members[i][k] for k in order)+(i!='p00',i))
        else:
            if rule!='CV' and any(v['full_validation'] is None for v in members.values()):selections[rule]=dict(status='missing_complete_pool_check');continue
            def value(i):
                return (members[i]['CV']+members[i]['full_validation'])/2 if rule=='equal_CV_and_full_validation' else members[i][rule]
            selected=min(members,key=lambda i:(value(i),i!='p00',i))
        final=members[selected]['final']
        selections[rule]=dict(status='scored',program=selected,final_error=float(final),selection_loss=float(final-oracle),
            harm=float(max(final-starting,0)),gain_capture=float(max(starting-final,0)/opportunity) if opportunity else None)
    rows.append(dict(**{k:v for k,v in pool.items() if k!='members'},candidate_count=len(members),
        members={i:{k:float(v) if v is not None else None for k,v in p.items()} for i,p in members.items()},
        oracle_error=float(oracle),selections=selections))
summary={}
primary=[p for p in rows if p['primary']];assert len(primary)==7
for rule in rules:
    values=[p['selections'][rule] for p in primary];assert all(v['status']=='scored' for v in values)
    summary[rule]=dict(pools=7,mean_final_error=sum(v['final_error'] for v in values)/7,
        mean_selection_loss=sum(v['selection_loss'] for v in values)/7,pools_at_own_oracle=sum(v['selection_loss']==0 for v in values),
        mean_harm=sum(v['harm'] for v in values)/7)
result=dict(status='complete',qualification=qualification,all_planned_pools=12,primary_pools=7,
    rules_frozen_before_execution=rules,primary_summary=summary,pools=rows,
    retrospective_opened_development_only=True,no_order_selected_as_new_method=True,
    related_pools_from_three_training_cohorts_one_rig=True,new_LLM_calls=0,new_fits=0,new_predictions=0,
    independent_confirmation_scenarios=0,majority_goal_complete=False)
(root/'RESULT353.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='complete',primary_summary=summary)))
