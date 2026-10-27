"""External scoring of sealed A329 choices; no access from experimental Agents."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from controlled_feedback import save,sha,utc
from continue_program_updates import read,bind_many,OUT,update
from evaluate_program_update_screen import classification,source_balanced
from program_selection_policies import choose


def main(root):
    assert not (root/'FINAL_RESULTS329.json').exists(),'No implicit rescore/restart'
    seal=read(root/'ALL_CHOICES_SEALED.json')
    assert seal['final_score_join_not_yet_run']
    for r in seal['files']:assert sha(root/r['path'])==r['sha256'],r['path']
    cases=read(root/'public/cases.json');scores={};audit=[]
    choices=read(root/'FIXED_CHOICES.json')['choices'];episodes=read(root/'AGENT_CHOICES.json')['episodes'];gates=[]
    for e in episodes:
        name=e['id']
        if not e['completed'] or not e['choice']:
            choices[name+'_fallback']=[dict(case=c['id'],incumbent=i,selected=i) for c in cases for i in c['candidate_ids']]
            continue
        choices[name]=e['choice']['choices'];choices[name+'_M10_gate']=[]
        for r in choices[name]:
            c=next(c for c in cases if c['id']==r['case']);_,evidence=choose(c,r['incumbent'],'M10_broad_support')
            accepted=r['selected'] in evidence['eligible'];selected=r['selected'] if accepted else r['incumbent']
            choices[name+'_M10_gate'].append(dict(case=r['case'],incumbent=r['incumbent'],selected=selected))
            gates.append(dict(agent=name,case=r['case'],incumbent=r['incumbent'],proposed=r['selected'],selected=selected,
                              accepted=accepted,eligible=evidence['eligible'],new_model_calls=0))
    # Deterministic gate choices are materialized before opening any private
    # outcome file. They depend only on the already frozen eligibility rule.
    save(root/'PAIRED_GATE_CHOICES_SEALED329.json',dict(utc=utc(),choices=choices,gates=gates,final_score_join_started=False,
        rule_sha256=sha(root/'code/program_selection_policies.py'),agent_choices_sha256=sha(root/'AGENT_CHOICES.json')))
    save(root/'audit/EVALUATION_START.json',dict(utc=utc(),all_choices_sealed=True,evaluator_sha256=sha(Path(__file__)),researcher_previously_knows_scores=True,
        gate_choices_sha256=sha(root/'PAIRED_GATE_CHOICES_SEALED329.json')))
    paths=[]
    for c in cases[:2]:
        cohort=c['id'][-1];paths.append(cohort+'/hidden/ELECTRICMOTORVIBRATIONS/labels.json')
        paths += ['controls/'+c['id']+'/'+name+'_train_external_predictions.json' for name in c['candidate_ids']]
    source=bind_many(root,'motor',paths)
    for c in cases[:2]:
        cohort=c['id'][-1];raw=read(source/cohort/'hidden/ELECTRICMOTORVIBRATIONS/labels.json')
        gold={r['sample_id']:r for r in raw};assert len(gold)==len(raw)
        scores[c['id']]={}
        for cid in c['candidate_ids']:
            pred=read(source/'controls'/c['id']/(cid+'_train_external_predictions.json'))['predictions']
            lookup={r['sample_id']:int(r['predicted_label']) for r in pred}
            assert len(lookup)==len(pred)==len(gold) and set(lookup)==set(gold)
            keys=sorted(gold);y=np.asarray([gold[i]['label'] for i in keys]);p=np.asarray([lookup[i] for i in keys]);g=np.asarray([gold[i]['group'] for i in keys])
            records=[]
            for group in sorted(set(g)):
                m=g==group;actual=np.unique(y[m]);assert len(actual)==1
                records.append(dict(group=str(group),n=int(m.sum()),correct=int((p[m]==y[m]).sum()),error=float(np.mean(p[m]!=y[m])),actual=int(actual[0]),majority_prediction=int(np.bincount(p[m],minlength=4).argmax())))
            metric=classification(y,p,list(range(4)))
            scores[c['id']][cid]=dict(loss=statistics.mean(r['error'] for r in records),unit='record-equal window error',records=records,
                record_majority_accuracy=statistics.mean(r['actual']==r['majority_prediction'] for r in records),**metric)
            audit.append(dict(case=c['id'],candidate=cid,row_identity_complete=True,independent_confusion_metrics=True,records=len(records)))
    c=cases[-1];catalog=read(root/'G2_CANDIDATE_BINDING.json');basis=catalog['ids'][:6]
    paths=['hidden/test.npz']+['evaluation282/basis/'+n+'/test/probabilities.json' for n in basis]
    gear=bind_many(root,'gear',paths)
    data=dict(np.load(gear/'hidden/test.npz',allow_pickle=False));arrays=[]
    for name in basis:
        r=read(gear/'evaluation282/basis'/name/'test/probabilities.json');assert r['classes']==list(range(6))
        order={str(i):j for j,i in enumerate(r['sample_ids'])};assert len(order)==len(data['labels']) and set(order)==set(data['sample_ids'])
        p=np.asarray(r['probabilities'])[[order[str(i)] for i in data['sample_ids']]]
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,rtol=0,atol=1e-8)
        arrays.append(p)
    arrays=np.stack(arrays,axis=1);scores[c['id']]={}
    for cid,w in zip(catalog['ids'],catalog['weights']):
        pred=np.einsum('nmc,m->nc',arrays,np.asarray(w)/sum(w)).argmax(1)
        metric=classification(data['labels'],pred,list(range(6)))
        scores[c['id']][cid]=dict(loss=1-metric['macro_f1'],unit='1-macroF1',**metric)
        audit.append(dict(case=c['id'],candidate=cid,row_identity_complete=True,independent_confusion_metrics=True))
    rows=[]
    for method,decisions in choices.items():
        for d in decisions:
            c=next(c for c in cases if c['id']==d['case']);s=scores[c['id']]
            losses=[v['loss'] for v in s.values()];lo,hi=min(losses),max(losses);scale=hi-lo
            old=s[d['incumbent']]['loss'];new=s[d['selected']]['loss']
            def normalize(x):return x/scale if scale>1e-12 else 0.
            rows.append(dict(method=method,case=c['id'],source=c['source'],incumbent=d['incumbent'],selected=d['selected'],
                loss=new,old_loss=old,best_loss=lo,range=scale,
                normalized_regret=normalize(new-lo),normalized_harm=normalize(max(0,new-old)),normalized_gain=normalize(max(0,old-new)),
                normalized_opportunity=normalize(old-lo),harmful=new>old+1e-12,updated=d['selected']!=d['incumbent'],
                oracle_hit=abs(new-lo)<1e-12,original_anchor=d['incumbent']==c['original_incumbent']))
    summaries={}
    for method in choices:
        rr=[r for r in rows if r['method']==method];summary=source_balanced(rr)
        summary.update(updates=sum(r['updated'] for r in rr),harmful_updates=sum(r['harmful'] for r in rr),oracle_hits=sum(r['oracle_hit'] for r in rr),
                       anchor_panel=source_balanced([r for r in rr if r['original_anchor']]))
        summaries[method]=summary
    normal=[e['id'] for e in episodes if e['completed']]
    m=summaries['M10_broad_support']
    promise=len(normal)==2 and all(m['normalized_regret']<summaries[n]['normalized_regret']-1e-12 and m['normalized_harm']<=summaries[n]['normalized_harm']+1e-12 for n in normal) and (m['gain_capture'] or 0)>0
    result=dict(utc=utc(),iteration='A329',cases=cases,scores=scores,summaries=summaries,rows=rows,paired_gate_decisions=gates,
        M10_meets_frozen_exploratory_promise_rule=bool(promise),new_fits=0,new_final_forwards=0,new_independent_physical_tasks=0,
        researcher_previously_knows_final_scores=True,confirmation_experiment=False,
        scope='19counterfactual incumbents in3 archival pools,2source families. Two Motor splits related;G2 one gear. No general prevalence or significance claim.')
    save(root/'FINAL_RESULTS329.json',result);save(OUT/'FINAL_RESULTS329.json',result)
    save(root/'INDEPENDENT_SCORE_AUDIT329.json',dict(passed=True,checks=audit,source_bindings_sha256=sha(root/'SOURCE_BINDINGS329.json')))
    save(OUT/'INDEPENDENT_SCORE_AUDIT329.json',read(root/'INDEPENDENT_SCORE_AUDIT329.json'))
    with (root/'decisions329.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import shutil
    shutil.copy2(root/'decisions329.csv',OUT/'decisions329.csv')
    update(root,status='scored_pending_execution_audit_and_archive',final_scores_joined=True,actual_running_agents=0)
    print(json.dumps(dict(methods={k:{a:v[a] for a in ['normalized_regret','normalized_harm','gain_capture','updates','harmful_updates']} for k,v in summaries.items()},promise=bool(promise))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
