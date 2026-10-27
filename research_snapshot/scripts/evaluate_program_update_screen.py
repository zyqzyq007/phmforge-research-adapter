"""External, independent scoring after all A328 choices have been sealed."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

from controlled_feedback import save, sha, utc
from screen_program_updates import SOURCES, OUT, update
from run_frozen_candidate_review import source_context


def classification(y, prediction, classes):
    index={int(k):i for i,k in enumerate(classes)}
    cm=np.zeros((len(classes),len(classes)),dtype=int)
    for a,b in zip(y,prediction):cm[index[int(a)],index[int(b)]]+=1
    den=cm.sum(0)+cm.sum(1)
    f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(len(classes)),where=den!=0).mean())
    acc=float(np.trace(cm)/cm.sum())
    assert np.array_equal(cm,confusion_matrix(y,prediction,labels=classes))
    assert abs(f1-f1_score(y,prediction,labels=classes,average='macro',zero_division=0))<1e-12
    return dict(accuracy=acc,macro_f1=f1,confusion=cm.tolist(),count=len(y),independent_recalculation_passed=True)


def source_balanced(rows):
    by_source=defaultdict(lambda:defaultdict(list))
    for r in rows:by_source[r['source']][r['case']].append(r)
    keys=['normalized_regret','normalized_harm','normalized_gain','normalized_opportunity','harmful','updated']
    summaries={}
    for source,pools in by_source.items():
        per_pool=[{k:statistics.mean(r[k] for r in rr) for k in keys} for rr in pools.values()]
        summaries[source]={k:statistics.mean(r[k] for r in per_pool) for k in keys}
        gain=summaries[source]['normalized_gain'];opp=summaries[source]['normalized_opportunity']
        summaries[source]['gain_capture']=gain/opp if opp>1e-12 else None
        summaries[source]['pools']=len(pools)
    overall={k:statistics.mean(x[k] for x in summaries.values()) for k in keys}
    overall['gain_capture']=overall['normalized_gain']/overall['normalized_opportunity'] if overall['normalized_opportunity']>1e-12 else None
    overall.update(sources=summaries,decisions=len(rows))
    return overall


def main(root):
    assert not (root/'FINAL_RESULTS328.json').exists(), 'No repeated score join'
    seal=json.loads((root/'ALL_CHOICES_SEALED.json').read_text())
    assert seal['final_score_join_not_yet_run']
    for r in seal['files']:assert sha(root/r['path'])==r['sha256'],r['path']
    save(root/'audit/EVALUATION_START.json',dict(utc=utc(),all_new_choices_sealed=True,
        researcher_previously_knows_scores=True,evaluator_sha256=sha(Path(__file__))))
    contexts={k:source_context(v) for k,v in SOURCES.items()};bindings=[]
    def bound(source,relative):
        p=SOURCES[source]/relative;record=contexts[source][0][relative]
        assert sha(p)==record['sha256']
        bindings.append(dict(source=source,path=relative,sha256=record['sha256']))
        return p
    def read(source,relative):return json.loads(bound(source,relative).read_text())
    cases=json.loads((root/'public/cases.json').read_text());scores={};audits=[]
    for case in cases:
        name=case['id'];scores[name]={}
        if case['source']=='rotor':
            block=name.removeprefix('rotor_')
            with np.load(bound('rotor',f'hidden/{block}/test.npz'),allow_pickle=False) as z:
                data={k:z[k] for k in ['sample_ids','labels','groups']}
            for cid in case['candidate_ids']:
                pred=read('rotor',f'evaluation/{block}/{cid}/predictions1.json')['predictions']
                second=read('rotor',f'evaluation/{block}/{cid}/predictions2.json')['predictions'];assert pred==second
                keyed={r['sample_id']:int(r['predicted_label']) for r in pred}
                assert len(keyed)==len(pred)==len(data['sample_ids']) and set(keyed)==set(data['sample_ids'])
                p=np.asarray([keyed[str(i)] for i in data['sample_ids']]);record_rows=[]
                for group in sorted(set(data['groups'])):
                    mask=data['groups']==group;actual=np.unique(data['labels'][mask]);assert len(actual)==1
                    voted=int(np.bincount(p[mask],minlength=5).argmax())
                    record_rows.append(dict(group=str(group),actual=int(actual[0]),predicted=voted))
                loss=sum(r['actual']!=r['predicted'] for r in record_rows)/len(record_rows)
                old_score=read('rotor',f'evaluation/{block}/{cid}/score.json')
                assert abs(loss-old_score['record_error'])<1e-12
                scores[name][cid]=dict(loss=loss,unit='whole-record error',records=record_rows,**classification(data['labels'],p,list(range(5))))
                audits.append(dict(case=name,candidate=cid,old_metric_matches=True,two_original_cold_predictions_match=True))
        elif case['source']=='gear':
            cohort=name.removeprefix('gear_')
            with np.load(bound('gear','hidden/test.npz'),allow_pickle=False) as z:
                ids=z['sample_ids'];labels=z['labels']
            for cid in case['candidate_ids']:
                rr=read('gear',f'evaluation282/{cohort}/{cid}/test/predictions.json')['predictions']
                lookup={r['sample_id']:r['predicted_label'] for r in rr}
                assert len(lookup)==len(rr)==len(ids) and set(lookup)==set(ids)
                pred=np.asarray([lookup[str(i)] for i in ids]);m=classification(labels,pred,[0,1,2])
                scores[name][cid]=dict(loss=1-m['macro_f1'],unit='1 - macro-F1',**m)
                audits.append(dict(case=name,candidate=cid,prediction_ids_match=True,independent_score=True))
        else:
            labels=read('rul','hidden/final_labels.json')
            assert isinstance(labels,list)
            for cid in case['candidate_ids']:
                rows=[]
                for gold in labels:
                    prediction=read('rul',f'evaluation278/bank_{cid}/final/{gold["sample_id"]}/predictions.json')['predictions']
                    assert len(prediction)==1 and prediction[0]['sample_id']==gold['sample_id']
                    p=prediction[0];assert p['unit_id']==gold['unit_id']
                    rows.append(dict(sample_id=gold['sample_id'],unit_id=gold['unit_id'],actual=gold['actual_rul'],predicted=p['predicted_rul']))
                losses={unit:statistics.mean(abs(r['predicted']-r['actual']) for r in rows if r['unit_id']==unit) for unit in sorted({r['unit_id'] for r in rows})}
                loss=statistics.mean(losses.values())
                independent=statistics.mean(mean_absolute_error([r['actual'] for r in rows if r['unit_id']==u],[r['predicted'] for r in rows if r['unit_id']==u]) for u in losses)
                assert abs(loss-independent)<1e-9
                scores[name][cid]=dict(loss=loss,unit='minutes MAE',per_bearing=losses,queries=rows,independent_recalculation_passed=True)
                audits.append(dict(case=name,candidate=cid,independent_score=True,queries=len(rows)))
    fixed=json.loads((root/'FIXED_CHOICES.json').read_text())['choices']
    choices=dict(fixed);agent_costs=[]
    for episode in seal['ordinary_episodes']:
        method=episode['id'];run=root/'episodes'/method
        rows=episode.get('choice',{}).get('choices') if episode.get('choice') else None
        if rows is None:rows=[dict(case=c['id'],incumbent=i,selected=i,reason='System fallback;missing Agent choice') for c in cases for i in c['candidate_ids']]
        choices[method]=rows
        path=run/'model_gateway'/method/'status.json';cost=json.loads(path.read_text()) if path.exists() else {}
        receipts=[json.loads(p.read_text()) for p in (run/'model_gateway').glob('*/*_receipt.json')]
        usage_missing=sum(r.get('upstream_status')==200 and not r.get('usage') for r in receipts)
        agent_costs.append(dict(id=method,completed=episode['completed'],fallback=episode['fallback'],seconds=episode.get('seconds'),
            model_requests=cost.get('requests',0),input_tokens=cost.get('input_tokens',0),output_tokens=cost.get('output_tokens',0),
            returned_model_ids=cost.get('returned_model_ids',[]),successful_receipts_missing_usage=usage_missing))
    rows=[];by_case={c['id']:c for c in cases}
    expected={(c['id'],i) for c in cases for i in c['candidate_ids']}
    for method,decisions in choices.items():
        assert {(r['case'],r['incumbent']) for r in decisions}==expected and len(decisions)==len(expected)
        for decision in decisions:
            c=by_case[decision['case']];bank=scores[c['id']]
            loss=bank[decision['selected']]['loss'];old=bank[decision['incumbent']]['loss']
            best=min(r['loss'] for r in bank.values());worst=max(r['loss'] for r in bank.values());scale=worst-best
            normalize=lambda x: x/scale if scale>1e-12 else 0.
            rows.append(dict(method=method,case=c['id'],source=c['source'],incumbent=decision['incumbent'],selected=decision['selected'],
                original_anchor=decision['incumbent']==c['original_incumbent'],loss=loss,incumbent_loss=old,oracle_loss=best,unit=bank[decision['selected']]['unit'],
                normalized_regret=normalize(loss-best),normalized_harm=normalize(max(loss-old,0)),normalized_gain=normalize(max(old-loss,0)),normalized_opportunity=normalize(old-best),
                harmful=float(loss>old+1e-12),updated=float(decision['selected']!=decision['incumbent'])))
    summaries={method:{'original_anchors':source_balanced([r for r in rows if r['method']==method and r['original_anchor']]),
                       'all_anchors':source_balanced([r for r in rows if r['method']==method])} for method in choices}
    ordinary=[r['id'] for r in agent_costs]
    promising=[]
    for method in fixed:
        if not method.startswith('M'):continue
        value=summaries[method]['original_anchors']
        if all(c['completed'] and not c['fallback'] for c in agent_costs) and all(
            value['normalized_regret']<summaries[o]['original_anchors']['normalized_regret']-1e-12 and
            value['normalized_harm']<=summaries[o]['original_anchors']['normalized_harm']+1e-12 for o in ordinary) and (value['gain_capture'] or 0)>0:
            promising.append(method)
    result=dict(iteration='A328',completed_utc=utc(),protocol_sha256=sha(root/'CAMPAIGN_CONFIG.json'),
        methods=10,controls=3,new_ordinary_agent_sessions=2,source_families=3,pools=9,
        original_anchor_decisions_per_method=9,all_anchor_decisions_per_method=88,
        actual_new_fits=0,new_final_forwards=0,independently_rescored_candidate_slots=88,
        researcher_previously_knows_final_scores=True,confirmation_experiment=False,
        summaries=summaries,agent_costs=agent_costs,candidate_scores=scores,decisions=rows,
        meets_frozen_exploratory_improvement_rule=promising,
        limits=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())['source_limitations'],
        interpretation='All ten proposals reported. Any advantage is retrospective development evidence; neither agent-specific novelty nor unobserved-condition generalization is established.')
    save(root/'audit/FINAL_SOURCE_BINDINGS.json',bindings);save(root/'audit/INDEPENDENT_SCORE_AUDIT.json',dict(passed=True,rows=audits))
    save(root/'FINAL_RESULTS328.json',result);save(OUT/'FINAL_RESULTS328.json',result)
    with (root/'decisions328.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil=__import__('shutil');shutil.copy2(root/'decisions328.csv',OUT/'decisions328.csv')
    update(root,status='complete_independently_scored_pending_archive',new_final_scores_joined=True,promising_methods=promising,
        normal_agent_sessions=sum(c['completed'] for c in agent_costs),new_fits=0)
    print(json.dumps(dict(promising=promising,summaries={k:v['original_anchors']['normalized_regret'] for k,v in summaries.items()},costs=agent_costs)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
