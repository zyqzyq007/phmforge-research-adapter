"""Recount paired program holdouts from saved probabilities, then seal choices.

No fitting, prediction, threshold tuning or new final-test evaluation occurs.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

CANDIDATES=['P00','P10','P01','P11','P11B']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def err(report):return Fraction(report['record_total']-report['record_correct'],report['record_total'])


def main(root):
    import numpy as np
    assert json.loads((root/'TERMINAL391.json').read_text())['exit_code']==0
    plan=json.loads((root/'PROTOCOL391.json').read_text());freeze=json.loads((root/'FREEZE391.json').read_text())
    assert sha(root/'PROTOCOL391.json')==freeze['protocol_sha256']
    assert sha(root/'code/probe_ordered_support.py')==freeze['code_sha256']
    assert sha(root/'MODEL_CONFIG.json')==freeze['model_config_sha256']
    assert sha(root/'EVIDENCE391.json')==json.loads((root/'EVIDENCE_SEAL391.json').read_text())['sha256']
    for item in plan['bindings']:assert sha(item['source_path'])==item['sha256']
    evidence=json.loads((root/'EVIDENCE391.json').read_text());audited=[]
    for task in plan['tasks']:
        v=evidence[task['id']];assert v['k']==task['k'] and v['kind']==task['kind']
        assert len(v['fit_groups'])==task['k']*10 and len(v['held_groups'])==(5-task['k'])*10
        assert set(v['fit_groups']).isdisjoint(v['held_groups'])
        loads=plan['geometry']['loads'];k=task['k']
        expected=loads[:k] if task['kind']=='low' else (loads[-k:] if task['kind']=='high' else plan['geometry']['interior'+str(k)])
        assert v['fit_loads']==expected and v['held_loads']==[l for l in loads if l not in expected]
        assert v['classifier_fits']==(0 if task['reuse'] else 5) and v['pipeline_prediction_calls']==(5 if task['reuse'] else 10)
        with np.load(root/'workers'/task['id']/'probabilities.npz',allow_pickle=False) as z:
            y=z['labels'];groups=z['groups'];load=z['loads'];assert len(set(z['sample_ids']))==len(y)
            assert np.allclose(z['P11B'],.9*z['P11']+.1*z['RAW_LR'],rtol=0,atol=1e-15)
            for l in set(load):assert set(y[load==l])==set(range(5))
            for c in CANDIDATES:
                probability=z[c];assert probability.shape==(len(y),5) and np.isfinite(probability).all() and np.allclose(probability.sum(1),1)
                pred=probability.argmax(1);report=v['reports'][c];records={r['group']:r for r in report['records']}
                assert set(records)==set(groups);ncorrect=0
                for g in set(groups):
                    m=groups==g;truth=int(np.unique(y[m]).item());l=float(np.unique(load[m]).item())
                    vote=int(np.bincount(pred[m],minlength=5).argmax());r=records[g]
                    assert (r['actual'],r['predicted'],r['load'],r['windows'],r['correct_windows'])==(truth,vote,l,int(m.sum()),int((y[m]==pred[m]).sum()))
                    ncorrect+=int(truth==vote)
                assert ncorrect==report['record_correct'] and len(records)==report['record_total']
                assert abs(float(err(report))-report['record_error'])<1e-12
                cm=[[int(np.sum((y==a)&(pred==b))) for b in range(5)] for a in range(5)]
                assert cm==report['window_confusion'];correct=sum(cm[i][i] for i in range(5));f1=[]
                for i in range(5):
                    den=sum(cm[i])+sum(row[i] for row in cm);f1.append(Fraction(2*cm[i][i],den) if den else Fraction(0))
                assert report['window_total']==len(y) and report['window_correct']==correct
                assert abs(report['window_accuracy']-correct/len(y))<1e-12
                assert abs(report['window_macro_f1']-float(sum(f1)/5))<1e-12
                for l,s in report['by_load'].items():
                    rr=[r for r in records.values() if r['load']==float(l)]
                    assert s==dict(record_correct=sum(r['actual']==r['predicted'] for r in rr),record_total=len(rr))
                audited.append(dict(task=task['id'],candidate=c,records=len(records),windows=len(y),correct=ncorrect))
    blocks={};choices={}
    for block in ['block78','block910']:
        original=json.loads((Path(plan['source'])/'bank'/block/'VALIDATION_REPORTS.json').read_text())
        rows={};exact={}
        for c in CANDIDATES:
            extra=[err(evidence[f'{block}_{direction}{k}']['reports'][c]) for direction in ['low','high'] for k in [2,3,4]]
            extra_by_k={str(k):sum(err(evidence[f'{block}_{d}{k}']['reports'][c]) for d in ['low','high'])/2 for k in [2,3,4]}
            mid=original[c]['cv']['by_load'];interior4=sum(Fraction(mid[str(l)]['record_total']-mid[str(l)]['record_correct'],mid[str(l)]['record_total']) for l in [1.,1.5,2.])/3
            interior={str(k):err(evidence[f'{block}_interior{k}']['reports'][c]) for k in [2,3]};interior['4']=interior4
            exact[c]=dict(extrapolation=sum(extra)/6,interior=sum(interior.values())/3)
            rows[c]=dict(extrapolation_mean=float(exact[c]['extrapolation']),extrapolation_by_k={k:float(v) for k,v in extra_by_k.items()},interior_mean=float(exact[c]['interior']),interior_by_k={k:float(v) for k,v in interior.items()},ordinary_CV_error=float(err(original[c]['cv'])))
        choice=min(CANDIDATES,key=lambda c:(exact[c]['extrapolation'],CANDIDATES.index(c)))
        contrast=(exact['P11B']['extrapolation']-exact['P00']['extrapolation'])-(exact['P11B']['interior']-exact['P00']['interior'])
        gates=dict(chose_original_equivalent=choice in ['P00','P01'],beneficial_intermediate_not_ranked_worse=exact['P11B']['extrapolation']<=exact['P11']['extrapolation'],geometry_contrast_positive=contrast>0)
        blocks[block]=dict(candidates=rows,selected=choice,geometry_difference_in_differences=float(contrast),geometry_difference_exact=dict(numerator=contrast.numerator,denominator=contrast.denominator),qualification_gates=gates,qualification_passed=all(gates.values()))
        choices[block]=choice
    save(root/'CHOICES391.json',dict(choices=choices,sealed_utc=datetime.now(timezone.utc).isoformat(),rule=plan['primary'],before_final_join=True))
    result=dict(blocks=blocks,joint_qualification_passed=all(b['qualification_passed'] for b in blocks.values()),new_classifier_fits=sum(v['classifier_fits'] for v in evidence.values()),new_scaler_fits=sum(v['scaler_fits'] for v in evidence.values()),pipeline_prediction_calls=sum(v['pipeline_prediction_calls'] for v in evidence.values()),reused_endpoint_models=20,new_LLM_calls=0,new_final_predictions=0,seconds=json.loads((root/'TERMINAL391.json').read_text())['seconds'],independent_confirmation=False,Agent_advantage_claim=False,novelty_claim=False,goal_achieved=False)
    assert result['new_classifier_fits']==60 and result['new_scaler_fits']==12 and result['pipeline_prediction_calls']==140
    save(root/'RESULT391.json',result)
    save(root/'RECOUNT391.json',dict(passed=True,source_bindings=len(plan['bindings']),recomputed_reports=len(audited),reports=audited,result_sha256=sha(root/'RESULT391.json'),choices_sha256=sha(root/'CHOICES391.json'),classifier_fits_not_repeated=True))
    # Outcomes were previously known to the researcher, never numerical workers.
    # This join occurs after the newly measured rule choices have been sealed.
    source=Path(plan['source']);path=source/'FINAL_RESULTS325.json'
    index={v['path']:v for v in json.loads((source/plan['source_manifest_filename']).read_text())['files']}
    assert sha(path)==index['FINAL_RESULTS325.json']['sha256']
    final=json.loads(path.read_text())['candidate_scores'];joined={}
    for block,b in blocks.items():
        # The interior-minimum selector is descriptive, added after qualification.
        interior=min(CANDIDATES,key=lambda c:(b['candidates'][c]['interior_mean'],CANDIDATES.index(c)))
        selections={'frozen_range_rule':b['selected'],'keep':'P00','historical_ordinary':'P11B','posthoc_interior_min':interior}
        entries={}
        for name,c in selections.items():
            s=final[block][c];error=Fraction(s['record_total']-s['record_correct'],s['record_total'])
            old=err(final[block]['P00']);oracle=min(err(v) for v in final[block].values())
            assert abs(float(error)-s['record_error'])<1e-12
            entries[name]=dict(candidate=c,correct=s['record_correct'],total=s['record_total'],error=float(error),regret=float(error-oracle),harm=float(max(Fraction(0),error-old)),gain=float(max(Fraction(0),old-error)))
        joined[block]=entries
    aggregate={name:{metric:sum(joined[b][name][metric] for b in joined)/len(joined) for metric in ['error','regret','harm','gain']} for name in ['frozen_range_rule','keep','historical_ordinary','posthoc_interior_min']}
    save(root/'HISTORICAL_JOIN391.json',dict(source=str(path),sha256=sha(path),blocks=joined,mean=aggregate,gain_capture='N/A: no better-than-P00 candidate in this fixed pool',not_a_fair_Agent_comparison='Historical ordinary lacked the new evidence; no new Agent sessions. Interior-minimum selector is posthoc descriptive.',new_final_predictions=0))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
