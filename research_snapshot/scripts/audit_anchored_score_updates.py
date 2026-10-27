"""Independently check A448 score equations, nullspace identity and accounting."""
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root):
    result=read(root/'RESULT448.json');source=read(root/'INPUTS403.json');sealed=read(root/'SEALED448.json')
    checks=0;effects=[];ranks=[];max_normal=0.;max_equal_fit=0.;max_projection=0.
    assert sha(root/'INPUTS403.json')==read(root/'BINDING448.json')['sha256']
    assert len(source['pools'])==len(result['rows'])==len(sealed['models'])==20
    for row,pool,seal in zip(result['rows'],source['pools'],sealed['models']):
        fit=read(root/'fits'/row['id']/'MODELS.json');process=read(root/'fits'/row['id']/'PROCESS.json')
        assert process['returncode']==0 and fit['fits']==3
        assert sha(root/'fits'/row['id']/'MODELS.json')==seal['models_sha256']
        command=process['command']; assert '--unshare-net' in command
        mounted=[command[i+1] for i,x in enumerate(command) if x=='--ro-bind']
        assert str(root/'legal'/(row['id']+'.npz')) in mounted
        assert all('/data/yyf-zyq' not in x and 'INPUTS403' not in x for x in mounted)
        assert fit['input_sha256']==sha(root/'legal'/(row['id']+'.npz'))
        names=row['names'];m=len(names);legal=sorted(pool['members']['p00']['legal'])
        # Explicit class-by-class indicators: separate from the primary reshape.
        a=np.asarray([[int(pool['members'][n]['legal'][g][1]==label) for n in names] for g in legal for label in range(4)],float)
        target=np.asarray([int(pool['members']['p00']['legal'][g][0]==label) for g in legal for label in range(4)],float)
        e=np.zeros(m);e[0]=1
        zero=np.asarray(fit['weights']['unanchored_ls']);anchor=np.asarray(fit['weights']['anchored_ls']);ridge=np.asarray(fit['weights']['anchored_ridge'])
        eigen,rotation=np.linalg.eigh(a.T@a)
        keep=eigen>max(float(eigen.max())*1e-12,1e-12)
        # Integer vote designs here have a wide nonzero singular-value gap.
        assert int(keep.sum())==fit['rank']
        projection=rotation[:,keep]@rotation[:,keep].T
        for name,weights,prior,penalty in [('unanchored_ls',zero,np.zeros(m),0),('anchored_ls',anchor,e,0),('anchored_ridge',ridge,e,1)]:
            normal=a.T@(a@weights-target)+penalty*(weights-prior)
            residual=float(np.max(np.abs(normal)));max_normal=max(max_normal,residual)
            assert residual<1e-8
            assert np.max(np.abs((np.eye(m)-projection)@(weights-prior)))<1e-9
            assert abs(float(np.sum((a@weights-target)**2))-fit['training_squared_error'][name])<1e-9
            checks+=3
        fit_diff=float(np.max(np.abs(a@anchor-a@zero)));max_equal_fit=max(max_equal_fit,fit_diff);assert fit_diff<1e-9
        delta=(np.eye(m)-projection)@e
        identity=float(np.max(np.abs(anchor-zero-delta)));max_projection=max(max_projection,identity);assert identity<1e-9
        assert np.max(np.abs(delta-np.asarray(fit['incumbent_null_component'])))<1e-9
        assert abs(float(np.linalg.norm(delta))-row['null_norm'])<1e-9
        ranks.append(dict(case=row['id'],rank=fit['rank'],columns=m,incumbent_unidentified=bool(np.linalg.norm(delta)>1e-9)))
        # Rebuild actual scores from scalar class tallies, including fixed tie rule.
        for i,g in enumerate(row['groups']):
            truth=pool['members']['p00']['final'][g][0];base=pool['members']['p00']['final'][g][1]
            assert truth==row['truth'][i] and base==row['base'][i]
            for method in ['anchored_ls','unanchored_ls','anchored_ridge','equal_vote']:
                weights=fit['weights'][method] if method!='equal_vote' else [1]*m
                scores=[sum(w for n,w in zip(names,weights) if pool['members'][n]['final'][g][1]==k) for k in range(4)]
                tied=[k for k in range(4) if abs(scores[k]-max(scores))<=1e-10]
                chosen=base if base in tied else min(tied)
                assert chosen==row['predictions'][method][i];checks+=1
            old=row['predictions']['unanchored_ls'][i];new=row['predictions']['anchored_ls'][i]
            if old!=new:effects.append(dict(case=row['id'],cohort=row['cohort'],group=g,actual=truth,incumbent=base,unanchored=old,anchored=new,improved=int(new==truth)-int(old==truth)))
        assert row['anchored_vs_unanchored_changes']==sum(e['case']==row['id'] for e in effects)
        for rule,pred in row['predictions'].items():
            n=len(pred)
            values=dict(error=Fraction(sum(x!=y for x,y in zip(pred,row['truth'])),n),harm=Fraction(sum(x!=y and b==y for x,y,b in zip(pred,row['truth'],row['base'])),n),gain=Fraction(sum(x==y and b!=y for x,y,b in zip(pred,row['truth'],row['base'])),n))
            assert all(Fraction(row['scores'][rule][k])==v for k,v in values.items());checks+=3
        for field,key in [('legal','legal_best'),('cv','cv_best')]:
            errors={n:Fraction(sum(y!=p for y,p in pool['members'][n][field].values()),len(pool['members'][n][field])) for n in names}
            winner=min(names,key=lambda n:(errors[n],names.index(n)))
            assert winner==row[key];checks+=1
        checks+=12
    # Check the preexisting ordinary trajectories, with A342 decision-stage choices.
    rows_by_id={r['id']:r for r in result['rows']}
    for n in result['native']:
        row=rows_by_id[n['case']];pool=source['pools'][int(n['case'])]
        pred=[pool['members'][n['selected']]['final'][g][1] for g in row['groups']]
        truth=row['truth'];base=row['base'];count=len(pred)
        expected=dict(error=Fraction(sum(p!=y for p,y in zip(pred,truth)),count),harm=Fraction(sum(p!=y and b==y for p,y,b in zip(pred,truth,base)),count),gain=Fraction(sum(p==y and b!=y for p,y,b in zip(pred,truth,base)),count))
        assert all(Fraction(n['scores']['native'][k])==v for k,v in expected.items());checks+=3
    panels=[('all',result['rows']),('eligible',[r for r in result['rows'] if r['complete'] and not r['registry_deviation']]),('native_aggregate',result['native'])]
    for field,rows in panels:
        groups=sorted({r['cohort'] for r in rows});expected=result[field]
        for rule in rows[0]['scores']:
            for metric in ['error','harm','gain']:
                values=[]
                for group in groups:
                    members=[r for r in rows if r['cohort']==group]
                    value=sum((Fraction(r['scores'][rule][metric]) for r in members),Fraction())/len(members)
                    assert value==Fraction(expected['cohorts'][group][rule][metric]);values.append(value);checks+=1
                assert sum(values,Fraction())/len(values)==Fraction(expected['means'][rule][metric]);checks+=1
    for comparison,g in result['gates'].items():
        panel=result['native_aggregate'] if comparison=='native' else result['all']
        diffs=[Fraction(v['anchored_ls']['error'])-Fraction(v[comparison]['error']) for v in panel['cohorts'].values()]
        assert g['wins']==sum(x<0 for x in diffs) and g['ties']==sum(x==0 for x in diffs) and g['losses']==sum(x>0 for x in diffs)
        before,after=panel['means'][comparison],panel['means']['anchored_ls']
        gate=sum(x<0 for x in diffs)>=2 and Fraction(after['error'])<Fraction(before['error']) and Fraction(after['harm'])<=Fraction(before['harm']) and Fraction(after['gain'])>=Fraction(before['gain'])
        assert gate==g['passed'];checks+=2
    assert result['primary_passed']==all(x['passed'] for x in result['gates'].values())
    assert len(result['native'])==13 and result['complete_non_deviating_pools']==13
    assert result['new_supervised_combiners']==sealed['supervised_combiners_fitted']==60
    for path,digest in read(root/'FREEZE448.json').items(): assert sha(root/path)==digest;checks+=1
    audit=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,parameter_audit='Independent symmetric-eigen projection, normal equations and nullspace constraints; not a second PHM fit',max_normal_equation_residual=max_normal,max_legal_score_difference=max_equal_fit,max_prior_projection_identity_error=max_projection,ranks=ranks,incumbent_unidentified_pools=sum(r['incumbent_unidentified'] for r in ranks),external_changes=effects,supervised_combiner_fits=60,new_audit_fits=0,new_LLM_calls=0,scope='Same archived measurement cache; independent algebra and scoring, not independent raw acquisition or source reader.')
    (root/'AUDIT448.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['passed','checks','max_normal_equation_residual','max_legal_score_difference','incumbent_unidentified_pools','external_changes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
