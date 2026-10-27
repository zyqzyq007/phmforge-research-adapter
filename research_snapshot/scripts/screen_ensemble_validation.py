"""EnsV selection and a frozen parent-reference contrast on real saved predictions."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def probs(path, classes):
    result=[]
    rows=load(path)['predictions']
    for row in rows:
        value=row.get('probabilities',row.get('class_probabilities'))
        vector=[value[str(k)] for k in classes] if isinstance(value,dict) else value
        assert str(classes[int(np.argmax(vector))])==str(row['predicted_label'])
        result.append(vector)
    return normalize(np.asarray(result)),[str(r['sample_id']) for r in rows]


def normalize(p):
    assert np.isfinite(p).all() and (p>=0).all()
    assert np.allclose(p.sum(-1),1,atol=1e-6,rtol=0)
    # Preserve original saved-program arithmetic, including floating point ties.
    # Renormalizing each parent can change an emitted class at a tied maximum.
    return p


def choose(run):
    started=time.monotonic()
    source=run/'inputs/a415'
    previous=load(source/'CHOICES_SEALED415.json')
    pairs=[]
    bundles={}
    for pair in previous['pairs']:
        ref,ids=probs(source/pair['paths']['reference'],pair['classes'])
        candidate,cids=probs(source/pair['paths']['Agent'],pair['classes'])
        assert ids==cids and len(ids)==len(set(ids))
        matrix=np.stack([ref,candidate])
        role=matrix.mean(0).argmax(-1)
        scores=(matrix.argmax(-1)==role).sum(-1)
        selected=['reference','Agent'][int(scores.argmax())]
        pairs.append(dict(id=pair['id'],source_cluster=pair['source_cluster'],n=len(ids),
                          agreement_counts=scores.tolist(),selected=selected,parent_balanced_selection=None,
                          reason='Opaque complete-program pair; internal parent predictions not available, so no parent-intervention claim.'))
        bundles[pair['id']]=(matrix,matrix.mean(0))
    parent_probs=[]
    ids=None
    for name in previous['bank_names']:
        value=load(source/'inputs/g2/evaluation282/basis'/name/'test/probabilities.json')
        assert value['classes']==previous['bank_classes']
        if ids is not None:assert ids==value['sample_ids']
        ids=value['sample_ids']
        parent_probs.append(normalize(np.asarray(value['probabilities'],dtype=float)))
    basis=np.stack(parent_probs)
    cache={}
    cases=[]
    parent_role=basis.mean(0)
    for case in previous['banks']:
        weights=np.array([c['weights'] for c in case['candidates']],dtype=float)/12
        # Identical to A298's original candidate-program evaluation expression.
        original_order=np.stack(parent_probs,axis=1)
        matrix=np.stack([np.einsum('nbc,b->nc',original_order,np.asarray(c['weights'],dtype=np.float64))/12
                         for c in case['candidates']])
        matrix=normalize(matrix)
        original_role=matrix.mean(0)
        roles={'EnsV':original_role,'parent_balanced_EnsV':parent_role}
        selections={}
        agreements={}
        for name,role in roles.items():
            counts=(matrix.argmax(-1)==role.argmax(-1)).sum(-1)
            selected=int(counts.argmax())
            selections[name]=case['candidates'][selected]['id']
            agreements[name]=counts.tolist()
            bundles[case['stage']+'/'+case['case']+'/'+name]=(matrix,role)
        effective_parent_weights=weights.mean(0)
        assert np.allclose(original_role,np.einsum('b,bnc->nc',effective_parent_weights,basis),atol=1e-12,rtol=0)
        cases.append(dict(stage=case['stage'],case=case['case'],candidate_ids=[c['id'] for c in case['candidates']],
                          candidate_weights=[c['weights'] for c in case['candidates']],ordinary=case['ordinary_choice'],
                          selected=selections,agreement_counts=agreements,n=len(ids),
                          original_effective_parent_weights=effective_parent_weights.tolist(),
                          parent_reference_weights=[1/len(basis)]*len(basis),
                          role_prediction_changes=int((original_role.argmax(-1)!=parent_role.argmax(-1)).sum()),
                          parent_balancing_changed_choice=selections['EnsV']!=selections['parent_balanced_EnsV']))
    # Execute exactly the author's three selection-score statements on actual probabilities.
    # Their upstream image training and evaluation logging are deliberately not represented as reproduced.
    import torch
    torch.set_num_threads(1)
    module=ast.parse((run/'literature/ensv.py').read_text())
    method=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='ensv')
    score_nodes=[node for node in ast.walk(method) if isinstance(node,(ast.Assign,ast.Expr)) and getattr(node,'lineno',0) in [200,201,202]]
    assert [node.lineno for node in sorted(score_nodes,key=lambda x:x.lineno)]==[200,201,202]
    score_code=compile(ast.Module(body=sorted(score_nodes,key=lambda x:x.lineno),type_ignores=[]),'pinned_author_score_kernel','exec')
    select_node=next(node for node in ast.walk(method) if isinstance(node,ast.Assign) and getattr(node,'lineno',0)==208)
    select_code=compile(ast.Module(body=[select_node],type_ignores=[]),'pinned_author_selection_kernel','exec')
    audit=[]
    for key,(matrix,role) in bundles.items():
        env=dict(torch=torch,ensem_pl=torch.from_numpy(role).argmax(-1),score_list=[])
        for candidate in matrix:
            env['idx_pred']=torch.from_numpy(candidate)
            exec(score_code,env)
        exec(select_code,env)
        independent=np.asarray(env['score_list'])
        expected=(matrix.argmax(-1)==role.argmax(-1)).mean(-1)
        assert np.allclose(independent,expected,atol=1e-7,rtol=0)
        assert env['best_index']==int(expected.argmax())
        audit.append(dict(key=key,author_kernel_agrees=True,candidates=len(matrix),selected_index=env['best_index']))
    seconds=time.monotonic()-started
    result=dict(utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(run/'PROTOCOL416.json'),
                official_code_sha256=sha(run/'literature/ensv.py'),pairs=pairs,banks=cases,
                author_scoring_kernel_parity=audit,seconds=seconds,final_score_metrics_read=False,
                new_LLM=0,new_fits=0,new_model_forwards=0,new_independent_scenarios=0,confirmation_opened=False)
    assert seconds<60
    save(run/'CHOICES_SEALED416.json',result)
    print(json.dumps(dict(seconds=seconds,pairs=pairs,bank_changes=[{k:c[k] for k in ['stage','case','selected','role_prediction_changes','parent_balancing_changed_choice']} for c in cases]),indent=2))


def metric_check(metric):
    cm=np.asarray(metric['confusion'],dtype=int)
    correct=int(np.trace(cm));n=int(cm.sum())
    assert abs(metric['accuracy']-correct/n)<1e-12
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(len(cm)),where=den!=0).mean()
    assert abs(metric['macro_f1']-f1)<1e-12


def score(run):
    started=time.monotonic()
    seal=load(run/'CHOICES_SEALED416.json')
    assert not seal['final_score_metrics_read']
    previous=load(run/'inputs/a415/RESULT415.json')
    projection=load(run/'inputs/a415/SOURCE_METRIC_PROJECTION415.json')
    pair_results=[]
    for row in seal['pairs']:
        old=next(r for r in previous['pairs'] if r['id']==row['id'])
        selected=row['selected']
        metrics=old['quality'][selected]
        pair_results.append(dict(id=row['id'],source_cluster=row['source_cluster'],selected=selected,
                                 EnsV_accuracy=metrics['accuracy'],ordinary_accuracy=old['outcomes']['ordinary'],
                                 keep_accuracy=old['outcomes']['keep_original'],
                                 selection_loss=max(v['accuracy'] for v in old['quality'].values())-metrics['accuracy'],
                                 harmful_change_loss=max(0.,old['outcomes']['keep_original']-metrics['accuracy']),
                                 gain_capture=old['gain_capture']['keep_original' if selected=='reference' else 'ordinary'],
                                 parent_balanced_result=None))
    bank_results=[]
    checks=0
    for row in seal['banks']:
        projected=next(r for r in projection['cases'] if r['stage']==row['stage'] and r['case']==row['case'])
        old=next(r for r in previous['banks'] if r['stage']==row['stage'] and r['case']==row['case'])
        byid={r['id']:r for r in projected['candidates']}
        for ident,weights in zip(row['candidate_ids'],row['candidate_weights']):
            assert byid[ident]['weights']==weights
            metric_check(byid[ident]['final']);checks+=1
        oracle=max(r['final']['macro_f1'] for r in byid.values())
        outcomes={key:dict(id=ident,accuracy=byid[ident]['final']['accuracy'],macro_f1=byid[ident]['final']['macro_f1'],
                           selection_loss=oracle-byid[ident]['final']['macro_f1']) for key,ident in row['selected'].items()}
        for key in ['ordinary','fixed_CV','fixed_development']:
            value=old['outcomes'][key]
            outcomes[key]=dict(id=value['id'],accuracy=value['accuracy'],macro_f1=value['macro_f1'],
                               selection_loss=old['macro_f1_selection_loss'][key])
        bank_results.append(dict(stage=row['stage'],case=row['case'],outcomes=outcomes,
                                 parent_balancing_changed_choice=row['parent_balancing_changed_choice'],
                                 available_gain_capture=None,reason='No original shared deployment incumbent for this candidate search; do not fabricate gain capture.'))
    aggregate={};gates=[]
    for phase in ['search','review']:
        rows=[r for r in bank_results if r['stage']==phase]
        assert len(rows)==6
        aggregate[phase]={key:{metric:float(np.mean([r['outcomes'][key][metric] for r in rows]))
                               for metric in ['accuracy','macro_f1','selection_loss']} for key in rows[0]['outcomes']}
        for comparator in ['ordinary','EnsV']:
            method=aggregate[phase]['parent_balanced_EnsV'];baseline=aggregate[phase][comparator]
            gates.append(dict(phase=phase,comparator=comparator,
                              passed=bool(method['macro_f1']>baseline['macro_f1']+1e-12 and method['accuracy']>=baseline['accuracy']-1e-12)))
    bindings=load(run/'RESTORED416.json')['bindings']
    for binding in bindings:
        p=run/binding['local'];assert p.stat().st_size==binding['bytes'] and sha(p)==binding['sha256'];checks+=1
    seconds=time.monotonic()-started
    assert seconds+seal['seconds']<60
    result=dict(utc=datetime.now(timezone.utc).isoformat(),choice_seal_sha256=sha(run/'CHOICES_SEALED416.json'),
                pairs=pair_results,banks=bank_results,aggregate=aggregate,primary_gate_components=gates,
                primary_gate_passed=all(g['passed'] for g in gates),candidate_metrics_rechecked=checks,
                seconds=seconds,new_LLM=0,new_fits=0,new_model_forwards=0,new_independent_scenarios=0,
                confirmation_opened=False,goal_complete=False,
                interpretation='Paper score-kernel reproduction and one-source dependency-weighting ablation only. Caches already scored historically; no novel method or independent superiority claim.')
    save(run/'RESULT416.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['banks']},indent=2))


def audit(run):
    started=time.monotonic()
    seal=load(run/'CHOICES_SEALED416.json');result=load(run/'RESULT416.json')
    assert result['choice_seal_sha256']==sha(run/'CHOICES_SEALED416.json')
    previous=load(run/'inputs/a415/CHOICES_SEALED415.json')
    basis=[]
    for name in previous['bank_names']:
        v=load(run/'inputs/a415/inputs/g2/evaluation282/basis'/name/'test/probabilities.json')
        basis.append([[float(x) for x in p] for p in v['probabilities']])
    n=len(basis[0]);nc=len(basis[0][0]);nb=len(basis)
    # Direct scalar sums independently reconstruct each real candidate's hard labels.
    unique={tuple(w) for c in seal['banks'] for w in c['candidate_weights']}
    hard={};rounding=[];original_order=np.stack(basis,axis=1)
    for weights in unique:
        original=np.einsum('nbc,b->nc',original_order,np.asarray(weights,dtype=np.float64))/12
        exact=np.asarray([[math.fsum(weights[b]*basis[b][i][j] for b in range(nb))/12
                           for j in range(nc)] for i in range(n)])
        bound=32*np.finfo(float).eps*np.maximum(1,np.abs(exact).max(1))
        assert np.all(np.max(abs(original-exact),axis=1)<=bound)
        oh=original.argmax(1);eh=exact.argmax(1)
        for i in np.flatnonzero(oh!=eh):
            assert exact[i,eh[i]]-exact[i,oh[i]]<=2*bound[i]
            rounding.append(dict(weights=list(weights),sample=int(i),original_label=int(oh[i]),scalar_label=int(eh[i]),
                                 scalar_top_gap=float(exact[i,eh[i]]-exact[i,oh[i]]),bound=float(2*bound[i])))
        hard[weights]=oh.tolist()
    comparisons=[]
    for row in seal['banks']:
        weights=row['candidate_weights'];m=len(weights)
        effective=[math.fsum(w[j]/12 for w in weights)/m for j in range(nb)]
        assert max(abs(x-y) for x,y in zip(effective,row['original_effective_parent_weights']))<1e-12
        original_matrix=np.stack([np.einsum('nbc,b->nc',original_order,np.asarray(w,dtype=np.float64))/12 for w in weights])
        for method,ww in [('EnsV',effective),('parent_balanced_EnsV',[1/nb]*nb)]:
            independent=np.asarray([[math.fsum(ww[b]*basis[b][i][j] for b in range(nb)) for j in range(nc)] for i in range(n)])
            role=original_matrix.mean(0) if method=='EnsV' else np.asarray(basis).mean(0)
            bound=32*np.finfo(float).eps*np.maximum(1,np.abs(independent).max(1))
            assert np.all(np.max(abs(role-independent),axis=1)<=bound)
            reference=role.argmax(1).tolist()
            for i,(actual,precise) in enumerate(zip(reference,independent.argmax(1))):
                assert independent[i,precise]-independent[i,actual]<=2*bound[i]
            counts=[sum(a==b for a,b in zip(hard[tuple(w)],reference)) for w in weights]
            assert counts==row['agreement_counts'][method]
            selected=row['candidate_ids'][counts.index(max(counts))]
            assert selected==row['selected'][method]
            comparisons.append(dict(stage=row['stage'],case=row['case'],method=method,scalar_reconstruction_matches=True))
    for phase,summary in result['aggregate'].items():
        rows=[x for x in result['banks'] if x['stage']==phase]
        for method,metrics in summary.items():
            for metric,value in metrics.items():
                assert abs(value-math.fsum(x['outcomes'][method][metric] for x in rows)/len(rows))<1e-12
    assert not result['primary_gate_passed'] or all(g['passed'] for g in result['primary_gate_components'])
    elapsed=time.monotonic()-started
    report=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,seconds=elapsed,
                scalar_candidate_prediction_vectors=len(hard),window_rows_per_candidate=n,comparisons=comparisons,
                original_author_kernel_comparisons=len(seal['author_scoring_kernel_parity']),
                candidate_confusions_and_bindings_rechecked=result['candidate_metrics_rechecked'],
                total_numeric_seconds=elapsed+result['seconds']+seal['seconds'],
                rounding_ambiguities=rounding,original_emitted_label_semantics_retained=True,
                new_LLM=0,new_fits=0,new_model_forwards=0,confirmation_opened=False,
                scope='Independent scalar arithmetic, exact author selection statements, and prior verified metric binding; no new raw-signal inference.')
    recovery=load(run/'IMPLEMENTATION_RECOVERY416.json')
    assert report['total_numeric_seconds']<recovery['remaining_numeric_seconds']
    report['cumulative_numeric_seconds_upper_bound']=60-recovery['remaining_numeric_seconds']+report['total_numeric_seconds']
    save(run/'AUDIT416.json',report)
    print({k:v for k,v in report.items() if k!='comparisons'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['choose','score','audit'])
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='choose':choose(args.run)
    elif args.mode=='score':score(args.run)
    else:audit(args.run)
