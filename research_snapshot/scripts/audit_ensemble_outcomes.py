"""Read-only audit of sealed,actually registered ensemble outcomes and traces.

Never fits a model,searches a new weight,or changes a scientific result. Cold
development predictions are checked at class level,not only allclose values.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from controlled_feedback import save,sha,utc


def audit(root,output='POSTSCORE_AUDIT298.json'):
    original=sha(root/'ENSEMBLE_RESULTS.json');result=json.loads((root/'ENSEMBLE_RESULTS.json').read_text())
    bank=dict(np.load(root/'bank/probabilities.npz',allow_pickle=False));classes=bank['classes'];cold={}
    for part in ['val','test']:
        with np.load(root/'hidden'/(part+'.npz'),allow_pickle=False) as data:ids=data['sample_ids']
        values=[]
        for name in bank['basis_names']:
            path=root/'evaluation282/basis'/str(name)/part/'probabilities.json';raw=json.loads(path.read_text())
            assert raw['classes']==classes.tolist();positions={v:i for i,v in enumerate(raw['sample_ids'])}
            assert len(positions)==len(ids) and set(positions)==set(ids.tolist())
            values.append(np.asarray(raw['probabilities'])[[positions[str(i)] for i in ids]])
        cold[part]=np.stack(values,axis=1)
    with np.load(root/'hidden/test.npz',allow_pickle=False) as data:final_labels=data['labels']
    assert np.allclose(cold['val'],bank['development'],atol=1e-12,rtol=0)
    # Post-score mechanism falsification,not a search over new weights. A true
    # class dominated by one wrong class in every basis cannot become the
    # first maximum under any permitted nonnegative mixture.
    positions={int(c):i for i,c in enumerate(classes)};witnesses=[];strict_rows=[]
    for i,label in enumerate(final_labels):
        actual=positions[int(label)];matches=[];strict=False
        for wrong in range(len(classes)):
            if wrong==actual:continue
            diff=cold['test'][i,:,wrong]-cold['test'][i,:,actual]
            if np.all(diff>=0) and (wrong<actual or np.all(diff>0)):
                matches.append(dict(competitor=int(classes[wrong]),minimum_basis_margin=float(diff.min()),tie_precedes_actual=wrong<actual))
            if np.all(diff>1e-12):strict=True
        if matches:witnesses.append(dict(row=i,actual=int(label),witnesses=matches))
        if strict:strict_rows.append(i)
    blocked=np.asarray([x['row'] for x in witnesses],dtype=int)
    actual={};rows=[];representatives=set();source_metric_conflicts=[];trace=[];code_flags=[];artifacts=[];errors=[];http=Counter();models=set()
    for case in result['cases']:
        for candidate in case['candidates']:
            key=tuple(candidate['weights']);previous=actual.get(key)
            compact={part:{k:candidate[part][k] for k in ['accuracy','macro_f1','correct','total','confusion']} for part in ['train','cv','development','final']}
            if previous is not None and previous!=compact:source_metric_conflicts.append(dict(case=case['id'],weights=key))
            actual[key]=compact
        representatives.add(tuple(case['chosen']['weights']))
        best=next(x for x in case['candidates'] if x['id']==case['posthoc_actual_pool_best']);representatives.add(tuple(best['weights']))
        representatives.update(tuple(x['weights']) for x in case['candidates'] if x['kind']=='basis_anchor')
        folder=(root if case['stage']=='search' else root/'review')/case['id']
        job=json.loads((folder/'PLAN187.json').read_text())['jobs'][0];workspace=folder/job['workspace']
        tools=json.loads((folder/'audit/closed_choices'/job['id']/'completed_tools.json').read_text())
        for tool in tools:
            if tool['type']=='command_execution':
                cmd=tool['command']
                if re.search(r'\.(?:fit|fit_transform|partial_fit)\s*\(|\.backward\s*\(',cmd):
                    code_flags.append(dict(stage=case['stage'],case=case['id'],source='command',event_line=tool['event_line'],text=cmd))
                if tool.get('exit_code') not in [0,None]:errors.append(dict(stage=case['stage'],case=case['id'],event_line=tool['event_line'],exit_code=tool['exit_code']))
            elif tool['type']=='mcp_tool_call':
                name=tool.get('tool',tool.get('name',''))
                if 'train' in str(name).lower():code_flags.append(dict(stage=case['stage'],case=case['id'],source='MCP_training_request',tool=tool))
        for path in workspace.rglob('*.py'):
            if path.is_symlink():continue
            for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
                if re.search(r'\.(?:fit|fit_transform|partial_fit)\s*\(|\.backward\s*\(',line):
                    code_flags.append(dict(stage=case['stage'],case=case['id'],source='workspace_python',path=str(path.relative_to(root)),line=number,text=line))
        artifacts += [dict(stage=case['stage'],case=case['id'],path=str(p.relative_to(root))) for p in workspace.rglob('*')
                      if p.is_file() and p.suffix.lower() in ['.joblib','.pkl','.pickle','.pt','.pth']]
        for path in (folder/'model_gateway'/job['id']).glob('*_receipt.json'):
            receipt=json.loads(path.read_text());http[str(receipt['upstream_status'])]+=1;models.update(receipt.get('returned_model_ids',[]))
        t=case['feedback_trace'];trace.append(dict(stage=case['stage'],case=case['id'],registered_batches=len(t['registration_batches']),
            batch_sizes=[x['vectors'] for x in t['registration_batches']],early_development_evidence_requests=t['early_development_evidence_requests'],
            report=str((workspace/'report.md').relative_to(root)),completed_tools=str((folder/'audit/closed_choices'/job['id']/'completed_tools.json').relative_to(root)),
            actual_adaptive_use_requires_manual_trajectory_check=True))
    for weights,metrics in actual.items():
        w=np.asarray(weights,dtype=np.float64)
        warm=classes[(np.einsum('nbc,b->nc',bank['development'],w)/12).argmax(1)]
        replay=classes[(np.einsum('nbc,b->nc',cold['val'],w)/12).argmax(1)]
        final=classes[(np.einsum('nbc,b->nc',cold['test'],w)/12).argmax(1)]
        assert not len(blocked) or not np.any(final[blocked]==final_labels[blocked]),'Counterexample to dominance certificate'
        checks=None
        if weights in representatives:
            checks=dict(accuracy=abs(accuracy_score(final_labels,final)-metrics['final']['accuracy'])<1e-12,
                macro_f1=abs(f1_score(final_labels,final,labels=classes,average='macro',zero_division=0)-metrics['final']['macro_f1'])<1e-12,
                confusion=confusion_matrix(final_labels,final,labels=classes).tolist()==metrics['final']['confusion'])
            assert all(checks.values())
        rows.append(dict(weights=list(weights),warm_cold_development_changed=int(np.sum(warm!=replay)),
            selected_or_basis_or_actual_pool_best=weights in representatives,independent_sklearn_checks=checks))
    assert not source_metric_conflicts and sha(root/'ENSEMBLE_RESULTS.json')==original
    summary=dict(utc=utc(),original_result_sha256=original,registered_case_slots=sum(len(x['candidates']) for x in result['cases']),
        unique_actual_vectors=len(actual),no_new_vectors_searched=True,no_new_fits_or_model_inference=True,new_LLM_calls=0,
        repeated_vector_metric_conflicts=source_metric_conflicts,
        maximum_warm_cold_basis_development_probability_delta=float(np.abs(cold['val']-bank['development']).max()),
        vectors_with_any_class_replay_change=sum(x['warm_cold_development_changed']>0 for x in rows),
        selected_or_best_or_basis_vectors_with_class_replay_change=[x for x in rows if x['selected_or_basis_or_actual_pool_best'] and x['warm_cold_development_changed']],
        independent_sklearn_representative_vectors=len(representatives),all_sklearn_representative_checks_passed=True,
        convex_capability_bound=dict(exploratory_after_final_scoring=True,unavoidable_error_rows=len(witnesses),
            strict_margin_gt_1e_12_unavoidable_rows=len(strict_rows),final_rows=len(final_labels),
            upper_bound_accuracy_any_nonnegative_mixture=(len(final_labels)-len(witnesses))/len(final_labels),
            by_actual_class={str(c):dict(total=int(np.sum(final_labels==c)),unavoidable=sum(x['actual']==int(c) for x in witnesses)) for c in classes},
            witnesses=witnesses,no_new_weight_search=True,
            scope='A sufficient impossibility certificate for the fixed basis space only;not a tight best-score bound or an Agent failure.'),
        fitting_call_flags=code_flags,additional_model_like_workspace_artifacts=artifacts,
        fitting_audit_scope='All retained workspace Python including hidden runtime paths,and all completed command/MCP items;flags need semantic review,not an assertion that every possible dynamic operation is observable.',
        nonzero_command_exits=errors,actual_HTTP=dict(http),actual_returned_models=sorted(models),
        feedback_traces=trace,candidate_replay_checks=rows,final_scores_already_opened=True,
        scientific_claim='No automatic causal attribution;inspect traces,poor-model-space and fixed-selector rivals before adjudication.')
    assert Path(output).name==output
    save(root/'audit'/output,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['feedback_traces','candidate_replay_checks','fitting_call_flags','nonzero_command_exits','convex_capability_bound']},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',default='POSTSCORE_AUDIT298.json')
    args=parser.parse_args();audit(args.run.resolve(),args.output)
