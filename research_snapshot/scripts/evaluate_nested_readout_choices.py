"""Join already-open exploration outcomes after nested source choices seal."""
import argparse
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np


def read(p): return json.loads(Path(p).read_text())


def evaluate(root):
    choice=read(root/'CHOICES_SEALED459.json');assert not choice['external_targets_joined']
    assert read(root/'AUDIT459.json')['passed']
    old=read(root/'evaluation_inputs/RESULT458.json')
    ref=read(root/'evaluation_inputs/NUMERICAL_REFERENCE_RESULT458.json')
    refs={r['program']:r for r in ref['rows']};selection={r['program']:r['selected'] for r in choice['choices']}
    rows=[]
    for p in old['programs']:
        name=p['program'];q=refs[name];y=np.array(p['truth']);parent=np.abs(np.array(p['parent_prediction'])-y)
        original=np.abs(np.array(p['original_prediction'])-y);learned=np.abs(np.array(q['prediction'])-y)
        assert p['ids']==q['ids'] and np.allclose(learned,q['errors'])
        error=dict(all_original=original,always_readout=learned,
            component_CV=np.array(p['metrics']['component_CV']['per_cell_error']),
            nested_CV=learned if selection[name]=='readout' else original)
        oracle_mean=min(float(original.mean()),float(learned.mean()))
        available_gain=float(np.maximum(parent-np.minimum(original,learned),0).mean())
        metrics={}
        for policy,e in error.items():
            delta=e-parent
            metrics[policy]=dict(MAE=float(e.mean()),harm=float(np.maximum(delta,0).mean()),
                gain=float(np.maximum(-delta,0).mean()),
                same_pair_selection_loss=None if policy=='component_CV' else float(e.mean()-oracle_mean),
                per_cell_error=e.tolist())
        rows.append(dict(program=name,ids=p['ids'],selected=selection[name],metrics=metrics,
            pair_oracle_mean_MAE=oracle_mean,available_pair_gain=available_gain,
            mean_win_vs_original=bool(error['nested_CV'].mean()<original.mean()-1e-8),
            mean_loss_vs_original=bool(error['nested_CV'].mean()>original.mean()+1e-8),
            query_wins=int(np.sum(error['nested_CV']<original-1e-8)),
            query_losses=int(np.sum(error['nested_CV']>original+1e-8))))
    denominator=float(np.mean([r['available_pair_gain'] for r in rows]));means={}
    for policy in ['all_original','always_readout','component_CV','nested_CV']:
        m={k:float(np.mean([r['metrics'][policy][k] for r in rows])) for k in ['MAE','harm','gain']}
        if policy!='component_CV':
            m['same_pair_selection_loss']=float(np.mean([r['metrics'][policy]['same_pair_selection_loss'] for r in rows]))
            m['improvement_utilization']=m['gain']/denominator
        means[policy]=m
    selected=means['nested_CV'];baseline=means['all_original']
    gate=dict(at_least_three_program_mean_wins=sum(r['mean_win_vs_original'] for r in rows)>=3,
        MAE_lower=selected['MAE']<baseline['MAE']-1e-8,
        selection_loss_lower=selected['same_pair_selection_loss']<baseline['same_pair_selection_loss']-1e-8,
        harm_not_higher=selected['harm']<=baseline['harm']+1e-8,
        gain_not_lower=selected['gain']>=baseline['gain']-1e-8,
        utilization_not_lower=selected['improvement_utilization']>=baseline['improvement_utilization']-1e-8)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,means=means,
        primary_exploratory_gate=gate,primary_exploratory_gate_passed=all(gate.values()),
        added_benefit_vs_always_readout=selected['MAE']<means['always_readout']['MAE']-1e-8,
        same_choices_as_always_readout=all(v=='readout' for v in selection.values()),
        original_success_preserved=not next(r for r in rows if r['program']=='A445__ordinary_2__p03')['mean_loss_vs_original'],
        available_pair_gain_denominator=denominator,independent_sources=1,exploration_cells=8,
        independent_confirmation=0,new_LLM_calls=0,new_Agent_stages=0,new_method_superiority=False,
        goal_achieved=False,full_predictions_reused_from='A458 exact numerical reference; no new external prediction',
        decision='Retain learned aggregation as an ordinary strong control. Nested validation adds no selection benefit here and fails to protect the successful program. Stop extending this splitter/threshold configuration; no novelty or majority claim.')
    (root/'RESULT459.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    evaluate(p.parse_args().run.resolve())
