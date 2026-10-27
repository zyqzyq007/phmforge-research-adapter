"""Rebuild A278 figures and descriptive contrasts from sealed numeric results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256


def main(root):
    path=root/'NUMERICAL_RESULTS278.json';data=json.loads(path.read_text())
    indexed={r['variant']['id']:r for r in data['variants']}
    final=[]
    for row in data['cohorts']:
        results=indexed.get(row['final_variant'],{}).get('results',{})
        final.append(dict(cohort=row['cohort'],arm=row['arm'],status=row['status'],
            candidate=row['selected'],variant=row['final_variant'],
            scores={split:(value['metrics'] if value['valid'] else None) for split,value in results.items()}))
    control_id='bank_'+data['selection_controls']['development_selected']
    control=indexed[control_id]['results']
    cv_id='bank_'+data['selection_controls']['training_cv_selected']
    comparisons=[]
    for item in final:
        available=[key for key in indexed if key.startswith('bank_')]
        if item['arm']=='adaptive':available.append('reference_mlp')
        if item['cohort']=='b':available+=['b_prior_90_10','b_tree','b_mlp','b_mlp035']
        if item['cohort']=='d':available += [key for key in indexed if key.startswith(('d50_','d100_','d200_'))]
        for other in available:
            candidate=indexed[other]['results']
            for split in ['development','final']:
                selected=item['scores'].get(split)
                if not selected or not candidate[split]['valid']:continue
                ref=candidate[split]['metrics'];delta=selected['mean_bearing_mae_minutes']-ref['mean_bearing_mae_minutes']
                by_ref={x['unit_id']:x for x in ref['per_bearing']}
                unit_deltas=[dict(unit_id=x['unit_id'],selected_mae_minutes=x['mae_minutes'],
                    reference_mae_minutes=by_ref[x['unit_id']]['mae_minutes'],
                    difference_minutes=x['mae_minutes']-by_ref[x['unit_id']]['mae_minutes']) for x in selected['per_bearing']]
                comparisons.append(dict(cohort=item['cohort'],split=split,reference_variant=other,
                    reference_scope=indexed[other]['variant']['scope'],
                    is_component_ablation=(other in ['b_tree','b_mlp','b_mlp035'] or
                        (other.startswith(('d50_','d100_','d200_')) and not other.endswith('_minimum'))),
                    selected_mae_minutes=selected['mean_bearing_mae_minutes'],reference_mae_minutes=ref['mean_bearing_mae_minutes'],
                    difference_minutes=delta,ratio=selected['mean_bearing_mae_minutes']/ref['mean_bearing_mae_minutes'] if ref['mean_bearing_mae_minutes'] else None,
                    meets_registered_material_loss=split=='final' and delta>=10 and selected['mean_bearing_mae_minutes']>=1.2*ref['mean_bearing_mae_minutes'],
                    per_bearing=unit_deltas))
    output=root/'analysis278';output.mkdir(exist_ok=True)
    save(output/'DESCRIPTIVE_CONTRASTS278.json',dict(source_sha256=sha256(path),
        final_choices=final,comparisons=comparisons,physical_allocation_count=1,
        final_bearings=3,final_queries=15,population_inference=False,
        attribution='Numeric losses alone do not establish an Agent-specific or repeated-feedback causal mechanism.'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    order=[x['cohort'].upper()+'\n'+x['arm'] for x in final]
    x=np.arange(len(final));width=.34
    for offset,split,color in [(-width/2,'development','#80b1d3'),(width/2,'final','#e69f00')]:
        vals=[item['scores'].get(split,{}).get('mean_bearing_mae_minutes',np.nan) if item['scores'].get(split) else np.nan for item in final]
        bars=axes[0].bar(x+offset,vals,width,label=split,color=color)
        for b,v in zip(bars,vals):
            if np.isfinite(v):axes[0].annotate(f'{v:.1f}',(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_xticks(x,order);axes[0].set_ylabel('Mean per-bearing MAE (minutes; lower is better)')
    axes[0].set_title('Frozen choices and adaptive programs');axes[0].legend(frameon=False)
    if control['final']['valid']:
        ref=control['final']['metrics']['mean_bearing_mae_minutes']
        axes[0].axhline(ref,color='#444444',linestyle=':',linewidth=1)
    units=sorted({r['unit_id'] for item in final if item['scores'].get('final') for r in item['scores']['final']['per_bearing']})
    ux=np.arange(len(units));bar_width=.18
    for k,item in enumerate(final):
        lookup={r['unit_id']:r for r in item['scores'].get('final',{}).get('per_bearing',[])} if item['scores'].get('final') else {}
        vals=[lookup[u]['mae_minutes'] if u in lookup else np.nan for u in units]
        axes[1].bar(ux+(k-1.5)*bar_width,vals,bar_width,label=item['cohort'].upper()+' '+str(item['candidate']))
    axes[1].set_xticks(ux,['Bearing '+str(u) for u in units]);axes[1].set_ylabel('Final MAE (minutes)')
    axes[1].set_title('All three held-out physical bearings');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('XJTU measured-data RUL adaptation: one split, four Agent trajectories\n15 final queries are repeated observations of 3 bearings',fontsize=12)
    for extension in ['png','svg']:fig.savefig(output/('final_comparison278.'+extension),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.5),constrained_layout=True)
    for row in data['variants']:
        if row['variant']['id'].startswith('wrapper_'):continue
        dev,hold=(row['results'][s] for s in ['development','final'])
        if not(dev['valid'] and hold['valid']):continue
        a=dev['metrics']['mean_bearing_mae_minutes'];b=hold['metrics']['mean_bearing_mae_minutes']
        is_agent=row['variant']['id'].startswith('agent_')
        ax.scatter(a,b,s=65 if is_agent else 35,color='#d55e00' if is_agent else '#0072b2',marker='D' if is_agent else 'o')
        ax.annotate(row['variant']['id'],(a,b),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Development MAE (3 bearings, one query each; minutes)')
    ax.set_ylabel('Final mean per-bearing MAE (minutes)')
    ax.set_title('All preregistered candidates, including counterevidence')
    ax.grid(alpha=.18)
    for extension in ['png','svg']:fig.savefig(output/('candidate_comparison278.'+extension),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(figures=['final_comparison278','candidate_comparison278'],descriptive_comparisons=len(comparisons))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
