"""Summarize both frozen budget layers;no refits or extra model forward passes."""
import argparse
import json
from pathlib import Path
import statistics

from controlled_feedback import save,sha,utc


def read(path):return json.loads(Path(path).read_text())


def mean(values):return statistics.mean(values) if values else None


def summarize(root):
    result=read(root/'RESULT342.json');cfg=read(root/'PROTOCOL342.json');cases=read(root/'public/cases.json')
    rows_by_style={}
    for style in ['ordinary','controlled']:
        episodes=[e for e in result['selectors'] if e['condition']==style]
        assert len(episodes)==2 and all(e['completed'] for e in episodes)
        rows=[r for e in episodes for r in e['rows']];assert len(rows)==8 and all(r['evaluation_status']=='complete' for r in rows)
        rows_by_style[style]=rows
    summaries={style:dict(mean_record_error=mean([r['record_error'] for r in rows]),
        mean_selection_loss=mean([r['selection_loss'] for r in rows]),mean_harm=mean([r['harm'] for r in rows]),
        mean_useful_gain_capture=mean([r['gain_capture'] for r in rows if r['gain_capture'] is not None])) for style,rows in rows_by_style.items()}
    pool=[]
    for case in cases:
        per={style:[r for r in rows if r['case']==case['id']] for style,rows in rows_by_style.items()}
        assert all(len(v)==2 for v in per.values())
        losses={style:mean([r['record_error'] for r in rows]) for style,rows in per.items()}
        final_interaction=None
        design=case['controlled_edit_evidence']
        if design['complete_structural_control']:
            values={cell:result['all_program_results']['common_p00' if ident=='p00' else case['id']+'__'+ident]['record_error'] for cell,ident in design['cells'].items()}
            final_interaction=values['11']-values['10']-values['01']+values['00']
        pool.append(dict(case=case['id'],ordinary_error=losses['ordinary'],controlled_error=losses['controlled'],
            choices={style:[r['selected'] for r in rows] for style,rows in per.items()},
            comparison='win' if losses['controlled']<losses['ordinary'] else 'loss' if losses['controlled']>losses['ordinary'] else 'tie',
            controlled_edit_cells=case['controlled_edit_evidence']['cells'],
            structural_control_complete=case['controlled_edit_evidence']['complete_structural_control'],
            legal_error_interaction=case['controlled_edit_evidence'].get('error_interaction'),final_error_interaction=final_interaction))
    fixed={name:dict(mean_record_error=mean([r['record_error'] for r in rows]),mean_selection_loss=mean([r['selection_loss'] for r in rows]),
        mean_harm=mean([r['harm'] for r in rows]),mean_useful_gain_capture=mean([r['gain_capture'] for r in rows if r['gain_capture'] is not None]))
        for name,rows in result['fixed_controls'].items()}
    baseline=result['all_program_results']['common_p00'];details=[]
    for case in cases:
        for ident,legal in zip(case['candidate_ids'],case['group_losses']):
            value=result['all_program_results']['common_p00' if ident=='p00' else case['id']+'__'+ident]
            assert value['status']=='complete'
            before={r['group']:r for r in baseline['records']};after={r['group']:r for r in value['records']};assert set(before)==set(after)
            negative=sum(before[g]['predicted']==before[g]['actual'] and after[g]['predicted']!=after[g]['actual'] for g in before)
            positive=sum(before[g]['predicted']!=before[g]['actual'] and after[g]['predicted']==after[g]['actual'] for g in before)
            details.append(dict(case=case['id'],program=ident,legal_errors=legal,final_error=value['record_error'],positive_record_flips=positive,negative_record_flips=negative,
                fault_as_healthy=sum(r['actual']!=0 and r['predicted']==0 for r in value['records']),healthy_as_fault=sum(r['actual']==0 and r['predicted']!=0 for r in value['records'])))
    return dict(iteration=cfg['iteration'],budget_profile=cfg.get('budget_profile','standard'),baseline_error=result['baseline_record_error'],
        common_pool_selectors=summaries,paired_pools=pool,fixed_controls=fixed,candidate_details=details,
        generator_selector_matrix=result['generator_selector_matrix'],cost=result['cost'],
        result_sha256=sha(root/'RESULT342.json'),protocol_sha256=sha(root/'PROTOCOL342.json'))


def main(low,high,output,no_plot=False):
    a=summarize(low);b=summarize(high)
    for key in ['record_error','confusion','records']:
        assert read(low/'RESULT342.json')['all_program_results']['common_p00'][key]==read(high/'RESULT342.json')['all_program_results']['common_p00'][key]
    result=dict(utc=utc(),profiles=[a,b],physical_development_cohorts=1,independent_confirmation_scenarios=0,
        total_model_requests=a['cost']['requests']+b['cost']['requests'],total_input_tokens=a['cost']['input_tokens']+b['cost']['input_tokens'],
        total_output_tokens=a['cost']['output_tokens']+b['cost']['output_tokens'],new_fit_entry_executions=a['cost']['actual_fit_entries']+b['cost']['new_fit_entry_executions'],
        completed_estimator_fits=a['cost']['completed_estimator_fits']+b['cost']['completed_estimator_fits']-b['cost']['reused_baseline_fit_entry_records'],
        final_prediction_processes=a['cost']['final_prediction_processes']+b['cost']['final_prediction_processes'],
        low_complete_native_selectors=4,high_complete_native_builders=4,high_complete_native_selectors=4,
        decision='R4doesnotimprovesame-poolselectionorfinaltaskloss;do notadvanceasaneffectiveprimarymethod. It generated useful alternatives but extra component reasoning did not choose them.',
        mechanism_boundary='Strongest high-layer pool has zero feature/classifier interaction. Another pool changes from-0.25infull-fitvalidationto+0.25infinalrecords. Fixed-condition component controls did not reliably transfer;this doesnot establish an Agent-specific reasoning failure.',
        no_net_harmful_replacement_in_this_cohort=True,novelty_established=False,majority_goal_complete=False,reserved338_unused=True,
        no_new_fits_or_forward_passes_for_this_summary=True)
    save(output,result)
    if not no_plot:plot(result,output)
    print(json.dumps(dict(requests=result['total_model_requests'],new_fit_entries=result['new_fit_entry_executions'],estimator_fits=result['completed_estimator_fits'],
        high_selector_comparison=b['common_pool_selectors'],decision=result['decision'])))


def plot(result,output):
    b=result['profiles'][1]
    # Descriptive scientific figure. Row order follows source/recipe chronology,
    # never a ranking selected by the final outcome.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw=dict(width_ratios=[1,2]))
    matrix=np.array([[b['generator_selector_matrix'][g][s]['mean_record_error'] for s in ['ordinary','controlled']] for g in ['ordinary_coding','controlled_edits']])
    axes[0].imshow(matrix,vmin=0,vmax=1,cmap='YlOrRd')
    axes[0].set_xticks([0,1],['Ordinary\nselector','R4\nselector']);axes[0].set_yticks([0,1],['Ordinary coding','R4 coding'])
    axes[0].set_title('Final record error: adequate budget')
    for (i,j),value in np.ndenumerate(matrix):axes[0].text(j,i,f'{100*value:g}%',ha='center',va='center',fontsize=14)
    pool4=[r for r in b['candidate_details'] if r['case']=='pool_4'];table=np.array([r['legal_errors']+[r['final_error']] for r in pool4]);im=axes[1].imshow(table,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto')
    axes[1].set_xticks(range(5),['CV@18\nfold-fit','CV@28\nfold-fit','Dev@38\nfull-fit','Check@54\nfull-fit','Final@44\nsealed'])
    axes[1].set_yticks(range(5),['p00: original','p01: features','p02: classifier','p03: both','p04: repair (selected)'])
    axes[1].set_title('Actual failure case: all four selectors chose p04')
    for (i,j),value in np.ndenumerate(table):axes[1].text(j,i,f'{100*value:g}%',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=axes[1],fraction=.04,pad=.03,label='Record error')
    fig.suptitle('R4 negative result: one measured cohort; no independent confirmation',fontsize=12)
    fig.text(.5,.01,'Each condition: four laboratory acquisitions. Related pools/repeats are not independent tasks.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--low',required=True,type=Path);p.add_argument('--high',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--no-plot',action='store_true');p.add_argument('--plot-only',action='store_true')
    a=p.parse_args()
    if a.plot_only:plot(read(a.output),a.output)
    else:main(a.low,a.high,a.output,a.no_plot)
