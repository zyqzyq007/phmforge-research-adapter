"""Recompute descriptive failure counts from the frozen public CoT-FD6 CSVs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def summarize(frame):
    correct = frame['llm_prediction'].eq(frame['true_label'])
    accepted = frame['disposition'].eq('accepted')
    return {
        'rows': len(frame), 'instances': frame['instance_id'].nunique(),
        'units': frame['group_id'].nunique(),
        'raw_correct': int(correct.sum()), 'raw_wrong': int((~correct).sum()),
        'raw_accuracy': float(correct.mean()),
        'mean_agreement': float(frame['agreement'].mean()),
        'accepted': int(accepted.sum()), 'coverage': float(accepted.mean()),
        'accepted_correct': int((accepted & correct).sum()),
        'accepted_wrong': int((accepted & ~correct).sum()),
        'selective_accuracy': float(correct[accepted].mean()) if accepted.any() else None,
        'unanimous_wrong': int((frame['agreement'].eq(1) & ~correct).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = ['cotfd6_main_predictions.csv', 'cotfd6_main_raw_traces.csv',
             'nasa_temporal_diagnostic_stability.csv', 'nasa_temporal_raw_traces.csv',
             'full100_memory_only_predictions.csv']
    frames = {name: pd.read_csv(args.results / name) for name in names}
    main_frame = frames[names[0]]
    temporal = frames[names[2]]
    assert not main_frame.duplicated(['dataset', 'instance_id', 'repeat']).any()
    assert not temporal.duplicated(['dataset', 'instance_id']).any()
    assert len(frames[names[1]]) == int(main_frame['valid_traces'].sum())
    assert len(frames[names[3]]) == int(temporal['valid_traces'].sum())
    nasa = main_frame[main_frame['dataset'].eq('nasa_fd001')].copy()
    assert len(nasa) == 500 and nasa['instance_id'].nunique() == 100
    assert nasa['repeat'].nunique() == 5
    nasa['wrong'] = nasa['llm_prediction'].ne(nasa['true_label'])
    nasa['accepted_wrong'] = nasa['wrong'] & nasa['disposition'].eq('accepted')
    stability = nasa.groupby(['instance_id', 'group_id', 'true_label']).agg(
        wrong_repeats=('wrong', 'sum'),
        accepted_wrong_repeats=('accepted_wrong', 'sum'),
        repeats=('repeat', 'count'),
    ).reset_index()
    assert stability['repeats'].eq(5).all()
    memory = frames[names[4]]
    assert not memory.duplicated(['dataset', 'instance_id']).any()
    matched = nasa.merge(memory[['dataset', 'instance_id', 'true_label',
                                 'memory_majority_prediction']],
                         on=['dataset', 'instance_id'], suffixes=('', '_memory'),
                         validate='many_to_one')
    assert len(matched) == len(nasa)
    assert matched['true_label'].eq(matched['true_label_memory']).all()
    matched['memory_correct'] = matched['memory_majority_prediction'].eq(matched['true_label'])
    summary = {
        'kind': 'independent_reanalysis_of_published_outputs_not_model_rerun',
        'model': sorted(main_frame['model'].unique().tolist()),
        'inputs': [{'file': name, 'sha256': hashlib.sha256((args.results / name).read_bytes()).hexdigest()}
                   for name in names],
        'main_by_dataset_repeat': [dict(dataset=dataset, repeat=int(repeat), **summarize(group))
                                   for (dataset, repeat), group in main_frame.groupby(['dataset', 'repeat'])],
        'temporal': summarize(temporal),
        'temporal_by_unit': [dict(unit=int(unit), **summarize(group))
                             for unit, group in temporal.groupby('unit')],
        'temporal_constant_degraded_accuracy': float(temporal['true_label'].eq('degraded').mean()),
        'temporal_prediction_counts': temporal['llm_prediction'].value_counts().to_dict(),
        'nasa_repeat_stability': {
            'distinct_windows': len(stability),
            'distinct_device_units': int(stability['group_id'].nunique()),
            'windows_all_five_repeats_wrong': int(stability['wrong_repeats'].eq(5).sum()),
            'units_with_all_five_repeats_wrong': int(stability.loc[stability['wrong_repeats'].eq(5), 'group_id'].nunique()),
            'windows_any_accepted_wrong': int(stability['accepted_wrong_repeats'].gt(0).sum()),
            'units_any_accepted_wrong': int(stability.loc[stability['accepted_wrong_repeats'].gt(0), 'group_id'].nunique()),
            'windows_all_five_accepted_wrong': int(stability['accepted_wrong_repeats'].eq(5).sum()),
        },
        'memory_only': [dict(dataset=dataset, rows=len(group),
                             majority_accuracy=float(group['memory_majority_prediction'].eq(group['true_label']).mean()),
                             weighted_accuracy=float(group['memory_weighted_prediction'].eq(group['true_label']).mean()))
                        for dataset, group in memory.groupby('dataset')],
        'nasa_memory_llm_paired_outputs': {
            'repeated_rows_not_independent_tasks': len(matched),
            'memory_correct_llm_wrong': int((matched['memory_correct'] & matched['wrong']).sum()),
            'memory_wrong_llm_correct': int((~matched['memory_correct'] & ~matched['wrong']).sum()),
            'both_correct': int((matched['memory_correct'] & ~matched['wrong']).sum()),
            'both_wrong': int((~matched['memory_correct'] & matched['wrong']).sum()),
            'memory_correct_accepted_wrong': int((matched['memory_correct'] & matched['accepted_wrong']).sum()),
        },
    }
    stability.to_csv(args.output / 'nasa_repeat_stability.csv', index=False)
    temporal.assign(wrong=temporal['llm_prediction'].ne(temporal['true_label']))[
        ['instance_id', 'unit', 'window_order', 'window_end', 'rul_end', 'true_label',
         'llm_prediction', 'agreement', 'verifier_prediction', 'disposition', 'wrong']
    ].to_csv(args.output / 'temporal_case_evidence.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'temporal': summary['temporal'], 'model': summary['model'],
                      'main_dataset_repeat_count': len(summary['main_by_dataset_repeat'])}, indent=2))


if __name__ == '__main__':
    main()
