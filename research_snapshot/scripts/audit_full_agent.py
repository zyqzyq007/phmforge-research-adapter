"""Score full coding-agent artifacts against host-only fixed evaluation labels."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--published-task-predictions', type=Path, required=True)
    parser.add_argument('--known-cases', type=Path, required=True)
    parser.add_argument('--published-cases', type=Path,
                        help='Optional first-iteration stable-error manifest for descriptive checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gold = pd.DataFrame(read(args.run / 'hidden_gold.json'))
    assert len(gold) == 100 and gold.case_id.is_unique
    known_ids = {x['instance_id'] for x in read(args.known_cases)['gold']}
    legacy_ids = ({x['instance_id'] for x in read(args.published_cases)['main_stable_accepted_errors']}
                  if args.published_cases else set())
    classical = pd.read_csv(args.published_task_predictions)
    classical_map = {(r.model, r.instance_id): r.prediction for r in classical.itertuples()}
    plans = {}
    for name in ['pilot_plan.json', 'pilot_recovered_plan.json', 'pilot_writable_runtime_plan.json',
                 'pilot_verified_plan.json', 'formal_plan.json']:
        path = args.run / name
        if path.is_file():
            plans.update({x['id']: x for x in read(path)['jobs']})
    initial = read(args.run / 'task_input_manifest.json')
    formal = read(args.run / 'formal_task_input_manifest.json')
    inventories = {**initial, **formal}
    jobs, prediction_rows, errors = [], [], []
    for status_path in sorted(args.run.rglob('status.json')):
        if 'clients-cache' in status_path.relative_to(args.run).parts:
            continue
        status = read(status_path)
        if 'job_id' not in status:
            continue
        record = {k: status.get(k) for k in ['job_id', 'condition', 'repeat', 'status',
                  'exit_code', 'response_valid', 'elapsed_seconds', 'timeout_seconds', 'usage']}
        record['status_path'] = str(status_path.relative_to(args.run))
        if status['status'] != 'finished' or not status.get('response_valid'):
            jobs.append(record)
            continue
        job = plans[status['job_id']]
        workspace = args.run / job['workspace']
        response = read(status_path.parent / 'response.json')
        assert status['filesystem_isolated'] and status['analysis_tools_enabled']
        for item in inventories[workspace.name]:
            path = workspace / item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
        ast.parse((workspace / 'analysis.py').read_text())
        assert (workspace / 'report.md').is_file()
        csv = pd.read_csv(workspace / 'predictions.csv')
        assert len(csv) == len(gold) and csv.case_id.is_unique
        assert set(csv.case_id) == set(gold.case_id)
        submitted = {x['case_id']: x['diagnosis'] for x in response['results']}
        assert len(submitted) == len(gold)
        assert all(submitted[r.case_id] == r.diagnosis for r in csv.itertuples())
        assert set(csv.diagnosis) <= {'healthy', 'degraded'}
        frame = gold.merge(csv[['case_id', 'diagnosis']], on='case_id', validate='one_to_one')
        frame['correct'] = frame.true_label.eq(frame.diagnosis)
        record.update(artifact_valid=True, windows=100, devices=int(frame.group_id.nunique()),
                      correct=int(frame.correct.sum()), accuracy=float(accuracy_score(frame.true_label, frame.diagnosis)),
                      macro_f1=float(f1_score(frame.true_label, frame.diagnosis, average='macro')),
                      balanced_accuracy=float(balanced_accuracy_score(frame.true_label, frame.diagnosis)),
                      confusion_matrix_healthy_degraded=confusion_matrix(frame.true_label, frame.diagnosis,
                                                                        labels=['healthy', 'degraded']).tolist(),
                      device_macro_accuracy=float(frame.groupby('group_id').correct.mean().mean()),
                      error_devices=int(frame.loc[~frame.correct, 'group_id'].nunique()),
                      method=response['method'], validation=response['validation'],
                      known_subset_correct=int(frame.loc[frame.instance_id.isin(known_ids), 'correct'].sum()),
                      known_subset_windows=int(frame.instance_id.isin(known_ids).sum()))
        if legacy_ids:
            record.update(legacy_stable_error_subset_correct=int(
                frame.loc[frame.instance_id.isin(legacy_ids), 'correct'].sum()),
                legacy_stable_error_subset_windows=int(frame.instance_id.isin(legacy_ids).sum()))
        commands = {}
        for line in (status_path.parent / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                commands[item['id']] = item
        record['completed_command_executions'] = len(commands)
        record['nonzero_command_exits'] = sum(x.get('exit_code') not in [0, None] for x in commands.values())
        for row in frame.to_dict('records'):
            row.update(job_id=status['job_id'], condition=status['condition'], repeat=status['repeat'])
            row['classical_predictions'] = {name: classical_map[(name, row['instance_id'])]
                                           for name in classical.model.unique()}
            prediction_rows.append(row)
            if not row['correct']:
                errors.append(row)
        jobs.append(record)
    scored = [x for x in jobs if x.get('artifact_valid')]
    paired = [x for x in scored if x['condition'] == 'complete_training_and_tools_correct_metadata']
    formal_errors = [x for x in errors if x['condition'] == 'complete_training_and_tools_correct_metadata']
    frequencies = Counter(x['instance_id'] for x in formal_errors)
    summary = dict(kind='full_coding_agent_discovery_not_cross_family_confirmation', jobs=jobs,
                   scored_jobs=len(scored), scored_predictions=100 * len(scored),
                   formal_jobs=len(paired), formal_predictions=100 * len(paired),
                   formal_accuracies=[x['accuracy'] for x in paired],
                   formal_known_subset_correct=[x['known_subset_correct'] for x in paired],
                   formal_legacy_stable_error_subset_correct=[x.get('legacy_stable_error_subset_correct')
                                                              for x in paired],
                   formal_error_frequencies=dict(sorted(frequencies.items())),
                   stable_error_windows_all_three=[key for key, count in sorted(frequencies.items())
                                                   if len(paired) == 3 and count == 3],
                   distinct_windows=100 if scored else 0, distinct_devices=25 if scored else 0,
                   dataset_families=1 if scored else 0, real_hardware_cases=0,
                   gold_outside_namespace=True, input_hashes_unchanged=True,
                   csv_and_json_predictions_identical=True,
                   limits=['one simulated family; no E3', 'original capped stratified paper subset',
                           'pilot dictionary mismatch separately retained',
                           'six known-error/control windows overlap the 100-window set',
                           'no service model identity echo; sampling seed not exposed',
                           'no independent confirmation devices beyond this original split'])
    for name, payload in [('summary.json', summary), ('predictions.json', prediction_rows),
                          ('error_cases.json', errors)]:
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['scored_jobs', 'formal_jobs', 'formal_accuracies',
                                             'formal_known_subset_correct']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
