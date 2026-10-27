"""Score frozen diagnostic replays without treating runtime failures as predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--protocol-gold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fidelity', type=Path)
    parser.add_argument('--task-predictions', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gold = {x['case_id']:x for x in json.loads((args.replay / 'hidden_gold.json').read_text())}
    protocol_gold = json.loads(args.protocol_gold.read_text())
    task = (args.replay / 'original/task.txt').read_text()
    payload = json.loads(task.split('\n\n', 1)[1])
    visible = {}
    for item in payload:
        visible[item['case_id']] = {k:float(v) for k,v in re.findall(
            r'^- (s\d+_(?:delta_mean|z_last|slope)): (-?\d+(?:\.\d+)?)$', item['user'], re.M)}
        assert len(visible[item['case_id']]) == 45
    statuses, predictions, synthetic = [], [], []
    for group in ['pilot_runs','retry_runs','followup_runs']:
        base = args.replay / group
        if not base.exists():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            p = folder / 'status.json'
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            statuses.append(s)
            if s['status'] != 'finished' or not s.get('response_valid'):
                continue
            result = json.loads((folder / 'response.json').read_text())['results']
            # A recovered transport notice is an error item, not a tool invocation.
            observed = s.get('observed_non_message_item_types', [])
            assert not [k for k in observed if k != 'error'], 'Unexpected tool use; inspect before scoring'
            for a in result:
                if s['condition'] == 'synthetic_protocol_filter':
                    expected = protocol_gold[a['case_id']]
                    synthetic.append(dict(job_id=s['job_id'], case_id=a['case_id'],
                                          correct=a['valid_output'] == expected['valid_output'] and a['point_rul_cycles'] == expected['point_rul_cycles'],
                                          prediction=a, expected=expected))
                    continue
                g = gold[a['case_id']]
                claims = a['evidence']
                supported, unsupported = 0, 0
                for claim in claims:
                    feature = claim['feature']
                    if feature not in visible[a['case_id']]:
                        unsupported += 1
                        continue
                    v = visible[a['case_id']][feature]
                    direction = 'positive' if v>0.25 else 'negative' if v< -0.25 else 'near_zero'
                    supported += claim['direction'] == direction
                    unsupported += claim['direction'] != direction
                predictions.append(dict(job_id=s['job_id'], condition=s['condition'], repeat=s['repeat'],
                                        timeout_seconds=s['timeout_seconds'], case_id=a['case_id'],
                                        group_id=g['group_id'], instance_id=g['instance_id'],
                                        truth=g['true_label'], prediction=a['diagnosis'],
                                        correct=a['diagnosis'] == g['true_label'],
                                        legacy_all5_wrong=g['legacy_all5_wrong'],
                                        grounded_claims=supported, unsupported_claims=unsupported,
                                        evidence_count_valid=2 <= len(claims) <= 4))
    aggregate = []
    for condition in ['original','label_defined']:
        rows = [r for r in predictions if r['condition']==condition and r['timeout_seconds']==480]
        jobs = {r['job_id'] for r in rows}
        by_case = defaultdict(list)
        for a in rows:
            by_case[a['case_id']].append(a)
        aggregate.append(dict(condition=condition, timeout_seconds=480,
                              valid_job_count=len(jobs), repeated_predictions=len(rows),
                              distinct_windows=len(by_case), distinct_devices=len({x['group_id'] for x in rows}),
                              correct=sum(x['correct'] for x in rows),
                              accuracy=sum(x['correct'] for x in rows)/len(rows) if rows else None,
                              by_case=[dict(case_id=k, instance_id=v[0]['instance_id'], true_label=v[0]['truth'],
                                            repeats=len(v), correct=sum(x['correct'] for x in v),
                                            predicted_labels=[x['prediction'] for x in v]) for k,v in sorted(by_case.items())]))
    checks = []
    if args.fidelity and args.task_predictions:
        fidelity = pd.read_csv(args.fidelity).set_index('instance_id')
        models = pd.read_csv(args.task_predictions)
        forest = models.loc[models['model'].eq('Random Forest')].set_index('instance_id')
        validators = {
            'published_prototype': fidelity['reconstructed_verifier'].to_dict(),
            'retrained_random_forest': forest['prediction'].to_dict(),
        }
        for condition in ['original','label_defined']:
            rows = [x for x in predictions if x['condition']==condition and x['timeout_seconds']==480]
            for name, labels in validators.items():
                accepted = [x for x in rows if x['prediction']==labels[x['instance_id']]]
                checks.append(dict(condition=condition, validator=name, predictions=len(rows),
                                   accepted=len(accepted), coverage=len(accepted)/len(rows),
                                   accepted_wrong=sum(not x['correct'] for x in accepted),
                                   selective_accuracy=sum(x['correct'] for x in accepted)/len(accepted),
                                   correct_referred_for_review=sum(x['correct'] for x in rows if x not in accepted)))
    summary = dict(kind='current_cli_discovery_probe_not_full_autonomous_agent_benchmark',
                   conditions=aggregate, scored_predictions_all_budgets=len(predictions),
                   runtime_states=[{k:s.get(k) for k in ['job_id','status','response_valid','timeout_seconds','elapsed_seconds','usage']} for s in statuses],
                   prepared_diagnostic_windows=6, distinct_devices=3, dataset_families=1,
                   synthetic_correct=sum(x['correct'] for x in synthetic), synthetic_scored=len(synthetic),
                   all_observed_runtime_tools_absent=True,
                   recovered_transport_notice_jobs=[s['job_id'] for s in statuses
                       if s.get('response_valid') and 'error' in s.get('observed_non_message_item_types', [])],
                   unsupported_claims=sum(x['unsupported_claims'] for x in predictions),
                   evidence_count_violations=sum(not x['evidence_count_valid'] for x in predictions),
                   offline_validator_counterfactuals=checks,
                   counterfactual_scope='Postprocess unchanged responses with the same agreement rule; no new agent-plus-validator run.',
                   limits=['known-error discovery subset plus matched controls',
                           'same three devices across repeats, not additional independent tasks',
                           'model sampling seed is not exposed',
                           'model identifier is CLI-requested; response events do not independently echo the service model id',
                           'no execution tools or full training data were supplied',
                           'six tasks shared one context; original paper generated per-window requests',
                           'timeouts without answers have no classification score',
                           'synthetic protocol cases do not count as physical-device evidence'])
    (args.output / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output / 'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    (args.output / 'synthetic_protocol_scores.json').write_text(json.dumps(synthetic,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['conditions','synthetic_correct','synthetic_scored','unsupported_claims']},indent=2))


if __name__ == '__main__':
    main()
