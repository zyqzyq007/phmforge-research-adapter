"""Descriptive A325 analysis from sealed choices and completed external scores.

No fitting, model calls, new predictions, or changes to selection decisions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def summarize(root, output):
    final = read(root / 'FINAL_RESULTS325.json')
    seal = read(root / 'ALL_CHOICES_SEALED.json')
    results = {row['id']: row for row in final['episodes']}
    episodes = []
    for row in seal['episodes']:
        outcome = results[row['id']]
        assert abs(outcome['selection_regret'] - (
            outcome['harm'] + outcome['opportunity'] - outcome['gain'])) < 1e-12
        folder = root / 'episodes' / row['id']
        access = [json.loads(line) for line in
                  (folder / 'broker/ACCESS.jsonl').read_text().splitlines()]
        actions, errors, checked_at, plan_at = Counter(), [], {}, []
        for event in access:
            request = event['request']
            if isinstance(request, str):
                request = json.loads(request)
            action = request.get('action')
            actions[action] += 1
            if not event['response'].get('ok'):
                errors.append(dict(action=action, response=event['response']))
            elif action == 'check':
                checked_at.setdefault(request['candidate_id'], event['started_utc'])
            elif action == 'plan':
                plan_at.append(event['ended_utc'])
        intermediates = [v for k, v in checked_at.items() if k not in ['P00', 'P11B']]
        order_ok = (bool(plan_at) and all(t >= min(plan_at) for t in intermediates))
        stage_rows = [dict(role=s['job_id'].split('_', 4)[-1],
                           status=s['status'], response_valid=s.get('response_valid'),
                           seconds=s.get('elapsed_seconds')) for s in row['stages']]
        episodes.append(dict(
            **outcome, proposed=(row.get('choice') or {}).get('proposed'),
            gate_reason=(row.get('choice') or {}).get('gate_reason'),
            elapsed_seconds=row.get('elapsed_seconds'), stages=stage_rows,
            tool_actions=dict(actions), tool_errors=errors,
            intermediate_checks_after_registered_plan=order_ok if row['arm'] == 'B' else None,
            hypothesis_sets=len(row.get('hypotheses', [])),
            plan_sets=len(row.get('plans', [])),
            choice_source=str((folder / 'EPISODE_RESULT.json').relative_to(root))))
    arms = {}
    for arm in ['ordinary', 'B']:
        rows = [r for r in episodes if r['arm'] == arm]
        blocks = sorted(set(r['block'] for r in rows))
        # Repeated decisions do not create additional physical test records.
        for metric in ['record_error', 'selection_regret']:
            block_mean = statistics.mean(statistics.mean(r[metric] for r in rows if r['block'] == b)
                                         for b in blocks)
            assert abs(block_mean - final['summary'][arm][metric]) < 1e-12
        arms[arm] = dict(
            episodes=len(rows), completed=sum(r['completed'] for r in rows),
            fallbacks=sum(r['fallback'] for r in rows),
            **{k: statistics.mean(r[k] for r in rows) for k in
               ['record_error', 'selection_regret', 'harm', 'gain', 'window_accuracy',
                'window_macro_f1', 'model_requests', 'input_tokens', 'output_tokens',
                'check_seconds', 'elapsed_seconds']},
            choices=dict(Counter(r['selected'] for r in rows)),
            total_requests=sum(r['model_requests'] for r in rows),
            total_input_tokens=sum(r['input_tokens'] for r in rows),
            total_output_tokens=sum(r['output_tokens'] for r in rows),
            opportunity=statistics.mean(r['opportunity'] for r in rows))
    candidate_rows = []
    for block, bank in final['candidate_scores'].items():
        validation = read(root / 'bank' / block / 'VALIDATION_REPORTS.json')
        for candidate, score in bank.items():
            candidate_rows.append(dict(
                block=block, candidate=candidate,
                cv_error=validation[candidate]['cv']['record_error'],
                development_error=validation[candidate]['development']['record_error'],
                final_error=score['record_error'], final_correct=score['record_correct'],
                final_total=score['record_total'], final_window_accuracy=score['window_accuracy'],
                final_window_macro_f1=score['window_macro_f1']))
    fixed = {}
    for method in sorted(set(x['method'] for x in final['fixed_controls'])):
        rows = [x for x in final['fixed_controls'] if x['method'] == method]
        fixed[method] = dict(
            record_error=statistics.mean(x['record_error'] for x in rows),
            selection_regret=statistics.mean(x['selection_regret'] for x in rows),
            choices={x['block']: x['selected'] for x in rows})
    payload = dict(
        iteration='A325', analysis_type='descriptive; no new model work',
        final_score_source='FINAL_RESULTS325.json', seal_source='ALL_CHOICES_SEALED.json',
        physical_final_records=40, independent_motors=1, physical_blocks=2,
        unique_candidates_per_block=5, episodes=episodes, arms=arms,
        candidates=candidate_rows, fixed=fixed,
        model_request_ratio_B_to_ordinary=arms['B']['model_requests']/arms['ordinary']['model_requests'],
        elapsed_ratio_B_to_ordinary=arms['B']['elapsed_seconds']/arms['ordinary']['elapsed_seconds'],
        no_inferential_p_values=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'arms': arms, 'fixed': fixed}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run.resolve(), args.output.resolve())
