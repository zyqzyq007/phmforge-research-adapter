"""Summarize all frozen paired sessions; never select a model from test scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from evaluate_forge_coding import terminal
from verified_benchmark_scoring import aligned, classification


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_classification(before, after, gold):
    a, g = aligned(before, gold, 'sample_id')
    b, _ = aligned(after, gold, 'sample_id')
    records = {}
    for x, y, t in zip(a, b, g):
        group = records.setdefault(t['group'], dict(n=0, before_correct=0,
            after_correct=0, fixed=0, broken=0))
        left = x['predicted_label'] == t['label']
        right = y['predicted_label'] == t['label']
        group['n'] += 1
        group['before_correct'] += int(left)
        group['after_correct'] += int(right)
        group['fixed'] += int(not left and right)
        group['broken'] += int(left and not right)
    ma, mb = classification(before, gold), classification(after, gold)
    return dict(fixed=sum(x['fixed'] for x in records.values()),
        broken=sum(x['broken'] for x in records.values()), records=records,
        delta={k: mb[k] - ma[k] for k in
               ['accuracy', 'macro_f1', 'balanced_accuracy', 'mean_recording_accuracy']})


def main(root):
    cohorts = ['classification', 'rul', 'classification_review', 'rul_review']
    jobs, scores, counts = {}, {}, {'planned': 6, 'finished': 0,
        'execution_error': 0, 'other_terminal': 0, 'requests': 0}
    # These checks precede all label reads and include the independently written
    # choice-review marker. Scores must already have been obtained by cold replay.
    for name in cohorts:
        run = root / name
        jobs[name] = terminal(run)
        assert read(run / 'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
    assert sum(len(v) for v in jobs.values()) == 6
    sealed = read(root / 'audit/motor_fixed_comparators/SEALED_PREDICTIONS.json')
    assert sealed['labels_read'] is False and sealed['refits'] == 0
    for item in sealed['output_files']:
        assert digest(root / item['path']) == item['sha256']
    prior_seal = read(root / 'audit/paderborn_fixed_prior/SEALED_PREDICTIONS.json')
    assert prior_seal['labels_read'] is False
    for name in cohorts:
        run = root / name
        scores[name] = read(run / 'CODING_SCORE.json')['rows']
        for row in scores[name]:
            state = row['status']
            counts[state if state in counts else 'other_terminal'] += 1
            counts['requests'] += read(run / 'model_gateway' / row['job'] / 'status.json')['requests']
    assert sum(len(v) for v in scores.values()) == 6
    result = dict(utc=datetime.now(timezone.utc).isoformat(), denominator=counts,
        exploratory_known_sources=True, research_adaptation=True,
        official_PHMForge_score=False, classification={}, rul={},
        motor_fixed_comparators=[], automatically_confirmed_problem=False)
    for dataset in ['ELECTRICMOTORVIBRATIONS', 'PADERBORN']:
        gold = read(root / 'classification/hidden' / dataset / 'labels.json')
        stages, predictions = {}, {}
        for cohort, stage in [('classification', 'initial'), ('classification_review', 'review')]:
            row = next(x for x in scores[cohort] if x['dataset'] == dataset)
            stages[stage] = row
            out = root / cohort / 'evaluation' / row['job'] / 'output'
            if row['final']['valid']:
                predictions[stage] = read(out / 'predictions.json')['predictions']
            ref = read(out / 'reference.json')['predictions']
            if 'reference' in predictions:
                assert {x['sample_id']: x['predicted_label'] for x in ref} == {
                    x['sample_id']: x['predicted_label'] for x in predictions['reference']}
            predictions['reference'] = ref
        contrasts = {}
        for before, after in [('reference', 'initial'), ('initial', 'review'), ('reference', 'review')]:
            contrasts[after + '_minus_' + before] = (
                paired_classification(predictions[before], predictions[after], gold)
                if before in predictions and after in predictions else
                {'unavailable': 'No valid cold prediction for one stage; retain missingness.'})
        result['classification'][dataset] = dict(stages=stages, contrasts=contrasts)
        if dataset != 'ELECTRICMOTORVIBRATIONS':
            registered = read(root / 'audit/pre_score_choices/PADERBORN_REVIEW_COUNTERFACTUAL267.json')
            assert registered['new_final_scores_read'] is False
            if prior_seal['exit_codes'].get('infer') == 0:
                path = root / 'audit/paderborn_fixed_prior/output/predictions.json'
                assert digest(path) == prior_seal['prediction_sha256']
                pred = read(path)['predictions']
                metrics = classification(pred, gold)
                p, g = aligned(pred, gold, 'sample_id')
                y, v = [x['label'] for x in g], [x['predicted_label'] for x in p]
                independent = dict(accuracy=accuracy_score(y, v),
                    macro_f1=f1_score(y, v, average='macro'),
                    balanced_accuracy=balanced_accuracy_score(y, v))
                assert all(abs(metrics[k] - value) < 1e-12 for k, value in independent.items())
                result['paderborn_fixed_prior'] = dict(metrics=metrics,
                    independent_sklearn_agrees=True, prediction_sha256=digest(path),
                    scope=registered['limitation'],
                    compared_with_review=paired_classification(predictions['review'], pred, gold))
            else:
                result['paderborn_fixed_prior'] = dict(valid=False, replay_status=prior_seal)
            continue
        registry = read(root / 'audit/pre_score_choices/MOTOR_INITIAL_CANDIDATE267.json')
        assert not registry['new_final_scores_read']
        for item in registry['fixed_contrasts']:
            assert digest(root / item['path']) == item['sha256']
            path = root / 'audit/motor_fixed_comparators/output' / (item['name'] + '.json')
            pred = read(path)['predictions']
            metrics = classification(pred, gold)
            p, g = aligned(pred, gold, 'sample_id')
            y, v = [x['label'] for x in g], [x['predicted_label'] for x in p]
            independent = dict(accuracy=accuracy_score(y, v),
                macro_f1=f1_score(y, v, average='macro'),
                balanced_accuracy=balanced_accuracy_score(y, v))
            assert all(abs(metrics[k] - value) < 1e-12 for k, value in independent.items())
            result['motor_fixed_comparators'].append(dict(name=item['name'],
                prediction_sha256=digest(path), metrics=metrics,
                independent_sklearn_agrees=True,
                compared_with_initial=paired_classification(predictions['initial'], pred, gold),
                compared_with_review=paired_classification(predictions['review'], pred, gold)))
    for cohort, stage in [('rul', 'initial'), ('rul_review', 'review')]:
        result['rul'][stage] = scores[cohort][0]
    contrasts = {}
    initial, review = result['rul']['initial'], result['rul']['review']
    for scope in ['consistent10', 'all11']:
        contrasts[scope] = {}
        operands = dict(initial=initial['results']['final'], review=review['results']['final'],
                        reference=initial['results']['reference'])
        left, right = initial['results']['reference'], review['results']['reference']
        if left.get('valid') and right.get('valid'):
            assert all(abs(left[scope][k] - right[scope][k]) < 1e-6
                       for k in ['mae', 'rmse', 'bias', 'challenge_score'])
        for before, after in [('reference', 'initial'), ('initial', 'review'), ('reference', 'review')]:
            contrasts[scope][after + '_minus_' + before] = ({
                k: operands[after][scope][k] - operands[before][scope][k]
                for k in ['mae', 'rmse', 'bias', 'challenge_score']}
                if operands[before].get('valid') and operands[after].get('valid') else
                {'unavailable': 'No valid cold prediction for one stage; retain missingness.'})
    result['rul']['contrasts'] = contrasts
    target = root / 'PAIRED_RESULTS267.json'
    assert not target.exists(), 'Do not overwrite a completed analysis'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(target), denominator=counts,
        classification={k: v['contrasts'] for k, v in result['classification'].items()},
        rul=contrasts), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
