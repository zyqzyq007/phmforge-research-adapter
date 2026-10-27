"""Score frozen cross-task artifacts only after both waves and scope audit end."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from prepare_validation_scope_review import live_readers


ARCH = Path('/data/yyf-zyq/PHM_20261010')
MILL = ARCH / '20261010T185641Z_crossfamily09'
ADMIT = ARCH / '20261017T235800Z_admit152/source/admit-bench'


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def average(values):
    return statistics.mean(values) if values else None


def artifact_json(ws, response):
    path = ws / 'predictions.json'
    if path.exists():
        if path.is_symlink() or not path.resolve().is_relative_to(ws.resolve()):
            raise ValueError('Prediction artifact must be a regular local file')
        return json.loads(path.read_text()), str(path), sha(path)
    text = response.read_text().strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    return json.loads(text), str(response), sha(response)


def milling_score(job, ws):
    task = job['case_ids'][0]
    with (MILL / 'hidden_gold' / (task + '.csv')).open() as handle:
        gold = {r['query_id']: r for r in csv.DictReader(handle)}
    with (MILL / 'fixed_RF_reference' / (task + '_test_predictions.csv')).open() as handle:
        reference = {r['query_id']: float(r['wear_estimate_mm']) for r in csv.DictReader(handle)}
    prediction = ws / 'predictions.json'
    assert prediction.is_file() and not prediction.is_symlink()
    assert prediction.resolve().is_relative_to(ws.resolve())
    data = json.loads(prediction.read_text())
    rows = data['results']
    ids = [r['case_id'] for r in rows]
    assert len(ids) == len(set(ids)) and set(ids) == set(gold), 'Prediction inventory mismatch'
    result = []
    for row in rows:
        ident = row['case_id']
        target = float(gold[ident]['VB'])
        point = row.get('wear_estimate_mm')
        low, high = row.get('interval_low_mm'), row.get('interval_high_mm')
        assert point is None or finite(point) and point >= 0
        assert (low is None and high is None) or finite(low) and finite(high) and 0 <= low <= high
        result.append({'query_id': ident, 'case': gold[ident]['case'], 'prediction_mm': point,
                       'gold_mm': target, 'absolute_error_mm': abs(point - target) if point is not None else None,
                       'reference_absolute_error_mm': abs(reference[ident] - target),
                       'interval_finite': low is not None,
                       'interval_covers': low <= target <= high if low is not None else None,
                       'interval_width_mm': high - low if low is not None else None})
    predicted = [r for r in result if r['prediction_mm'] is not None]
    intervals = [r for r in result if r['interval_finite']]
    return {'valid_inventory': True, 'planned_queries': len(result), 'predicted_queries': len(predicted),
            'abstentions': len(result) - len(predicted), 'MAE_mm': average([r['absolute_error_mm'] for r in predicted]),
            'reference_MAE_same_predicted_subset_mm': average([r['reference_absolute_error_mm'] for r in predicted]),
            'reference_MAE_all_queries_mm': average([r['reference_absolute_error_mm'] for r in result]),
            'finite_intervals': len(intervals), 'covered_intervals': sum(r['interval_covers'] for r in intervals),
            'mean_interval_width_mm': average([r['interval_width_mm'] for r in intervals]), 'cases': result,
            'limits': 'Researcher-defined original-data task; contemporaneous wear, not RUL. Conditional MAE and small correlated-query coverage do not establish general calibration.'}


def battery_score(job, ws, response, gold):
    base = job.get('original_job_id', job['id'])
    data, path, digest = artifact_json(ws, response)
    row = data['results'][0] if 'results' in data else data
    assert row['case_id'] == base
    point, eol = row.get('predicted_rul_cycles'), row.get('predicted_eol_cycle')
    low, high = row.get('interval_90pct_low_cycles'), row.get('interval_90pct_high_cycles')
    assert point is None or finite(point) and point >= 0
    assert eol is None or finite(eol) and eol >= 0
    assert (low is None and high is None) or finite(low) and finite(high) and 0 <= low <= high
    g = gold[base]
    with (ws / 'reference_manifest.csv').open() as handle:
        refs = list(csv.DictReader(handle))
    meta = json.loads((ws / 'target_metadata.json').read_text())
    same = [float(r['eol_cycle_at_0.90']) for r in refs
            if r['eol_cycle_at_0.90'] and int(r['cell_temperature_C']) == meta['cell_temperature_C']]
    baseline = max(0., statistics.median(same) - meta['current_cycle']) if same else None
    truth = g['rul_cycles']
    return {'valid_prediction': True, 'source_artifact': path, 'source_sha256': digest,
            'prediction': row, 'gold': g, 'rul_absolute_error': abs(point - truth) if point is not None and truth is not None else None,
            'eol_consistency_error': abs(eol - (meta['current_cycle'] + point)) if eol is not None and point is not None else None,
            'interval_covers': low <= truth <= high if low is not None and truth is not None else None,
            'reference_same_temperature_median_rul': baseline,
            'reference_absolute_error': abs(baseline - truth) if baseline is not None and truth is not None else None,
            'censored_target': truth is None, 'fixed_cutoff_independent_of_eol': True,
            'limits': 'One target per scenario, two temperatures; no population calibration or official leaderboard claim. Baseline uses only allowed complete reference EOLs.'}


def score_artifact(run, job, ws, out, battery_gold):
    if job['family'] == 'milling':
        return milling_score(job, ws)
    if job['family'] == 'battery':
        return battery_score(job, ws, out / 'response.txt', battery_gold)
    if job['family'] == 'admit':
        destination = run / 'scoring/admit' / (job['id'] + '.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(Path(__file__).with_name('score_admit162.py')),
                   '--source-root', str(ADMIT), '--response', str(out / 'response.txt'),
                   '--output', str(destination), '--case-id', job['original_case'], '--cartridge', job['cartridge']]
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        destination.with_suffix('.stdout').write_text(p.stdout)
        destination.with_suffix('.stderr').write_text(p.stderr)
        assert p.returncode == 0, p.stderr[-1000:]
        native = json.loads(destination.read_text())
        return {k: native.get(k) for k in ['case_id', 'parse_error', 'diagnosis_correct', 'record', 'gates', 'score']}
    return {'status': 'separate_original_IMP_MARL_evaluation_required', 'planned_final_seeds': 64}


def score_completed_families(run):
    """Outcome-blind scheduling amendment; no interaction with active subjects."""
    destination = run / 'SCORE_NON_IMPRL185.json'
    assert not destination.exists() and not (run / 'SCORE185.json').exists()
    amendment = run / 'audit/SCORING_PIPELINE_AMENDMENT185.json'
    assert json.loads(amendment.read_text())['before_any_new_hidden_scores']
    assert len(json.loads((run / 'LAUNCH_REVIEW185.json').read_text())['runners']) == 8
    notes_path = run / 'SCOPE_NOTES185.json'
    audit = json.loads(notes_path.read_text())
    jobs = []
    for name in ['PLAN185.json', 'REVIEW_PLAN185.json']:
        jobs += [j for j in json.loads((run / name).read_text())['jobs'] if j['family'] != 'imprl']
    assert len(jobs) == 12
    records, paths = [], []
    for job in jobs:
        annotation = audit['jobs'][job['id']]
        assert annotation['before_hidden_scores']
        for entry in annotation['artifacts']:
            p = run / entry['path']
            assert not p.is_symlink() and sha(p) == entry['sha256']
        out = run / job.get('output_directory', 'primary_' + job['family'] + '/' + job['id'])
        status = json.loads((out / 'status.json').read_text())
        assert status['status'] in ['finished', 'timeout', 'execution_error']
        assert not Path('/proc/' + str(status['cli_pid'])).exists()
        paths += [run / job['workspace'], out, run / 'source_history' / job['id']]
        records.append((job, status, out, annotation))
    assert not live_readers(paths)
    put(run / 'PRE_NON_IMPRL_SCORE_AUDIT185.json', {'checked_utc': datetime.now(timezone.utc).isoformat(),
         'selected12_terminal_and_readers_gone': True, 'all8reviews_launched': True,
         'scope_notes_sha256': sha(notes_path), 'amendment_sha256': sha(amendment),
         'selected_families_frozen_before_their_scores': True, 'no_score_feedback_to_subjects': True})
    gold = json.loads((run / 'score_data/BATTERY_GOLD185.json').read_text())
    rows = []
    for job, status, out, annotation in records:
        row = {'job': job['id'], 'family': job['family'], 'phase': job.get('phase', 'initial'),
               'native_status': status['status'], 'response_valid': status.get('response_valid'),
               'elapsed_seconds': status.get('elapsed_seconds'), 'usage': status.get('usage'),
               'scope_annotation': annotation}
        try:
            row['scientific'] = score_artifact(run, job, run / job['workspace'], out, gold)
        except (OSError, ValueError, KeyError, TypeError, AssertionError, subprocess.SubprocessError) as error:
            row['scientific'] = {'status': 'unscorable_or_invalid_artifact', 'error': type(error).__name__ + ': ' + str(error)}
        rows.append(row)
    put(destination, {'scored_utc': datetime.now(timezone.utc).isoformat(), 'rows': rows,
         'families': ['milling', 'battery', 'admit'], 'planned_sessions': 12,
         'remaining_family_not_scored': 'imprl', 'no_feedback_to_any_subject': True})
    put(run / 'NON_IMPRL_SCORE_RECEIPT185.json', {'sha256': sha(destination),
         'scoring_source_sha256': sha(Path(__file__)), 'new_scores_cannot_be_edited': True})
    print(json.dumps({'scored_completed_sessions': len(rows), 'remaining_family_not_scored': 'imprl'}))


def score(run):
    assert not (run / 'SCORE185.json').exists(), 'Refuse an edited rescore'
    initial = json.loads((run / 'PLAN185.json').read_text())['jobs']
    review = json.loads((run / 'REVIEW_PLAN185.json').read_text())['jobs']
    jobs = initial + review
    assert len(jobs) == 16
    scope_path = run / 'SCOPE_AUDIT185.json'
    audit = json.loads(scope_path.read_text())
    assert (audit['before_hidden_scores'] or audit.get('per_family_scope_frozen_before_own_scores'))
    assert set(audit['jobs']) == {j['id'] for j in jobs}
    for item in audit['jobs'].values():
        for artifact in item['artifacts']:
            p = run / artifact['path']
            assert p.resolve().is_relative_to(run) and not p.is_symlink()
            assert sha(p) == artifact['sha256'], 'Frozen submission changed after scope annotation'
    launches = [json.loads((run / name).read_text()) for name in ['LAUNCH185.json', 'LAUNCH_REVIEW185.json']]
    for launch in launches:
        for item in launch['runners']:
            assert not Path('/proc/' + str(item['runner_pid'])).exists(), 'A runner is active'
    statuses = {}
    for job in jobs:
        out = run / job['output_directory'] if job.get('output_directory') else run / ('primary_' + job['family']) / job['id']
        data = json.loads((out / 'status.json').read_text())
        assert data['status'] in ['finished', 'timeout', 'execution_error']
        assert not Path('/proc/' + str(data['cli_pid'])).exists()
        statuses[job['id']] = (data, out)
    paths = [run / 'workspaces', run / 'review_workspaces', run / 'review_primary_jobs', run / 'source_history']
    paths += [run / ('primary_' + f) for f in ['milling', 'battery', 'imprl', 'admit']]
    assert not live_readers(paths), 'Scientific reader/writer remains'
    put(run / 'PRE_SCORE_AUDIT185.json', {'checked_utc': datetime.now(timezone.utc).isoformat(),
         'all16_terminal': True, 'all_readers_gone': True, 'scope_audit_sha256': sha(scope_path),
         'scope_frozen_before_score': True, 'no_score_feedback_to_review': True})
    # These labels are first consumed for scoring only after the guard above.
    battery_gold = json.loads((run / 'score_data/BATTERY_GOLD185.json').read_text())
    cached = {}
    if (run / 'SCORE_NON_IMPRL185.json').is_file():
        receipt = json.loads((run / 'NON_IMPRL_SCORE_RECEIPT185.json').read_text())
        assert sha(run / 'SCORE_NON_IMPRL185.json') == receipt['sha256']
        cached = {r['job']: r for r in json.loads((run / 'SCORE_NON_IMPRL185.json').read_text())['rows']}
    rows = []
    for job in jobs:
        status, out = statuses[job['id']]
        ws = run / job['workspace']
        row = {'job': job['id'], 'family': job['family'], 'phase': job.get('phase', 'initial'),
               'native_status': status['status'], 'response_valid': status.get('response_valid'),
               'elapsed_seconds': status.get('elapsed_seconds'), 'usage': status.get('usage'),
               'scope_annotation': audit['jobs'][job['id']]}
        try:
            if job['id'] in cached:
                assert cached[job['id']]['scope_annotation'] == audit['jobs'][job['id']]
                row['scientific'] = cached[job['id']]['scientific']
            else:
                row['scientific'] = score_artifact(run, job, ws, out, battery_gold)
        except (OSError, ValueError, KeyError, TypeError, AssertionError, subprocess.SubprocessError) as error:
            row['scientific'] = {'status': 'unscorable_or_invalid_artifact', 'error': type(error).__name__ + ': ' + str(error)}
        rows.append(row)
    put(run / 'SCORE185.json', {'scored_utc': datetime.now(timezone.utc).isoformat(), 'subjects_planned': 16,
        'source_families': 4, 'task_scenarios': 8, 'rows': rows,
        'imprl_score_pending': True, 'common_mechanism_confirmed': False,
        'limits': ['This output does not decide causal scope failure from low numerical scores.',
                   'Initial/review pair is one task, not two independent benchmarks.',
                   'No population PHM failure rate or paper novelty is estimated.']})
    print(json.dumps({'scored_non_imprl_subjects': sum(r['family'] != 'imprl' for r in rows),
                      'imprl_subjects_pending': 4, 'hidden_feedback_to_subjects': False}))


def summarize(run):
    """Combine immutable native scores without deciding mechanism from errors."""
    destination = run / 'RESULT185.json'
    assert not destination.exists()
    source = json.loads((run / 'SCORE185.json').read_text())
    numerical_root = run / 'imprl_scoring185'
    numerical = {}
    for path in sorted(numerical_root.glob('*/SUMMARY.json')):
        data = json.loads(path.read_text())
        assert len(data['episodes']) == 64 and len({e['seed'] for e in data['episodes']}) == 64
        numerical[data['group']] = data
    assert len(numerical) == 6, 'All4submitted policies and2references required'
    jobs = []
    for name in ['PLAN185.json', 'REVIEW_PLAN185.json']:
        jobs += json.loads((run / name).read_text())['jobs']
    by_job = {j['id']: j for j in jobs}
    rows = []
    for row in source['rows']:
        row = dict(row)
        if row['family'] == 'imprl':
            item = numerical[row['job']]
            setting = by_job[row['job']]['setting']
            reference = numerical[setting + '_native_reference']
            successes = {e['seed']: e for e in item['episodes'] if e['complete']}
            refs = {e['seed']: e for e in reference['episodes'] if e['complete']}
            common = sorted(successes.keys() & refs.keys())
            costs = [e['discounted_cost'] for e in successes.values()]
            reference_costs = [e['discounted_cost'] for e in refs.values()]
            differences = [successes[s]['discounted_cost'] - refs[s]['discounted_cost'] for s in common]
            row['scientific'] = {
                'planned_episodes': 64, 'complete': len(successes), 'incomplete': 64 - len(successes),
                'reference_complete': len(refs), 'mean_cost_complete_only': average(costs),
                'reference_mean_cost_complete_only': average(reference_costs),
                'paired_complete_seeds': len(common), 'paired_mean_cost_difference': average(differences),
                'paired_difference_MC_SE': statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None,
                'max_manual_reward_difference': max([e['max_manual_difference'] for e in successes.values()], default=None),
                'episode_statuses': item['episodes'],
                'limits': 'Negative original shared discounted reward; lower better.64pseudo-random seeds are simulation repetitions, not64independent devices. No field-loss claim.'}
        rows.append(row)
    control = json.loads((run / 'controls185/battery_program_r3/RESULT.json').read_text())
    original = next(r for r in rows if r['job'] == 'battery185_case01')['scientific']
    truth = original.get('gold', {}).get('eol_cycle')
    comparisons = []
    for item in control['summaries']:
        value = item['matched_program_target_eol']
        error = abs(value - truth) if value is not None and truth is not None else None
        original_error = original.get('rul_absolute_error')
        comparisons.append(dict(item, target_absolute_error_cycles=error,
                                original_absolute_error_cycles=original_error,
                                error_change_vs_original=error - original_error if error is not None and original_error is not None else None))
    put(destination, {'finished_utc': datetime.now(timezone.utc).isoformat(), 'planned_sessions': 16,
         'source_families': 4, 'task_scenarios': 8, 'rows': rows,
         'battery_public_control_all_four_variants': comparisons,
         'mechanism_decision': 'Requires evidence assessment; numerical error is not scope overreach.',
         'scoped_problem_confirmed': False, 'common_mechanism_confirmed': False,
         'limitations': source['limits']})
    print(json.dumps({'result': str(destination), 'subjects': len(rows),
                      'imprl_planned_episode_attempts': 384, 'new_model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--completed-families', action='store_true')
    args = parser.parse_args()
    assert not (args.summarize and args.completed_families)
    action = summarize if args.summarize else score_completed_families if args.completed_families else score
    action(args.run.resolve())
