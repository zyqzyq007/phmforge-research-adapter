"""Independent post-seal arithmetic/provenance audit for whole-program trials.

Reads genuine cold window predictions, reconstructs acquisition votes with
Counter/Fraction, and recounts harms, repairs, fixed votes, gates and costs.
It does not run any fitted program or call a model. Archive regression checks
must write to a separate live output directory, never into the source archive.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def audit(root, iteration, output):
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = read(root / ('RESULT' + iteration + '.json'))
    cfg = read(root / ('PROTOCOL' + iteration + '.json'))
    seal = read(root / 'ALL_CHOICES_SEALED337.json')
    assert read(root / 'HANDLE337.json')['status'] == 'terminal'
    assert seal['final_scores_joined'] is False and len(seal['episodes']) == 4
    assert sha(root / ('PROTOCOL' + iteration + '.json')) == seal['protocol_sha256']
    assert sha(root / 'EFFECTIVE_CODE_FREEZE337.json') == seal['code_freeze_sha256']
    checks = 4
    for name, digest in read(root / 'EFFECTIVE_CODE_FREEZE337.json').items():
        assert sha(root / name) == digest; checks += 1
    source = Path(cfg['source_archive'])
    path = source / 'qualification/final.npz'
    assert sha(path) == read(source / 'QUALIFICATION346.json')['files']['final.npz']['sha256']; checks += 1
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in z['sample_ids']]
        groups = [str(x) for x in z['groups']]
        truth = [int(x) for x in z['labels']]
    grouped_truth = {}
    for group in sorted(set(groups)):
        ys = {truth[i] for i, g in enumerate(groups) if g == group}
        assert len(ys) == 1
        grouped_truth[group] = ys.pop()
    predictions = {}; n = len(grouped_truth)
    for key, reported in result['all_program_results'].items():
        if reported['status'] != 'complete': continue
        versions = []
        for direction in ['forward', 'cold_repeat']:
            p = root / 'evaluation' / key / direction / 'PREDICTIONS.npz'
            with np.load(p, allow_pickle=False) as z:
                assert [str(x) for x in z['sample_ids']] == ids
                yhat = [int(x) for x in z['prediction']]
            assert len(yhat) == len(truth) and set(yhat) <= {0, 1, 2, 3}; checks += 2
            versions.append(yhat)
        assert versions[0] == versions[1]; checks += 1
        voted = {}
        for group in grouped_truth:
            counts = Counter(versions[0][i] for i, g in enumerate(groups) if g == group)
            voted[group] = min(counts, key=lambda k: (-counts[k], k))
        predictions[key] = voted
        assert {r['group']: r['predicted'] for r in reported['records']} == voted
        assert {r['group']: r['actual'] for r in reported['records']} == grouped_truth
        assert Fraction(reported['record_error']) == Fraction(sum(voted[g] != y for g, y in grouped_truth.items()), n)
        checks += 3
    base = predictions['common_p00']
    def stats(pred):
        return dict(record_error=Fraction(sum(pred[g] != y for g, y in grouped_truth.items()), n),
                    record_harm=Fraction(sum(base[g] == y and pred[g] != y for g, y in grouped_truth.items()), n),
                    record_gain=Fraction(sum(base[g] != y and pred[g] == y for g, y in grouped_truth.items()), n))
    best = min(stats(p)['record_error'] for p in predictions.values())
    collected = {}; vote_errors = {}
    for episode in result['episodes']:
        prefix = episode['id'] + '__'
        state = read(root / 'subjects' / episode['id'] / 'broker/STATE.json')
        lookup = lambda ident: predictions['common_p00' if ident == 'p00' else prefix + ident]
        for ident, entry in state['programs'].items():
            location = root / 'reference/p00' if ident == 'p00' else root / 'subjects' / episode['id'] / 'programs' / ident
            for name, digest in entry['report']['artifacts'].items():
                assert sha(location / name) == digest; checks += 1
        for name, choice in episode['methods'].items():
            if choice['evaluation_status'] != 'complete': continue
            actual = stats(lookup(choice['program_id']))
            actual['common_union_selection_loss'] = actual['record_error'] - best
            for metric, value in actual.items():
                assert abs(float(value) - choice[metric]) < 1e-12, (episode['id'], name, metric); checks += 1
            if name == 'actor': collected[episode['id']] = actual
        control = episode['same_bank_equal_vote']
        if control['status'] == 'complete':
            anchor = lookup(control['anchor']); voted = {}
            for group in grouped_truth:
                counts = Counter(lookup(ident)[group] for ident in control['base_programs'])
                tied = [k for k, count in counts.items() if count == max(counts.values())]
                voted[group] = anchor[group] if anchor[group] in tied else min(tied)
            assert voted == {r['group']: r['predicted'] for r in control['records']}
            vote_errors[episode['id']] = stats(voted)['record_error']
            assert abs(float(vote_errors[episode['id']]) - control['record_error']) < 1e-12; checks += 2
    arms = list(result['primary']); assert len(arms) == 2
    means = {}
    for arm in arms:
        episodes = [e for e in result['episodes'] if e['arm'] == arm]
        summary = result['primary'][arm]
        valid = all(e['completed'] and e['id'] in collected for e in episodes)
        assert summary['valid'] == valid and summary['completed'] == sum(e['completed'] for e in episodes); checks += 2
        means[arm] = None
        if valid:
            means[arm] = {k: sum((collected[e['id']][k] for e in episodes), Fraction()) / len(episodes) for k in collected[episodes[0]['id']]}
            for k, value in means[arm].items(): assert abs(float(value) - summary[k]) < 1e-12; checks += 1
        else:
            assert all(summary[k] is None for k in ['record_error', 'record_harm', 'record_gain', 'common_union_selection_loss']); checks += 1
    a, b = (means[arm] for arm in arms)
    gate = bool(a is not None and b is not None and len(predictions) == len(result['all_program_results']) and b['record_error'] < min(a['record_error'], stats(base)['record_error']) and b['record_harm'] <= a['record_harm'] and b['record_gain'] >= a['record_gain'])
    assert gate == result['joint_pilot_gate_passed']; checks += 1
    requests = list(root.glob('subjects/*/model_gateway/*/*_request.json')) + list(root.glob('chains/*/model_gateway/*/*_request.json'))
    receipts = []; hash_checks = 0
    for p in [*root.glob('subjects/*/model_gateway/*/*_receipt.json'), *root.glob('chains/*/model_gateway/*/*_receipt.json')]:
        receipt = read(p); prefix = str(receipt['sequence']).zfill(3)
        assert sha(p.parent / (prefix + '_request.json')) == receipt['request_sha256']; hash_checks += 1
        if receipt.get('response_sha256'):
            assert sha(p.parent / (prefix + '_response.sse')) == receipt['response_sha256']; hash_checks += 1
        assert receipt['model_requested'] == 'deepseek-flash'; receipts.append(receipt)
    fits = [read(p) for p in root.glob('subjects/*/programs/*/**/training_audit.json') if read(p).get('mode') == 'fit']
    cost = dict(actual_requests=len(requests), receipts=len(receipts), actual_new_fit_entries=len(fits), actual_estimator_fits=sum(r['completed_estimator_fits'] for r in fits),
                input_tokens=sum(r.get('usage', {}).get('input_tokens', 0) for r in receipts), output_tokens=sum(r.get('usage', {}).get('output_tokens', 0) for r in receipts))
    for k, value in cost.items(): assert value == result['cost'][k]; checks += 1
    report = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks, receipt_hash_checks=hash_checks,
                  audited_iteration='A' + iteration, audited_source=str(root), source_result_sha256=sha(root / ('RESULT' + iteration + '.json')),
                  cold_window_predictions_recounted=True, source_labels_rebound=True, costs=cost, new_LLM_calls=0, new_fits=0, new_forwards=0,
                  scope='Numerical/provenance audit; behavioral attribution, unregistered command review, and novelty remain separate.')
    output.parent.mkdir(parents=True, exist_ok=True); assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--iteration', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.run.resolve(), a.iteration, a.output.resolve())
