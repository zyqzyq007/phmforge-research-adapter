"""Independent numerical and provenance checks for the frozen A452 probe."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(run):
    start = time.monotonic()
    checks = 0

    def check(value, label):
        nonlocal checks
        checks += 1
        assert value, label

    def close(a, b, label):
        check(np.allclose(a, b, atol=1e-9, rtol=1e-9), label)

    def rank(a):
        return np.asarray([(sum(y < x-1e-12 for y in a)
                            + (sum(abs(y-x) <= 1e-12 for y in a)-1)/2)/(len(a)-1)
                           for x in a])

    def norm(a):
        a = np.asarray(a)
        return (a-min(a))/(max(a)-min(a)) if max(a)-min(a) > 1e-12 else a*0

    def pick(scores, anchor):
        eligible = np.flatnonzero(np.asarray(scores) <= min(scores)+1e-12)
        return anchor if anchor in eligible else int(eligible[0])

    cases = read(run/'LEGAL_FEATURES452.json')
    legal = read(run/'corpus/analysis/LEGAL_EVIDENCE450.json')
    episodes = read(run/'corpus/analysis/EPISODES450.json')
    mapping = read(run/'SOURCE_TO_FACT452.json')
    fact_seal = read(run/'FACTS_SEALED452.json')
    facts = {r['id']: r['facts'] for r in fact_seal['jobs']}
    definitions = read(run/'FACT_DEFINITIONS452.json')
    gold = {r['id']: r['final_loss'] for r in read(run/'private_outcomes/analysis/HISTORICAL_OUTCOMES450.json')}
    result = read(run/'RESULT452.json')
    protocol = read(run/'PROTOCOL452.json')
    seal = read(run/'ALL_CHOICES_SEALED452.json')
    check(len(cases) == len(episodes) == 16 and len(legal) == 134, 'complete denominator')
    check(len({r['content_identity'] for r in legal}) == len(facts) == 59, 'unique artifacts')
    check(sum(e['completed'] for e in episodes) == 9, 'native incompletes retained')
    check(sha(run/'LEGAL_FEATURES452.json') == seal['feature_sha256'] == read(run/'FEATURES_FROZEN452.json')['sha256'], 'feature seals')
    check(read(run/'FEATURES_FROZEN452.json')['historical_outcomes_absent'], 'features frozen before outcomes restore')
    check(sha(run/'DECISIONS452.json') == seal['sha256'], 'choice seal')
    check(sha(Path(protocol['corpus_archive'])/'SOURCE_MANIFEST.json') == protocol['corpus_manifest_sha256'], 'canonical corpus')
    check(read(run/'SEMANTIC_REVIEW452.json')['review_complete'], 'pre-score semantic sample')
    for binding in read(run/'PREPARATION452.json')['bindings']:
        parent = Path(binding['archive'])
        roots = [run/'corpus', run/'tasks'] + list((run/'helpers').iterdir())
        matches = [p/binding['member'] for p in roots if (p/binding['member']).is_file()]
        check(any(sha(p) == binding['sha256'] for p in matches), 'restored binding '+binding['member'])
    check(sha(run/'private_outcomes/analysis/HISTORICAL_OUTCOMES450.json') == 'de70fe56aa5fd741668b9e9dd615a51ccb39d4d8fd4774034f7709f2d6c66a57', 'outcomes binding')

    unknowns = 0
    for ident, values in facts.items():
        visible = read(run/'visible'/(ident+'.json'))
        terminal = read(run/'extractions'/ident/'TERMINAL.json')
        check(sha(run/'extractions'/ident/'TERMINAL.json') == fact_seal['hashes'][ident], 'fact seal')
        check(values == terminal['facts'] and set(values) == set(definitions), 'schema')
        for name, value in values.items():
            check(value['value'] is None or type(value['value']) is bool, 'bool or unknown')
            if value['value'] is None:
                unknowns += 1
            if value['value'] is True:
                check(bool(value['evidence']), 'positive quote required')
            for quote in value['evidence']:
                check(quote['file'] in visible['source_files'] and 0 < len(quote['quote']) <= 180
                      and quote['quote'] in visible['source_files'][quote['file']], 'exact source quote')
        check(not terminal['EOF_delimiter_suffix'] or set(terminal['EOF_delimiter_suffix']) <= set('}]'), 'delimiter-only recovery')

    receipts = list((run/'model_gateway').glob('*/001_receipt.json'))
    requests = list((run/'model_gateway').glob('*/001_request.json'))
    check(len(receipts) == len(requests) == 56, '56 new requests only')
    usage = Counter()
    for path in receipts:
        receipt = read(path)
        request = read(path.with_name('001_request.json'))
        check(sha(path.with_name('001_request.json')) == receipt['request_sha256'], 'request binding')
        check(sha(path.with_name('001_response.sse')) == receipt['response_sha256'], 'response binding')
        check(receipt['model_requested'] == 'deepseek-flash' and receipt['returned_model_ids'] == ['deepseek-flash'] and receipt['upstream_status'] == 200, 'actual model')
        check(request['tools'] == [] and json.loads(request['input']) == read(run/'visible'/(receipt['job']+'.json')), 'exact visible request')
        usage.update({k: receipt['usage'][k] for k in ['input_tokens', 'output_tokens']})
    reuse = read(run/'REUSE_BINDINGS452.json')
    check(len(reuse) == 3 and sum(t['reused_prior_response'] for t in fact_seal['jobs']) == 3, 'three reused')
    for ident, binding in reuse.items():
        check(sha(run/'reused_extractions'/ident/'RESPONSE.json') == binding['response_sha256'], 'cached response')
        check(sha(run/'reused_extractions'/ident/'REQUEST.json') == binding['request_sha256'], 'cached request')
    for p in (run/'model_gateway').glob('*/status.json'):
        s = read(p)
        check(not s['active'] and not s['inflight'] and s['inflight_count'] == 0, 'gateway terminal')

    for c in cases:
        rows = [r for r in legal if r['episode'] == c['id']]
        check([r['program_id'] for r in rows] == c['ids'], 'candidate order')
        views = []
        for r in rows:
            report = r['legal_report']
            if c['source'] == 'HUST_BATTERY':
                v = {str(x['cell']): x['MAE'] for f in report['cv'] for x in f['cells']}
                t = {}
            elif c['source'] == 'CMAPSS_FD001':
                v = {str(x['engine']): x['MAE'] for x in report['development']['engine_errors']}
                t = {str(x['engine']): x['MAE'] for x in r['additional_legal_check']['engine_errors']} if r['additional_legal_check'] else {}
            else:
                v = {str(x['condition_hz']): x['record_error'] for x in report['cv']}
                records = defaultdict(list)
                for x in report['development']['records']:
                    records[str(x['condition_hz'])].append(x['actual'] != x['predicted'])
                t = {k: float(np.mean(a)) for k, a in records.items()}
                if r['additional_legal_check']:
                    t['additional_check'] = r['additional_legal_check']['record_error']
            views.append((v, t))
        vk = sorted(set.intersection(*(set(v) for v, _ in views)))
        tk = sorted(set.intersection(*(set(t) for _, t in views)))
        check(vk == c['primary_group_ids'] and tk == c['secondary_group_ids'], 'common view groups')
        a = np.array([[v[k] for k in vk] for v, t in views])
        b = np.array([[t[k] for k in tk] for v, t in views]) if tk else a.copy()
        check(c['secondary_fallback_to_primary'] == (not tk), 'whole-pool fallback')
        both = np.concatenate((a, b), axis=1)
        ar, br = rank(a.mean(1)), rank(b.mean(1))
        n = np.array([ar, br, rank(a.max(1)), rank(b.max(1)), rank(both.std(1)),
                      -(a <= a.min(0)+1e-12).mean(1), -(b <= b.min(0)+1e-12).mean(1),
                      (ar-br)*(b.mean()-a.mean())/(b.mean()+a.mean()+1e-12)]).T
        n -= n.mean(0)
        close(n, c['numeric'], 'numeric reconstruction')
        z = np.array([[value for name in definitions for value in
                       [facts[mapping[r['id']]][name]['value'] is True, facts[mapping[r['id']]][name]['value'] is not None]] for r in rows], dtype=float)/np.sqrt(24)
        z -= z.mean(0)
        close(z, c['facts'], 'descriptor encoding')
        interaction = np.concatenate((z, z*(ar-br)[:, None], z*n[:, 4, None]), axis=1)
        interaction -= interaction.mean(0)
        close(interaction, c['conditioned'], 'interactions')
        for name, values in [('primary', a.mean(1)), ('secondary', b.mean(1)), ('all_mean', both.mean(1)), ('worst', both.max(1))]:
            close(values, c['controls'][name], 'fixed control')

    decisions = {(d['case'], d['method']): d['selected'] for d in read(run/'DECISIONS452.json')}
    check(len(decisions) == 160, 'all decisions')
    dimensions, variance, max_delta = {}, {}, 0.0
    for path in sorted((run/'fold_inputs').glob('*.json')):
        fold = read(path)
        held = fold['held_source']
        train, test = fold['train'], fold['test']
        sources = {c['source'] for c in train}
        check(len(sources) == 2 and held not in sources and {c['source'] for c in test} == {held}, 'source exclusion')
        check(all(not any(k in c for k in ['target', 'historical_selected', 'native_completed', 'native_arm']) for c in test), 'no held labels/decisions')
        for c in train:
            target = norm([gold[c['id']+'/'+p] for p in c['ids']])
            close(target-target.mean(), c['target'], 'training target')
        counts = Counter((c['source'], h) for c in train for h in c['contents'])
        unique = {s: len({h for c in train if c['source'] == s for h in c['contents']}) for s in sources}
        weights = np.array([1/(2*unique[c['source']]*counts[c['source'], h]) for c in train for h in c['contents']])
        out = read(run/'fold_outputs'/path.name)
        vocab, idf = out['tfidf_vocabulary'], np.asarray(out['tfidf_idf'])
        docs = {h: t for c in train for h, t in zip(c['contents'], c['texts'])}
        token_docs = [set(re.findall(r'(?u)\b[A-Za-z_0-9]+\b', t)) for t in docs.values()]
        for token, ix in vocab.items():
            df = sum(token in t for t in token_docs)
            check(df > 0, 'vocabulary only training sources')
            close(idf[ix], np.log((1+len(token_docs))/(1+df))+1, 'training-only idf')

        def matrix(c, method):
            if method == 'numeric':
                return np.asarray(c['numeric'])
            if method == 'numeric_tfidf':
                z = np.zeros((len(c['texts']), len(vocab)))
                for i, text in enumerate(c['texts']):
                    for token, count in Counter(re.findall(r'(?u)\b[A-Za-z_0-9]+\b', text)).items():
                        if token in vocab:
                            z[i, vocab[token]] = (1+np.log(count))*idf[vocab[token]]
                    length = np.linalg.norm(z[i])
                    if length:
                        z[i] /= length
                z -= z.mean(0)
            else:
                z = np.asarray(c['facts' if method == 'numeric_facts' else 'conditioned'])
            return np.column_stack((c['numeric'], z))

        dimensions[held], variance[held] = {}, {}
        for method, model in out['models'].items():
            x = np.vstack([matrix(c, method) for c in train])
            y = np.concatenate([c['target'] for c in train])
            close(x, model['train_matrix'], 'training features')
            close(weights, model['weights'], 'duplicate-aware source weights')
            close(y, model['y'], 'training labels')
            prior = np.zeros(x.shape[1]); prior[:2] = .5
            close(prior, model['prior'], 'same prior')
            design = np.vstack((np.sqrt(weights)[:, None]*x, np.sqrt(fold['lambda'])*np.eye(x.shape[1])))
            target = np.concatenate((np.sqrt(weights)*y, np.sqrt(fold['lambda'])*prior))
            beta = np.linalg.lstsq(design, target, rcond=None)[0]
            close(beta, model['beta'], 'independent augmented least squares')
            max_delta = max(max_delta, float(np.max(np.abs(beta-model['beta']))))
            dimensions[held][method] = x.shape[1]
            variance[held][method] = dict(training_zero_columns=int((np.ptp(x, axis=0) < 1e-12).sum()),
                zero_train_varying_test_columns=int(((np.ptp(x, axis=0) < 1e-12) & (np.ptp(np.vstack([matrix(c, method) for c in test]), axis=0) > 1e-12)).sum()))
            for c in test:
                xt = matrix(c, method)
                close(xt, model['test_matrices'][c['id']], 'held features')
                close(xt@beta, model['scores'][c['id']], 'held scores')
                selected = c['ids'][pick(xt@beta, c['ids'].index('p00'))]
                check(selected == decisions[c['id'], method], 'independent choice')
        history = {}
        for rule in ['primary', 'secondary', 'all_mean', 'worst']:
            by = defaultdict(list)
            for c in train:
                target = norm([gold[c['id']+'/'+p] for p in c['ids']])
                by[c['source']].append(target[pick(c['controls'][rule], c['ids'].index('p00'))])
            history[rule] = np.mean([np.mean(v) for v in by.values()])
        best_rule = min(history, key=history.get)
        check(best_rule == fold['history_rule'], 'history baseline training only')
        for c in test:
            old = c['ids'].index('p00')
            for rule, scores in c['controls'].items():
                check(c['ids'][pick(scores, old)] == decisions[c['id'], rule], 'rule selection')
            check(decisions[c['id'], 'history_rule'] == decisions[c['id'], best_rule], 'history choice')
            check(decisions[c['id'], 'keep'] == 'p00', 'keep choice')

    lookup = {c['id']: c for c in cases}
    for row in result['rows']:
        c = lookup[row['case']]
        loss = np.array([gold[c['id']+'/'+p] for p in c['ids']]); target = norm(loss)
        i, old = c['ids'].index(row['selected']), c['ids'].index('p00')
        close([row[k] for k in ['loss', 'anchor', 'regret', 'harm', 'gain']],
              [loss[i], loss[old], target[i], max(0, target[i]-target[old]), max(0, target[old]-target[i])], 'score recomputation')
    for method, summary in result['summaries'].items():
        for source, values in summary['by_source'].items():
            rows = [r for r in result['rows'] if r['method'] == method and r['source'] == source]
            for key, value in values.items():
                close(value, np.mean([r[key] for r in rows]), 'source mean')
        for key in ['regret', 'harm', 'gain']:
            close(summary[key], np.mean([v[key] for v in summary['by_source'].values()]), 'equal-source mean')
    summary = result['summaries']; main = summary['conditioned_facts']
    for method, comparison in result['comparisons'].items():
        deltas = [main['by_source'][s]['regret']-summary[method]['by_source'][s]['regret'] for s in main['by_source']]
        check(comparison == dict(wins=sum(d < -1e-12 for d in deltas), ties=sum(abs(d) <= 1e-12 for d in deltas), losses=sum(d > 1e-12 for d in deltas)), 'source comparison')
    gate = all(main['regret'] < v['regret']-1e-12 for k, v in summary.items() if k not in ['conditioned_facts', 'keep'])
    gate = gate and all(result['comparisons'][k]['wins'] >= 2 and main['harm'] <= summary[k]['harm']+1e-12 and main['gain'] >= summary[k]['gain']-1e-12 for k in ['numeric', 'numeric_tfidf', 'history_rule'])
    check(result['primary_gate_passed'] == gate, 'frozen gate')
    identical = {m: sum(decisions[c['id'], m] == decisions[c['id'], 'conditioned_facts'] for c in cases) for m in ['numeric', 'numeric_tfidf', 'numeric_facts']}
    report = dict(utc=datetime.now(timezone.utc).isoformat(),checks=checks,passed=True,seconds=time.monotonic()-start,
        independent_augmented_lstsq_solves=12,max_beta_difference=max_delta,dimensions=dimensions,
        feature_variance_diagnostic=variance,identical_choices_out_of_16=identical,unknown_fact_fields=unknowns,
        semantic_truth_certified=False,gate_passed=gate,requests=56,reused_responses=3,**usage,
        scientific_meta_fits=12,unpersisted_meta_fits_before_serialization_failure=12,total_meta_fit_executions=24,
        TFIDF_fits_including_recovery=6,new_PHM_fits=0,new_Agent_stages=0,independent_confirmations=0)
    (run/'AUDIT452.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
