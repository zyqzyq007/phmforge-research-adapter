"""Inventory complete historical PHM batches without outcome-based row selection.

Reuses archived scores. It does not fit a policy, execute submissions, or grant
Agent access to retrospective evaluation labels. A manifest binding is narrower
than a new replay or a semantic audit of legal decision-time evidence.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FILES = {
    267: 'results/validation267/PAIRED_RESULTS267.json',
    268: 'results/selection268/NUMERICAL_RESULTS268.json',
    269: 'results/overfit269/NUMERICAL_RESULTS269.json',
    270: 'results/selection270/NUMERICAL_RESULTS270.json',
    274: 'results/validation274/NUMERICAL_RESULTS274.json',
    276: 'results/pipeline276/NUMERICAL_RESULTS276.json',
    278: 'results/feedback278/NUMERICAL_RESULTS278.json',
    281: 'results/review281/NUMERICAL_RESULTS281.json',
    284: 'results/evidence284/NUMERICAL_RESULTS281.json',
    305: 'results/rotor305/NUMERICAL_RESULTS305.json',
}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def metric(stage, kind='final'):
    result = stage.get(kind, stage.get('results', {}).get(kind, {}))
    values = result.get('metrics', result.get('all11', {}))
    key = 'accuracy' if 'accuracy' in values else 'mae'
    return values.get(key), bool(result.get('valid')), key


def complete(stage):
    return stage.get('status') == 'finished' and stage.get('formal_submission') is True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    assert Path('/tmp') not in run.parents
    started = time.monotonic()
    protocol = json.loads((run / 'PROTOCOL449.json').read_text())
    assert list(FILES) == protocol['batches']
    index = json.loads((PROJECT / 'results/archive_index.json').read_text())['archives']
    bindings, inventories, data, manifests, roots = [], {}, {}, {}, {}

    def bind(n):
        candidates = [r for r in index if str(r.get('run_id', '')).endswith(str(n))]
        assert len(candidates) == 1, (n, len(candidates))
        rec = candidates[0]
        root = Path(rec['archive_path'])
        manifest_name = rec.get('manifest_filename')
        if not manifest_name:
            receipt = json.loads((root / 'ARCHIVE_VERIFIED.json').read_text())
            assert receipt['manifest_sha256'] == rec['manifest_sha256']
            manifest_name = receipt.get('manifest_filename', 'SOURCE_MANIFEST.json')
        raw_manifest = (root / manifest_name).read_bytes()
        assert digest(raw_manifest) == rec['manifest_sha256'], n
        manifest = json.loads(raw_manifest)['files']
        raw = (PROJECT / FILES[n]).read_bytes()
        matches = [r for r in manifest if r['sha256'] == digest(raw) and r['bytes'] == len(raw)]
        assert matches, ('summary_not_bound', n)
        selected = min(matches, key=lambda r: (r['path'].count('/'), r['path']))
        (run / 'sources' / f'SUMMARY{n}.json').write_bytes(raw)
        b = dict(batch=n, project=FILES[n], archive=str(root), member=selected['path'],
                 sha256=digest(raw), bytes=len(raw), manifest_sha256=digest(raw_manifest),
                 manifest_filename=manifest_name,
                 binding='project bytes match verified canonical archive manifest')
        return n, json.loads(raw), manifest, root, b

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, obj, manifest, root, binding in pool.map(bind, FILES):
            bindings.append(binding)
            data[n], manifests[n], roots[n] = obj, manifest, root

    def inventory(n, job):
        key = f'A{n}:{job}'
        if key in inventories:
            return key
        mf = manifests[n]
        ws = [r for r in mf if f'/workspaces/{job}/' in r['path']
              and not r['path'].startswith(('workspace_snapshot/', 'audit/'))]
        plans = [r for r in mf if r['path'].endswith(f'/plans/{job}.txt')]
        events = [r for r in mf if r['path'].endswith(f'/{job}/{job}/events.jsonl')]
        statuses = [r for r in mf if r['path'].endswith(f'/{job}/{job}/status.json')]
        code = [r for r in ws if r['path'].endswith('.py') and '/.runtime/' not in r['path']]
        reports = [r for r in ws if r['path'].endswith('/report.md')]
        refs = [r for r in mf if f'/references/{job}/' in r['path']
                and not r['path'].startswith(('workspace_snapshot/', 'audit/'))]
        inventories[key] = dict(batch=n, job=job, archive=str(roots[n]),
                               code=code, reports=reports, plans=plans, events=events,
                               statuses=statuses, reference_members=refs,
                               workspace_manifest=ws, legal_feature_semantics_verified=False)
        return key

    rows = []

    def add(n, case, source, category, before, after, key, eligible, job,
            parent_job=None, arm=None, status=None, reason=None):
        higher = key in ('accuracy', 'macro_f1')
        tol = 1e-12 if higher else 1e-6
        gain = ((after - before) if higher else (before - after)) \
            if before is not None and after is not None else None
        direction = ('unpaired' if gain is None else
                     'unchanged' if abs(gain) <= tol else
                     'improved' if gain > 0 else 'worse')
        rows.append(dict(id=f'A{n}:{case}:{category}', batch=n, case=case,
                         source_family=source, category=category, arm=arm,
                         status=status, metric=key, before=before, after=after,
                         gain=gain, diagnostic_direction=direction,
                         eligible_complete_comparison=bool(eligible and gain is not None),
                         exclusion_or_scope=reason, evidence_key=inventory(n, job),
                         parent_evidence_key=inventory(n, parent_job) if parent_job else None,
                         independently_sampled_example=False))

    # Initial construction and later review are different populations.
    for n in [267, 268, 269, 270, 274, 276]:
        d = data[n]
        if n == 267:
            cases = [dict(case=k, dataset=k, stages=v['stages'])
                     for k, v in d['classification'].items()]
            cases.append(dict(case='femto', dataset='FEMTO',
                              stages={k: d['rul'][k] for k in ['initial', 'review']}))
        else:
            cases = d['cases']
        for c in cases:
            initial = c['stages']['initial']
            before, valid0, key0 = metric(initial, 'reference')
            after, valid1, key1 = metric(initial)
            scope = 'FEMTO all11 includes known label ambiguity; do not treat as clean training target' \
                if c['dataset'] == 'FEMTO' else None
            recovered = c['case'] in d.get('recovery_context_restart_cases', [])
            if recovered:
                scope = 'Historical infrastructure/context restart; retain but separate sensitivity'
            add(n, c['case'], c['dataset'], 'initial_vs_provided_reference',
                before, after, key0 if after is None else key1,
                complete(initial) and valid0 and valid1, initial['job'],
                arm=c.get('arm', 'ordinary_or_validation_setting'),
                status=initial['status'], reason=scope)
            if 'review' in c['stages']:
                review = c['stages']['review']
                new, valid2, key2 = metric(review)
                add(n, c['case'], c['dataset'], 'free_review_vs_initial', after, new,
                    key1 if after is not None else key2,
                    complete(initial) and complete(review) and valid1 and valid2,
                    review['job'], initial['job'], arm=c.get('arm', 'ordinary_or_validation_setting'),
                    status=review['status'], reason=scope)

    # A305 explicitly records stage outcomes; verify native status bytes as well.
    d = data[305]
    by_cohort = {r['cohort']: r for r in d['rows']}
    for row in d['rows']:
        job = f"s305_{row['block']}_{row['arm']}_{row['stage']}"
        inv = inventories[inventory(305, job)]
        assert len(inv['statuses']) == 1, job
        ref = inv['statuses'][0]
        raw = (roots[305] / ref['path']).read_bytes()
        assert digest(raw) == ref['sha256'] and len(raw) == ref['bytes']
        native = json.loads(raw)
        inv['verified_native_status'] = native['status']
        old = by_cohort[row['cohort'].replace('_review', '')]
        initial = row['stage'] == 'initial'
        before = row['reference_accuracy'] if initial else old['final_accuracy']
        parent = None if initial else f"s305_{row['block']}_{row['arm']}_initial"
        add(305, row['cohort'], 'ROTORBROKENBAR',
            'initial_vs_provided_reference' if initial else 'free_review_vs_initial',
            before, row['final_accuracy'], 'accuracy', native['status'] == 'finished',
            job, parent, row['arm'], native['status'])

    for n in [281, 284]:
        for c in data[n]['cases']:
            key = c['primary_metric']
            add(n, c['case'], c['dataset'], 'fixed_pool_review',
                c['original'].get(key), c['reviewed'].get(key), key,
                c['complete'] and c['status'] == 'finished', f"s{n}_{c['case']}",
                status=c['status'], reason='Repeated fixed candidate selection; not a newly fitted program')
    for c in data[278]['cohorts']:
        variant = next(v for v in data[278]['variants'] if v['variant']['id'] == c['final_variant'])
        score = variant['results']['final']['metrics']['mean_bearing_mae_minutes']
        add(278, c['case'], 'XJTU', 'unpaired_candidate_selection', None, score,
            'mean_bearing_mae_minutes', False, c['job'], arm=c['arm'], status=c['status'],
            reason='Multiple offered candidates; no unique registered parent, do not invent a revision pair')

    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['category'], row['source_family'])].append(row)
    counts = []
    for (category, source), group in sorted(groups.items()):
        qualified = [r for r in group if r['eligible_complete_comparison']]
        counts.append(dict(category=category, source_family=source, total=len(group),
                           complete_paired=len(qualified), unpaired_or_incomplete=len(group)-len(qualified),
                           directions=dict(collections.Counter(r['diagnostic_direction'] for r in qualified))))
    reviews = [r for r in rows if r['category'] == 'free_review_vs_initial'
               and r['eligible_complete_comparison']]
    folds = []
    for source in sorted({r['source_family'] for r in reviews}):
        train = [r for r in reviews if r['source_family'] != source]
        ct = collections.Counter(r['diagnostic_direction'] for r in train)
        folds.append(dict(held_out_family=source, train_rows=len(train), directions=dict(ct),
                          both_gain_and_harm=ct['improved'] > 0 and ct['worse'] > 0))
    result = dict(iteration='A449', scope=protocol['inclusion'], total_rows=len(rows),
                  source_families=sorted({r['source_family'] for r in rows}),
                  batch_count=len(data), counts=counts, leave_source_out_free_review_support=folds,
                  free_review_complete=len(reviews),
                  free_review_directions=dict(collections.Counter(r['diagnostic_direction'] for r in reviews)),
                  all_folds_gain_and_harm=all(f['both_gain_and_harm'] for f in folds),
                  program_inventory_count=len(inventories), summary_bindings=len(bindings),
                  fits=0, LLM_calls=0, new_predictions=0, new_independent_confirmations=0,
                  decision_time_feature_semantics_qualified=False,
                  policy_training_admitted=False,
                  decision='Do not train a general revision critic on this corpus yet. Keep fixed selection, construction and free review separate; qualify code/evidence and expand independent revision outcomes before source-disjoint learned validation.',
                  elapsed_seconds=time.monotonic()-started)
    dump(run/'analysis/ROWS449.json', rows)
    dump(run/'analysis/RESULT449.json', result)
    dump(run/'analysis/BINDINGS449.json', bindings)
    dump(run/'analysis/PROGRAM_INVENTORY449.json', inventories)
    with (run/'analysis/rows449.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['total_rows', 'source_families', 'free_review_complete',
          'free_review_directions', 'all_folds_gain_and_harm', 'elapsed_seconds']}))


if __name__ == '__main__':
    main()
