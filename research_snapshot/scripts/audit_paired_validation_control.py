"""One fixed conventional rule on archived CV evidence, no model execution."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as st
import time

from screen_score_blinding import read, save, sha, restore


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=st.mean(values), median=st.median(values),
                sd=st.stdev(values), conditional_se=st.stdev(values)/math.sqrt(len(values)),
                improved=sum(x < -1e-10 for x in values), worse=sum(x > 1e-10 for x in values),
                tied=sum(abs(x) <= 1e-10 for x in values))


def run(root):
    started = time.monotonic()
    cfg = read(root/'PROTOCOL465.json')
    assert not (root/'CHOICES_SEALED465.json').exists()
    bindings, banks = [], []
    for entry in cfg['archives']:
        archive = Path(entry['path'])
        assert sha((archive/'SOURCE_MANIFEST.json').read_bytes()) == entry['manifest_sha256']
        dest = root/'inputs'/entry['arm']
        bindings += restore(archive, dest, lambda p:
            p.startswith('subjects/'+entry['arm']+'_') and p.endswith('/REPORT.json') and '/programs/' in p
            or p in ['incumbents/1/REPORT.json', 'incumbents/2/REPORT.json'])
        for repeat in [1, 2]:
            episode = entry['arm']+'_'+str(repeat)
            reports = {'p00': read(dest/'incumbents'/str(repeat)/'REPORT.json')}
            for path in sorted((dest/'subjects'/episode/'programs').glob('*/REPORT.json')):
                report = read(path)
                if report['status'] == 'complete':
                    reports[path.parent.name] = report
            values, folds = {}, {}
            for ident, report in reports.items():
                values[ident] = {str(c['cell']): float(c['MAE']) for f in report['cv'] for c in f['cells']}
                folds[ident] = {str(c['cell']): f['fold'] for f in report['cv'] for c in f['cells']}
                assert len(values[ident]) == 47
                assert folds[ident] == folds['p00'], 'Paired errors must use matching held-out folds'
            ids = sorted(values['p00'])
            order = ['p00'] + sorted(i for i in values if i != 'p00')
            minimum = min(order, key=lambda i: (st.mean(values[i].values()), order.index(i)))
            differences = {i: describe(values[i][c]-values[minimum][c] for c in ids) for i in order}
            selected = next(i for i in order if differences[i]['mean'] <= differences[i]['conditional_se']+1e-10)
            comparisons = {}
            for ident in order:
                if ident == 'p00':
                    continue
                delta = {c:values[ident][c]-values['p00'][c] for c in ids}
                groups = sorted(set(folds[ident].values()))
                by_fold = {f: describe(delta[c] for c in ids if folds[ident][c] == f) for f in groups}
                gains = sorted([-x for x in delta.values() if x < 0], reverse=True)
                comparisons[ident] = dict(delta_by_cell=delta, summary=describe(delta.values()), by_fold=by_fold,
                    top1_fraction_of_total_gain=gains[0]/sum(gains) if gains else None,
                    top5_fraction_of_total_gain=sum(gains[:5])/sum(gains) if gains else None,
                    leave_one_cell_mean={c:st.mean(v for k,v in delta.items() if k != c) for c in ids},
                    leave_one_fold_mean={f:st.mean(delta[c] for c in ids if folds[ident][c] != f) for f in groups})
            banks.append(dict(id=episode, order=order, cv_errors=values, folds=folds['p00'],
                CV_minimum=minimum, paired_1SE_earliest=selected, keep='p00',
                paired_differences_from_minimum=differences, comparisons_to_incumbent=comparisons))
    save(root/'SOURCE_BINDINGS465.json', bindings)
    save(root/'CHOICES_SEALED465.json', dict(utc=datetime.now(timezone.utc).isoformat(), banks=banks,
         selection_uses_only_CV=True, rule_sha256=sha(Path(__file__).read_bytes())))
    # Previously opened outcomes are read only after all fixed decisions persist.
    outcome_path = Path(cfg['outcomes_path'])
    assert sha(outcome_path.read_bytes()) == cfg['outcomes_sha256']
    save(root/'HISTORICAL_OUTCOMES462.json', read(outcome_path))
    lookup = {r['id']:r for r in read(outcome_path)['rows']}
    rows = []
    for bank in banks:
        row = lookup[bank['id']]
        assert set(row['bank']) == set(bank['order'])
        for ident in bank['order']:
            assert abs(st.mean(bank['cv_errors'][ident].values())-row['bank'][ident]['CV']) < 1e-9
        anchor = row['bank']['p00']['errors']
        oracle = min(v['MAE'] for v in row['bank'].values())
        decisions = {}
        for method in ['keep','CV_minimum','paired_1SE_earliest','native']:
            ident = row['selected'] if method == 'native' else bank[method]
            if ident is None:
                decisions[method] = dict(status='incomplete', selected=None)
                continue
            errors = row['bank'][ident]['errors']
            decisions[method] = dict(status='complete', selected=ident, MAE=st.mean(errors),
                bank_selection_loss=st.mean(errors)-oracle,
                harm=st.mean(max(a-b,0) for a,b in zip(errors,anchor)),
                gain=st.mean(max(b-a,0) for a,b in zip(errors,anchor)))
        rows.append(dict(id=bank['id'], native_completed=row['completed'], decisions=decisions))
    prior = Path(cfg['prior_control_path'])
    assert sha(prior.read_bytes()) == cfg['prior_control_sha256']
    save(root/'REUSED_PRIOR369.json', read(prior))
    primary = next(r for r in rows if r['id']=='ordinary_2')
    result = dict(id='A465', utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        primary_harmful_replacement_blocked=primary['decisions']['paired_1SE_earliest']['selected']=='p00',
        primary=primary, elapsed_seconds=time.monotonic()-started,
        new_model_requests=0, new_PHM_fits=0, new_predictions=0, new_native_stages=0,
        physical_sources=1, related_incumbents=2, physical_exploratory_cells=8,
        independent_confirmation=0, novel_method=False, goal_complete=False,
        conditional_heuristic_not_confidence=True, prior369_reused_not_new=True)
    save(root/'RESULT465.json', result)
    print(json.dumps(dict(primary=result['primary_harmful_replacement_blocked'], rows=rows,
                         elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)
