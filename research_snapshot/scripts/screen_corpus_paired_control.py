"""Fixed paired-1SE selector. Selection reads legal evidence only."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics as st
import time

from screen_score_blinding import read, save, sha


def records(rows):
    loss, signature = {}, {}
    for row in rows:
        key = str(row['group'])
        assert key not in loss
        loss[key] = float(row['actual'] != row['predicted'])
        signature[key] = (row['actual'], row['condition_hz'])
    return loss, signature


def evidence(row, prefixes):
    report = row['legal_report']
    assert report['status'] == 'complete' and report['final_data_read'] is False
    family = row['source_family']
    if family == 'HUST_BATTERY':
        loss, signature = {}, {}
        for fold in report['cv']:
            for cell in fold['cells']:
                key = str(cell['cell'])
                assert key not in loss
                loss[key], signature[key] = float(cell['MAE']), fold['fold']
        assert len(loss) == 47
        return {'primary': (loss, signature)}
    if family == 'COMFAULDA':
        primary = records([r for fold in report['cv'] for r in fold['records']])
        assert len(primary[0]) == 8
        full = records(report['development']['records'] + row['additional_legal_check']['records'])
        assert len(full[0]) == 12 and not (set(primary[0]) & set(full[0]))
        return {'primary': primary, 'full_fit_supplement': full}
    assert family == 'CMAPSS_FD001'
    check = prefixes[row['content_identity']]['check']
    assert check['observations_only_through_query'] and len(check['rows']) == 60
    loss, signature = {}, {}
    for query in check['rows']:
        key = str(query['engine'])
        loss.setdefault(key, []).append(abs(query['prediction'] - query['truth']))
        signature.setdefault(key, []).append((query['cycle'], query['cut'], query['truth']))
    assert len(loss) == 20 and all(len(v) == 3 for v in loss.values())
    loss = {k: st.mean(v) for k, v in loss.items()}
    assert all(abs(loss[str(e['engine'])]-e['MAE']) < 1e-8 for e in check['engine_errors'])
    return {'primary': (loss, signature)}


def choose(matrix, order):
    keys = sorted(matrix['p00'])
    means = {i: st.mean(matrix[i].values()) for i in order}
    best_value = min(means.values())
    best = next(i for i in order if means[i] <= best_value + 1e-12)
    differences = {}
    for ident in order:
        delta = [matrix[ident][k]-matrix[best][k] for k in keys]
        differences[ident] = dict(mean=st.mean(delta), sd=st.stdev(delta),
                                  conditional_se=st.stdev(delta)/math.sqrt(len(delta)))
    selected = next(i for i in order if differences[i]['mean'] <= differences[i]['conditional_se']+1e-10)
    return dict(minimum=best, paired_1SE=selected, keep='p00', means=means,
                differences_to_minimum=differences, entities=keys)


def main(root):
    started = time.monotonic()
    assert not (root/'CHOICES_SEALED466.json').exists()
    cfg = read(root/'PROTOCOL466.json')
    legal = read(root/'corpus/analysis/LEGAL_EVIDENCE450.json')
    prefixes = read(root/'prefix453/COMPLETE_EVIDENCE453.json')
    assert len(legal) == cfg['expected']['memberships']
    assert len({r['content_identity'] for r in legal}) == cfg['expected']['unique_programs']
    # No episode table, prior native selection, or final loss enters this stage.
    bank_ids = sorted({r['episode'] for r in legal})
    assert len(bank_ids) == cfg['expected']['episodes']
    inputs, decisions, qualification = [], [], []
    for episode in bank_ids:
        members = [r for r in legal if r['episode'] == episode]
        original = {r['program_id']: ix for ix,r in enumerate(members)}
        initial = [r['program_id'] for r in members if r['initial'] and r['program_id'] != 'p00']
        fresh = sorted([r for r in members if not r['initial']], key=lambda r: (r['available_utc'], original[r['program_id']]))
        order = ['p00'] + initial + [r['program_id'] for r in fresh]
        assert set(order) == set(original) and len(order) == len(members)
        try:
            extracted = {r['program_id']: evidence(r, prefixes) for r in members}
            views = set(extracted['p00'])
            assert all(set(e) == views for e in extracted.values())
            local_inputs, local_decisions = [], []
            for view in sorted(views):
                matrix = {i: extracted[i][view][0] for i in order}
                signatures = {i: extracted[i][view][1] for i in order}
                assert all(s == signatures['p00'] for s in signatures.values())
                assert all(set(v) == set(matrix['p00']) for v in matrix.values())
                packet = dict(episode=episode, family=members[0]['source_family'], view=view,
                              order=order, losses=matrix, signature=signatures['p00'])
                local_inputs.append(packet)
                local_decisions.append(dict(episode=episode, family=packet['family'], view=view,
                                      status='qualified', **choose(matrix, order)))
            inputs.extend(local_inputs)
            decisions.extend(local_decisions)
            qualification.append(dict(episode=episode, qualified=True, members=len(members), views=sorted(views)))
        except (AssertionError, KeyError, ValueError, TypeError) as error:
            qualification.append(dict(episode=episode, qualified=False, members=len(members),
                                      error_type=type(error).__name__, error=str(error)))
    save(root/'LEGAL_MATRICES466.json', inputs)
    save(root/'QUALIFICATION466.json', qualification)
    save(root/'CHOICES466.json', decisions)
    save(root/'CHOICES_SEALED466.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        all16accounted=len(qualification)==16, qualified=sum(q['qualified'] for q in qualification),
        wall_seconds=time.monotonic()-started, selector_reads_outcomes=False,
        inputs_sha256=sha((root/'LEGAL_MATRICES466.json').read_bytes()),
        choices_sha256=sha((root/'CHOICES466.json').read_bytes()),
        protocol_sha256=sha((root/'PROTOCOL466.json').read_bytes()),
        clarification_sha256=sha((root/'SCOPE_CLARIFICATION466.json').read_bytes()),
        selector_sha256=sha(Path(__file__).read_bytes())))
    print({'qualified':sum(q['qualified'] for q in qualification), 'total':len(qualification),
           'views':len(decisions), 'seconds':time.monotonic()-started}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
