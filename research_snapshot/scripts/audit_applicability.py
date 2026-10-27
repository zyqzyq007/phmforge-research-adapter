"""Mechanical audit of frozen A66 references; claim semantics need separate review."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def boolean(x):
    if isinstance(x, bool): return x
    if isinstance(x, str) and x.lower() in ('true','false'): return x.lower() == 'true'
    raise ValueError('not a declared boolean')


def compare(rows, expected, keys, fields):
    mapped = {tuple(str(row[k]) for k in keys):row for row in rows}
    errors = []
    refkeys = {tuple(str(row[k]) for k in keys) for row in expected}
    if len(mapped) != len(rows) or set(mapped) != refkeys:
        errors.append({'type':'row_ids','actual_rows':len(rows),'unique_rows':len(mapped),
            'missing':sorted(refkeys-set(mapped)),'extra':sorted(set(mapped)-refkeys)})
    correct = 0
    for gold in expected:
        rid = tuple(str(gold[k]) for k in keys); got = mapped.get(rid, {})
        bad = []
        for name, kind, tol in fields:
            try:
                actual = boolean(got[name]) if kind == 'bool' else float(got[name])
                target = gold[name]
                ok = actual == target if kind == 'bool' else abs(actual-target) <= tol
            except (KeyError, ValueError, TypeError):
                ok = False
            if not ok: bad.append({'field':name,'actual':got.get(name),'expected':gold[name]})
        if bad: errors.append({'id':rid,'fields':bad})
        else: correct += 1
    return {'correct_rows':correct,'expected_rows':len(expected),'errors':errors}


def audit(run):
    hidden = run/'hidden'
    configs = json.loads((hidden/'configuration_reference.json').read_text())
    curves = json.loads((hidden/'trajectory_reference.json').read_text())
    curves = [dict(family=r['family'],specimen=r['specimen'],n_observations=r['source_rows'],
        first_exceedance_mm=r['first_observed_exceedance']['x_mm'],
        later_nonexceedance_count=len(r['later_nonexceedances']),published_lad_mm=r['published_lad_mm'],
        published_lad_matches=r['published_lad_matches_raw']) for r in curves]
    steel = json.loads((hidden/'steel_reference.json').read_text())
    out = []
    for job in json.loads((run/'native_plan66.json').read_text())['jobs']:
        ws = run/job['workspace']; log = run/'native'/job['id']
        status = json.loads((log/'status.json').read_text())
        item = {'job_id':job['id'],'status':status['status'],'strict_response_valid':status.get('response_valid'),
                'elapsed_seconds':status.get('elapsed_seconds'),'usage':status.get('usage')}
        for name, ref, keys, fields in [
            ('predictions.csv',configs,['case_id'],[('reported_a90_95_mm','float',1e-6),('reported_target_met','bool',0)]),
            ('trajectories.csv',curves,['family','specimen'],[(x,'float',0 if 'count' in x or x=='n_observations' else .012)
                for x in ['n_observations','first_exceedance_mm','later_nonexceedance_count','published_lad_mm']]+[('published_lad_matches','bool',0)]),
            ('steel.csv',steel,['sensor'],[(x,'float',.012) for x in ['initial_crack_mm','sensor_position_mm',
                'total_crack_at_detection_mm','published_growth_mm','computed_growth_mm']]+[('total_below_12_7_mm','bool',0)])]:
            try: item[name] = compare(read_csv(ws/name),ref,keys,fields)
            except (FileNotFoundError,KeyError,ValueError) as e:
                item[name] = {'error_type':type(e).__name__,'expected_rows':len(ref),'correct_rows':0}
        if (ws/'qualification.json').exists():
            try:
                q = json.loads((ws/'qualification.json').read_text())
                item['qualification_raw'] = q
            except ValueError: item['qualification_error'] = 'invalid_JSON'
        # Record strict parsing separately from predeclared unique-fence recovery.
        responses = list(log.glob('*response*'))
        item['response_files'] = [x.name for x in responses if x.is_file()]
        replay = run/'scientific_replay'/job['id']
        if replay.exists():
            names = ['predictions.csv','trajectories.csv','steel.csv','qualification.json','report.md']
            item['cold_replay_byte_equal'] = {n:(ws/n).is_file() and (replay/n).is_file() and
                hashlib.sha256((ws/n).read_bytes()).digest()==hashlib.sha256((replay/n).read_bytes()).digest() for n in names}
        out.append(item)
    return {'iteration':66,'jobs':out,'semantic_status':'Manual source-grounded review required; raw boolean is not an adjudicated scientific failure.',
        'evidence_unit':'One bounded source-audit task, not 213 independent maintenance decisions.',
        'new_E3':0,'natural_confirmation_gate_passed':False}


if __name__ == '__main__':
    run = Path(sys.argv[1]).resolve()
    result = audit(run)
    out = run/'mechanical_audit66.json'
    assert not out.exists(), 'Preserve original audit'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{'job':r['job_id'],'status':r['status'],**{k:r[k]['correct_rows'] for k in
        ['predictions.csv','trajectories.csv','steel.csv']}} for r in result['jobs']]))
