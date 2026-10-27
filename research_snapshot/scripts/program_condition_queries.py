"""Actual calibration-label queries and frozen, rotation-only program policies."""
import itertools
import json
from pathlib import Path

from controlled_feedback import save, sha, utc, remaining_budget

PROGRAMS = ['p00', 'p01', 'p04']


def validate_policy(value, centers):
    assert isinstance(value, dict)
    if value.get('kind') == 'global':
        assert set(value) == {'kind', 'program'} and value['program'] in PROGRAMS
        return dict(kind='global', program=value['program'])
    assert value.get('kind') == 'nearest' and set(value) == {'kind', 'assignments'}
    assignments = value['assignments']
    assert isinstance(assignments, dict) and set(assignments) == set(centers), 'Assign every available center exactly once'
    assert all(v in PROGRAMS for v in assignments.values())
    return dict(kind='nearest', assignments={key: assignments[key] for key in sorted(centers)})


def route(policy, centers, rotation_hz):
    if policy['kind'] == 'global':
        return policy['program']
    center = min(centers, key=lambda c: (abs(float(rotation_hz) - float(centers[c])), float(centers[c]), c))
    return policy['assignments'][center]


def best_program(errors):
    return min(PROGRAMS, key=lambda p: (errors[p], p != 'p00', p))


def evidence_policy(packets, kind):
    centers = {key: p['condition_hz'] for key, p in packets.items()}
    if kind == 'global':
        means = {p: sum(v['errors'][p] for v in packets.values()) / len(packets) for p in PROGRAMS}
        return dict(kind='global', program=best_program(means)), centers
    return dict(kind='nearest', assignments={key: best_program(p['errors']) for key, p in packets.items()}), centers


def policy_family(base_centers, calibration_centers, max_queries):
    """Common predeclared finite action family;no outcome-dependent pruning."""
    for program in PROGRAMS:
        yield dict(kind='global', program=program), dict(base_centers)
    for number in range(max_queries + 1):
        for queried in itertools.combinations(sorted(calibration_centers), number):
            centers = dict(base_centers, **{k: calibration_centers[k] for k in queried})
            keys = sorted(centers)
            for values in itertools.product(PROGRAMS, repeat=len(keys)):
                yield dict(kind='nearest', assignments=dict(zip(keys, values))), centers


class QueryBroker:
    def __init__(self, run, config):
        self.run = Path(run)
        self.config = config
        self.public = Path(config['public'])
        self.case = json.loads((self.public / 'cases.json').read_text())[0]
        self.private = json.loads(Path(config['query_private']).read_text())
        self.state_file = self.run / 'broker/QUERY_STATE.json'
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else dict(checked=False, queried=[], choice=None)

    def dispatch(self, request):
        action = request['action']
        if action == 'cases':
            return dict(cases=[self.case], final_choice=self.state['choice'], queried_slots=self.state['queried'],
                        episode_budget=remaining_budget(self.run))
        if action == 'evidence':
            assert request['case_id'] == 'update'
            self.state['checked'] = True
            save(self.state_file, self.state)
            return dict(self.case, queried_slots=self.state['queried'], episode_budget=remaining_budget(self.run))
        assert self.state['choice'] is None, 'Choice is already sealed'
        if action == 'query':
            slot = request['slot']
            assert slot in self.private['calibration'] and self.state['checked']
            cached = slot in self.state['queried']
            if not cached:
                assert len(self.state['queried']) < self.config['max_queries'], 'Calibration label-query budget exhausted'
                self.state['queried'].append(slot)
                save(self.run / 'broker/queries' / (slot + '.json'), dict(utc=utc(), request=request,
                     packet_sha256=sha(Path(self.config['query_private'])), packet=self.private['calibration'][slot],
                     real_author_record_labels=True, final_data=False))
                save(self.state_file, self.state)
            return dict(packet=self.private['calibration'][slot], cached=cached,
                        labels_consumed=4 * len(self.state['queried']), remaining_queries=self.config['max_queries'] - len(self.state['queried']),
                        episode_budget=remaining_budget(self.run))
        if action == 'select':
            rows = request['choices']
            assert len(rows) == 1 and rows[0]['case'] == 'update' and rows[0]['incumbent'] == 'p00'
            assert self.state['checked'] and isinstance(rows[0].get('reason'), str) and rows[0]['reason'].strip()
            selected = rows[0]['selected']
            centers = dict(self.private['base_centers'], **{k: self.private['calibration'][k]['condition_hz'] for k in self.state['queried']})
            if selected in PROGRAMS:
                policy = dict(kind='global', program=selected)
            else:
                assert selected == 'policy.json', 'Only the declared policy.json artifact is allowed'
                p = self.run / 'app/policy.json'
                assert not p.is_symlink() and p.is_file() and p.stat().st_size < 16384
                raw = p.read_bytes()
                policy = json.loads(raw)
            policy = validate_policy(policy, centers)
            save(self.run / 'broker/POLICY.json', policy)
            result = dict(utc=utc(), choices=rows, policy=policy, centers=centers, queried_slots=list(self.state['queried']),
                          labels_consumed=4 * len(self.state['queried']), checked_cases=['update'],
                          policy_sha256=sha(self.run / 'broker/POLICY.json'), final_scores_joined=False)
            self.state['choice'] = result
            save(self.state_file, self.state)
            save(self.run / 'broker/FINAL_CHOICE.json', result)
            return result
        raise ValueError('Unknown action')
