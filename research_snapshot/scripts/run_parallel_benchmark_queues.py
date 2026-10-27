"""Resume the two frozen PHM queues under an explicitly approved quota amendment.

Reuses the unchanged primary launcher, capacity census, isolation, tool cleanup
and per-task budgets. Default operation is a metadata-only preparation check.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import run_full_benchmark_campaign as original


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('/srv/yyf-zyq/PHM_20261010')
RUN_IDS = ('20261018T185541Z_assethf190', '20261019T032531Z_assetextra202')
AUDIT = ROOT / RUN_IDS[1] / 'audit/acceleration213'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(pid):
    path = Path('/proc') / str(pid) / 'stat'
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def queues():
    result = []
    for name in RUN_IDS:
        run = ROOT / name
        protocol = json.loads((run / 'PROTOCOL187.json').read_text())
        assert protocol['model'] == 'deepseek-flash'
        assert protocol['max_concurrency'] == 8
        assert protocol['max_new_primary_rolling24h'] == 120
        assert protocol['subject_timeout_seconds'] == 1200
        assert protocol['original_tool_timeout_seconds'] == 180
        assert json.loads((run / 'FIRST_SUBJECT_GATE187.json').read_text())['passed']
        jobs = json.loads((run / 'PLAN187_LAUNCH.json').read_text())['jobs']
        assert len({job['id'] for job in jobs}) == len(jobs)
        pending, states = [], Counter()
        for job in jobs:
            path = original.status_path(run, job)
            if path.exists():
                states[json.loads(path.read_text())['status']] += 1
            else:
                pending.append(job)
        result.append(dict(run=run, protocol=protocol, jobs=jobs,
                           pending=pending, states=dict(states),
                           env=original.clean_env(run), active={}))
    return result


def prepare(items):
    bindings = []
    for item in items:
        run = item['run']
        controller = json.loads((run / 'CONTROLLER_STATUS187.json').read_text())
        pipeline_file = run / ('PIPELINE_STATUS190.json' if run.name == RUN_IDS[0]
                               else 'PIPELINE_STATUS202.json')
        pipeline = json.loads(pipeline_file.read_text())
        bindings.append(dict(run=str(run), jobs=len(item['jobs']),
                             states=item['states'], pending=len(item['pending']),
                             protocol_sha256=digest(run / 'PROTOCOL187.json'),
                             plan_sha256=digest(run / 'PLAN187_LAUNCH.json'),
                             controller_pid=controller['controller_pid'],
                             pipeline_pid=pipeline['pipeline_pid']))
    return dict(utc=original.now().isoformat(), phase='prepared_not_authorized',
                queues=bindings, proposed_max_concurrency=8,
                proposed_max_new_primary_rolling24h=400,
                primary_launcher_sha256=digest(Path(original.__file__)),
                capacity_census_sha256=digest(PROJECT / 'scripts/recount_subject_capacity.py'),
                dispatcher_sha256=digest(Path(__file__)),
                new_LLM_calls=0, frozen_task_files_changed=False)


def take_over(prepared, items, approval):
    assert approval.get('user_approved') is True
    assert approval['max_concurrency'] == 8
    assert approval['max_new_primary_rolling24h'] == 400
    assert approval['run_ids'] == list(RUN_IDS)
    assert approval['remaining_original_tasks_only'] is True
    assert approval.get('user_message')
    lock_path = PROJECT / 'logs/background_research.lock'
    with lock_path.open('a+') as lock:
        deadline = time.monotonic() + 50
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Admission lock busy; no takeover performed')
                time.sleep(0.2)
        records = []
        for binding, item in zip(prepared['queues'], items):
            run = item['run']
            assert digest(run / 'PROTOCOL187.json') == binding['protocol_sha256']
            assert digest(run / 'PLAN187_LAUNCH.json') == binding['plan_sha256']
            assert not any(json.loads(original.status_path(run, job).read_text())['status'] == 'running'
                           for job in item['jobs'] if original.status_path(run, job).exists())
            for kind in ('pipeline', 'controller'):
                pid = binding[kind + '_pid']
                command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
                expected = ('pipeline190.py' if run.name == RUN_IDS[0] else 'pipeline202.py')
                if kind == 'controller':
                    expected = 'run_full_benchmark_campaign.py'
                assert any(Path(os.fsdecode(arg)).name == expected for arg in command if arg)
                if kind == 'controller':
                    assert os.fsencode(str(run)) in command
                else:
                    assert any(str(run) in os.fsdecode(arg) for arg in command)
            snapshot = AUDIT / (run.name + '_before_takeover.json')
            original.save(snapshot, dict(controller=json.loads((run / 'CONTROLLER_STATUS187.json').read_text()),
                                         ownership=json.loads((run / 'OWNERSHIP.json').read_text())))
        for binding in prepared['queues']:
            # Stop only idle scheduling parents, preserving brokers and all data.
            for kind in ('pipeline', 'controller'):
                pid = binding[kind + '_pid']
                os.kill(pid, signal.SIGTERM)
                records.append(dict(run=binding['run'], role=kind, pid=pid, signal='SIGTERM'))
        deadline = time.monotonic() + 8
        while any(alive(record['pid']) for record in records):
            if time.monotonic() > deadline:
                raise RuntimeError('Old idle scheduler did not exit; no new admission')
            time.sleep(0.1)
        original.save(AUDIT / 'TAKEOVER.json', dict(utc=original.now().isoformat(),
                      old_idle_processes=records, new_controller_pid=os.getpid(),
                      brokers_preserved=True, frozen_protocols_unchanged=True))


def heartbeat(item, waiting):
    run = item['run']
    states = Counter()
    for job in item['jobs']:
        path = original.status_path(run, job)
        if path.exists():
            states[json.loads(path.read_text())['status']] += 1
    fields = dict(controller_pid=os.getpid(), updated_utc=original.now().isoformat(),
                  planned=len(item['jobs']), pending=len(item['pending']),
                  active_jobs=list(item['active']), waiting=waiting,
                  terminal_counts=dict(states), controller='parallel_original_queues',
                  scheduling_amendment=str(AUDIT / 'APPROVAL.json'),
                  original_task_and_per_task_budgets_unchanged=True)
    if not item['pending'] and not item['active']:
        fields.update(status='all_selected_subjects_terminal_no_scoring',
                      ended_utc=original.now().isoformat())
    original.save(run / 'CONTROLLER_STATUS187.json', fields)
    ownership = json.loads((run / 'OWNERSHIP.json').read_text())
    ownership.update(controller_pid=os.getpid(), writers_stopped=False,
                     lease_expires_utc=(original.now() + timedelta(hours=1)).isoformat(),
                     status='original_subjects_running_or_quota_wait',
                     heartbeat_utc=original.now().isoformat())
    original.save(run / 'OWNERSHIP.json', ownership)
    return dict(run=run.name, states=dict(states), pending=len(item['pending']),
                active_jobs=list(item['active']), waiting=waiting)


def execute(items):
    serials = {x['run'].name: len(list((x['run'] / 'audit').glob('CAPACITY_LAUNCH_*.json')))
               for x in items}
    for item in items:
        item['effective_protocol'] = dict(item['protocol'], max_new_primary_rolling24h=400)
    launches = 0
    cursor = 0
    while any(item['pending'] or item['active'] for item in items):
        for item in items:
            for job_id, (process, job) in list(item['active'].items()):
                if process.poll() is None:
                    continue
                try:
                    original.close_worker(item['run'], job)
                except Exception as error:
                    original.save(item['run'] / 'audit' / ('CLEANUP_ERROR_' + job_id + '.json'),
                                  dict(utc=original.now().isoformat(), error_type=type(error).__name__,
                                       error=str(error), terminal_subject=True))
                del item['active'][job_id]
        waiting = None
        while sum(len(x['active']) for x in items) < 8 and any(x['pending'] for x in items):
            candidates = [items[(cursor + offset) % len(items)] for offset in range(len(items))]
            item = next(x for x in candidates if x['pending'])
            cursor = (items.index(item) + 1) % len(items)
            run = item['run']
            serials[run.name] += 1
            process, record = original.launch(run, PROJECT, item['effective_protocol'],
                                                item['pending'][0], serials[run.name], item['env'])
            if process is None:
                waiting = record
                break
            job = item['pending'].pop(0)
            item['active'][job['id']] = (process, job)
            launches += 1
        snapshots = [heartbeat(x, waiting) for x in items]
        original.save(AUDIT / 'STATUS.json', dict(utc=original.now().isoformat(),
                      controller_pid=os.getpid(), queues=snapshots, new_original_tasks_started=launches,
                      max_concurrency=8, max_new_primary_rolling24h=400,
                      scores_read=False, goal_complete=False))
        if any(x['pending'] or x['active'] for x in items):
            time.sleep(2 if waiting is None else 5)
    original.save(AUDIT / 'PRIMARY_PASS_CLOSED.json', dict(utc=original.now().isoformat(),
                  primary_queues_terminal=True, scoring_started=False, brokers_closed=False,
                  next_action='Close owned brokers/backends; run original all333 quiescence gate before judges.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    items = queues()
    prepared = prepare(items)
    if not args.execute:
        original.save(AUDIT / 'PREPARED.json', prepared)
        print(json.dumps(prepared, ensure_ascii=False))
        return
    approval = json.loads((AUDIT / 'APPROVAL.json').read_text())
    prior = json.loads((AUDIT / 'PREPARED.json').read_text())
    for field in ('primary_launcher_sha256', 'capacity_census_sha256', 'dispatcher_sha256'):
        assert prior[field] == prepared[field], 'Prepared code differs: ' + field
    take_over(prepared, items, approval)
    execute(items)


if __name__ == '__main__':
    main()
