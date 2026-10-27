"""Evaluate every frozen maintenance submission after its native run terminates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan', default='plan88.json')
    p.add_argument('--native-directory', default='native_runs')
    p.add_argument('--batch-file')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--tag', default='')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--impact-deps', type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    jobs = json.loads((run/args.plan).read_text())['jobs']
    logs = run/'evaluator_logs'
    logs.mkdir(exist_ok=True)
    started = set()
    futures = []

    def evaluate(job, native):
        name = job['id'] + ('_' + args.tag if args.tag else '')
        output = run/'evaluation'/name
        if not (run/job['workspace']/'policy.py').is_file():
            output.mkdir(parents=True)
            result = dict(name=name, status='missing_policy', native_status=native['status'],
                          planned_episodes=args.episodes, completed_episodes=0)
            (output/'status.json').write_text(json.dumps(result, indent=2)+'\n')
        else:
            if job.get('family') == 'IMP-act':
                if args.impact_deps is None:
                    raise ValueError('IMP-act evaluation requires --impact-deps')
                command = [sys.executable, str(Path(__file__).with_name('evaluate_impact_policy.py')),
                           '--run', str(run), '--job', job['id'], '--episodes', str(args.episodes),
                           '--seed-start', str(args.seed_start), '--label', name, '--deps', str(args.impact_deps)]
            else:
                command = [sys.executable, str(Path(__file__).with_name('evaluate_imprl_policy.py')),
                           '--run', str(run), '--setting', job['setting'], '--policy', 'submitted', '--job', job['id'],
                           '--steps', str(args.steps), '--episodes', str(args.episodes), '--seed-start', str(args.seed_start),
                           '--label', name]
            with (logs/(name+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output/'status.json').read_text())
            (output/'native_status_binding.json').write_text(json.dumps(dict(
                native_status=native['status'], native_completion=native['status']=='finished',
                terminal_draft_only=native['status']!='finished'), indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return result

    if not 1 <= args.workers <= 16:
        p.error('workers must be 1..16')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(started) < len(jobs):
            for job in jobs:
                status_path = run/args.native_directory/job['id']/'status.json'
                if job['id'] in started or not status_path.exists():
                    continue
                try:
                    native = json.loads(status_path.read_text())
                except ValueError:
                    continue
                if native['status'] != 'running':
                    started.add(job['id'])
                    futures.append(pool.submit(evaluate, job, native))
            if len(started) < len(jobs):
                time.sleep(5)
        results = [future.result() for future in futures]
    suffix = '_' + args.tag if args.tag else ''
    name = args.batch_file or ('evaluation_batch88'+suffix+'.json')
    (run/name).write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
