"""Clone terminal public artifacts for an outcome-blind ordinary review wave."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def live_readers(paths):
    excluded = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[1])
        except (OSError, ValueError):
            break
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            refs = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0')]
            try:
                refs.append(os.readlink(proc / 'cwd'))
            except OSError:
                pass
            try:
                for fd in (proc / 'fd').iterdir():
                    try:
                        refs.append(os.readlink(fd))
                    except OSError:
                        pass
            except OSError:
                pass
            if any(s == str(p) or s.startswith(str(p) + '/') for p in paths for s in refs):
                matches.append(int(proc.name))
        except OSError:
            pass
    return sorted(set(matches))


def prepare(run, selected=None):
    assert not (run / 'SCORE185.json').exists()
    plan = json.loads((run / 'PLAN185.json').read_text())['jobs']
    chosen = [j for j in plan if selected is None or j['id'] in selected]
    assert chosen and (selected is None or {j['id'] for j in chosen} == set(selected))
    for job in chosen:
        out = run / ('primary_' + job['family']) / job['id']
        status = json.loads((out / 'status.json').read_text())
        assert status['status'] in ['finished', 'timeout', 'execution_error']
        assert not Path('/proc/' + str(status['cli_pid'])).exists()
    paths = [run / j['workspace'] for j in chosen]
    paths += [run / ('primary_' + j['family']) / j['id'] for j in chosen]
    paths += [run / 'source_history' / j['id'] for j in chosen]
    readers = live_readers(paths)
    assert not readers, readers
    # Review starts from public prior artifacts, never the scorer or hidden results.
    root = run / 'review_workspaces'
    root.mkdir(exist_ok=True)
    reviews, lineage = [], []
    for old in plan:
        job_id = old['id'] + '_review'
        reviews.append(dict(old, id=job_id, original_job_id=old['id'], phase='ordinary_review',
                            workspace=str((root / job_id).relative_to(run)),
                            prompt_file='review_plans/' + job_id + '.txt',
                            output_directory='review_primary_jobs/' + job_id + '/' + job_id,
                            condition='same_public_inputs_blind_ordinary_review'))
    review_plan = {'jobs': reviews}
    p = run / 'REVIEW_PLAN185.json'
    if p.exists():
        assert json.loads(p.read_text()) == review_plan, 'Review cohort must not change'
    else:
        put(p, review_plan)
    for old in chosen:
        job_id = old['id'] + '_review'
        source = run / old['workspace']
        ws = root / job_id
        shutil.copytree(source, ws, symlinks=True)
        prior = ws / 'prior_evidence'
        prior.mkdir()
        out = run / ('primary_' + old['family']) / old['id']
        response = out / 'response.txt'
        if response.is_file():
            shutil.copy2(response, prior / 'response.txt')
        commands = []
        for line_number, line in enumerate((out / 'events.jsonl').read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                commands.append({'event_line': line_number, 'command': item.get('command'),
                                 'exit_code': item.get('exit_code'),
                                 'output': item.get('aggregated_output', '')})
        log = json.dumps(commands, ensure_ascii=False, indent=2) + '\n'
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{24,}', log), 'Credential-like output must not be propagated'
        (prior / 'tool_log.json').write_text(log)
        history = run / 'source_history' / old['id']
        if history.is_dir():
            shutil.copytree(history, prior / 'source_history', symlinks=True)
        readonly = json.loads((ws / 'readonly_inputs.json').read_text())
        readonly.append('prior_evidence')
        put(ws / 'readonly_inputs.json', readonly)
        original = (run / old['prompt_file']).read_text()
        prompt = ('Review the existing submission for the ORIGINAL TASK below. The workspace contains its current '
                  'program/results and prior_evidence contains the previous response, completed code-tool log and '
                  'observed source versions. These contain only previous work on the allowed public inputs; no '
                  'hidden evaluation results are supplied. Independently check whether the actual validation '
                  'supports the final program and its use in this task. Keep a sound submission or correct it '
                  'using the same data and available tools. Record actual checks, comparisons and limitations. '
                  'Do not infer a hidden score or assume the previous submission is wrong. Submit the same '
                  'required deliverables and response format. Total budget3600seconds; no extension or retry.\n\n'
                  'ORIGINAL TASK:\n' + original)
        path = run / 'review_plans' / (job_id + '.txt')
        path.parent.mkdir(exist_ok=True)
        path.write_text(prompt)
        new = next(j for j in reviews if j['id'] == job_id)
        item = dict(new, workspace='../' + new['workspace'], prompt_file=path.name)
        put(run / 'review_plans' / (job_id + '.json'), {'jobs': [item]})
        lineage.append({'original_job': old['id'], 'review_job': job_id,
                        'prior_response_exists': response.is_file(),
                        'prior_response_sha256': sha(response) if response.is_file() else None,
                        'prior_commands': len(commands), 'hidden_feedback': False,
                        'prior_status': json.loads((out / 'status.json').read_text())['status']})
    prepared_ids = {row['review_job'] for row in lineage}
    files = [p for ident in prepared_ids for p in (root / ident).rglob('*') if p.is_file() and not p.is_symlink()]
    frozen = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'new_hidden_scores_read': False,
              'selected_initial_writers_and_readers_gone': True, 'lineage': lineage,
              'files': [{'path': str(p.relative_to(run)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)],
              'scope': 'Ordinary operational review with public prior work, not a pure one-sentence causal ablation.'}
    put(run / 'audit' / ('REVIEW_FREEZE185_' + '_'.join(sorted(prepared_ids)) + '.json'), frozen)
    print(json.dumps({'review_jobs_prepared_this_call': len(chosen), 'review_cohort_fixed': len(reviews), 'no_hidden_feedback': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--jobs', nargs='+')
    args = parser.parse_args()
    prepare(args.run.resolve(), args.jobs)
