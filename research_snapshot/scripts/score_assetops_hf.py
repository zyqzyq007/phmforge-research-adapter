"""Use the unchanged HF152 author scorer; keep judge failure separate from task failure.

No credential is loaded here. Real judge calls go only through the stable CLI
runner, after both original primary campaigns are terminal and quiescent.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def alive(pid, started_utc=None):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return False
    fields = path.read_text().split(') ', 1)[1].split()
    if fields[0] == 'Z':
        return False
    if started_utc is not None:
        boot = next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines()
                    if x.startswith('btime '))
        actual = boot + int(fields[19]) / os.sysconf('SC_CLK_TCK')
        expected = datetime.fromisoformat(started_utc.replace('Z', '+00:00')).timestamp()
        if abs(actual - expected) > 15:
            return False  # The operating system has reused this old PID.
    return True


def quiescent(run):
    jobs = json.loads((run / 'PLAN187_LAUNCH.json').read_text())['jobs']
    expected = json.loads((run / 'PROTOCOL187.json').read_text())['published_questions']
    if len(jobs) != expected or len({j['id'] for j in jobs}) != expected:
        raise RuntimeError('Declared complete primary inventory is inconsistent')
    controller_path = run / 'CONTROLLER_STATUS187.json'
    if not controller_path.exists():
        raise RuntimeError('Full primary controller has not finished or started: ' + str(run))
    controller = json.loads(controller_path.read_text())
    if controller.get('status') != 'all_selected_subjects_terminal_no_scoring':
        raise RuntimeError('Full primary controller is not terminal: ' + str(run))
    if alive(controller['controller_pid'], controller.get('started_utc')):
        raise RuntimeError('Primary controller still live: ' + str(run))
    # A queued, abandoned request can restart a worker after an earlier close
    # receipt. Stop trusted brokers after the complete pass, then audit actual
    # processes/containers; controller completion alone is not quiescence.
    for start in run.glob('BROKER_START*.json'):
        broker = json.loads(start.read_text())
        if alive(broker['pid'], broker.get('started_utc')):
            raise RuntimeError('Original tool broker still live; stop it after the full pass: ' + str(start))
    proof = []
    for job in jobs:
        folder = run / 'primary' / job['id'] / job['id']
        status = json.loads((folder / 'status.json').read_text())
        if status['status'] == 'running' or alive(status['cli_pid'], status['started_utc']):
            raise RuntimeError('A primary CLI is still live: ' + job['id'])
        worker = run / 'traces' / job['id'] / 'worker_status.json'
        if worker.exists():
            record = json.loads(worker.read_text())
            closed = worker.with_name('worker_closed.json')
            if alive(record['pid'], record['started_utc']):
                raise RuntimeError('Original tool worker is not closed: ' + job['id'])
            if closed.exists() and json.loads(closed.read_text())['pid'] != record['pid']:
                raise RuntimeError('Worker closure receipt identity differs: ' + job['id'])
            command = record.get('command', [])
            if command and command[0] == 'docker' and '--name' in command:
                name = command[command.index('--name') + 1]
                result = subprocess.run(['docker', 'container', 'inspect', '--format',
                                         '{{.State.Running}}', name], capture_output=True,
                                        text=True, timeout=10)
                absent = result.returncode and 'no such' in result.stderr.lower()
                if not absent and (result.returncode or result.stdout.strip() != 'false'):
                    raise RuntimeError('Tool container live or observation uncertain: ' + name)
            proof.append(dict(job=job['id'], worker_pid=record['pid'],
                              native_closure_receipt=closed.exists(),
                              live_original_process=False,
                              note='Missing legacy receipt is retained; current PID/container audit proves quiescence, not task validity.'))
    save(run / 'audit/SCORING_QUIESCENCE_AUDIT.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), workers=proof,
        subjects=len(jobs), subject_task_validity='requires separate audit'))
    return jobs


def native(mode):
    """Executed without network in the pinned Python runtime, not a new scorer."""
    from evaluation.models import Scenario, PersistedTrajectory
    from evaluation.evaluator import _trajectory_to_text
    from evaluation.scorers.llm_judge import LLMJudgeScorer

    class Captured(BaseException):
        pass

    class Backend:
        def __init__(self, row):
            self.row = row
            self.prompt = None

        def generate(self, prompt):
            self.prompt = prompt
            if mode == 'capture':
                # Bypass the author's Exception handler without computing a
                # fabricated score for an actual subject.
                raise Captured()
            if self.row.get('backend_error'):
                raise RuntimeError(self.row['backend_error'])
            return self.row['judge_raw']

    output = []
    for row in json.load(sys.stdin):
        scenario = Scenario.from_raw(row['scenario'])
        trajectory = PersistedTrajectory.from_raw(row['trajectory'])
        backend = Backend(row)
        try:
            score = LLMJudgeScorer(backend)(
                scenario, trajectory.answer, _trajectory_to_text(trajectory))
        except Captured:
            output.append(dict(job=row['job'], prompt=backend.prompt,
                               native_score_computed=False))
        else:
            output.append(dict(job=row['job'], prompt=backend.prompt,
                               score=score.model_dump(mode='json')))
    print(json.dumps(output, ensure_ascii=False))


def original_source(run, source):
    destination = run / 'scoring_author'
    if not destination.exists():
        destination.mkdir()
        for name in ['evaluation', 'llm']:
            shutil.copytree(source / name, destination / name,
                            ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc'))
        save(run / 'SCORING_AUTHOR_SOURCE.json', dict(
            source=str(source), source_commit='c6b96755b8a182aa090ee028852cbc83279b2982',
            files=[dict(path=str(p.relative_to(destination)), sha256=sha(p))
                   for p in sorted(destination.rglob('*')) if p.is_file()]))
    for item in json.loads((run / 'SCORING_AUTHOR_SOURCE.json').read_text())['files']:
        if sha(destination / item['path']) != item['sha256']:
            raise RuntimeError('Frozen original scorer changed: ' + item['path'])
    return destination


def call_native(run, source, mode, rows, label):
    project = Path(__file__).resolve().parents[1]
    from run_full_benchmark_campaign import clean_env
    output = run / 'scoring_runtime'
    output.mkdir(exist_ok=True)
    if any((output / (label + suffix)).exists() for suffix in ['_COMMAND.json', '.json', '.stderr']):
        raise FileExistsError('Retain the earlier native scorer operation: ' + label)
    for name in ['tmp', 'mpl']:
        (output / name).mkdir(exist_ok=True)
    resources = json.loads((run / 'runtime/RESOURCES190.json').read_text())
    command = ['docker', 'run', '--rm', '-i', '--network', 'none',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', str(run / 'runtime/asset_deps') + ':/deps:ro',
               '-v', str(run / 'runtime/python312_standalone/python') + ':/python312:ro',
               '-v', str(source) + ':/author:ro',
               '-v', str(Path(__file__).resolve()) + ':/adapter.py:ro',
               '-v', str(output) + ':/runtime',
               '-e', 'PYTHONPATH=/deps:/author', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'TMPDIR=/runtime/tmp', '-e', 'TMP=/runtime/tmp',
               '-e', 'TEMP=/runtime/tmp', '-e', 'MPLCONFIGDIR=/runtime/mpl',
               resources['base_OS_image'], '/python312/bin/python3.12',
               '-B', '/adapter.py', 'native-' + mode]
    save(output / (label + '_COMMAND.json'), command)
    result = subprocess.run(command, input=json.dumps(rows), capture_output=True,
                            text=True, timeout=120, env=clean_env(run), cwd=project)
    (output / (label + '.stderr')).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Original scorer adapter failed: ' + result.stderr[-2000:])
    value = json.loads(result.stdout)
    save(output / (label + '.json'), value)
    return value


def scenarios(raw_path, run):
    raw = [json.loads(x) for x in raw_path.read_text().splitlines() if x.strip()]
    rows = {str(x['id']): x for x in raw}
    jobs = json.loads((run / 'PLAN187_LAUNCH.json').read_text())['jobs']
    if len(raw) != 152 or len(rows) != 152 or set(rows) != {j['task_id'] for j in jobs}:
        raise ValueError('Original complete HF152 inventory mismatch')
    for job in jobs:
        question = (run / job['workspace'] / 'inputs/question.txt').read_text()
        if question != rows[job['task_id']]['text'] + '\n':
            raise ValueError('Original question changed: ' + job['id'])
    return rows, jobs


def observable(run, job):
    calls = run / 'traces' / job / 'calls.jsonl'
    tools = [json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
    commands = []
    events = run / 'primary' / job / job / 'events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                commands.append({key: item.get(key) for key in
                                 ['id', 'command', 'aggregated_output', 'exit_code']})
    # Native evaluator serializes this with its unchanged indent and truncation.
    # No agent_message, reasoning item, or hidden chain of thought is copied.
    return dict(original_tool_calls=tools, observable_command_executions=commands)


def snapshot_companion(run, companion):
    """Retain only terminal metadata before the earlier finished run is archived."""
    jobs = quiescent(companion)
    destination = run / 'scoring_companion' / companion.name
    destination.mkdir(parents=True, exist_ok=False)
    paths = [companion / name for name in
             ['PLAN187_LAUNCH.json', 'PROTOCOL187.json', 'CONTROLLER_STATUS187.json']]
    paths += list(companion.glob('BROKER_START*.json'))
    for job in jobs:
        paths.append(companion / 'primary' / job['id'] / job['id'] / 'status.json')
        paths += list((companion / 'traces' / job['id']).glob('worker_*.json'))
    records = []
    for source in paths:
        relative = source.relative_to(companion)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(dict(path=str(relative), sha256=sha(source)))
    (destination / 'audit').mkdir(exist_ok=True)
    quiescent(destination)
    save(destination / 'SOURCE_METADATA_PROVENANCE.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), original_run=str(companion),
        original_subjects=len(jobs), files=records,
        purpose='Closed primary metadata only; preserves grading prerequisite after original archive migration'))
    return destination


def prepare(run, source, raw_path):
    jobs = quiescent(run)
    rows, _ = scenarios(raw_path, run)
    destination = run / 'judge_workspaces'
    destination.mkdir(exist_ok=False)
    payload, frozen = [], []
    for job in jobs:
        folder = run / 'primary' / job['id'] / job['id']
        answer = folder / 'response.txt'
        payload.append(dict(job=job['id'], scenario=rows[job['task_id']], trajectory=dict(
            run_id=run.name, scenario_id=job['task_id'], runner='codex0.163_original_tools',
            model='deepseek-flash', question=rows[job['task_id']]['text'],
            answer=answer.read_text() if answer.exists() else '', trajectory=observable(run, job['id']))))
        for path in [folder / 'status.json', answer, folder / 'events.jsonl',
                     run / 'traces' / job['id'] / 'calls.jsonl']:
            if path.exists():
                frozen.append(dict(path=str(path), sha256=sha(path)))
    captured = {x['job']: x for x in call_native(run, source, 'capture', payload, 'capture_original_prompts')}
    judges, unavailable = [], []
    for row in payload:
        prompt = captured[row['job']].get('prompt')
        if prompt is None:
            unavailable.append(dict(job=row['job'], native=captured[row['job']]))
            continue
        judge_id = row['job'] + '_judge'
        app = destination / judge_id
        (app / 'inputs').mkdir(parents=True)
        (app / 'inputs/prompt.txt').write_text(prompt)
        judges.append(dict(id=judge_id, source_job=row['job'], condition='original_HF152_flash_judge',
                           repeat=1, answer_mode='text', numerical_threads=1,
                           workspace=str(app), prompt_file=str(app / 'inputs/prompt.txt')))
    save(run / 'SCORER_INPUTS190.json', payload)
    save(run / 'PLAN_JUDGES190.json', dict(jobs=judges, unscorable=unavailable,
                                          complete_subject_denominator=152))
    for path in [run / 'SCORER_INPUTS190.json', run / 'PLAN_JUDGES190.json',
                 run / 'SCORING_AUTHOR_SOURCE.json', raw_path]:
        frozen.append(dict(path=str(path), sha256=sha(path)))
    frozen.extend(dict(path=j['prompt_file'], sha256=sha(Path(j['prompt_file']))) for j in judges)
    save(run / 'PRE_JUDGE_ARTIFACT_FREEZE190.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), files=frozen,
        raw_HF_sha256=sha(raw_path), source_manifest_sha256=sha(run / 'SCORING_AUTHOR_SOURCE.json'),
        observable_only=True, native_trajectory_character_limit=8000))


def launch(run, companion):
    from run_full_benchmark_campaign import clean_env
    project = Path(__file__).resolve().parents[1]
    quiescent(run)
    quiescent(companion)
    if (run / 'judges').exists():
        raise FileExistsError('No automatic re-judging or overwriting')
    lock = (project / 'logs/background_research.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Both authorized primary campaigns are terminal. The new batch is declared
    # while holding the shared admission lock; metadata updates remain available
    # while the tools-disabled eight-worker batch runs.
    protocol = json.loads((run / 'PROTOCOL190.json').read_text())
    if protocol['model'] != 'deepseek-flash' or sha(Path(protocol['cli'])) != protocol['cli_sha256']:
        raise RuntimeError('Frozen judge model/CLI identity changed')
    command = [sys.executable, '-B', str(project / 'scripts/replay_diagnostic_cli.py'),
               '--plan', str(run / 'PLAN_JUDGES190.json'), '--cli', protocol['cli'],
               '--model', 'deepseek-flash', '--output', str(run / 'judges'),
               '--workers', '8', '--timeout', '120',
               '--permission-profile', 'iot-offline',
               '--isolation-config', str(run / 'cli_config187.toml'),
               '--model-catalog', str(run / 'model_catalog187.json'),
               '--workspace-alias', '/app', '--cleanup-runtime-cache',
               '--credential-env-file', '/home/jupyter-yyf-zyq/.bashrc',
               '--credential-env-name', 'DEEPSEEK_API_KEY']
    save(run / 'JUDGE_COMMAND190.json', command)
    try:
        with (run / 'logs/judges.stdout').open('wb') as out, (run / 'logs/judges.stderr').open('wb') as err:
            process = subprocess.Popen(command, env=clean_env(run), cwd=project,
                                       stdout=out, stderr=err, start_new_session=True)
            save(run / 'JUDGE_BATCH_START190.json', dict(
                pid=process.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                model='deepseek-flash', workers=8, both_primary_campaigns_terminal=True))
            lock.close()
            code = process.wait(timeout=3000)
        save(run / 'JUDGE_EXIT190.json', dict(exit_code=code,
                                             ended_utc=datetime.now(timezone.utc).isoformat()))
    finally:
        if not lock.closed:
            lock.close()


def collect(run, source):
    quiescent(run)
    if (run / 'RESULT190.json').exists():
        raise FileExistsError('Preserve existing native scores')
    batch_start = run / 'JUDGE_BATCH_START190.json'
    batch_exit = run / 'JUDGE_EXIT190.json'
    if not batch_start.exists() or not batch_exit.exists():
        raise RuntimeError('Judge batch termination is not yet proven; do not grade queued cases as missing')
    batch = json.loads(batch_start.read_text())
    if alive(batch['pid'], batch['started_utc']):
        raise RuntimeError('Judge batch supervisor is still live')
    plan = json.loads((run / 'PLAN_JUDGES190.json').read_text())
    rows = json.loads((run / 'SCORER_INPUTS190.json').read_text())
    expected = {j['id'] for j in json.loads((run / 'PLAN187_LAUNCH.json').read_text())['jobs']}
    if len(rows) != 152 or len({x['job'] for x in rows}) != 152 or {x['job'] for x in rows} != expected:
        raise RuntimeError('Scorer input lost or duplicated an original subject')
    declared = [x['source_job'] for x in plan['jobs']] + [x['job'] for x in plan['unscorable']]
    if len(declared) != 152 or set(declared) != expected:
        raise RuntimeError('Judge plan does not retain the complete denominator')
    for item in json.loads((run / 'PRE_JUDGE_ARTIFACT_FREEZE190.json').read_text())['files']:
        if sha(Path(item['path'])) != item['sha256']:
            raise RuntimeError('Primary evidence changed after score preparation')
    validity = {}
    prompt_hashes = {}
    for row in rows:
        folder = run / 'judges' / (row['job'] + '_judge')
        status_path = folder / 'status.json'
        if not status_path.exists():
            row['backend_error'] = 'Judge status missing; not an Agent task failure'
            validity[row['job']] = 'judge_not_run_or_missing_status'
            continue
        status = json.loads(status_path.read_text())
        if status['status'] == 'running' or alive(status['cli_pid'], status['started_utc']):
            raise RuntimeError('Judge still live')
        if status['status'] != 'finished' or not status.get('response_valid'):
            row['backend_error'] = 'Judge session ' + status['status']
            validity[row['job']] = 'judge_runtime_error'
        else:
            row['judge_raw'] = (folder / 'response.txt').read_text()
            validity[row['job']] = 'completed_same_model_judge_not_independent'
            prompt_hashes[row['job']] = status['prompt_sha256']
    scores = call_native(run, source, 'evaluate', rows, 'unchanged_author_scores')
    results = []
    for row in scores:
        if row['job'] in prompt_hashes:
            actual = hashlib.sha256(row['prompt'].encode()).hexdigest()
            if actual != prompt_hashes[row['job']]:
                raise RuntimeError('Judge input differs from unchanged author prompt: ' + row['job'])
        score = row['score']
        criteria = ['task_completion', 'data_retrieval_accuracy', 'generalized_result_verification',
                    'agent_sequence_correct', 'clarity_and_justification', 'hallucinations']
        available = (validity[row['job']].startswith('completed_') and
                     all(isinstance(score.get('details', {}).get(key), bool) for key in criteria))
        results.append(dict(job=row['job'], native_score=score, judge_state=validity[row['job']],
                            scoring_available=available, physical_PHM_correctness='not established by LLM rubric'))
    output = dict(subjects=152, rows=results, native_passed=sum(x['native_score']['passed'] for x in results),
                  scoring_available=sum(x['scoring_available'] for x in results),
                  judge_states=dict(Counter(validity.values())),
                  caveats=['Unchanged original scorer with same deepseek-flash judge: not independent.',
                           'Native8000-character observable-trajectory truncation retained.',
                           'Source/data/tool failures and physical correctness require separate audit.',
                           'A failed/unparseable judge is not an Agent task failure.'],
                  utc=datetime.now(timezone.utc).isoformat())
    save(run / 'RESULT190.json', output)


def main():
    if len(sys.argv) == 2 and sys.argv[1].startswith('native-'):
        native(sys.argv[1].removeprefix('native-'))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'launch', 'collect', 'record-companion'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--author-src', type=Path)
    parser.add_argument('--raw-scenarios', type=Path)
    parser.add_argument('--companion-run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.mode in ['launch', 'record-companion']:
        if args.companion_run is None:
            parser.error('--companion-run is required before any judge LLM calls')
        if args.mode == 'launch':
            launch(run, args.companion_run.resolve())
        else:
            snapshot_companion(run, args.companion_run.resolve())
    else:
        if args.author_src is None:
            parser.error('--author-src is required')
        source = original_source(run, args.author_src.resolve())
        if args.mode == 'prepare':
            if args.raw_scenarios is None:
                parser.error('--raw-scenarios is required')
            prepare(run, source, args.raw_scenarios.resolve())
        else:
            collect(run, source)


if __name__ == '__main__':
    main()
