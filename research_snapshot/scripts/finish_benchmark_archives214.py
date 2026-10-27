"""Finish the two closed benchmark archives, register verified receipts once."""
import fcntl, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LIVE = Path('/srv/yyf-zyq/PHM_20261010')
ARCHIVE = Path('/data/yyf-zyq/PHM_20261010')
RESULTS = PROJECT / 'results/fullbench187'
RUNS = [('20261018T185541Z_assethf190', 152, 152, 69),
        ('20261019T032531Z_assetextra202', 103, 102, 84)]
from run_full_benchmark_campaign import clean_env

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def register(run_id, subjects, score_rows, passed):
    source, destination = LIVE / run_id, ARCHIVE / run_id
    receipt_path = destination / 'ARCHIVE_VERIFIED.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'verified' and receipt['source_cleanup'] == 'complete'
    assert receipt['run_id'] == run_id and receipt['archive_path'] == str(destination)
    assert not source.exists()
    manifest = destination / receipt['manifest_filename']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == receipt['manifest_sha256']
    assert receipt['logical_files_verified'] == receipt['files_verified'] + 1
    assert (destination / receipt['bundle_filename']).stat().st_size == receipt['bundle_bytes']
    row = {**receipt, 'classification': 'Closed original benchmark cohort; native grades preserved, capability inference quarantined for truncated observable trajectories.',
        'primary_episode_rows': subjects, 'native_LLM_score_rows_available': score_rows,
        'native_numeric_passed': passed, 'requested_model': 'deepseek-flash',
        'independent_PHM_accuracy_established': False, 'scoped_problem_confirmed': False,
        'full_user_goal_complete': False, 'archive_admission_shared_lock_used': True}
    with (PROJECT / 'logs/background_research.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = PROJECT / 'results/archive_index.json'
        index = json.loads(index_path.read_text())
        existing = [r for r in index['archives'] if r['run_id'] == run_id]
        if existing:
            assert len(existing) == 1 and existing[0]['manifest_sha256'] == row['manifest_sha256']
        else:
            index['archives'].append(row)
            index['updated_utc'] = datetime.now(timezone.utc).isoformat()
            temp = index_path.with_name('archive_index214.next.json')
            assert not temp.exists()
            save(temp, index)
            os.replace(temp, index_path)
    save(RESULTS / ('A190_archive_receipt.json' if subjects == 152 else 'A202_archive_receipt.json'), row)
    return row

def main():
    start = RESULTS / 'archive_finish214_START.json'
    assert not start.exists()
    save(start, {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Only finalize closed A190/A202 archives; zero model calls, zero primary/scoring retries.'})
    first = LIVE / RUNS[0][0]
    deadline = time.monotonic() + 5400
    while first.exists() or not (ARCHIVE / first.name / 'ARCHIVE_VERIFIED.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('First canonical archive not closed; no retry or overwrite')
        error = RESULTS / 'A190_archive_stderr.txt'
        if error.exists() and error.stat().st_size:
            raise RuntimeError('First canonical archive reported an error; source retained for review')
        time.sleep(5)
    receipts = [register(*RUNS[0])]
    print(json.dumps({'run': RUNS[0][0], 'archive_registered': True}), flush=True)
    run = LIVE / RUNS[1][0]
    gate = [sys.executable, '-B', str(run / 'source/score202.py'), 'gate']
    with (RESULTS / 'post_A190_archive_gate214.stdout').open('wb') as out, (RESULTS / 'post_A190_archive_gate214.stderr').open('wb') as err:
        code = subprocess.call(gate, cwd=PROJECT, env=clean_env(run), stdout=out, stderr=err)
    assert code == 0, 'Companion closure metadata must remain valid after archive migration'
    command = [sys.executable, '-B', str(PROJECT / 'scripts/archive_run_locked.py'),
        '--workspace', str(PROJECT), '--run', str(run), '--destination', str(ARCHIVE / run.name), '--portable-bundle']
    save(RESULTS / 'A202_archive_COMMAND.json', command)
    with (RESULTS / 'A202_archive_stdout.json').open('wb') as out, (RESULTS / 'A202_archive_stderr.txt').open('wb') as err:
        code = subprocess.call(command, cwd=PROJECT, env=clean_env(run), stdout=out, stderr=err)
    assert code == 0, 'Canonical archive failed; retain source and inspect logs'
    receipts.append(register(*RUNS[1]))
    now = datetime.now(timezone.utc).isoformat()
    save(RESULTS / 'archive_finish214_DONE.json', {'utc': now, 'cohorts': receipts,
        'files_verified': sum(r['files_verified'] for r in receipts),
        'bytes_verified': sum(r['retained_bytes'] for r in receipts),
        'sources_cleared_only_after_verification': True, 'model_calls': 0,
        'full_user_goal_complete': False, 'new_confirmed_research_problems': 0})
    path = RESULTS / 'current_execution_checkpoint.json'
    checkpoint = json.loads(path.read_text())
    checkpoint.update(utc=now, phase='selected333_closed_scored_archived; research_goal_unachieved',
        archived_cohorts=checkpoint.get('archived_cohorts', []) + receipts,
        final_archives_verified=True, actual_running_primary_tasks=0, actual_running_judges=0,
        next_action='No queued work remains. Research problem not confirmed; source gaps and discriminating control plan retained. Do not restart old primaries or rescore.')
    save(path, checkpoint)
    for name in ['briefing20261020.md', 'full_benchmark_progress187_190.md']:
        path = PROJECT / 'docs' / name
        text = path.read_text().replace('本轮两批归档正在按逐文件SHA验证流程完成；最终收据见', '本轮两批已逐文件SHA验证归档并清理原副本；最终收据见')
        if name.startswith('full_benchmark_progress'):
            text += '\nA190/A202均已逐文件校验归档并清源，收据见results/fullbench187/archive_finish214_DONE.json。\n'
        path.write_text(text)
    print(json.dumps({'both_archives_verified_and_registered': True, 'model_calls': 0}), flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(RESULTS / 'archive_finish214_FAILED.json', {'utc': datetime.now(timezone.utc).isoformat(),
            'exception_type': type(exc).__name__, 'message': str(exc), 'automatic_retry': False})
        raise
