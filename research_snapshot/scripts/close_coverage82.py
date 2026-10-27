"""Append an independently hashed closing checkpoint after immutable A82 archive."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parent.parent
    archive = Path('/data/yyf-zyq/PHM_20261010/20261014T163123Z_coverage82')
    source = Path('/srv/yyf-zyq/PHM_20261010') / archive.name
    results = project / 'results/coverage82'
    receipt = json.loads((archive / 'ARCHIVE_VERIFIED.json').read_text())
    manifest = json.loads((archive / 'SOURCE_MANIFEST.json').read_text())
    assert not source.exists()
    assert sha(archive / 'SOURCE_MANIFEST.json') == receipt['manifest_sha256']
    assert receipt['source_cleanup'] == 'complete' and receipt['status'] == 'verified'
    assert len(manifest['files']) == receipt['files_verified'] == 1857
    assert all((archive / x['path']).stat().st_size == x['bytes'] and sha(archive / x['path']) == x['sha256'] for x in manifest['files'])
    save(results / 'archive_receipt82.json', receipt)
    state_path = project / 'docs/discovery_state.json'
    state = json.loads(state_path.read_text())
    now = datetime.now(timezone.utc).isoformat()
    state.update({
        'updated_utc': now, 'last_update_utc': now,
        'goal_status': 'active', 'goal_complete': False,
        'archive_status': 'verified', 'archive_manifest_sha256': receipt['manifest_sha256'],
        'source_run_cleanup': 'complete', 'source_run_cleanup_scope': archive.name,
        'active_run_id': None, 'active_run_handles': [], 'active_run_workdir': None,
        'active_workdir': None, 'active_archive_destination': None,
        'active_run_phase': 'A82_closed; next real-data qualification pending',
        'phase': 'A82_completed_archived_goal_active', 'active_phase': 'real_data_failure_qualification_next',
        'last_completed_iteration': 82,
        'last_verified_archive': receipt, 'latest_archive': receipt,
        'A82_archive': receipt,
        'closing_checkpoint82': str(archive / 'checkpoint_iteration82'),
    })
    if 'A82_1857file_archive_verified_source_cleaned_goal_active' not in state['completed_iterations']:
        state['completed_iterations'].append('A82_1857file_archive_verified_source_cleaned_goal_active')
    save(state_path, state)
    iteration = project / 'docs/iteration82.md'
    with iteration.open('a') as out:
        out.write('\n## 归档完成\n\n')
        out.write('归档至 `/data/yyf-zyq/PHM_20261010/20261014T163123Z_coverage82`，1857个制品、35,572,102字节，逐文件核验通过，srv运行副本已清理。SOURCE_MANIFEST SHA256为 `' + receipt['manifest_sha256'] + '`。最终收口另存独立checkpoint，不改原manifest或实验输入。总体目标保持active；下一轮实际模型实验尚未启动。\n')
    spec = importlib.util.spec_from_file_location('reader82', project / 'scripts/replay_original_smartgrid.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    secret = mod.literal_credential(Path('/home/jupyter-yyf-zyq/.bashrc')).encode()
    key_matches = []
    paths = [p for p in archive.rglob('*') if p.is_file()]
    for name in ['AGENTS.md', 'AGNETS.md', 'docs', 'scripts', 'cases', 'logs', 'results']:
        p = project / name
        paths.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file() and not q.is_symlink()])
    for p in paths:
        if secret in p.read_bytes():
            key_matches.append(str(p))
    del secret
    assert not key_matches
    closing = {
        'run_id': archive.name, 'closed_utc': now,
        'immutable_archive_files_reverified': len(manifest['files']),
        'immutable_source_manifest_sha256': receipt['manifest_sha256'],
        'immutable_archive_not_modified': True, 'source_cleanup_verified': not source.exists(),
        'retained_bytes': receipt['retained_bytes'], 'secret_scan_files': len(paths),
        'secret_matches': key_matches, 'key_value_or_hash_recorded': False,
        'frozen_inputs_verified': 63, 'main_sessions_terminal': 16,
        'source_supported_main_outputs': 16, 'confirmed_primary_task_failures': 0,
        'readonly_matches': 135, 'WO_state_checks': 17, 'cold_code_replay_jobs': 3,
        'commonness_established': False, 'challenge_established': False,
        'physical_damage_measured': False, 'E3': 0, 'goal_status': 'active',
        'next_model_sessions_started': False,
        'status': 'round_complete; overall_discovery_goal_active',
    }
    save(results / 'closing_audit82.json', closing)
    checkpoint = archive / 'checkpoint_iteration82'
    checkpoint.mkdir(exist_ok=False)
    for name in ['AGENTS.md', 'AGNETS.md', 'docs/discovery_state.json', 'docs/iteration82.md',
                 'docs/fast_problem_discovery_goal.md', 'docs/next_iteration_protocol.md',
                 'docs/research_goal.md', 'docs/problem_discovery.md', 'docs/hidden_important_problems.md',
                 'docs/exclusion_evidence_validation_goal.md', 'scripts/close_coverage82.py']:
        target = checkpoint / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    shutil.copytree(results, checkpoint / 'results/coverage82')
    files = [{'path': str(p.relative_to(checkpoint)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(checkpoint.rglob('*')) if p.is_file()]
    checkpoint_manifest = {'run_id': archive.name, 'scope': 'closing appendix; separate from immutable original SOURCE_MANIFEST',
                           'files': files, 'file_count': len(files), 'bytes': sum(x['bytes'] for x in files)}
    save(checkpoint / 'CLOSING_MANIFEST.json', checkpoint_manifest)
    assert all(sha(checkpoint / x['path']) == x['sha256'] for x in files)
    final = {'status': 'verified', 'checkpoint_path': str(checkpoint),
             'files_verified': len(files), 'manifest_sha256': sha(checkpoint / 'CLOSING_MANIFEST.json'),
             'original_manifest_unchanged': sha(archive / 'SOURCE_MANIFEST.json') == receipt['manifest_sha256'],
             'goal_status': 'active'}
    save(results / 'checkpoint_receipt82.json', final)
    save(checkpoint / 'CHECKPOINT_VERIFIED.json', final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == '__main__':
    main()
