"""Finish a complete regular-file copy interrupted only by closed IPC sockets.

No files from the original manifest are overwritten or omitted. This is a
recovery for the canonical archiver, not a way to waive a failed hash check.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import stat

from archive_run import digest


def main(run, destination, workspace):
    assert run.parent == Path('/srv/yyf-zyq/PHM_20261010') and run.is_dir()
    assert destination.parent == Path('/data/yyf-zyq/PHM_20261010') and destination.is_dir()
    assert run.name == destination.name
    assert not (destination/'ARCHIVE_VERIFIED.json').exists()
    manifest_path = run/'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['run_id'] == run.name and manifest['symlink_count'] == 0
    assert digest(manifest_path) == digest(destination/manifest_path.name)
    assert json.loads((run/'audit/PREARCHIVE305.json').read_text())['active_experiment_processes'] == []
    for cohort in ['a','b','c','d','a_review','b_review','c_review','d_review']:
        state=json.loads((run/cohort/'CONTROLLER_STATUS251.json').read_text())
        assert state['status']=='all_selected_terminal_no_scores'
        assert not Path('/proc',str(state['controller_pid'])).exists()
        for job in json.loads((run/cohort/'PLAN187.json').read_text())['jobs']:
            status=json.loads((run/cohort/'primary'/job['id']/job['id']/'status.json').read_text())
            assert status['status']!='running' and not Path('/proc',str(status['cli_pid'])).exists()
    expected={item['path'] for item in manifest['files']} | {manifest_path.name}
    assert len(expected)==len(manifest['files'])+1
    sockets=[]
    for root in [run,destination]:
        actual=set()
        for path in root.rglob('*'):
            rel=path.relative_to(root)
            if root==run and rel.parts[0]=='clients-cache':continue
            assert not path.is_symlink(), str(path)
            mode=path.stat().st_mode
            if stat.S_ISREG(mode):actual.add(str(rel))
            elif stat.S_ISSOCK(mode):
                assert root==run and len(rel.parts)==3 and rel.parts[1]=='ipc' and rel.suffix=='.sock'
                sockets.append(str(rel))
            else:assert stat.S_ISDIR(mode), str(path)
        assert actual==expected, dict(root=str(root),missing=sorted(expected-actual),extra=sorted(actual-expected))
    def verify(item):
        rel=Path(item['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        for root in [run,destination]:
            path=root/rel
            assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'], str(path)
        return item['bytes']
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes=list(pool.map(verify,manifest['files']))
    assert sum(sizes)==manifest['bytes']
    assert digest(manifest_path)==digest(destination/manifest_path.name)
    recovery=dict(utc=datetime.now(timezone.utc).isoformat(),original_failed_session=95804,
        failure='copytree could not copy8terminated IPC sockets;all regular files were copied',
        original_source_and_destination_regular_files_verified=len(sizes),
        excluded_closed_runtime_sockets=sockets,no_experiment_rerun=True,
        manifest_unchanged=True,no_manifest_entry_dropped=True)
    (destination/'ARCHIVE_COPY_RECOVERY.json').write_text(json.dumps(recovery,indent=2)+'\n')
    driver=destination/'ARCHIVE_RECOVERY_DRIVER.py'
    shutil.copy2(Path(__file__),driver)
    policy=json.loads((run/'retention_policy.json').read_text())
    receipt=dict(status='verified',run_id=run.name,archive_path=str(destination),
        files_verified=len(sizes),retained_bytes=sum(sizes),manifest_sha256=digest(manifest_path),
        verified_utc=recovery['utc'],manifest_filename=manifest_path.name,symlinks_verified=0,
        shared_dependencies=policy['shared_dependencies'],source_cleanup='pending',
        copy_recovery='verified complete partial copy;closed runtime sockets have no dataset content',
        recovery_evidence_sha256=digest(destination/'ARCHIVE_COPY_RECOVERY.json'),
        recovery_driver_sha256=digest(driver))
    path=destination/'ARCHIVE_VERIFIED.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.rmtree(run)
    receipt['source_cleanup']='complete'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    with (args.workspace/'logs/background_research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main(args.run.resolve(),args.destination.resolve(),args.workspace.resolve())
