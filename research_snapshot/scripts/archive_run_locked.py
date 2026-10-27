"""Run the canonical archiver under the primary admission lock.

An admission census scans both archive and live roots. Keep archive publication
and source removal mutually exclusive with that entire census/admission step.
"""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys


def run_archive_command(command, workspace):
    lock = Path(workspace).resolve() / 'logs/background_research.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        # The child retains the same open-file-description lock if its
        # supervisor exits; closing this parent FD cannot unlock a live child.
        return subprocess.call(command, pass_fds=(handle.fileno(),))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--workspace', type=Path, required=True)
    args, canonical_args = parser.parse_known_args()
    command = [sys.executable, '-B', str(Path(__file__).with_name('archive_run.py')),
               '--workspace', str(args.workspace.resolve()), *canonical_args]
    return run_archive_command(command, args.workspace)


if __name__ == '__main__':
    raise SystemExit(main())
