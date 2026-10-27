"""Bootstrap the saved CPU environment without resolving new pip dependencies.

python3 scripts/frozen_runtime.py --store /path/to/store --cache /path/to/cache -- environment list --store /path/to/store
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--store', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo/'src'))
from phmforge_bench.environment import Environment

env = Environment(args.store)
cache = args.cache.resolve()
python_root = env.materialize('runtime/cpython-3.12.11-linux-x86_64-gnu', cache/'python')
deps = env.materialize('runtime/verified_benchmarks246_v4/deps', cache/'deps')
for directory in ['tmp', 'mpl', 'xdg']: (cache/directory).mkdir(parents=True, exist_ok=True)
process_env = dict(os.environ)
process_env.update(PYTHONPATH=str(repo/'src')+os.pathsep+str(deps), PYTHONDONTWRITEBYTECODE='1',
                   TMPDIR=str(cache/'tmp'), TMP=str(cache/'tmp'), TEMP=str(cache/'tmp'),
                   MPLCONFIGDIR=str(cache/'mpl'), XDG_CACHE_HOME=str(cache/'xdg'))
command = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
if not command: command = ['environment', 'list', '--store', str(env.root)]
raise SystemExit(subprocess.call([str(python_root/'bin/python3.12'), '-B', '-m', 'phmforge_bench', *command], env=process_env))
