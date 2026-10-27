"""Belief-only policy interface, run in a namespace without evaluator files."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def main():
    root = Path.cwd()
    sys.path.insert(0, str(root))
    protocol_stdout = sys.stdout
    np.random.seed(880000)  # Policy RNG, unrelated to environment seeds.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location('submitted_policy', root/'policy.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            policy = module.Policy(json.loads((root/'inputs/config.json').read_text()))
        print(json.dumps({'status': 'ready'}), file=protocol_stdout, flush=True)
        for line in sys.stdin:
            msg = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if msg['op'] == 'reset':
                    if hasattr(policy, 'reset'):
                        policy.reset()
                    reply = {'status': 'ready'}
                else:
                    reply = {'action': np.asarray(policy.act(np.array(msg['belief'], dtype=float))).tolist()}
            print(json.dumps(reply, allow_nan=False), file=protocol_stdout, flush=True)
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}), file=protocol_stdout, flush=True)


if __name__ == '__main__':
    main()
