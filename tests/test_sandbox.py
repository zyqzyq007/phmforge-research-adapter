"""Integration check for cold-inference data and network isolation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from phmforge_bench.runner import _predict_independent_prefixes, _sandbox_command


@unittest.skipUnless(shutil.which('bwrap'), 'bubblewrap is required for the isolation test')
class ColdSandbox(unittest.TestCase):
    def test_each_prefix_of_one_unit_is_in_a_fresh_sandbox(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-prefix-sandbox-') as folder:
            base = Path(folder)
            data, submission, output = base/'data', base/'submission', base/'evaluation'
            for path in (data/'holdout', submission, output):
                path.mkdir(parents=True)
            np.savez_compressed(data/'holdout/test.npz', dataset=np.asarray('FIXTURE'),
                sample_ids=np.asarray(['early', 'late']), unit_ids=np.asarray([1, 1]),
                features=np.asarray([[1.], [2.]]))
            (submission/'solution.py').write_text('''import argparse, json
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--input'); p.add_argument('--output'); a = p.parse_args()
z = np.load(a.input, allow_pickle=False)
assert len(z['sample_ids']) == 1
assert len(list(Path('/input').iterdir())) == 1
assert not Path('/output/predictions.json').exists()
Path(a.output).write_text(json.dumps({'predictions': [{'sample_id': str(z['sample_ids'][0]), 'predicted_rul': 0.0}]}))
''')
            predictions = _predict_independent_prefixes(submission, data, data/'holdout/test.npz', output, 60)
            self.assertEqual([x['sample_id'] for x in predictions], ['early', 'late'])

    def test_private_and_training_files_are_not_visible(self):
        # These files are software fixtures, not PHM observations or results.
        with tempfile.TemporaryDirectory(prefix='phmforge-sandbox-') as folder:
            base = Path(folder)
            data, submission, output = base/'data', base/'submission', base/'output'
            for directory in [data/'public', data/'private', data/'holdout', submission, output]:
                directory.mkdir(parents=True)
            for name in ['tmp', 'mpl', 'xdg', 'home']:
                (output/name).mkdir()
            (data/'public/train.txt').write_text('training fixture')
            (data/'private/labels.json').write_text('private fixture')
            (data/'holdout/test.npz').write_text('observation fixture')
            code = '''import json, os, socket
from pathlib import Path
private = Path(%r)
checks = {
    'observation_visible': Path('/input/observations').read_text() == 'observation fixture',
    'training_absent': not Path('/data/train.txt').exists(),
    'private_absent': not private.exists(),
    'credential_env_absent': 'DEEPSEEK_API_KEY' not in os.environ,
}
try:
    Path('/app/write_probe').write_text('probe')
    checks['submission_read_only'] = False
except OSError:
    checks['submission_read_only'] = True
try:
    with socket.create_connection(('1.1.1.1', 443), timeout=1):
        checks['network_isolated'] = False
except OSError:
    checks['network_isolated'] = True
Path('/output/checks.json').write_text(json.dumps(checks))
''' % str(data/'private/labels.json')
            (submission/'solution.py').write_text(code)
            command = _sandbox_command(submission, data, data/'holdout/test.npz', output)
            environment = dict(os.environ, DEEPSEEK_API_KEY='fixture-only-not-a-credential')
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            checks = json.loads((output/'checks.json').read_text())
            self.assertTrue(all(checks.values()), checks)


if __name__ == '__main__':
    unittest.main()
