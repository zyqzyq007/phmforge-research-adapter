"""Integrity and relocation software fixtures; no PHM performance data."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phmforge_bench.environment import Environment, _equal, _parse_command


class FrozenEnvironment(unittest.TestCase):
    def make_store(self, root):
        root.mkdir()
        content = b'original fitted artifact fixture'
        digest = hashlib.sha256(content).hexdigest()
        obj = root/'objects'/digest[:2]/digest[2:]
        obj.parent.mkdir(parents=True)
        obj.write_bytes(content)
        manifest = {'schema_version': 1, 'files': {'runs/a/model': {'sha256': digest, 'bytes': len(content)}},
                    'aliases': {'/original/run': 'runs/a'}, 'archives': [], 'scope': {}, 'symlinks': {}}
        path = root/'environment.json.gz'
        with gzip.open(path, 'wt') as f: json.dump(manifest, f)
        (root/'READY.json').write_text(json.dumps({'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
        return obj

    def test_original_paths_relocate_and_writes_do_not_change_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = self.make_store(root/'store')
            env = Environment(root/'store')
            self.assertEqual(env.translate('/original/run/model'), 'runs/a/model')
            env.materialize('runs/a', root/'working', writable=True)
            (root/'working/model').write_bytes(b'changed by a fresh run')
            self.assertEqual(obj.read_bytes(), b'original fitted artifact fixture')
            self.assertEqual(env.verify(full=True)['objects'], 1)
            with self.assertRaises(FileNotFoundError): env.translate('/etc/passwd')

    def test_corrupt_object_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = self.make_store(root/'store')
            obj.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): Environment(root/'store').object('runs/a/model')
            path = root/'store/environment.json.gz'
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError): Environment(root/'store')

    def test_snapshot_cannot_overwrite_its_own_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_store(root/'store')
            env = Environment(root/'store')
            with self.assertRaises(ValueError): env.materialize('runs/a', root/'store/replacement')
            with self.assertRaises(ValueError): env.materialize('../escape', root/'out')

    def test_recorded_command_is_parsed_without_shell_interpolation(self):
        cmd = ['bwrap', '--unshare-net', '--ro-bind', '/old/data', '/input', '--bind', '/old/output', '/work',
               '--chdir', '/input', '/opt/python/bin/python', '-B', '/input/program.py', '--label', '$(not-a-shell)']
        mounts, options, inner = _parse_command({'command': cmd})
        self.assertEqual(mounts[0], ('--ro-bind', '/old/data', '/input'))
        self.assertEqual(options, ['--chdir', '/input'])
        self.assertEqual(inner[-1], '$(not-a-shell)')
        with self.assertRaises(ValueError): _parse_command(['sh', '-c', 'echo unsupported'])

    def test_prediction_comparison_aligns_sample_ids(self):
        a = [{'sample_id': 'a', 'predicted_rul': 5.0}, {'sample_id': 'b', 'predicted_rul': 6.0}]
        self.assertTrue(_equal(a, list(reversed(a))))
        self.assertFalse(_equal(a, [{'sample_id': 'a', 'predicted_rul': 9}, a[1]]))
        self.assertFalse(_equal([float('nan')], [float('nan')]))


if __name__ == '__main__': unittest.main()
