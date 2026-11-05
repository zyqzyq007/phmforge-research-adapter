"""Contracts for explicit alternatives, source separation and reference reuse."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phmforge_bench.data import resource_json
from phmforge_bench.engine import _native_records
from phmforge_bench.runner import training_fingerprint
from phmforge_bench.scenario import case_data, create_workspace, inventory, run
from verified_phmforge import VerifiedForge


class ResearchAlternativeContracts(unittest.TestCase):
    def test_mcp_checks_declared_classes_even_when_development_lacks_a_class(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-dev-classes-') as directory:
            base = Path(directory)
            data = base / 'public/datasets/CWRU'
            data.mkdir(parents=True)
            (base / 'public/dataset_registry.json').write_text(json.dumps({'CWRU': {
                'task': 'classification', 'path': 'datasets/CWRU', 'available_splits': ['train', 'val']}}))
            (data / 'dataset.json').write_text(json.dumps({'class_names': [0, 1]}))
            np.savez_compressed(data / 'val.npz', features=np.ones((1, 2)), labels=np.array([0]),
                                sample_ids=np.array(['sample']), groups=np.array(['record']))
            backend = VerifiedForge(base / 'public', base / 'artifacts')
            truth = backend.call('load_ground_truth', {'dataset_name': 'CWRU', 'file': 'val'})
            rows = json.loads(Path(truth['artifact']).read_text())
            self.assertEqual(rows[0]['dataset'], 'CWRU')
            verified = backend.call('verify_classification', {'ground_truth': truth['artifact'],
                'predictions': [{'sample_id': 'sample', 'predicted_label': 1}]})
            self.assertEqual(verified['metrics']['accuracy'], 0)
            with self.assertRaisesRegex(ValueError, 'declared task universe'):
                backend.call('verify_classification', {'ground_truth': truth['artifact'],
                    'predictions': [{'sample_id': 'sample', 'predicted_label': 2}]})

    def test_opt_in_preserves_unavailable_original_targets(self):
        original = inventory(include_unavailable=True)['tasks']
        alternatives = inventory(include_alternatives=True)['tasks']
        self.assertEqual(len(alternatives), 75)
        self.assertEqual(sum(row['status'] == 'unavailable' for row in original), 4)
        chosen = [row for row in alternatives if row['status'] == 'explicit_research_alternative']
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(row['original_target_status'] == 'unavailable' for row in chosen))
        self.assertTrue(all(row['adapter_id'] and row['adapter_id'] != row['task_id'] for row in chosen))
        self.assertEqual(inventory('pdm_fault_005')['status'], 'unavailable')
        self.assertEqual(inventory('pdm_fault_005', include_alternatives=True)['dataset'], 'SNU_PLANETARY')

    def test_alternative_workspace_requires_opt_in_and_keeps_original_scope(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-alternative-') as directory:
            base = Path(directory)
            root = base / 'data'
            data = root / 'public/datasets/SNU_PLANETARY'
            data.mkdir(parents=True)
            (data / 'dataset.json').write_text(json.dumps({'class_names': ['healthy', 'planet_gear', 'sun_gear'],
                                                         'group_definition': 'fixture', 'feature_count': 15}))
            (root / 'READY.json').write_text(json.dumps({'status': 'prepared', 'dataset': 'SNU_PLANETARY',
                                                        'task_id': 'research_snu_planetary_component_classification'}))
            (root / 'private').mkdir()
            (root / 'private/labels.json').write_text('private-sentinel')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                case_data('pdm_fault_005', root)
            create_workspace('pdm_fault_005', root, base / 'workspace', research_alternative=True)
            text = (base / 'workspace/TASK.md').read_text()
            meta = json.loads((base / 'workspace/WORKSPACE.json').read_text())
            self.assertIn('original target remains unavailable', text)
            self.assertIn('Current experiment question', text)
            self.assertEqual(meta['original_target_status'], 'unavailable')
            self.assertFalse(meta['complete_original_task_claimed'])
            self.assertFalse((base / 'workspace/data/private').exists())

    def test_alternative_engine_records_are_disjoint_and_frozen(self):
        original = resource_json('engine_source_manifest.json')
        alternative = resource_json('engine_alternatives.json')
        reserved = set(original['full_record_array_sha256']) | set(original['prefix_record_array_sha256'])
        ids = [ident for row in alternative['cases'].values() for ident in row]
        self.assertEqual(len(alternative['cases']), 2)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 20)
        self.assertFalse(reserved.intersection(ids))
        self.assertEqual(set(ids), set(alternative['prefix_record_array_sha256']))

    def test_snu_record_split_and_fault_semantics(self):
        source = resource_json('snu_sources.json')
        groups = {part: {row['path'] for row in source['members'] if row['split'] == part}
                  for part in ('train', 'val', 'test')}
        self.assertEqual([len(groups[p]) for p in ('train', 'val', 'test')], [4, 3, 7])
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse((groups['train'] | groups['val']) & groups['test'])
        self.assertEqual(set(source['classes']), {'healthy', 'planet_gear', 'sun_gear'})
        self.assertTrue(all(row['label'] == 'planet_gear' for row in source['members'] if '_pr_' in row['path']))
        self.assertTrue(all(row['condition'] == 'nonstationary' for row in source['members'] if row['split'] == 'test'))

    def test_engine_accepts_only_hash_verified_offline_record_ranges(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-engine-offline-') as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'records').mkdir(parents=True)
            body = np.zeros((600, 33), dtype='<f8').tobytes()
            full_hash = hashlib.sha256(body).hexdigest()
            prefix_hash = hashlib.sha256(body[:60 * 33 * 8]).hexdigest()
            native = source / 'records/7-600.bin'
            native.write_bytes(body)
            manifest = {'full_record_array_sha256': {'7': full_hash},
                        'prefix_record_array_sha256': {'7': prefix_hash}}
            arrays, audit = _native_records(root / 'prepared', source, manifest, 1)
            self.assertEqual(arrays['7'].shape, (600, 33))
            self.assertEqual(audit['downloaded_or_read_records'], 0)
            native.write_bytes(body[:-1] + b'x')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                _native_records(root / 'rejected', source, manifest, 1)

    def test_reference_fingerprint_uses_values_not_npz_compression(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-fit-binding-') as directory:
            root = Path(directory)
            data = root / 'public/datasets/CWRU'
            data.mkdir(parents=True)
            (root / 'READY.json').write_text(json.dumps({'status': 'prepared', 'dataset': 'CWRU'}))
            values = {'features': np.ones((2, 3)), 'labels': np.array([0, 1]), 'groups': np.array(['a', 'b'])}
            np.savez_compressed(data / 'train.npz', **values)
            first = training_fingerprint(root)
            np.savez(data / 'train.npz', **values)
            self.assertEqual(first, training_fingerprint(root))
            values['features'][0, 0] = 2
            np.savez_compressed(data / 'train.npz', **values)
            self.assertNotEqual(first, training_fingerprint(root))
            submission = root.parent / (root.name + '-submission')
            submission.mkdir()
            (submission / 'SUBMISSION.json').write_text(json.dumps({'dataset': 'CWRU', 'training_fingerprint': first}))
            with self.assertRaisesRegex(ValueError, 'training/development'):
                run('pdm_fault_001', root, root.parent / (root.name + '-run'), submission=submission)


if __name__ == '__main__':
    unittest.main()
