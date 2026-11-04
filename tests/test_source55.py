"""Source identities, annotation parsing and boundaries for the 55 case entries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phmforge_bench.data import resource_json
from phmforge_bench.engine import _annotation, _native_records
from phmforge_bench.scenario import case_data, create_workspace, inventory
from verified_phmforge import VerifiedForge


class Source55Contracts(unittest.TestCase):
    def test_cases_are_unique_and_do_not_imply_complete_original_answers(self):
        cases = inventory()['tasks']
        self.assertEqual(len(cases), 55)
        self.assertEqual(len({row['task_id'] for row in cases}), 55)
        self.assertTrue(all(row['status'] == 'partial_numerical_research_adaptation' for row in cases))
        self.assertEqual(len(resource_json('engine_source_manifest.json')['cases']), 28)
        self.assertIn('not a calibrated', ' '.join(inventory('engine_health_combined_013')['limitations']))
        with self.assertRaisesRegex(ValueError, 'Unknown historical task'):
            inventory('pdm_fault_999')

    def test_engine_training_records_exclude_all_original_cases(self):
        source = resource_json('engine_source_manifest.json')
        held = {value for case in source['cases'].values() for value in case}
        health = source['health_training_records']
        trend = source['trend_training_tuples']
        self.assertFalse(held & {item['source_id'] for item in health})
        self.assertFalse(held & {ident for item in trend for ident in item['source_ids']})
        train = {ident for item in trend if item['part'] == 'train' for ident in item['source_ids']}
        val = {ident for item in trend if item['part'] == 'val' for ident in item['source_ids']}
        self.assertFalse(train & val)
        self.assertFalse(any('label' in item for item in health + trend))
        for name in ('full_record_array_sha256', 'prefix_record_array_sha256'):
            self.assertTrue(all(len(value) == 64 for value in source[name].values()))

    def test_engine_qa_parser_preserves_publisher_answers_and_missing_values(self):
        conversations = []
        for component, answer in zip(('fan', 'lpc', 'hpc', 'hpt', 'lpt'), ('a', 'b', 'a', 'b', 'a')):
            conversations.extend([{'from': 'human', 'stage': '2', 'value': f'What is the health status of {component}? a: Health\n b: Fault\n'},
                                  {'from': 'gpt', 'value': answer}])
        conversations.extend([{'from': 'human', 'stage': '2', 'value': 'What is the potential health state? a: Operating Normally\n b: Abnormal Degradation\n'},
                              {'from': 'gpt', 'value': 'b'},
                              {'from': 'human', 'stage': '2', 'value': 'Was the timing issue caused by efficiency? a: HPT efficiency modifier\n b: LPT flow modifier\n'},
                              {'from': 'gpt', 'value': 'a,b'}])
        labels = _annotation({'id': '1', 'conversations': conversations})
        self.assertEqual(labels, {'joint': '01010', 'state': 'Abnormal Degradation',
                                  'timing': 'HPT efficiency modifier|LPT flow modifier'})
        labels = _annotation({'id': '1', 'conversations': conversations[:10]})
        self.assertEqual((labels['state'], labels['timing']), ('', ''))

    def test_tampered_cached_record_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-engine-digest-') as directory:
            root = Path(directory)
            cache = root / 'raw/records'
            cache.mkdir(parents=True)
            body = bytes(600 * 33 * 8)
            (cache / '7-600.bin').write_bytes(body[:-1] + b'x')
            manifest = {'full_record_array_sha256': {'7': hashlib.sha256(body).hexdigest()},
                        'prefix_record_array_sha256': {}}
            with self.assertRaisesRegex(ValueError, 'failed SHA-256'):
                _native_records(root, None, manifest, 1)

    def test_scenario_rejects_mismatched_dataset(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-scenario-binding-') as directory:
            root = Path(directory)
            (root / 'READY.json').write_text(json.dumps({'status': 'prepared', 'dataset': 'CWRU'}))
            with self.assertRaisesRegex(ValueError, 'requires CMAPSS_FD001'):
                case_data('pdm_rul_001', root)

    def test_scenario_workspace_gives_original_question_but_not_private_labels(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-scenario-workspace-') as directory:
            base = Path(directory)
            root = base / 'data'
            data = root / 'public/datasets/CWRU'
            data.mkdir(parents=True)
            (data / 'dataset.json').write_text(json.dumps({'class_names': [0, 1], 'group_definition': 'fixture'}))
            (root / 'READY.json').write_text(json.dumps({'status': 'prepared', 'dataset': 'CWRU',
                'task_id': 'research_cwru_fault_classification'}))
            (root / 'private').mkdir()
            (root / 'private/labels.json').write_text('private-label-sentinel')
            (root / 'holdout').mkdir()
            (root / 'holdout/test.npz').write_bytes(b'private-input-sentinel')
            create_workspace('pdm_cost_benefit_002', root, base / 'workspace')
            text = (base / 'workspace/TASK.md').read_text()
            self.assertIn(inventory('pdm_cost_benefit_002')['original_question'], text)
            self.assertIn('not supplied', text)
            self.assertFalse((base / 'workspace/data/private').exists())
            self.assertFalse((base / 'workspace/data/holdout').exists())

    def test_engine_development_observations_exclude_auxiliary_labels(self):
        with tempfile.TemporaryDirectory(prefix='phmforge-engine-dev-') as directory:
            root = Path(directory)
            public = root / 'public'
            data = public / 'datasets/ENGINE_QA_HEALTH'
            data.mkdir(parents=True)
            np.savez_compressed(data / 'val.npz', features=np.ones((1, 2)), labels=np.array(['00000']),
                                groups=np.array(['g']), states=np.array(['state']), timing=np.array(['cause']),
                                sample_ids=np.array(['s']))
            (public / 'dataset_registry.json').write_text(json.dumps({'ENGINE_QA_HEALTH': {
                'task': 'classification', 'path': 'datasets/ENGINE_QA_HEALTH', 'available_splits': ['train', 'val']}}))
            backend = VerifiedForge(public, root / 'output')
            result = backend.call('load_dataset', {'dataset_name': 'ENGINE_QA_HEALTH', 'split': 'val'})
            with np.load(result['data_path'], allow_pickle=False) as observations:
                self.assertFalse(set(observations.files) & {'labels', 'groups', 'states', 'timing'})


if __name__ == '__main__':
    unittest.main()
