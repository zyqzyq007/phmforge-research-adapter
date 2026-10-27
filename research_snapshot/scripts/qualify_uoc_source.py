"""Bind the published PHMForge UoC table to its original measured MAT file.

No fitting, synthetic observations, split construction or performance scoring.
This check does not invent acquisition identities absent from the source.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from acquire_forge_pdmb_sources244 import save, sha256


def qualify(run):
    source = run / 'source'
    metadata = json.loads((source / 'uoc_metadata.json').read_text())
    csv_path = run / 'inputs/pdmb_attempt2/UoC.csv'
    receipt = json.loads(csv_path.with_suffix('.receipt.json').read_text())
    assert receipt['pinned_hash_verified']
    assert sha256(csv_path) == receipt['actual_sha256']
    native = source / 'DataForClassification_TimeDomain.mat'
    file_metadata = next(x for x in metadata['files'] if x['name'] == native.name)
    assert native.stat().st_size == file_metadata['size']
    assert hashlib.md5(native.read_bytes()).hexdigest() == file_metadata['computed_md5']
    mat = loadmat(native)
    assert [k for k in mat if not k.startswith('__')] == ['AccTimeDomain']
    signals = mat['AccTimeDomain'].T
    assert signals.shape == (936, 3600) and np.isfinite(signals).all()
    csv.field_size_limit(32 * 1024**2)
    with csv_path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['features', 'label']
        rows = list(reader)
    measured = np.asarray([np.fromstring(x['features'].strip('[]'), sep=',') for x in rows])
    assert measured.shape == signals.shape and np.array_equal(measured, signals)
    classes = ['healthy', 'missing', 'crack', 'spall', 'chip5a', 'chip4a', 'chip3a', 'chip2a', 'chip1a']
    assert [x['label'] for x in rows] == np.repeat(classes, 104).tolist()
    fingerprints = [hashlib.sha256(row.astype('<f8').tobytes()).hexdigest() for row in signals]
    assert len(set(fingerprints)) == len(signals)
    official = next(source.rglob('phm_scenarios.json'))
    task = next(x for x in json.loads(official.read_text())['pdm_scenarios']
                if x['task_id'] == 'pdm_fault_010')
    result = dict(
        iteration='A275', source_qualified_as_measured=True,
        native_sha256=sha256(native), csv_sha256=sha256(csv_path),
        publisher_MD5_verified=True, publisher_DOI=metadata['doi'],
        observations=936, points_per_observation=3600, exact_matched_values=int(signals.size),
        maximum_absolute_difference=float(np.max(np.abs(measured-signals))),
        source_label_counts=dict(Counter(x['label'] for x in rows)),
        published_columns_are_unique_complete_observations=True,
        physical_hardware_or_acquisition_ids_available=False,
        nonoverlap_of_original_acquisitions_independently_certified=False,
        raw_acquisition_frequency_hz=20000,
        published_matrix_axis='Published AccTimeDomain; paper describes angle resampling/TSA. Do not assert a 20 kHz clock on this matrix without its transformation lineage.',
        source_description_arithmetic_error='The publisher page says 9x104=903; both original MAT and matched table contain 936 observations.',
        task_id=task['task_id'], original_task_sha256=hashlib.sha256(task['input_question'].encode()).hexdigest(),
        original_prompt=task['input_question'],
        original_expected_classes=task['ground_truth']['expected_fault_types'],
        task_mismatch='Nine states of the input pinion, not labels identifying distinct damaged gearbox components or bearing races.',
        precomputed_stage0_used=False,
        precomputed_stage0_exclusion='No fold-specific fit lineage for the published PCA-derived representation; do not use as a leakage-safe model input.',
        strict_new_condition_experiment_ready=False,
        permitted_scope='A separately declared within-published-collection classification control could use whole columns. It cannot establish cross-device/recording generalization.',
        decision='Retain a genuinely measured source. Do not launch the proposed strict new-condition Agent experiment on guessed recording groups.',
        new_LLM_calls=0, new_training_jobs=0, new_test_scores=0,
        no_new_Agent_failure_claim=True)
    save(run / 'audit/UOC_QUALIFICATION275.json', result)
    save(run / 'audit/UOC_NATIVE_COLUMN_BINDINGS275.json', [dict(
        published_row=n, native_column=n, label=rows[n]['label'], signal_sha256=h)
        for n, h in enumerate(fingerprints)])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    result = qualify(parser.parse_args().run.resolve())
    print(json.dumps({k: result[k] for k in ['source_qualified_as_measured',
        'observations', 'exact_matched_values', 'strict_new_condition_experiment_ready',
        'new_LLM_calls', 'new_training_jobs']}))
