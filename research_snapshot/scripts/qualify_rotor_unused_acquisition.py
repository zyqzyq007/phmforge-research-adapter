"""Recover the unused second acquisition of the measured rotor benchmark.

The old CSV is bound to acquisition one. This exports acquisition two without
inventing observations, labels or physical-device independence. No fitting.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import loadmat

from acquire_forge_pdmb_sources244 import save, sha256


def signature(arrays):
    return [(len(a), hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest())
            for a in arrays]


def qualify(run, source):
    binding = json.loads((source / 'SOURCE_BINDINGS271.json').read_text())
    previous = json.loads((source / 'INDEPENDENT_SOURCE_CHECK271.json').read_text())
    physical = json.loads((source / 'PHYSICAL_INFO271.json').read_text())
    zip_path = source / 'experimental_database_short.zip'
    bound = next(x for x in binding['files'] if x['path'].endswith(zip_path.name))
    assert zip_path.stat().st_size == bound['bytes'] and sha256(zip_path) == bound['sha256']
    assert previous['passed'] and previous['experiment_indices_used'] == [1]
    unused = {(x['member'], x['load']): x for x in previous['unused_upstream_experiments']}
    assert len(unused) == 40 and all(x['experiment'] == 2 for x in unused.values())
    prior_signatures = {tuple(tuple(t) for t in x['channel_fingerprints']) for x in previous['rows']}
    channels = physical['channels']
    destination = run / 'prepared/ROTOR_ACQUISITION2'
    destination.mkdir(parents=True, exist_ok=False)
    saved = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            payload = archive.read(member)  # ZIP CRC checked on read.
            primary = loadmat(io.BytesIO(payload), simplify_cells=True)
            independent = loadmat(io.BytesIO(payload), struct_as_record=False, squeeze_me=True)
            key, = [k for k in primary if not k.startswith('__')]
            assert [k for k in independent if not k.startswith('__')] == [key]
            for load, experiments in primary[key].items():
                assert len(experiments) == 2
                expected = unused[(member, load)]
                arrays = [np.asarray(experiments[1][c], dtype='<f8').reshape(-1) for c in channels]
                assert len(set(len(a) for a in arrays)) == 1
                assert all(np.isfinite(a).all() for a in arrays)
                actual = signature(arrays)
                assert actual == [tuple(x) for x in expected['channel_fingerprints']]
                assert tuple(actual) not in prior_signatures
                other = getattr(independent[key], load)[1]
                for c, a in zip(channels, arrays):
                    assert np.array_equal(a, np.asarray(getattr(other, c)).reshape(-1))
                output = destination / (Path(member).stem + '_' + load + '_acquisition2.npz')
                np.savez_compressed(output, signals=np.stack(arrays, axis=1),
                    channels=np.asarray(channels), broken_bars=np.asarray(expected['severity']),
                    load_nm=np.asarray(int(load[6:])/10), acquisition=np.asarray(2),
                    sampling_hz=np.asarray(physical['vibration_sampling_hz_publisher_declared']))
                with np.load(output, allow_pickle=False) as reloaded:
                    for i, a in enumerate(arrays):
                        assert np.array_equal(reloaded['signals'][:, i], a)
                saved.append(dict(member=member, member_sha256=hashlib.sha256(payload).hexdigest(),
                    load=load, acquisition=2, broken_bars=expected['severity'],
                    points_per_channel=len(arrays[0]), channel_fingerprints=actual,
                    file=str(output.relative_to(run)), file_sha256=sha256(output)))
            print(json.dumps(dict(member=member, recovered_records=len(saved))), flush=True)
            del primary, independent, payload, experiments, other, arrays
            gc.collect()
    assert len(saved) == 40
    assert len({tuple(x['channel_fingerprints']) for x in saved}) == 40
    save(run / 'audit/ROTOR_UNUSED_BINDINGS275.json', dict(source_zip=bound, records=saved))
    result = dict(iteration='A275', source_qualified=True, source_zip_sha256=bound['sha256'],
        source_binding='Official measured MAT archive, second acquisition; native expansion of the PHMForge/PDMBench first-acquisition CSV.',
        records=40, acquisition=2, broken_bar_classes=[0, 1, 2, 3, 4],
        load_levels_nm=[.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
        measured_values_verified=sum(x['points_per_channel']*len(channels) for x in saved),
        prior_unused_record_fingerprints_all_match=True,
        no_complete_record_identical_to_prior_used_acquisition=True,
        two_MAT_traversals_agree=True, exported_signals_bit_identical=True,
        input_units=physical['calibrated_amplitude_units'],
        independent_physical_motor_claim=False, source_known_to_researcher=True,
        no_new_source_claim=True, not_original_40_row_benchmark_score=True,
        no_new_LLM_calls=True, no_new_training=True, no_final_predictions_or_scores=True,
        next_gate='Freeze complete source-record/load splits and validation controls before any fitting. Keep final observations and labels outside every Agent stage.')
    save(run / 'audit/ROTOR_UNUSED_QUALIFICATION275.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    result = qualify(args.run.resolve(), args.source.resolve())
    print(json.dumps(result))
