"""Prepare a frozen load split from qualified, unused measured acquisitions."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import secrets
import shutil

import numpy as np
import torch

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_fault_extensions import features
from prepare_selection_expansion import controls, utc
from real_fault_model import train


def prepare(root):
    config = json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    tag = config['iteration'].removeprefix('A')
    binding = json.loads((root/config['source_binding_file']).read_text())
    assert binding['source_zip']['sha256'] == config['source_zip_sha256']
    width = config['window_samples']
    rows = defaultdict(list)
    private = []
    for record in binding['records']:
        path = root/'source/ROTOR_ACQUISITION2'/Path(record['file']).name
        assert sha256(path) == record['file_sha256']
        with np.load(path, allow_pickle=False) as data:
            assert int(data['acquisition']) == config['acquisition'] == 2
            signal = data['signals']
            label, load = int(data['broken_bars']), float(data['load_nm'])
            assert label == record['broken_bars']
            part, = [p for p, loads in config['load_allocation'].items() if load in loads]
            group = 'group_'+secrets.token_hex(10)
            for start in range(0, len(signal)-width+1, width):
                observation = signal[start:start+width]
                assert np.isfinite(observation).all()
                identity = secrets.token_hex(16)
                rows[part].append(dict(features=features(observation), waveforms=observation.T.copy(),
                    labels=label, groups=group, sample_ids=identity, sampling_hz=int(data['sampling_hz']),
                    operating_load_Nm=load))
                private.append(dict(sample_id=identity, group=group, label=label, split=part,
                    native_file=record['member'], native_load=record['load'], acquisition=2,
                    start=start, length=width))
    common = root/'common/restored/ROTORBROKENBAR'
    data = common/'data'
    data.mkdir(parents=True, exist_ok=False)
    group_sets = {}
    for part, records in rows.items():
        secrets.SystemRandom().shuffle(records)
        group_sets[part] = set(r['groups'] for r in records)
        assert set(r['labels'] for r in records) == set(range(5))
        payload = {k: np.asarray([r[k] for r in records]) for k in records[0]
                   if part != 'test' or k not in ['labels', 'groups']}
        payload['dataset'] = np.asarray('ROTORBROKENBAR')
        np.savez_compressed(data/(part+'.npz'), **payload)
        with np.load(data/(part+'.npz'), allow_pickle=False) as saved:
            assert all(np.array_equal(saved[k], v) for k, v in payload.items())
        if part == 'test':
            save(common/'hidden/labels.json', [dict(sample_id=r['sample_ids'],
                group=r['groups'], label=r['labels']) for r in records])
    assert all(not group_sets[a] & group_sets[b] for a,b in [('train','val'),('train','test'),('val','test')])
    info = dict(dataset='ROTORBROKENBAR', seed=config['numerical_seed'], class_names=list(range(5)),
        architectures=['mlp'], source_sha256=config['source_zip_sha256'],
        source_qualification='Native acquisition 2; whole measured files bound to A275 source checks.',
        group_definition='One complete acquired source record. Complete operating load also held out; no independent-motor claim.',
        label_definition='Documented original broken-bar counts:0healthy,1..4broken bars.',
        channels=['Vib_acpe','Vib_acpi','Vib_axial','Vib_base','Vib_carc'],
        sampling_hz=7600, amplitude_units='Calibration not verified; original measured values retained',
        operating_load_units='Nm', cv_group_field='operating_load_Nm',
        rows_by_split={p:len(r) for p,r in rows.items()},
        groups_by_split={p:len(g) for p,g in group_sets.items()},
        train_validation_groups={p:sorted(group_sets[p]) for p in ['train','val']},
        observation_layout='waveforms[query,channel,2048],features[query,75],sampling_hz[query],operating_load_Nm[query]',
        preprocessing='Nonoverlapping2048point windows; seven time-domain and eight FFT-band statistics per channel. No waveform normalization, imputation or augmentation.',
        feature_order='statistic-major then channel',
        split_scope='All source records and loads disjoint across fitting,development and final evaluation.',
        identity_policy='Opaque record and sample IDs; common private shuffled order; no source ordinal or label-bearing filenames.',
        labels_in_test_input=False, official_original_benchmark_score_claimed=False)
    save(data/'dataset.json', info)
    save(root/'audit'/('PRIVATE_ALLOCATION'+tag+'.json'), private)
    del rows, payload, signal, observation
    training = train(data, common/'model', epochs=config['reference_epochs'])
    assert training['optimizer_steps'] == training['nonzero_gradient_steps'] > 0
    assert training['changed_parameter_tensors'] > 0 and not training['test_labels_accessed']
    checkpoint = torch.load(common/'model/model.pt', weights_only=True, map_location='cpu')
    with np.load(data/'val.npz', allow_pickle=False) as dev:
        x = (dev['features'].astype(np.float32)-checkpoint['mean'].numpy())/checkpoint['scale'].numpy()
        for n in [0,2,4]:
            x = x@checkpoint['state_dict'][f'net.{n}.weight'].numpy().T+checkpoint['state_dict'][f'net.{n}.bias'].numpy()
            if n != 4: x=np.maximum(x,0)
        predictions=np.asarray(checkpoint['class_names'])[x.argmax(1)]
        accuracy=float(np.mean(predictions==dev['labels']))
        assert abs(accuracy-training['selected_validation_accuracy'])<1e-12
    save(root/'audit'/('REFERENCE_FORWARD'+tag+'.json'), dict(passed=True,
        development_accuracy=accuracy, optimizer_steps=training['optimizer_steps'],
        independent_numpy_forward=True, final_test_used=False))
    first = config['cases'][0]['case']
    controls(root, first, data)
    for case in config['cases']:
        shutil.copytree(common, root/case['cohort']/'restored/ROTORBROKENBAR', copy_function=os.link)
        if case['case'] != first:
            shutil.copytree(root/'controls'/first, root/'controls'/case['case'], copy_function=os.link)
    output = dict(utc=utc(), passed=True, source_acquisition=2, native_records=40,
        rows=info['rows_by_split'], groups=info['groups_by_split'],
        optimizer_steps=training['optimizer_steps'], reference_fitted_once=True,
        reference_development_accuracy=accuracy, final_test_scores_opened=False,
        new_LLM_calls=0, protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'))
    save(root/('PREPARATION'+tag+'.json'), output)
    save(Path(__file__).resolve().parents[1]/config['result_directory']/('PREPARATION'+tag+'.json'), output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    prepare(parser.parse_args().run.resolve())
