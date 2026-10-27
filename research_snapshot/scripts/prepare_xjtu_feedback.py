"""Prepare the frozen A278 real-data RUL comparison, without Agent calls.

The private allocation/labels stay outside Agent mounts. All bank fits finish
before development metrics are calculated; final predictions are never made.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from real_xjtu_rul_model import BANK, fit_bank_model, predict_bank, metrics, train


def utc():
    return datetime.now(timezone.utc).isoformat()


def state(root, **changes):
    path = root/'PREPARATION_STATUS278.json'
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(utc=utc(), run=str(root), **changes)
    save(path, value)
    save(Path(__file__).resolve().parents[1]/'results/feedback278/current_run.json', value)


def opaque(name, kind):
    return kind+'_'+hashlib.sha256(('278-private-'+name).encode()).hexdigest()[:16]


def records(raw, unit, scenario):
    n = len(raw['features'])
    features = np.column_stack([raw['features'], raw['nominal_elapsed_minutes'],
        np.repeat(np.eye(3)[scenario-1][None, :], n, axis=0)])
    return dict(dataset=np.asarray('XJTU'), features=features,
        sample_ids=np.asarray([opaque(str(unit)+':'+str(i), 'q') for i in range(n)]),
        unit_ids=np.full(n, unit), scenario_ids=np.full(n, scenario),
        elapsed_minutes=raw['nominal_elapsed_minutes'], labels=raw['nominal_remaining_minutes'],
        channel_rms=raw['channel_rms'], channel_max_abs=raw['channel_max_abs'])


def concatenate(rows):
    return {k: rows[0][k] if k == 'dataset' else np.concatenate([r[k] for r in rows]) for k in rows[0]}


def query(raw, index, sample_id):
    values = {k:(v if k == 'dataset' else v[index-1:index]) for k,v in raw.items() if k != 'labels'}
    values['sample_ids'] = np.asarray([sample_id])
    values.update(history_features=raw['features'][:index],
        history_elapsed_minutes=raw['elapsed_minutes'][:index],
        history_channel_rms=raw['channel_rms'][:index],
        history_channel_max_abs=raw['channel_max_abs'][:index],
        history_offsets=np.asarray([0, index]))
    return values


def main(root):
    config = json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert not (root/'PREPARATION_STATUS278.json').exists(), 'No automatic preparation rerun'
    state(root, status='binding_data', pid=os.getpid(), goal_status='active', new_model_sessions=0,
          new_training_jobs=0, final_scores_read=False)
    source = Path(config['source_archive'])
    receipt = json.loads((source/'ARCHIVE_VERIFIED.json').read_text())
    assert receipt['manifest_sha256'] == config['source_manifest_sha256']
    qualification = json.loads((source/'audit/XJTU_SOURCE_QUALIFICATION277.json').read_text())
    assert qualification['passed'] and qualification['rul_measured_source_qualified']
    pin = {x['bearing']:x for x in qualification['bearings']}
    rng = np.random.default_rng(config['numerical_seed'])
    units = sorted(pin)
    aliases = {name: int(i+101) for name,i in zip(units, rng.permutation(len(units)))}
    # The split RNG is separate: alias generation must not advance its state.
    rng = np.random.default_rng(config['numerical_seed'])
    splits = dict(train=[], val=[], test=[])
    folds = [[], [], []]
    for scenario in (1,2,3):
        chosen = rng.permutation([f'Bearing{scenario}_{i}' for i in range(1,6)]).tolist()
        splits['train'] += chosen[:3]
        splits['val'].append(chosen[3]); splits['test'].append(chosen[4])
        for f in range(3): folds[f].append(aliases[chosen[f]])
    assert len(set(sum(splits.values(), []))) == 15
    arrays, bindings = {}, []
    for name in units:
        path = source/pin[name]['path']
        assert sha256(path) == pin[name]['sha256']
        original = dict(np.load(path, allow_pickle=False))
        assert np.array_equal(original['nominal_remaining_minutes'],
            len(original['features'])-original['nominal_elapsed_minutes'])
        arrays[name] = records(original, aliases[name], int(name[7]))
        bindings.append(dict(native_bearing=name, opaque_unit=aliases[name],
            source_path=str(path), source_sha256=pin[name]['sha256']))
    save(root/'hidden/ALLOCATION278.json', dict(splits=splits, aliases=aliases, folds=folds,
        bindings=bindings, source_qualification_sha256=sha256(source/'audit/XJTU_SOURCE_QUALIFICATION277.json')))
    public = root/'public_data/datasets/XJTU'; public.mkdir(parents=True)
    fit = concatenate([arrays[name] for name in splits['train']])
    np.savez_compressed(public/'train.npz', **fit)
    truncation = np.random.default_rng(279)
    dev_queries, dev_labels, final_labels, prefix_audits = [], [], [], []
    for split in ('val','test'):
        for name in splits[split]:
            raw = arrays[name]; n = len(raw['labels']); indices = []
            wanted = 1 if split == 'val' else 5
            while len(indices) < wanted:
                index = int(np.clip(np.floor(n*truncation.uniform(.15,.95)), 1, n-1))
                if index not in indices: indices.append(index)
            for order,index in enumerate(indices):
                ident = opaque(name+':'+split+':'+str(order), 'query')
                q = query(raw, index, ident)
                row = dict(sample_id=ident, unit_id=aliases[name], scenario_id=int(name[7]),
                    actual_rul=float(n-index), elapsed_minutes=index, native_bearing=name,
                    terminal_damage_locations=pin[name]['terminal_damage_locations'], lifetime_minutes=n)
                if split == 'val':
                    dev_queries.append(q); dev_labels.append(row)
                else:
                    path = root/'evaluation_data/queries'/(ident+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path, **q); row['input_path']=str(path.relative_to(root))
                    final_labels.append(row)
                assert q['history_elapsed_minutes'][-1] == index
                assert len(q['history_features']) == index
                assert not any(k in q for k in ['labels','lifetime_minutes','native_bearing'])
                prefix_audits.append(dict(split=split, sample_id=ident, observations=index,
                    maximum_time=index, current_time=index, future_records_absent=True))
    dev = concatenate([{k:v for k,v in q.items() if not k.startswith('history_')} for q in dev_queries])
    dev['labels'] = np.asarray([r['actual_rul'] for r in dev_labels])
    for key in ('history_features','history_elapsed_minutes','history_channel_rms','history_channel_max_abs'):
        dev[key] = np.concatenate([q[key] for q in dev_queries])
    dev['history_offsets'] = np.cumsum([0]+[len(q['history_features']) for q in dev_queries])
    np.savez_compressed(public/'val.npz', **dev)
    save(root/'hidden/final_labels.json', final_labels)
    save(root/'hidden/development_provenance.json', dev_labels)
    save(root/'audit/PREFIX_BOUNDARY278.json', dict(passed=True, rows=prefix_audits,
        final_inference_contract='One query/causal prefix per fresh process, without other inputs or labels'))
    feature_names = [f'{feature}_{channel}' for feature in
        ['mean','std','rms','max_abs','peak_to_peak','skewness','kurtosis']+[f'log1p_fftband{k}' for k in range(1,9)]
        for channel in ['horizontal','vertical']]+['elapsed_minutes','condition1','condition2','condition3']
    save(public/'dataset.json', dict(dataset='XJTU', format='xjtu_npz', task='rul', target_unit='minutes',
        input_features=feature_names, training_bearings=9, development_bearings=3,
        conditions=[dict(id=1,rpm=2100,radial_load_kn=12),dict(id=2,rpm=2250,radial_load_kn=11),dict(id=3,rpm=2400,radial_load_kn=10)],
        acquisition=dict(hz=25600,samples=32768,channels=2,nominal_interval_minutes=1),
        label='N-i nominal minutes to the original terminal acquisition, not a continuous physical failure-time measurement',
        source='Published measured XJTU-SY accelerated bearing experiments, author-hash verified native records',
        information_boundary='Only current and earlier observations; opaque IDs; final inputs/endpoints/labels absent',
        histories='Train rows can be grouped by unit_id/time. Each development query has only its own observed prefix, indexed by history_offsets.',
        limitations=['30 measured signal statistics, not the full waveform interface',
            'Laboratory accelerated lifetimes, not field maintenance costs',
            'Terminal damage is not a per-minute observed state or an inference feature',
            'Public-source memorization remains possible; native identities and final lengths are not provided'] ))
    train_config = dict(config['reference_mlp'], fit_unit_ids=sorted(set(fit['unit_ids'].tolist())))
    save(public/'training_config.json', train_config)
    save(root/'public_data/dataset_registry.json', {'XJTU':dict(task='rul',format='xjtu_npz',
        path='datasets/XJTU',architectures=['mlp'],test_inputs_visible=False,available_splits=['train','val'])})
    save(root/'references/training_cv_folds.json', dict(folds=[dict(fold=i,
        heldout_unit_ids=held,fit_unit_ids=sorted(set(fit['unit_ids'].tolist())-set(held))) for i,held in enumerate(folds)]))
    bank = root/'references/candidates'; bank.mkdir()
    fitted = 0; rows = []
    for name in BANK:
        oof = np.full(len(fit['labels']),np.nan)
        for number,held in enumerate(folds):
            hold = np.isin(fit['unit_ids'],held)
            subsets = [{k:v if k=='dataset' else v[mask] for k,v in fit.items()} for mask in (~hold,hold)]
            model = fit_bank_model(name,subsets[0],config['numerical_seed'])
            path = bank/'cv_models'/(name+'_fold'+str(number)+'.joblib');path.parent.mkdir(exist_ok=True)
            joblib.dump(model,path,compress=3)
            oof[hold] = predict_bank(model,subsets[1]); fitted += 1
        model = fit_bank_model(name,fit,config['numerical_seed'])
        joblib.dump(model,bank/(name+'.joblib'),compress=3); fitted += 1
        np.savez_compressed(bank/(name+'_OOF.npz'),sample_ids=fit['sample_ids'],predictions=oof)
        rows.append(dict(name=name,checkpoint=name+'.joblib',
            training_cv=metrics(oof,fit['labels'],fit['unit_ids']),
            training=metrics(predict_bank(model,fit),fit['labels'],fit['unit_ids'])))
        state(root,status='fitting_frozen_bank',new_training_jobs=fitted,bank_models_finished=len(rows))
    save(root/'audit/BANK_FITS_SEALED278.json',dict(utc=utc(),models=[dict(name=r['name'],
        sha256=sha256(bank/r['checkpoint'])) for r in rows],new_fits=fitted,
        fitting_data_sha256=sha256(public/'train.npz'),development_feedback_used=False))
    reference = train(public,root/'references/mlp',epochs_override=train_config['epochs'])
    fitted += 1
    # Development data enter predictor assessment only after every bank fit is sealed.
    for row in rows:
        model = joblib.load(bank/row['checkpoint'])
        row['development'] = metrics(predict_bank(model,dev),dev['labels'],dev['unit_ids'])
    chosen = min(rows,key=lambda r:(r['development']['pooled_mae_minutes'],
        r['training_cv']['mean_bearing_mae_minutes'],BANK.index(r['name'])))['name']
    save(bank/'manifest.json',dict(models=rows,starting_model='rf200',
        development_selected=chosen,selection_rule='development MAE, training-CV MAE, fixed bank order',
        fitting_scope='train_only; all bank fits sealed before development assessment'))
    save(root/'audit/DATA_PREPARATION278.json',dict(passed=True,train_bearings=9,development_bearings=3,
        final_bearings=3,final_queries=15,training_rows=len(fit['labels']),
        measured_source=True,synthetic_training_records=0,all_units_disjoint=True,
        final_scores_read=False,reference_training=reference,new_training_jobs=fitted,
        protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json')))
    for name in ['prepare_xjtu_feedback.py','real_xjtu_rul_model.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'code'/name)
    state(root,status='real_fits_completed_waiting_independent_qualification',
        new_training_jobs=fitted,new_model_sessions=0,final_scores_read=False,
        reference_optimizer_steps=reference['optimizer_steps'],reference_nonzero_updates=reference['nonzero_gradient_steps'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
