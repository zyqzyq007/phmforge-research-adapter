"""Prepare one frozen real HUST split, shared by two validation-evidence arms.

No test predictions or performance are computed. Source labels are used only
to construct the private evaluator, never copied into public observations.
"""
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
from prepare_selection_expansion import controls, utc
from real_fault_model import train


def main(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert config['iteration']=='A274'
    source=root/'source/public/datasets/HUST'
    original=json.loads((source/'dataset.json').read_text())
    assert original['source_sha256']==config['source_CSV_SHA256']
    parts={p:dict(np.load(source/(p+'.npz'),allow_pickle=False)) for p in ['train','val','test']}
    gold={r['sample_id']:r for r in json.loads((root/'source/hidden/HUST/labels.json').read_text())}
    parts['test']['labels']=np.asarray([gold[str(i)]['label'] for i in parts['test']['sample_ids']])
    parts['test']['groups']=np.asarray([gold[str(i)]['group'] for i in parts['test']['sample_ids']])
    keys=set(parts['train'])
    assert all(set(v)==keys for v in parts.values())
    full={}
    for k in sorted(keys):
        arrays=[v[k] for v in parts.values()]
        if arrays[0].ndim:
            assert all(len(v[k])==len(v['sample_ids']) for v in parts.values())
            full[k]=np.concatenate(arrays)
        else:
            assert all(np.array_equal(a,arrays[0]) for a in arrays)
            full[k]=arrays[0]
    del parts
    assert len(full['sample_ids'])==19095==len(set(full['sample_ids']))
    group_families=defaultdict(set)
    group_classes=defaultdict(set)
    for group,family,label in zip(full['groups'],full['bearing_family'],full['labels']):
        group_families[str(group)].add(str(family));group_classes[str(group)].add(int(label))
    assert len(group_families)==33
    assert all(len(v)==1 for v in group_families.values())
    assert all(len(v)==1 for v in group_classes.values())
    # Identity/display randomization is private and common to every evidence arm.
    aliases={g:'group_'+secrets.token_hex(10) for g in group_families}
    ids=[secrets.token_hex(16) for _ in full['sample_ids']]
    common=root/'common/restored/HUST';data=common/'data'
    data.mkdir(parents=True,exist_ok=False)
    counts={};groups={};mapping=[];split_indices={}
    for part,families in config['geometry_allocation'].items():
        indices=np.flatnonzero(np.isin(full['bearing_family'],families)).tolist()
        secrets.SystemRandom().shuffle(indices)
        split_indices[part]=indices
        assert set(full['labels'][indices])==set(original['class_names'])
        payload={k:(v[indices] if v.ndim else v) for k,v in full.items()
                 if k not in ['sample_ids','groups','labels']}
        payload['sample_ids']=np.asarray([ids[i] for i in indices])
        if part!='test':
            payload.update(labels=full['labels'][indices],
                           groups=np.asarray([aliases[str(full['groups'][i])] for i in indices]))
        else:
            save(common/'hidden/labels.json',[dict(sample_id=ids[i],label=int(full['labels'][i]),
                 group=aliases[str(full['groups'][i])]) for i in indices])
        np.savez_compressed(data/(part+'.npz'),**payload)
        counts[part]=len(indices)
        groups[part]=sorted(set(full['groups'][indices]))
        mapping.extend(dict(part=part,old_id=str(full['sample_ids'][i]),new_id=ids[i],
             old_group=str(full['groups'][i]),new_group=aliases[str(full['groups'][i])],
             bearing_family=str(full['bearing_family'][i])) for i in indices)
        # Numerical identity, not a reconstructed or synthetic waveform.
        with np.load(data/(part+'.npz'),allow_pickle=False) as written:
            for k,v in payload.items():
                assert np.array_equal(written[k],v), (part,k)
        print(json.dumps(dict(stage='source_split_verified',part=part,rows=len(indices))),flush=True)
    assert sum(counts.values())==19095
    assert all(not set(groups[a])&set(groups[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    excluded={'groups_per_class','train_validation_test_groups','train_validation_groups',
              'groups_by_split','rows_by_split','seed'}
    info={k:v for k,v in original.items() if k not in excluded}
    info.update(seed=config['numerical_seed'],rows_by_split=counts,
        groups_by_split={p:len(g) for p,g in groups.items()},
        train_validation_groups={p:[aliases[g] for g in groups[p]] for p in ['train','val']},
        split_scope='Complete bearing-family holdout: train6204/6205/6206,development6207; final family absent from training/development. All loads retained together.',
        previous_model_weights_reused=False,
        identity_policy='New opaque sample/group IDs,common private display order across validation arms; no source ordinal or class prefix')
    save(data/'dataset.json',info)
    save(root/'audit/PRIVATE_ALLOCATION274.json',dict(families=config['geometry_allocation'],
        rows=counts,groups=groups,identities=mapping,source_signals_labels_unchanged=True,
        no_final_predictions_or_scores_computed=True))
    del full,payload
    training=train(data,common/'model',epochs=30)
    assert training['optimizer_steps']==training['nonzero_gradient_steps']>0
    assert training['changed_parameter_tensors']>0
    # Independent NumPy forward calculation checks the actual saved neural weights.
    ckpt=torch.load(common/'model/model.pt',weights_only=True,map_location='cpu')
    dev=dict(np.load(data/'val.npz',allow_pickle=False))
    x=(dev['features'].astype(np.float32)-ckpt['mean'].numpy())/ckpt['scale'].numpy()
    state=ckpt['state_dict']
    for n in [0,2,4]:
        x=x@state[f'net.{n}.weight'].numpy().T+state[f'net.{n}.bias'].numpy()
        if n!=4:x=np.maximum(x,0)
    labels=np.asarray(ckpt['class_names'])[x.argmax(1)]
    accuracy=float(np.mean(labels==dev['labels']))
    assert abs(accuracy-training['selected_validation_accuracy'])<1e-12
    save(root/'audit/REFERENCE_FORWARD274.json',dict(passed=True,scope='development only',
        numpy_accuracy=accuracy,reported_accuracy=training['selected_validation_accuracy'],
        checkpoint_sha256=sha256(common/'model/model.pt'),test_accessed=False))
    first=config['cases'][0]['case']
    controls(root,first,data)
    prepared=[]
    for n,spec in enumerate(config['cases']):
        dest=root/spec['cohort']/'restored/HUST'
        shutil.copytree(common,dest,copy_function=os.link)
        if spec['case']!=first:
            shutil.copytree(root/'controls'/first,root/'controls'/spec['case'],copy_function=os.link)
        source_hashes={str(p.relative_to(common)):sha256(p) for p in common.rglob('*') if p.is_file()}
        assert all(sha256(dest/p)==h for p,h in source_hashes.items())
        row=dict(case=spec['case'],cohort=spec['cohort'],dataset='HUST',task_id='pdm_fault_012',
            rows=counts,groups={p:len(g) for p,g in groups.items()},
            optimizer_steps=training['optimizer_steps'] if n==0 else 0,
            reference_optimizer_steps_reused=training['optimizer_steps'] if n else 0,
            reference_validation_accuracy=accuracy,source_sha256=original['source_sha256'],
            split_files={p:sha256(data/(p+'.npz')) for p in ['train','val','test']},
            shared_source_model_hashes=source_hashes,all_groups_disjoint=True,
            real_observations=True,new_test_scores_read=False)
        save(root/'audit'/(spec['case']+'_PREPARATION.json'),row);prepared.append(row)
    result=dict(utc=utc(),passed=True,cases=prepared,new_model_sessions=0,new_test_scores_read=False,
        physical_splits=1,new_reference_models=1,
        unique_reference_optimizer_steps=training['optimizer_steps'],new_fixed_control_models=6)
    save(root/'PREPARATION274.json',result)
    project=Path(__file__).resolve().parents[1]
    save(project/config['result_directory']/'PREPARATION274.json',result)
    for name in [Path(__file__).name,'prepare_selection_expansion.py','prepare_classifier_bank.py',
                 'prepare_forge_coding_baseline.py','run_validation_transfer.py','real_fault_model.py','frozen_classifier.py']:
        shutil.copy2(project/'scripts'/name,root/'code'/name)
    save(root/'audit/PREPARATION_CODE274.json',dict(protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'),
        files={p.name:sha256(p) for p in (root/'code').glob('*.py')}))
    print(json.dumps(dict(passed=True,physical_splits=1,planned_stages=8,
        new_reference_optimizer_steps=training['optimizer_steps'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
