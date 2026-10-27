"""Freeze fresh record-group splits and numerical controls, without test scoring.

The raw observations and source labels are immutable. Old task outcomes played
no role in the split algorithm or control hyperparameters. These remain
exploratory allocations of public sources already known to the researcher.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import secrets
import shutil

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_verified_benchmarks import extract_bound
from real_fault_model import train
from frozen_classifier import estimators as ordinary_estimators

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVES = Path('/data/yyf-zyq/PHM_20261010')
CASES = [
    ('motor_a', 'ELECTRICMOTORVIBRATIONS', 'pdm_fault_015', 26801, 'a'),
    ('motor_b', 'ELECTRICMOTORVIBRATIONS', 'pdm_fault_015', 26802, 'b'),
    ('paderborn', 'PADERBORN', 'pdm_fault_011', 26803, 'a'),
    ('mfpt', 'MFPT', 'pdm_fault_013', 26804, 'a'),
    ('hust', 'HUST', 'pdm_fault_012', 26805, 'a'),
    ('mendeley', 'MENDELEY', 'pdm_fault_014', 26806, 'a'),
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze_protocol(root, config_path=None):
    config = dict(
        iteration='A268', frozen_utc=utc(), result_directory='results/selection268',
        authorization='用户2026-10-21明确要求：用更多的真实案例去验证该问题。',
        purpose='Six prespecified real-data classification cases from five sources; initial development and outcome-blind ordinary review, with fixed development-only numerical selection controls. No automatic retries.',
        planned_sessions=12, source_count=5, independent_source_count=5,
        independent_physical_case_count_not_claimed=True,
        prior_source_outcomes_known_to_researcher=True, untouched_confirmation=False,
        cohorts={k:dict(family='classification', specs=[dict(id='s268_'+case,
            task_id=task, dataset=name, split_seed=seed)
            for case,name,task,seed,cohort in CASES if cohort==k]) for k in ['a','b']},
        session_budget=dict(subject_timeout_seconds=2400, max_model_requests_per_subject=120,
            max_total_input_tokens_per_subject=16000000, max_output_tokens_per_request=32768,
            max_total_output_tokens_per_subject=262144, mcp_tool_timeout_seconds=180,
            max_concurrency=8, artifact_budget_bytes=40*1024**3),
        prompt_addendum='\nFor this research adaptation, the primary task metric is macro-F1; also report accuracy, balanced accuracy and per-class results. There is no known class-count quota for the test batch. Record the actual validation evidence for your choice, and retain runnable useful candidate artifacts when practical.\n',
        split_rule='Within each source class, shuffle complete existing source groups using the prespecified private seed; first floor(n/5), at least one, for development, next same number for test, remainder for fitting. No window crosses groups. Relabeling forbidden. New opaque IDs and private display order.',
        cases=[dict(case=c,dataset=d,task_id=t,split_seed=s,cohort=h) for c,d,t,s,h in CASES],
        reference='New MLP from scratch,30 epochs,2 numerical threads; epoch selected only on development accuracy; no old weights reused.',
        fixed_controls=dict(names=['lda_auto','lda_01','logistic_c1','svc_c1','rf300','hgb100'],
            selection='Fit only on training; select lexicographic development macro-F1, accuracy, then fixed name order. Freeze chosen identity before evaluating. Report same chosen algorithm trained on train only and train+development. Report all candidates as secondary, not oracle selected headline.',
            shared_features='Existing source-qualified numerical features; no test-derived transforms.',
            visible_to_agent=False, role='Same-information fixed workflow counterevidence; it is not itself an Agent-rejected candidate.'),
        estimands=['review minus initial','initial/review minus reference',
            'initial/review minus fixed development-selected control','all source-group results'],
        evidence_rule='A selection-loss case requires an actually attempted/available runnable candidate plus recorded rejection/retention rationale and independent replay. External control superiority alone is not evidence of discarding that candidate. Report all metric tradeoffs; highlight >=5 percentage points accuracy loss with no macro-F1 gain, or >=0.05 macro-F1 loss with no accuracy gain. Smaller losses remain descriptive.',
        qualification=['source hashes','complete group disjointness','new real reference fitting',
            'hidden-label filesystem/network isolation','native CLI discovers required MCP tools',
            'output budget','absence of target-prior metadata'],
        test_use='Unlabelled observations readable for inference; preprocessing/model choices use training/development only. Audit any test-driven selection separately as a protocol deviation.',
        scoring='No external scores until all twelve sessions are terminal, choices audited and candidate contrasts registered. No candidate parameters changed after scores.',
        interpretation='Windows are correlated; source groups and sources reported separately. Motor splits overlap and do not count as independent sources. Sources already used by researcher; no population failure-rate claim.',
        disk_estimate_gib=15, disk_cap_gib=40,
        archive=str(ARCHIVES/root.name), no_tmp=True, new_goal_claimed=False)
    if config_path is not None:
        config=json.loads(config_path.read_text())
        assert config['planned_sessions']==2*len(config['cases'])
        assert config['source_count']==len({r['dataset'] for r in config['cases']})
        assert all(r['cohort'] in config['cohorts'] for r in config['cases'])
    tag=config['iteration'].removeprefix('A')
    assert tag.isdigit() and not (root/'CAMPAIGN_CONFIG.json').exists()
    save(root/'CAMPAIGN_CONFIG.json',config)
    save(PROJECT/config['result_directory']/('PROTOCOL'+tag+'.json'),config)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    return config


def restore(root, names=None):
    old = ARCHIVES/'20261021T044345Z_contract261'
    newer = ARCHIVES/'20261021T031137Z_coding260'
    wanted=set(names or ['ELECTRICMOTORVIBRATIONS','MFPT','HUST','PADERBORN','MENDELEY'])
    old_names=[n for n in ['ELECTRICMOTORVIBRATIONS','MFPT','HUST'] if n in wanted]
    newer_jobs=[(n,j) for n,j in [('PADERBORN','forge260_pdm_fault_011_r4'),
        ('MENDELEY','forge260_pdm_fault_014_r5')] if n in wanted]
    assert wanted==set(old_names)|{n for n,j in newer_jobs}
    requests = [(old, root/'sources/261', [part for name in old_names for part in
        ['public/datasets/'+name+'/', 'hidden/'+name+'/labels.json',
         'audit/'+name+'_OBSERVATION_REPAIR261.json']]),
        (newer, root/'sources/260', [part for name,job in newer_jobs for part in
            ['public_data/'+job+'/datasets/'+name+'/', 'hidden/'+name+'/labels.json']])]
    requests=[row for row in requests if row[2]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        bindings = list(pool.map(lambda args:extract_bound(*args),requests))
    tag=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())['iteration'].removeprefix('A')
    save(root/'audit'/('SOURCE_BINDINGS'+tag+'.json'),sum(bindings,[]))
    sources = {}
    for name in old_names:
        sources[name]=(root/'sources/261/public/datasets'/name,
                       root/'sources/261/hidden'/name/'labels.json')
    for name,job in newer_jobs:
        sources[name]=(root/'sources/260/public_data'/job/'datasets'/name,
                       root/'sources/260/hidden'/name/'labels.json')
    return sources


def allocate(root, case, name, task, seed, cohort, source, labels):
    original = json.loads((source/'dataset.json').read_text())
    parts = {part:dict(np.load(source/(part+'.npz'),allow_pickle=False)) for part in ['train','val','test']}
    gold = {row['sample_id']:row for row in json.loads(labels.read_text())}
    test = parts['test']
    test['labels']=np.asarray([gold[str(i)]['label'] for i in test['sample_ids']])
    test['groups']=np.asarray([gold[str(i)]['group'] for i in test['sample_ids']])
    keys=set(parts['train'])
    assert all(set(v)==keys for v in parts.values()), 'Different observation fields across original partitions'
    full={}
    for key in keys:
        arrays=[v[key] for v in parts.values()]
        if arrays[0].ndim==0:
            assert all(np.array_equal(x,arrays[0]) for x in arrays)
            full[key]=arrays[0]
        else:
            assert all(len(v[key])==len(v['sample_ids']) for v in parts.values())
            full[key]=np.concatenate(arrays)
    assert len(set(full['sample_ids']))==len(full['sample_ids'])
    by_group=defaultdict(set)
    for g,y in zip(full['groups'],full['labels']):by_group[str(g)].add(y.item())
    assert all(len(v)==1 for v in by_group.values())
    by_label=defaultdict(list)
    for g,values in by_group.items():by_label[next(iter(values))].append(g)
    split={k:[] for k in parts}
    for label,groups in sorted(by_label.items(),key=lambda x:str(x[0])):
        selected=sorted(groups)
        assert len(selected)>=3
        random.Random(str(seed)+':'+name+':'+str(label)).shuffle(selected)
        n=max(1,len(selected)//5)
        split['val'].extend(selected[:n]);split['test'].extend(selected[n:2*n]);split['train'].extend(selected[2*n:])
    aliases={g:'group_'+secrets.token_hex(10) for g in by_group}
    identities=[secrets.token_hex(16) for _ in full['sample_ids']]
    selected_root=root/cohort/'restored'/name
    data=selected_root/'data';data.mkdir(parents=True,exist_ok=False)
    records=source/'records'
    if records.exists():
        shutil.copytree(records,data/'records',copy_function=__import__('os').link)
    counts={};mapping=[];new_gold=[]
    old_test_groups=set(parts['test']['groups'].astype(str))
    for part,groups in split.items():
        indices=np.flatnonzero(np.isin(full['groups'],groups)).tolist()
        secrets.SystemRandom().shuffle(indices)
        payload={k:(v[indices] if v.ndim else v) for k,v in full.items() if k not in ['sample_ids','groups','labels']}
        payload['sample_ids']=np.asarray([identities[i] for i in indices])
        if part!='test':
            payload.update(labels=full['labels'][indices],groups=np.asarray([aliases[str(full['groups'][i])] for i in indices]))
        else:
            new_gold=[dict(sample_id=identities[i],label=full['labels'][i].item(),
                group=aliases[str(full['groups'][i])]) for i in indices]
        np.savez_compressed(data/(part+'.npz'),**payload)
        counts[part]=len(indices)
        mapping.extend(dict(part=part,old_id=str(full['sample_ids'][i]),new_id=identities[i],
            old_group=str(full['groups'][i]),new_group=aliases[str(full['groups'][i])]) for i in indices)
    info={k:v for k,v in original.items() if k not in ['groups_per_class','train_validation_test_groups',
        'train_validation_groups','groups_by_split','rows_by_split','seed']}
    numerical_seed=json.loads((root/'CAMPAIGN_CONFIG.json').read_text()).get('numerical_seed',268)
    info.update(seed=numerical_seed, rows_by_split=counts, groups_by_split={p:len(v) for p,v in split.items()},
        train_validation_groups={p:[aliases[g] for g in split[p]] for p in ['train','val']},
        split_scope='New prespecified source-group allocation; original observations and labels unchanged. Entire original source groups kept together.',
        previous_model_weights_reused=False, identity_policy='Opaque IDs and private display order; no source ordinal or class prefix')
    save(data/'dataset.json',info)
    save(selected_root/'hidden/labels.json',new_gold)
    save(root/'audit'/(case+'_PRIVATE_ALLOCATION.json'),dict(case=case,seed=seed,
        split=split,identities=mapping,old_test_groups_reused=sorted(set(split['test'])&old_test_groups)))
    audit=train(data,selected_root/'model',epochs=30)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    # Training audit contains only fitting/development evidence. Do not score final observations here.
    row=dict(case=case,dataset=name,task_id=task,cohort=cohort,rows=counts,
        groups={p:len(v) for p,v in split.items()},source_sha256=original['source_sha256'],
        optimizer_steps=audit['optimizer_steps'],reference_validation_accuracy=audit['selected_validation_accuracy'],
        split_files={p:sha256(data/(p+'.npz')) for p in parts},
        all_groups_disjoint=all(not set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')]),
        real_observations=True,new_test_scores_read=False)
    save(root/'audit'/(case+'_PREPARATION.json'),row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return data,row


def controls(root,case,data):
    fit=dict(np.load(data/'train.npz',allow_pickle=False));dev=dict(np.load(data/'val.npz',allow_pickle=False))
    x,y=fit['features'].astype(float),fit['labels'];vx,vy=dev['features'].astype(float),dev['labels']
    seed=json.loads((root/'CAMPAIGN_CONFIG.json').read_text()).get('numerical_seed',268)
    estimators=ordinary_estimators(seed)
    scopes=json.loads((root/'CAMPAIGN_CONFIG.json').read_text()).get('fixed_controls',{}).get('fit_scopes',['train','train_dev'])
    assert scopes in [['train'],['train','train_dev']]
    out=root/'controls'/case;out.mkdir(parents=True)
    rows=[]
    for order,(name,estimator) in enumerate(estimators):
        row=dict(name=name,order=order)
        try:
            estimator.fit(x,y);p=estimator.predict(vx)
            row.update(valid=True,macro_f1=float(f1_score(vy,p,average='macro',zero_division=0)),
                accuracy=float(accuracy_score(vy,p)),balanced_accuracy=float(balanced_accuracy_score(vy,p)),
                mean_group_accuracy=float(np.mean([np.mean(p[dev['groups']==g]==vy[dev['groups']==g]) for g in set(dev['groups'])])))
            joblib.dump(estimator,out/(name+'_train.joblib'))
            if 'train_dev' in scopes:
                both=clone(estimator).fit(np.concatenate([x,vx]),np.concatenate([y,vy]))
                joblib.dump(both,out/(name+'_train_dev.joblib'))
        except Exception as error:
            row.update(valid=False,error_type=type(error).__name__,error=str(error))
        rows.append(row)
    valid=[r for r in rows if r['valid']];assert valid
    chosen=max(valid,key=lambda r:(r['macro_f1'],r['accuracy'],-r['order']))['name']
    save(out/'FROZEN_SELECTION.json',dict(utc=utc(),case=case,candidates=rows,selected=chosen,
        test_inputs_used_for_selection=False,test_labels_or_scores_read=False,
        selection_rule='development macro-F1, accuracy, then fixed model order',
        model_hashes={p.name:sha256(p) for p in out.glob('*.joblib')}))
    print(json.dumps(dict(case=case,fixed_control_selected=chosen,new_test_scores_read=False)),flush=True)


def main(root, config_path=None):
    config=freeze_protocol(root, config_path)
    tag=config['iteration'].removeprefix('A')
    sources=restore(root,{r['dataset'] for r in config['cases']})
    prepared=[]
    # Serial allocation avoids keeping several full waveform collections in memory.
    for spec in config['cases']:
        case,name,task,seed,cohort=(spec[k] for k in ['case','dataset','task_id','split_seed','cohort'])
        data,row=allocate(root,case,name,task,seed,cohort,*sources[name])
        controls(root,case,data);prepared.append(row)
    preparation='PREPARATION'+tag+'.json'
    save(root/preparation,dict(utc=utc(),passed=all(r['all_groups_disjoint'] for r in prepared),
        cases=prepared,new_model_sessions=0,new_test_scores_read=False))
    save(PROJECT/config['result_directory']/preparation,json.loads((root/preparation).read_text()))
    save(root/'audit'/('PREPARATION_CODE'+tag+'.json'),dict(protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'),
        files={name:sha256(PROJECT/'scripts'/name) for name in [Path(__file__).name,'run_validation_transfer.py','prepare_forge_coding_baseline.py','real_fault_model.py']}))
    print(json.dumps(dict(passed=True,cases=len(prepared),planned_sessions=config['planned_sessions'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path)
    args=p.parse_args();main(args.run.resolve(), args.config)
