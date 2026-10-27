"""One fixed representation decomposition; reuse real training and audit kernels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from snu_order_control import read, save, sha, utc, packet
from rotor_patch_factorial import feature_matrix, source_file, worker, metrics
from audit_rotor_patch_factorial import forest_probability, summary


def matrices(data, cfg):
    original = feature_matrix(data)['all']
    assert np.array_equal(original[:, :75], original[:, 75:150])
    return {j: original[:, info['columns']] for j, info in cfg['jobs'].items()}


def prepare(root, cfg):
    for split in ['train', 'val']:
        data = packet(source_file(cfg, split))
        np.savez(root/'legal'/(split+'.npz'), **matrices(data, cfg),
                 y=data['labels'], groups=data['groups'], sample_ids=data['sample_ids'],
                 loads=data['operating_load_Nm'])
    train, val = packet(root/'legal/train.npz'), packet(root/'legal/val.npz')
    assert not set(train['groups']) & set(val['groups'])
    save(root/'PREPARED426.json', dict(utc=utc(), duplicate_raw_columns_exact=True,
         rows={s:len(d['y']) for s,d in [('train',train),('val',val)]},
         legal_hashes={s:sha(root/'legal'/(s+'.npz')) for s in ['train','val']},
         external_accessed=False))


def seal(root, cfg):
    reports = {j:read(root/'reports'/(j+'.json')) for j in cfg['jobs']}
    order = list(cfg['jobs'])
    selected = {v:min(order,key=lambda j:(-reports[j][v]['macro_f1'],order.index(j)))
                for v in ['oof','development']}
    save(root/'CHOICES426.json', dict(utc=utc(), selected=selected, legal=reports,
         model_hashes={p.name:sha(p) for p in (root/'models').glob('*.joblib')},
         evaluation_not_started=True))


def contrasts(values):
    loss = {j:v['record_mean_error'] for j,v in values.items()}
    return dict(spatial_gain_without_raw=loss['P']-loss['SP'],
                spatial_gain_with_raw=loss['RP']-loss['RSP'],
                raw_gain_without_spatial=loss['P']-loss['RP'],
                raw_gain_with_spatial=loss['SP']-loss['RSP'])


def evaluate(root, cfg):
    choice = read(root/'CHOICES426.json')
    data = packet(source_file(cfg,'test'))
    assert 'labels' not in data and 'groups' not in data
    X = matrices(data,cfg)
    np.savez(root/'external/features.npz',**X,sample_ids=data['sample_ids'])
    probabilities = {}
    for job in cfg['jobs']:
        p=root/'models'/f'{job}_5.joblib'
        assert sha(p)==choice['model_hashes'][p.name]
        probabilities[job]=joblib.load(p).predict_proba(X[job])
        np.save(root/'external'/(job+'_probability.npy'),probabilities[job])
    save(root/'PREDICTIONS_SEALED426.json',dict(utc=utc(),hashes={j:sha(root/'external'/(j+'_probability.npy'))for j in cfg['jobs']}))
    gold={e['sample_id']:e for e in read(source_file(cfg,'gold'))}
    y=np.array([gold[str(s)]['label'] for s in data['sample_ids']])
    groups=np.array([gold[str(s)]['group'] for s in data['sample_ids']])
    np.savez(root/'external/truth.npz',y=y,groups=groups,sample_ids=data['sample_ids'])
    final={j:metrics(y,p.argmax(1),groups)for j,p in probabilities.items()}
    effects=contrasts(final)
    views={v:contrasts({j:choice['legal'][j][v]for j in cfg['jobs']})for v in ['oof','development']}
    save(root/'RESULT426.json',dict(utc=utc(),final=final,final_contrasts=effects,
         legal_contrasts=views,selected=choice['selected'],
         prespecified_spatial_gain_both_contexts=effects['spatial_gain_without_raw']>0 and effects['spatial_gain_with_raw']>0,
         new_classifier_fits=24,new_Agent_stages=0,new_LLM_calls=0,
         source_clusters=1,final_previously_opened=True,formal_confirmation_opened=False,
         method_success_not_tested=True))
    print({j:round(v['accuracy'],6)for j,v in final.items()},flush=True)


def audit(root, cfg):
    started=time.monotonic();checks=rows=trees=0
    def check(value):
        nonlocal checks
        assert value,f'check {checks+1}'
        checks+=1
    def equal(a,b):check(bool(np.allclose(a,b,atol=1e-12,rtol=0)))
    choice,result=read(root/'CHOICES426.json'),read(root/'RESULT426.json')
    check(choice['utc']<read(root/'PREDICTIONS_SEALED426.json')['utc']<result['utc'])
    check(sha(Path(cfg['source'])/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256'])
    train,val=packet(root/'legal/train.npz'),packet(root/'legal/val.npz')
    external,truth=packet(root/'external/features.npz'),packet(root/'external/truth.npz')
    check(not set(train['groups'])&set(val['groups']))
    check(not set(train['groups'])&set(truth['groups']))
    check(np.array_equal(external['sample_ids'],truth['sample_ids']))
    # Audit block identities against independently concatenated output views.
    for data in [train,val,external]:
        equal(data['RSP'][:,:75],data['RP'][:,:75])
        equal(data['RSP'][:,75:225],data['SP'][:,:150])
        for j,offset in [('RSP',225),('RP',75),('SP',150)]:equal(data[j][:,offset:],data['P'])
    independent={}
    for job,info in cfg['jobs'].items():
        oof=np.zeros((len(train['y']),5))
        for fold,load in enumerate([*cfg['loads'],None]):
            p=root/'models'/f'{job}_{fold}.joblib';check(sha(p)==choice['model_hashes'][p.name])
            model=joblib.load(p);report=read(root/'models'/f'{job}_{fold}_REPORT.json')
            check(all(model.get_params()[k]==v for k,v in cfg['tree_recipes']['fixed'].items()))
            check(all(t.max_features_==53 for t in model.estimators_))
            check(model.n_features_in_==len(info['columns']))
            check(sum(t.tree_.node_count for t in model.estimators_)==report['actual_nodes'])
            check(len(model.estimators_)==800);trees+=len(model.estimators_)
            if load is not None:
                held=train['loads']==load;pred=forest_probability(model,train[job][held]);oof[held]=pred;rows+=len(pred)
                check(report['train_rows']==int((~held).sum()))
            else:
                pred=forest_probability(model,val[job]);rows+=len(pred);equal(pred,np.load(root/'predictions'/(job+'_val.npy')))
                dev=summary(val['y'],pred,val['groups'])
                pred=forest_probability(model,external[job]);rows+=len(pred);equal(pred,np.load(root/'external'/(job+'_probability.npy')))
                independent[job]=summary(truth['y'],pred,truth['groups'])
        equal(oof,np.load(root/'predictions'/(job+'_oof.npy')))
        for view,value in [('oof',summary(train['y'],oof,train['groups'])),('development',dev)]:
            for k,v in value.items():equal(v,choice['legal'][job][view][k])
        for k,v in independent[job].items():equal(v,result['final'][job][k])
    # Independent pair differences, no producer contrast helper.
    losses={j:independent[j]['record_mean_error']for j in independent}
    pairs={'spatial_gain_without_raw':('P','SP'),'spatial_gain_with_raw':('RP','RSP'),
           'raw_gain_without_spatial':('P','RP'),'raw_gain_with_spatial':('SP','RSP')}
    for name,(a,b)in pairs.items():equal(losses[a]-losses[b],result['final_contrasts'][name])
    check(rows==23480 and trees==19200)
    check(result['prespecified_spatial_gain_both_contexts']==(losses['P']>losses['SP'] and losses['RP']>losses['RSP']))
    save(root/'INDEPENDENT_AUDIT426.json',dict(utc=utc(),passed=True,checks=checks,
         parameter_forward_rows=rows,actual_trees=trees,classifier_fits_verified=24,audit_fits=0,
         seconds=time.monotonic()-started,result_sha256=sha(root/'RESULT426.json')))


def run(root,cfg):
    start=time.monotonic()
    def invoke(phase,job=None):
        label=phase+('_'+job if job else '');cmd=[sys.executable,'-B',str(Path(__file__)),phase,'--run',str(root)]
        if job:cmd+=['--job',job]
        begin=time.monotonic();save(root/'logs'/(label+'_STARTED.json'),dict(utc=utc(),command=cmd))
        with (root/'logs'/(label+'.stdout')).open('wb')as out,(root/'logs'/(label+'.stderr')).open('wb')as err:
            p=subprocess.run(cmd,stdout=out,stderr=err,timeout=cfg['budget'][phase+'_seconds'])
        status=dict(utc=utc(),exit_code=p.returncode,seconds=time.monotonic()-begin)
        save(root/'logs'/(label+'_TERMINAL.json'),status);print(label,status,flush=True)
        assert p.returncode==0,label
    try:
        invoke('prepare')
        with ThreadPoolExecutor(max_workers=4)as ex:
            futures=[ex.submit(invoke,'worker',j)for j in cfg['jobs']]
            for f in as_completed(futures):f.result()
        invoke('seal');invoke('evaluate');invoke('audit');status=dict(exit_code=0,science_terminal=True,audit_terminal=True)
    except Exception as exc:status=dict(exit_code=1,error=repr(exc),science_terminal=True)
    save(root/'TERMINAL426.json',dict(utc=utc(),seconds=time.monotonic()-start,**status))
    if status['exit_code']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','worker','seal','evaluate','audit','run']);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job');args=parser.parse_args()
    root=args.run.resolve();cfg=read(root/'PROTOCOL426.json')
    for name,digest in cfg['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest,name
    if args.phase=='worker':worker(root,cfg,args.job)
    else:globals()[args.phase](root,cfg)
