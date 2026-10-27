"""Independent source, prediction, and selection audit for A453."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(run):
    started=time.monotonic();checks=0
    def check(value,message):
        nonlocal checks
        checks+=1
        assert value,message
    def equal(a,b,message):
        check(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),message)
    plan=read(run/'PROTOCOL453.json');pre=read(run/'PREPARATION453.json')
    for b in pre['bindings']:
        root=run/('corpus' if b['archive']==plan['corpus_archive'] else 'original')
        check(sha(root/b['member'])==b['sha256'],'input binding')
    for name,digest in read(run/'CODE_FREEZE453.json').items():
        check(sha(run/'code'/name)==digest,'scientific code unchanged')
    check(sha(run/'original/inputs/development.npz')==pre['development_sha256'],'data unchanged')
    with np.load(run/'original/inputs/development.npz',allow_pickle=False) as z:
        values,columns=z['values'],z['columns']
    meta=read(run/'original/inputs/metadata.json')
    raw=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks240/public/datasets/CMAPSS_FD001/train.txt')
    check(sha(raw)==meta['raw_train_sha256'],'original NASA train source')
    full=np.loadtxt(raw);ids=np.asarray(meta['development_engine_ids'])
    check(np.array_equal(values,full[np.isin(full[:,0],ids)]),'original development units and sensor rows')
    frames={int(i):values[values[:,0]==i] for i in ids}
    expected={(engine,max(1,int(len(frame)*fraction))):(frame,max(1,int(len(frame)*fraction)))
              for engine,frame in frames.items() for fraction in plan['prefix_fractions']}
    check(len(expected)==60,'all60 unique prefix queries')
    for frame in frames.values():
        check(np.array_equal(frame[:,1],np.arange(1,len(frame)+1)),'contiguous original cycles')
    program=run/'original'/plan['missing_program']
    for path,digest in read(program/'REPORT.json')['artifacts'].items():
        check(sha(program/path)==digest,'source and weights unchanged')
    terminal=read(run/'NEW_CHECK_TERMINAL453.json');new=read(run/'new_check/CHECK.json')
    check(terminal['status']=='complete' and terminal['new_predictions']==60,'prediction terminal')
    check(sha(run/'new_check/CHECK.json')==terminal['check_sha256'],'prediction seal')
    check(len(list((run/'new_check').glob('*/PROCESS.json')))==60,'60 actual processes')
    for row in new['rows']:
        frame,cut=expected[row['engine'],row['cut']];stem=f"e{row['engine']}_c{cut}"
        with np.load(run/'new_check'/(stem+'.npz'),allow_pickle=False) as z:
            check(set(z.files)=={'values','columns'},'no labels in mounted input')
            check(np.array_equal(z['values'],frame[:cut]) and np.array_equal(z['columns'],columns),'real prefix only')
        output=run/'new_check'/stem;proc=read(output/'PROCESS.json');audit=read(output/'training_audit.json')
        check(proc['returncode']==0 and proc['mode']=='predict' and '--unshare-net' in proc['command'],'successful isolated inference')
        check(audit['completed_estimator_fits']==audit['optimizer_steps']==0,'no fits in prediction')
        pred=read(output/'predictions.json')['rul'];check(len(pred)==cut and np.isfinite(pred).all(),'actual complete output')
        equal(row['prediction'],pred[-1],'endpoint from output')
        equal(row['truth'],len(frame)-cut,'legal development truth')
    available=read(run/'COMPLETE_EVIDENCE453.json');legal=read(run/'LEGAL_MEMBERS453.json')
    check(len(available)==14 and len(legal)==23,'all candidate denominator')
    for ident,item in available.items():
        report=item['check'];check(report==read(run/item['binding']),'complete check binding')
        check({(x['engine'],x['cut']) for x in report['rows']}==set(expected),'same queries for all14')
        group=defaultdict(list)
        for row in report['rows']:
            frame,cut=expected[row['engine'],row['cut']]
            equal(row['truth'],len(frame)-cut,'cache same development target')
            equal(row['cycle'],frame[cut-1,1],'cache same query time')
            group[row['engine']].append(abs(row['truth']-row['prediction']))
        computed={engine:math.fsum(errors)/len(errors) for engine,errors in group.items()}
        check(len(computed)==20,'20 engine groups')
        for row in report['engine_errors']:
            equal(row['MAE'],computed[row['engine']],'engine MAE independent')
        equal(report['engine_mean_MAE'],math.fsum(computed.values())/20,'engine mean')
        equal(report['MAE'],math.fsum(x for v in group.values() for x in v)/60,'query mean')
    seal=read(run/'ALL_CHOICES_SEALED453.json')
    check(sha(run/'DECISIONS453.json')==seal['decision_sha256'],'selection seal')
    check(sha(run/'COMPLETE_EVIDENCE453.json')==seal['evidence_sha256'],'evidence seal')
    choices={(x['case'],x['method']):x['selected'] for x in read(run/'DECISIONS453.json')}
    check(len(choices)==16,'all16 rule decisions')
    for pool in read(run/'VISIBLE_SELECTION453.json'):
        rows=[r for r in legal if r['episode']==pool['case']]
        check(pool['ids']==[r['program_id'] for r in rows],'original candidate order')
        prefix=[math.fsum(e['MAE'] for e in available[r['content_identity']]['check']['engine_errors'])/20 for r in rows]
        full=[math.fsum(e['MAE'] for e in r['legal_report']['development']['engine_errors'])/20 for r in rows]
        prior=prefix if all(r['additional_legal_check'] for r in rows) else full
        for method,scores in [('old_shared_secondary',prior),('complete_prefix_mean',prefix),('full_development_mean',full)]:
            for a,b in zip(scores,pool['rules'][method]):equal(a,b,'legal score reconstruction')
            eligible=[i for i,v in enumerate(scores) if v<=min(scores)+1e-12]
            old=pool['ids'].index('p00');selected=pool['ids'][old if old in eligible else eligible[0]]
            check(selected==choices[pool['case'],method],'independent selection')
        check(choices[pool['case'],'keep']=='p00','keep rule')
    result=read(run/'RESULT453.json');gold={r['id']:r['final_loss'] for r in read(run/'outcomes/analysis/HISTORICAL_OUTCOMES450.json')}
    for row in result['rows']:
        members=[r for r in legal if r['episode']==row['case']];ids=[r['program_id'] for r in members]
        losses=[gold[r['id']] for r in members];new=losses[ids.index(row['selected'])];old=losses[ids.index('p00')]
        equal(row['MAE'],new,'original final score');equal(row['pool_best_MAE'],min(losses),'complete pool oracle')
        equal(row['raw_selection_loss'],new-min(losses),'raw regret')
        scale=max(losses)-min(losses)
        for key,value in [('regret',new-min(losses)),('harm',max(0,new-old)),('gain',max(0,old-new))]:
            equal(row[key],value/scale if scale>1e-12 else 0.,'normalized metric')
    for method,summary in result['summary'].items():
        rows=[r for r in result['rows'] if r['method']==method];check(len(rows)==4,'all4 pools')
        for key,value in summary.items():equal(value,math.fsum(r[key] for r in rows)/4,'rule mean')
    a,b=result['summary']['complete_prefix_mean'],result['summary']['old_shared_secondary']
    gate=a['MAE']<b['MAE']-1e-12 and a['harm']<=b['harm']+1e-12 and a['gain']>=b['gain']-1e-12
    check(gate==result['ordinary_completion_control_gate'],'precommitted gate')
    old={r['case']:r for r in result['rows'] if r['method']=='old_shared_secondary'}
    new={r['case']:r for r in result['rows'] if r['method']=='complete_prefix_mean'}
    change=[dict(case=c,old_selected=old[c]['selected'],new_selected=new[c]['selected'],
        delta_MAE=new[c]['MAE']-old[c]['MAE']) for c in old]
    out=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,seconds=time.monotonic()-started,
        actual_new_prediction_processes=60,reused_unique_checks=13,new_PHM_fits=0,new_LLM_calls=0,new_Agent_stages=0,
        independent_confirmations=0,ordinary_control_gate=gate,choice_changes=change,
        residual_nonoptimal_pools=sum(r['raw_selection_loss']>1e-10 for r in new.values()))
    (run/'AUDIT453.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
