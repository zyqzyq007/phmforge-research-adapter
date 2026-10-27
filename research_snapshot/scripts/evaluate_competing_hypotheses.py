"""Sealed-choice, label-free cold prediction and independent numerical scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from controlled_feedback import save, sha, utc
from controlled_program_checks import CANDIDATES, designs, feature_kind, replay_check
from run_full_benchmark_campaign import clean_env


def cold_predict(root,block,candidate,input_file,output):
    import joblib
    import numpy as np
    with np.load(input_file,allow_pickle=False) as z:
        assert not {'labels','groups','actual_rul'} & set(z.files)
        data=dict(z)
    x=designs(root,data);bank=root/'bank'/block
    if candidate=='P11B':
        p=.9*joblib.load(bank/'P11/fold_full.joblib').predict_proba(x['spectral'])
        p+=.1*joblib.load(bank/'RAW_LR/fold_full.joblib').predict_proba(x['raw'])
    else:
        assert candidate in CANDIDATES
        p=joblib.load(bank/candidate/'fold_full.joblib').predict_proba(x[feature_kind(candidate)])
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    save(output,dict(candidate_id=candidate,predictions=[dict(sample_id=str(i),predicted_label=int(v))
        for i,v in zip(data['sample_ids'],p.argmax(1))],new_fits=0,input_labels_present=False))


def independent_score(truth,rows):
    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    keyed={r['sample_id']:r['predicted_label'] for r in rows}
    assert len(keyed)==len(rows)==len(truth['sample_ids']) and set(keyed)==set(truth['sample_ids'])
    prediction=np.asarray([keyed[str(i)] for i in truth['sample_ids']]);y=truth['labels']
    assert set(prediction.tolist())<=set(range(5))
    confusion=np.zeros((5,5),dtype=int)
    for a,b in zip(y,prediction):confusion[int(a),int(b)]+=1
    accuracy=float(np.trace(confusion)/confusion.sum())
    denominators=confusion.sum(0)+confusion.sum(1)
    f1=float(np.divide(2*np.diag(confusion),denominators,out=np.zeros(5),where=denominators!=0).mean())
    assert accuracy==accuracy_score(y,prediction)
    assert abs(f1-f1_score(y,prediction,labels=range(5),average='macro',zero_division=0))<1e-12
    assert np.array_equal(confusion,confusion_matrix(y,prediction,labels=range(5)))
    records=[]
    for group in sorted(set(truth['groups'])):
        indices=np.flatnonzero(truth['groups']==group);labels=set(y[indices].tolist())
        assert len(labels)==1
        counts=[sum(int(prediction[i])==c for i in indices) for c in range(5)]
        voted=max(range(5),key=lambda c:counts[c])
        assert voted==int(np.bincount(prediction[indices],minlength=5).argmax())
        records.append(dict(group=str(group),actual=int(next(iter(labels))),predicted=voted,
            load=float(truth['operating_load_Nm'][indices[0]]),windows=len(indices),window_correct=sum(y[i]==prediction[i] for i in indices)))
    correct=sum(r['actual']==r['predicted'] for r in records)
    for r in records:r['window_correct']=int(r['window_correct'])
    return dict(window_accuracy=accuracy,window_macro_f1=f1,window_correct=int(np.trace(confusion)),
        window_total=int(confusion.sum()),window_confusion=confusion.tolist(),records=records,
        record_correct=correct,record_total=len(records),record_error=1-correct/len(records),
        independent_metric_checks_passed=True)


def evaluate(root):
    import numpy as np
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    seal=json.loads((root/'ALL_CHOICES_SEALED.json').read_text())
    assert len(seal['episodes'])==8 and seal['final_scores_opened'] is False
    assert not (root/'FINAL_RESULTS325.json').exists(), 'Do not repeat opened evaluation'
    assert seal['protocol_sha256']==sha(root/'CAMPAIGN_CONFIG.json')
    assert sha(Path(__file__))==sha(root/'code'/Path(__file__).name)
    # Every retained file was sealed only after all model/gateway/broker writers stopped.
    for row in seal['files']:
        p=root/row['path']
        assert sha(p)==row['sha256'], 'Changed sealed file: '+row['path']
    evaluation=root/'evaluation';evaluation.mkdir(exist_ok=False)
    save(evaluation/'START.json',dict(utc=utc(),all_choices_previously_sealed=True,protocol_sha256=seal['protocol_sha256']))
    # Fixed selectors were sealed from genuine preparation validation. Measure
    # their full five-check execution cost now, before opening any final input.
    fixed_check_costs=[]
    for block in config['blocks']:
        started=time.monotonic();replays=0
        for cid in CANDIDATES:
            result=replay_check(root,block['id'],cid,evaluation/'fixed_checks'/block['id']/cid)
            replays+=result['actual_cold_models_replayed']
        fixed_check_costs.append(dict(block=block['id'],distinct_checks=5,cold_model_replays=replays,
            seconds=time.monotonic()-started,model_requests=0,new_fits=0,
            choice_previously_sealed=True,final_input_read=False))
    env=clean_env(root);env.update(PYTHONPATH='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps:'+str(Path(__file__).parent),PYTHONDONTWRITEBYTECODE='1')
    truth={}
    for block in config['blocks']:
        name=block['id']
        with np.load(root/'hidden'/name/'test.npz',allow_pickle=False) as z:truth[name]=dict(z)
        assert len(set(truth[name]['groups']))==20
        folder=evaluation/name;folder.mkdir()
        np.savez_compressed(folder/'observations.npz',**{k:v for k,v in truth[name].items() if k not in ['labels','groups']})
    def score_one(block,candidate):
        folder=evaluation/block/candidate;folder.mkdir();outputs=[]
        for repetition in [1,2]:
            output=folder/('predictions'+str(repetition)+'.json')
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'predict','--run',str(root),
                '--block',block,'--candidate',candidate,'--input',str(evaluation/block/'observations.npz'),'--output',str(output)]
            started=time.monotonic()
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
            (folder/('stderr'+str(repetition)+'.txt')).write_text(p.stderr)
            assert p.returncode==0,p.stderr[-1000:]
            rows=json.loads(output.read_text())['predictions'];outputs.append(rows)
            save(folder/('execution'+str(repetition)+'.json'),dict(command=command,exit_code=p.returncode,seconds=time.monotonic()-started))
        assert outputs[0]==outputs[1]
        result=independent_score(truth[block],outputs[0]);result['two_cold_replays_identical']=True
        save(folder/'score.json',result)
        return block,candidate,result
    scores={b['id']:{} for b in config['blocks']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(score_one,b['id'],c) for b in config['blocks'] for c in CANDIDATES]
        for f in futures:
            block,candidate,result=f.result();scores[block][candidate]=result
    outcomes=[]
    for episode in seal['episodes']:
        bank=scores[episode['block']];selected=bank[episode['selected']];old=bank['P00']['record_error']
        best=min(v['record_error'] for v in bank.values());loss=selected['record_error']
        path=root/'episodes'/episode['id'];gateway=path/'model_gateway'/episode['id']/'status.json'
        cost=json.loads(gateway.read_text()) if gateway.exists() else {}
        state=json.loads((path/'broker/STATE.json').read_text()) if (path/'broker/STATE.json').exists() else {}
        outcomes.append(dict(id=episode['id'],block=episode['block'],arm=episode['arm'],repeat=episode['repeat'],
            completed=episode['completed'],fallback=episode['fallback'],selected=episode['selected'],
            record_correct=selected['record_correct'],record_total=selected['record_total'],record_error=loss,
            selection_regret=loss-best,harm=max(loss-old,0),gain=max(old-loss,0),opportunity=old-best,
            window_accuracy=selected['window_accuracy'],window_macro_f1=selected['window_macro_f1'],
            model_requests=cost.get('requests',0),input_tokens=cost.get('input_tokens',0),output_tokens=cost.get('output_tokens',0),
            returned_model_ids=cost.get('returned_model_ids',[]),distinct_checks=len(state.get('checks',{})),
            check_replays=sum(v['actual_cold_models_replayed'] for v in state.get('checks',{}).values()),
            check_seconds=sum(v['elapsed_seconds'] for v in state.get('checks',{}).values())))
    summary={}
    for arm in ['ordinary','B']:
        rows=[r for r in outcomes if r['arm']==arm]
        assert len(rows)==4
        metrics={k:float(np.mean([r[k] for r in rows])) for k in ['selection_regret','record_error','harm','gain','window_accuracy','window_macro_f1','model_requests','distinct_checks','check_seconds']}
        opportunity=sum(r['opportunity'] for r in rows)
        summary[arm]=dict(metrics,completed=sum(r['completed'] for r in rows),episodes=4,
            harmful_updates=sum(r['harm']>1e-12 for r in rows),
            improvement_capture=sum(r['gain'] for r in rows)/opportunity if opportunity>0 else None)
    fixed=[]
    for block,choices in seal['fixed_controls'].items():
        for method in ['keep_P00','development_best','CV_best','complete_fixed_grid']:
            cid=choices[method];bank=scores[block]
            fixed.append(dict(block=block,method=method,selected=cid,record_correct=bank[cid]['record_correct'],
                record_total=bank[cid]['record_total'],record_error=bank[cid]['record_error'],
                selection_regret=bank[cid]['record_error']-min(x['record_error'] for x in bank.values())))
    result=dict(iteration='A325',completed_utc=utc(),protocol_sha256=seal['protocol_sha256'],
        all_choices_sealed_before_scoring=True,actual_PHM_component_fits=60,final_cold_processes=20,
        candidate_programs_per_block=5,physical_blocks=2,source_count=1,independent_motor_claim=False,
        candidate_scores=scores,episodes=outcomes,summary=summary,fixed_controls=fixed,fixed_check_costs=fixed_check_costs,
        B_minus_ordinary_regret=summary['B']['selection_regret']-summary['ordinary']['selection_regret'],
        B_minus_ordinary_record_error=summary['B']['record_error']-summary['ordinary']['record_error'],
        limits=['Related acquisitions of one laboratory motor','Fixed candidate selection,not free coding',
            'Two model trajectories per arm/block; descriptive only','Component recipes derived from known discovery case; final observations on acquisitions7-10 were unopened'])
    save(root/'FINAL_RESULTS325.json',result)
    out=Path(config['project'])/config['result_directory'];save(out/'FINAL_RESULTS325.json',result)
    print(json.dumps({'summary':summary,'B_minus_ordinary_regret':result['B_minus_ordinary_regret']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','predict']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--block');p.add_argument('--candidate');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='evaluate':evaluate(a.run.resolve())
    else:cold_predict(a.run.resolve(),a.block,a.candidate,a.input,a.output)
