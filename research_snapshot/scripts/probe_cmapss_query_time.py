"""Query-time sensitivity of fixed actual Agent submissions; no new training."""
import argparse,json,os,time,math
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from controlled_feedback import save,sha,utc
from program_revision_runtime import Runtime


def main(root):
    started=time.monotonic();plan=json.loads((root/'PROTOCOL437.json').read_text())
    assert not (root/'HANDLE437.json').exists(), 'No implicit scientific restart'
    for path,digest in plan['code_hashes'].items(): assert sha(root/path)==digest
    for path,digest in plan['restored_hashes'].items(): assert sha(root/path)==digest
    save(root/'HANDLE437.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    raw_path=Path(plan['test_source']);assert sha(raw_path)==plan['test_sha256']
    raw=np.loadtxt(raw_path);assert raw.shape[1]==26
    columns=np.load(root/'restored/inputs/development.npz',allow_pickle=False)['columns']
    engines=sorted(set(raw[:,0].astype(int).tolist()));assert engines==list(range(1,101))
    jobs=[]
    for program in plan['programs']:
        for engine in engines:
            frame=raw[raw[:,0]==engine];assert np.array_equal(frame[:,1],np.arange(1,len(frame)+1))
            for fraction in plan['fractions']:
                cut=max(1,int(len(frame)*fraction))
                jobs.append((program,engine,fraction,frame[:cut].copy()))
    runtime=Runtime(root);runtime.deadline=time.monotonic()+plan['wall_seconds']
    def one(job):
        program,engine,fraction,frame=job
        spec=plan['programs'][program];folder=root/'predictions'/program/f'{engine}_{fraction}'
        folder.mkdir(parents=True,exist_ok=False);inp=folder/'observations.npz'
        np.savez_compressed(inp,values=frame,columns=columns)
        _,audit=runtime.command(root/spec['source'],root/spec['artifacts'],inp,folder/'cold','predict')
        values=np.asarray(json.loads((folder/'cold/predictions.json').read_text())['rul'],dtype=float)
        assert values.shape==(len(frame),) and np.isfinite(values).all() and np.all(values>=0)
        return dict(program=program,engine=engine,fraction=fraction,cycle=int(frame[-1,1]),
            prediction=float(values[-1]),input_sha256=sha(inp),audit_sha256=sha(folder/'cold/training_audit.json'))
    rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=plan['parallel']) as pool:
        futures={pool.submit(one,j):j[:3] for j in jobs}
        for f in as_completed(futures):
            try:rows.append(f.result())
            except Exception as error:failures.append(dict(job=futures[f],error_type=type(error).__name__,error=str(error)))
            if (len(rows)+len(failures))%20==0:
                save(root/'LIVE437.json',dict(utc=utc(),completed=len(rows),failed=len(failures),planned=len(jobs)))
    rows.sort(key=lambda r:(r['program'],r['engine'],r['fraction']))
    save(root/'PREDICTIONS_SEALED437.json',dict(utc=utc(),rows=rows,failures=failures,expected=len(jobs)))
    if failures:raise RuntimeError('Prediction failures retained; no complete-panel claim')
    for path,digest in plan['restored_hashes'].items():assert sha(root/path)==digest
    label=Path(plan['label_source']);assert sha(label)==plan['label_sha256']
    truth=np.loadtxt(label);assert truth.shape==(100,)
    ends={g:int(raw[raw[:,0]==g,1].max()) for g in engines}
    baseline={};allrows=[]
    for program,spec in plan['programs'].items():
        cached=json.loads((root/spec['endpoint_result']).read_text())['rows']
        for r in cached:
            assert r['cycle']==ends[r['engine']]
            baseline[(program,r['engine'])]=r['prediction']
            allrows.append(dict(program=program,engine=r['engine'],fraction=1.0,cycle=r['cycle'],
                prediction=r['prediction'],truth=float(truth[r['engine']-1]),prediction_reused=True))
    max_context=0.0
    for r in rows:
        spec=plan['programs'][r['program']]
        cached=json.loads((root/spec['cached_full_root']/str(r['engine'])/'predictions.json').read_text())['rul']
        context=abs(r['prediction']-cached[r['cycle']-1]);max_context=max(max_context,context)
        allrows.append(dict(**r,truth=float(truth[r['engine']-1]+ends[r['engine']]-r['cycle']),
            prediction_reused=False,absolute_context_change=context))
    scores={}
    for program in plan['programs']:
        scores[program]={}
        for fraction in plan['fractions']+[1.0]:
            selected=[r for r in allrows if r['program']==program and r['fraction']==fraction]
            assert len(selected)==100 and len({r['engine'] for r in selected})==100
            before=[abs(r['prediction']-r['truth']) for r in selected]
            after=[abs(min(r['prediction'],125)-r['truth']) for r in selected]
            raw_mae=math.fsum(before)/100;clipped=math.fsum(after)/100
            scores[program][str(fraction)]=dict(MAE=raw_mae,capped_prediction_MAE=clipped,
                cap_delta=clipped-raw_mae,MAE_numpy_check=float(np.mean(before)),
                cap_harm=math.fsum(max(a-b,0) for a,b in zip(after,before))/100,
                cap_gain=math.fsum(max(b-a,0) for a,b in zip(after,before))/100,
                truth_min=min(r['truth'] for r in selected),truth_max=max(r['truth'] for r in selected),
                mean_truth=math.fsum(r['truth'] for r in selected)/100,
                truth_above125=sum(r['truth']>125 for r in selected),
                clipping_wins=sum(a<b for a,b in zip(after,before)),
                clipping_losses=sum(a>b for a,b in zip(after,before)))
            assert math.isclose(raw_mae,float(np.mean(before)),abs_tol=1e-10)
    reversal={p:v['0.25']['cap_delta']>0 and v['1.0']['cap_delta']<0 for p,v in scores.items()}
    result=dict(utc=utc(),seconds=time.monotonic()-started,scores=scores,earliest_vs_endpoint_reversal=reversal,
        primary_reversal_gate=all(reversal.values()),max_context_change=max_context,
        actual_cold_predictions=len(rows),cached_endpoints=200,new_fits=0,new_LLM_calls=0,
        independent_confirmation=0,source_clusters=1,new_method_success=False,
        scope='Same physical100engines and both normal A435submissions;new earlier observation times,not new independent tasks.',
        labels_derived_from_official_endpoint_plus_elapsed_difference=True,
        query_labels_never_mounted=True,allrows=allrows,limitations=plan['limitations'])
    save(root/'RESULT437.json',result)
    save(root/'TERMINAL437.json',dict(utc=utc(),status='completed',actual_predictions=len(rows)))
    print(json.dumps({k:result[k] for k in ['seconds','scores','primary_reversal_gate','max_context_change','actual_cold_predictions']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:main(a.run.resolve())
    except Exception as e:
        save(a.run/'TERMINAL437.json',dict(utc=utc(),status='failed',error_type=type(e).__name__,error=str(e)))
        raise
