"""Freeze completed coding submissions, then replay and score externally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from evaluate_real_training_reviews import alive
from run_full_benchmark_campaign import clean_env, save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal(run):
    config=json.loads((run/'PROTOCOL187.json').read_text())
    controller=json.loads((run/config['controller_status_name']).read_text())
    assert controller.get('status')=='all_selected_terminal_no_scores'
    assert not alive(controller['controller_pid'])
    jobs=json.loads((run/'PLAN187.json').read_text())['jobs']
    for job in jobs:
        status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        assert status['status']!='running' and not alive(status['cli_pid'])
        assert not json.loads((run/'model_gateway'/job['id']/'status.json').read_text())['inflight']
    return jobs


def files(workspace):
    rows=[]
    for p in sorted(workspace.rglob('*')):
        relative=p.relative_to(workspace)
        if any(x in ['__pycache__','.git'] for x in relative.parts):continue
        if p.is_symlink():rows.append({'path':str(relative),'symlink':os.readlink(p)})
        elif p.is_file():rows.append({'path':str(relative),'bytes':p.stat().st_size,'sha256':sha(p)})
    return rows


def audit(run, reuse_existing=False):
    target=run/'CODING_CHOICE_AUDIT.json'
    jobs=terminal(run)
    if target.exists():
        assert reuse_existing, 'Choice audit already exists'
        assert not (run/'CODING_SCORE.json').exists(), 'Cannot reuse a prescore gate after scoring'
        previous=json.loads(target.read_text())
        assert previous['all_submissions_terminal'] and previous['new_hidden_test_scores_not_read']
        indexed={row['job']:row for row in previous['rows']}
        assert len(indexed)==len(previous['rows'])==len(jobs)
        assert set(indexed)=={job['id'] for job in jobs}
        for job in jobs:
            frozen=indexed[job['id']]
            status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
            assert frozen['status']==status['status'] and frozen['response_valid']==status.get('response_valid')
            assert frozen['files']==files(run/job['workspace']), 'Submission changed after early audit'
        print(json.dumps({'existing_choice_audit_verified':len(jobs),'original_audit_preserved':True}))
        return previous
    rows=[]
    for job in jobs:
        ident=job['id'];folder=run/'primary'/ident/ident
        status=json.loads((folder/'status.json').read_text())
        commands=[];errors=[];kinds={}
        # JSONL uses physical LF records; tool output may contain valid Unicode
        # separators that str.splitlines() would incorrectly treat as records.
        for n,line in enumerate((folder/'events.jsonl').open(),1):
            event=json.loads(line);item=event.get('item',{})
            if event['type']=='item.completed':
                kind=item.get('type');kinds[kind]=kinds.get(kind,0)+1
                if kind in ['command_execution','mcp_tool_call']:
                    commands.append({'event_line':n,**item})
            elif event['type'] in ['error','turn.failed']:errors.append(event)
        destination=run/'audit/closed_choices'/ident;destination.mkdir(parents=True)
        save(destination/'completed_tools.json',commands)
        inventory=files(run/job['workspace']);save(destination/'SUBMISSION_MANIFEST.json',inventory)
        rows.append({'job':ident,'dataset':job['dataset'],'status':status['status'],
                     'response_valid':status.get('response_valid'),'elapsed_seconds':status['elapsed_seconds'],
                     'completed_item_kinds':kinds,'errors':errors,'files':inventory,
                     'has_predictions':(run/job['workspace']/'test_predictions.json').is_file(),
                     'has_validation':(run/job['workspace']/'validation.json').is_file(),
                     'has_report':(run/job['workspace']/'report.md').is_file()})
    result={'utc':datetime.now(timezone.utc).isoformat(),'all_submissions_terminal':True,
            'new_hidden_test_scores_not_read':True,'rows':rows}
    save(target,result)
    print(json.dumps({'audited':len(rows),'rows':[{k:r[k] for k in ['job','status','has_predictions','has_validation','has_report']} for r in rows]}))


def namespace(run,job,workspace,output,command):
    argv=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts',
          '--unshare-net','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():argv+=['--ro-bind',p,p]
    for mount in json.loads(Path(job.get('evaluation_mounts',job['mounts'])).read_text())['mounts']:
        argv+=['--ro-bind',mount['source'],mount['destination']]
    argv+=['--bind',str(workspace),'/app','--bind',str(output),'/output',
           '--proc','/proc','--dev','/dev','--chdir','/app','--remount-ro','/']
    return argv+command


def score(run):
    selected=json.loads((run/'PLAN187.json').read_text())['jobs']
    if selected and all(x.get('data_format')=='femto_npz' for x in selected):
        return score_femto(run)
    import numpy as np
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
    from verified_benchmark_scoring import classification,aligned
    jobs=terminal(run);assert not (run/'CODING_SCORE.json').exists()
    audited=json.loads((run/'CODING_CHOICE_AUDIT.json').read_text())
    assert audited['new_hidden_test_scores_not_read']
    assert json.loads((run/'OBSERVABLE_REVIEW_COMPLETE.json').read_text())['reviewed_before_new_test_scores']
    indexed={x['job']:x for x in audited['rows']}
    def replay(job):
        ident=job['id'];ws=run/job['workspace'];frozen=indexed[ident]
        assert files(ws)==frozen['files'],'Submission changed after audit'
        destination=run/'evaluation'/ident;destination.mkdir(parents=True)
        app=destination/'app';shutil.copytree(ws,app,symlinks=True)
        for p in ['.runtime/tmp','.runtime/mpl']:(app/p).mkdir(parents=True,exist_ok=True)
        output=destination/'output';output.mkdir()
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',
            PYTHONDONTWRITEBYTECODE='1',PATH='/opt/python312/bin:/usr/bin:/bin',
            TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
            MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        observation='/data/datasets/'+job['dataset']+'/test.npz'
        final=['/opt/python312/bin/python3.12','-B','/app/solution.py','--input',observation,'--output','/output/predictions.json']
        reference=['/opt/python312/bin/python3.12','-B','-c',
            "import sys;sys.path.insert(0,'/tools');from real_fault_model import infer;infer('/reference/model.pt',sys.argv[1],'/output/reference.json')",observation]
        codes={}
        for kind,inner in [('final',final),('reference',reference)]:
            argv=namespace(run,job,app,output,inner);save(destination/(kind+'_command.json'),argv)
            with (destination/(kind+'.stdout')).open('wb') as out,(destination/(kind+'.stderr')).open('wb') as err:
                try:codes[kind]=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
                except subprocess.TimeoutExpired:codes[kind]='timeout'
        row={k:frozen[k] for k in ['job','dataset','status','response_valid','elapsed_seconds']}
        row.update(formal_submission=frozen['status']=='finished' and frozen['response_valid'] is True,cold_exit_codes=codes)
        gold=json.loads((run/'hidden'/job['dataset']/'labels.json').read_text())
        info=json.loads((run/'public_data'/ident/'datasets'/job['dataset']/'dataset.json').read_text())
        for kind,filename in [('final','predictions.json'),('reference','reference.json')]:
            try:
                assert codes[kind]==0,'Cold execution failed'
                rows=json.loads((output/filename).read_text())['predictions']
                assert all(isinstance(x['sample_id'],str) and any(type(x['predicted_label']) is type(c) and x['predicted_label']==c for c in info['class_names']) for x in rows),'ID/label type invalid'
                metrics=classification(rows,gold)
                pred,truth=aligned(rows,gold,'sample_id')
                y=[x['label'] for x in truth];p=[x['predicted_label'] for x in pred]
                independent={'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
                assert all(abs(metrics[k]-v)<1e-12 for k,v in independent.items())
                assert metrics['confusion_matrix']==confusion_matrix(y,p,labels=metrics['class_order']).tolist()
                row[kind]={'valid':True,'metrics':metrics,'independent_sklearn_agrees':True,'prediction_sha256':sha(output/filename)}
                if kind=='final' and (ws/'test_predictions.json').is_file():
                    submitted=json.loads((ws/'test_predictions.json').read_text())['predictions']
                    aligned(submitted,gold,'sample_id')
                    row['submitted_matches_cold_predictions']={x['sample_id']:x['predicted_label'] for x in submitted}=={x['sample_id']:x['predicted_label'] for x in rows}
            except (AssertionError,OSError,ValueError,KeyError,TypeError) as error:
                row[kind]={'valid':False,'error_type':type(error).__name__,'error':str(error)}
        if row['final']['valid'] and row['reference']['valid']:
            row['change_from_reference']={k:row['final']['metrics'][k]-row['reference']['metrics'][k] for k in ['accuracy','macro_f1','balanced_accuracy']}
        save(destination/'SCORE.json',row);return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(replay,jobs))
    result={'utc':datetime.now(timezone.utc).isoformat(),'research_adaptation':True,'official_benchmark_score':False,'denominator':len(jobs),'rows':rows,'confirmed_Agent_problem_inferred_automatically':False}
    save(run/'CODING_SCORE.json',result)
    print(json.dumps(result,ensure_ascii=False))


def bearing_metrics(predictions,gold):
    """Challenge percent-error score and independent bearing-level summaries."""
    from verified_benchmark_scoring import aligned
    p,g=aligned(predictions,gold,'sample_id')
    residual=[x['predicted_rul']-y['actual_rul'] for x,y in zip(p,g)]
    relative=[100*(y['actual_rul']-x['predicted_rul'])/y['actual_rul'] for x,y in zip(p,g)]
    return {'n':len(g),'unit':'seconds','mae':sum(abs(x) for x in residual)/len(g),
        'rmse':math.sqrt(sum(x*x for x in residual)/len(g)), 'bias':sum(residual)/len(g),
        'overestimates':sum(x>0 for x in residual),'underestimates':sum(x<0 for x in residual),
        'challenge_score':sum(math.exp(math.log(.5)*abs(x)/(5 if x<=0 else 20)) for x in relative)/len(g)}


def score_femto(run):
    import numpy as np
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    from verified_benchmark_scoring import aligned
    jobs=terminal(run);assert not (run/'CODING_SCORE.json').exists()
    assert json.loads((run/'OBSERVABLE_REVIEW_COMPLETE.json').read_text())['reviewed_before_new_test_scores']
    audited=json.loads((run/'CODING_CHOICE_AUDIT.json').read_text());index={r['job']:r for r in audited['rows']}
    source=Path(json.loads((run/'PROTOCOL187.json').read_text())['source_run'])
    # This read occurs only after both submissions and observable choices freeze.
    gold=json.loads((source/'hidden/FEMTO/labels.json').read_text())
    assert len(gold)==11 and sum(x['native_continuation_consistency'] for x in gold)==10
    def replay(job):
        ident=job['id'];ws=run/job['workspace'];frozen=index[ident]
        assert files(ws)==frozen['files']
        dest=run/'evaluation'/ident;dest.mkdir(parents=True)
        app=dest/'app';shutil.copytree(ws,app,symlinks=True)
        output=dest/'output';output.mkdir()
        for p in ['.runtime/tmp','.runtime/mpl']:(app/p).mkdir(parents=True,exist_ok=True)
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',
            PYTHONDONTWRITEBYTECODE='1',PATH='/opt/python312/bin:/usr/bin:/bin',
            TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
            MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        commands={'final':['/opt/python312/bin/python3.12','-B','/app/solution.py','--input',
            '/data/datasets/FEMTO/test.npz','--output','/output/predictions.json'],
            'reference':['/opt/python312/bin/python3.12','-B','/tools/real_bearing_rul_model.py',
                'infer','--model','/reference/model.pt','--input','/data/datasets/FEMTO/test.npz','--output','/output/reference.json']}
        row={k:frozen[k] for k in ['job','dataset','status','response_valid','elapsed_seconds']}
        row.update(formal_submission=frozen['status']=='finished' and frozen['response_valid'] is True,
                   results={})
        for kind,inner in commands.items():
            argv=namespace(run,job,app,output,inner);save(dest/(kind+'_command.json'),argv)
            with (dest/(kind+'.stdout')).open('wb') as out,(dest/(kind+'.stderr')).open('wb') as err:
                try:code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
                except subprocess.TimeoutExpired:code='timeout'
            result={'cold_exit_code':code};row['results'][kind]=result
            try:
                assert code==0,'Cold program execution failed'
                path=output/('predictions.json' if kind=='final' else 'reference.json')
                pred=json.loads(path.read_text())['predictions'];a,b=aligned(pred,gold,'sample_id')
                assert len({x['unit_id'] for x in a})==11
                for x,y in zip(a,b):
                    assert type(x['unit_id']) is int and type(x['scenario_id']) is int
                    assert (x['unit_id'],x['scenario_id'])==(y['unit_id'],y['scenario_id'])
                    assert isinstance(x['predicted_rul'],(float,int)) and not isinstance(x['predicted_rul'],bool)
                    assert math.isfinite(x['predicted_rul']) and x['predicted_rul']>=0
                complete=bearing_metrics(pred,gold)
                clean=[x for x in gold if x['native_continuation_consistency']]
                clean_ids={x['sample_id'] for x in clean}
                consistent=bearing_metrics([x for x in pred if x['sample_id'] in clean_ids],clean)
                per={}
                for scenario in [1,2,3]:
                    truth=[x for x in gold if x['scenario_id']==scenario];ids={x['sample_id'] for x in truth}
                    per[str(scenario)]=bearing_metrics([x for x in pred if x['sample_id'] in ids],truth)
                y=[x['actual_rul'] for x in b];p=[x['predicted_rul'] for x in a]
                assert abs(complete['mae']-mean_absolute_error(y,p))<1e-8
                assert abs(complete['rmse']-math.sqrt(mean_squared_error(y,p)))<1e-8
                result.update(valid=True,all11=complete,consistent10=consistent,per_scenario=per,
                    independent_sklearn_agrees=True,prediction_sha256=sha(path),aligned_bearing_predictions=a)
                if kind=='final' and (ws/'test_predictions.json').is_file():
                    saved=json.loads((ws/'test_predictions.json').read_text())['predictions'];c,d=aligned(saved,pred,'sample_id')
                    differences=[abs(x['predicted_rul']-y['predicted_rul']) for x,y in zip(c,d)]
                    result.update(submitted_matches_cold=bool(np.allclose([x['predicted_rul'] for x in c],[x['predicted_rul'] for x in d],rtol=1e-6,atol=1e-6)),max_submitted_cold_difference_seconds=max(differences))
            except (AssertionError,OSError,ValueError,KeyError,TypeError) as error:
                result.update(valid=False,error_type=type(error).__name__,error=str(error))
        if all(row['results'][k].get('valid') for k in ['final','reference']):
            row['change_from_reference']={scope:{m:row['results']['final'][scope][m]-row['results']['reference'][scope][m] for m in ['mae','rmse','bias']} for scope in ['all11','consistent10']}
        save(dest/'SCORE.json',row);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(replay,jobs))
    result={'research_adaptation':True,'official_original_PHMForge_score':False,'task':'pdm_rul_003',
        'sessions':len(jobs),'unit':'seconds','label_ambiguity_kept':'Bearing1_4 official339s retained; exclude from Agent-failure attribution',
        'rows':rows,'new_hard_Agent_problem_not_inferred_automatically':True}
    save(run/'CODING_SCORE.json',result)
    print(json.dumps({'rows':[{'job':r['job'],'status':r['status'],'results':{k:{m:v.get(m) for m in ['valid','all11','consistent10','submitted_matches_cold','error']} for k,v in r['results'].items()}} for r in rows]}))


def probe(run):
    """Exploratory input-layout check; original scores stay unchanged."""
    selected=json.loads((run/'PLAN187.json').read_text())['jobs']
    if selected and all(x.get('data_format')=='femto_npz' for x in selected):
        return probe_femto(run)
    import numpy as np
    from verified_benchmark_scoring import classification,aligned
    jobs=terminal(run);assert not (run/'LAYOUT_DIAGNOSTIC.json').exists()
    audit=json.loads((run/'CODING_CHOICE_AUDIT.json').read_text())
    frozen={r['job']:r for r in audit['rows']}
    scores=json.loads((run/'CODING_SCORE.json').read_text())
    baseline={r['job']:r for r in scores['rows']}
    save(run/'audit/LAYOUT_DIAGNOSTIC_PLAN.json',{
        'seed':252,'variants':['fixed_seed_permutation','two_contiguous_halves'],
        'same_observations_and_ids':True,'programs_unchanged':True,
        'posthoc_exploratory_diagnostic':True,'not_an_original_benchmark_failure_rule':True,
        'no_Agent_feedback_or_tuning':True})
    def one(job):
        ident=job['id'];ws=run/job['workspace'];assert files(ws)==frozen[ident]['files']
        dest=run/'layout_diagnostic'/ident;dest.mkdir(parents=True)
        app=dest/'app';shutil.copytree(ws,app,symlinks=True)
        output=dest/'output';output.mkdir()
        test=dict(np.load(run/'public_data'/ident/'datasets'/job['dataset']/'test.npz',allow_pickle=False))
        n=len(test['sample_ids']);index=np.arange(n)
        variants={'permuted':np.random.default_rng(252).permutation(index),
                  'half1':index[:n//2],'half2':index[n//2:]}
        result={'job':ident,'original':baseline[ident]['final']['metrics'],'variants':{}}
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',
            PYTHONDONTWRITEBYTECODE='1',PATH='/opt/python312/bin:/usr/bin:/bin',
            TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
            MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        records={}
        for variant,indices in variants.items():
            observations={k:(v[indices] if v.ndim>0 and len(v)==n else v) for k,v in test.items()}
            np.savez_compressed(output/(variant+'.npz'),**observations)
            command=namespace(run,job,app,output,['/opt/python312/bin/python3.12','-B','/app/solution.py',
                '--input','/output/'+variant+'.npz','--output','/output/'+variant+'.json'])
            save(dest/(variant+'_command.json'),command)
            with (dest/(variant+'.stdout')).open('wb') as out,(dest/(variant+'.stderr')).open('wb') as err:
                try:code=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
                except subprocess.TimeoutExpired:code='timeout'
            result['variants'][variant]={'exit_code':code}
            if code==0:records[variant]=json.loads((output/(variant+'.json')).read_text())['predictions']
        gold=json.loads((run/'hidden'/job['dataset']/'labels.json').read_text())
        original=json.loads((run/'evaluation'/ident/'output/predictions.json').read_text())['predictions']
        for name,rows in [('permuted',records.get('permuted')),('two_halves',
            records['half1']+records['half2'] if {'half1','half2'}<=records.keys() else None)]:
            if rows is None:continue
            metrics=classification(rows,gold);a,b=aligned(rows,original,'sample_id')
            variant_result=result['variants'].setdefault(name,{})
            variant_result.update(metrics=metrics,
                changed_predictions=sum(x['predicted_label']!=y['predicted_label'] for x,y in zip(a,b)),
                accuracy_change=metrics['accuracy']-result['original']['accuracy'])
        save(dest/'RESULT.json',result);return result
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,jobs))
    result={'posthoc_diagnostic':True,'same_test_not_unseen_confirmation':True,
            'not_a_new_official_failure_rule':True,'new_model_calls':0,'rows':rows}
    save(run/'LAYOUT_DIAGNOSTIC.json',result)
    print(json.dumps({'rows':[{'job':r['job'],'original_accuracy':r['original']['accuracy'],
        'variants':{k:{j:v.get(j) for j in ['changed_predictions','accuracy_change']} for k,v in r['variants'].items() if 'metrics' in v}} for r in rows]}))


def probe_femto(run):
    import numpy as np
    from verified_benchmark_scoring import aligned
    jobs=terminal(run);assert not (run/'LAYOUT_DIAGNOSTIC.json').exists()
    source=Path(json.loads((run/'PROTOCOL187.json').read_text())['source_run'])
    test=dict(np.load(source/'public/datasets/FEMTO/test.npz',allow_pickle=False))
    n=len(test['sample_ids']);assert n==11
    def one(job):
        ident=job['id'];dest=run/'layout_diagnostic'/ident;dest.mkdir(parents=True)
        app=dest/'app';shutil.copytree(run/job['workspace'],app,symlinks=True)
        output=dest/'output';output.mkdir()
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',
            PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
        indices=np.arange(n);parts={'permuted':np.random.default_rng(254).permutation(indices),
            'first_half':indices[:n//2],'second_half':indices[n//2:]}
        observations={}
        for name,index in parts.items():
            np.savez_compressed(output/(name+'.npz'),**{k:v[index] if v.ndim and len(v)==n else v for k,v in test.items()})
            cmd=namespace(run,job,app,output,['/opt/python312/bin/python3.12','-B','/app/solution.py',
                '--input','/output/'+name+'.npz','--output','/output/'+name+'.json'])
            save(dest/(name+'_command.json'),cmd)
            with (dest/(name+'.stdout')).open('wb') as out,(dest/(name+'.stderr')).open('wb') as err:
                completed=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180)
            assert completed.returncode==0
            observations[name]=json.loads((output/(name+'.json')).read_text())['predictions']
        baseline=json.loads((run/'evaluation'/ident/'output/predictions.json').read_text())['predictions']
        result={'job':ident,'variants':{}}
        for name,rows in [('permuted',observations['permuted']),('two_halves',observations['first_half']+observations['second_half'])]:
            a,b=aligned(rows,baseline,'sample_id');p=[x['predicted_rul'] for x in a];q=[x['predicted_rul'] for x in b]
            result['variants'][name]={'equivalent_within_numerical_precision':bool(np.allclose(p,q,rtol=1e-6,atol=1e-6)),
                'max_absolute_prediction_difference_seconds':float(np.max(np.abs(np.asarray(p)-np.asarray(q))))}
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,jobs))
    save(run/'LAYOUT_DIAGNOSTIC.json',{'same_observations_and_ids':True,'posthoc_diagnostic':True,'new_LLM_calls':0,'rows':rows})
    print(json.dumps(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['audit','score','probe'],required=True)
    a=p.parse_args();{'audit':audit,'score':score,'probe':probe}[a.mode](a.run.resolve())
