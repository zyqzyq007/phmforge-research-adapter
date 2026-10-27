"""Seal A278 choices, then cold-predict one causal prefix per fresh process.

No labels, other final prefixes, or model API are mounted into inference.
Every prediction is written before external labels are read for scoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import files, terminal, audit
from prepare_forge_coding_baseline import PYTHON, RUNTIME
from run_full_benchmark_campaign import clean_env


PROJECT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def seal(root, alternatives=None):
    assert not (root/'ALL_CHOICES_SEALED278.json').exists()
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert json.loads((root/'CAMPAIGN_STATUS.json').read_text())['status']=='all_sessions_terminal_waiting_choice_review_and_external_scores'
    variants=[];cohorts=[]
    bank=json.loads((root/'references/candidates/manifest.json').read_text())
    for row in bank['models']:
        variants.append(dict(id='bank_'+row['name'],kind='bank',checkpoint='candidates/'+row['checkpoint'],
            candidate_name=row['name'],scope='All preprovided frozen bank models, not test-selected'))
    variants.append(dict(id='reference_mlp',kind='bank',checkpoint='model.pt',
        candidate_name='reference_mlp',scope='Separate fixed gradient-trained reference'))
    for case in config['cases']:
        run=root/case['cohort'];job=terminal(run)[0];audit(run,reuse_existing=True)
        ws=run/job['workspace'];status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        selection=None;error=None
        try:
            selection=json.loads((ws/'selection.json').read_text())['candidate']
            assert selection in config['fixed_bank']+(['custom'] if case['arm']=='adaptive' else [])
        except (OSError,ValueError,KeyError,AssertionError) as e:error=type(e).__name__+': '+str(e)
        output_variant=None
        if error is None:
            if selection=='custom':
                output_variant='agent_'+case['cohort']
                variants.append(dict(id=output_variant,kind='program',workspace=str(ws.relative_to(root)),
                    entry='solution.py',cohort=case['cohort'],scope='Exact sealed Agent final program'))
            else:output_variant='bank_'+selection
        # Check the submitted wrappers as well as the declared bank choice.
        if (ws/'solution.py').is_file():
            variants.append(dict(id='wrapper_'+case['cohort'],kind='program',workspace=str(ws.relative_to(root)),
                entry='solution.py',cohort=case['cohort'],scope='Unmodified submitted wrapper/program; compare to declared bank choice'))
        cohorts.append(dict(case,job=job['id'],status=status['status'],response_valid=status.get('response_valid'),
            elapsed_seconds=status.get('elapsed_seconds'),selected=selection,selection_error=error,
            final_variant=output_variant,workspace=str(ws.relative_to(root)),submission_files=files(ws)))
    if alternatives:
        extra=json.loads(alternatives.read_text())
        assert extra['registered_before_final_scores'] and extra['new_test_scores_read'] is False
        variants.extend(extra['variants'])
        shutil.copy2(alternatives,root/'REGISTERED_ALTERNATIVES278.json')
    assert len({v['id'] for v in variants})==len(variants)
    for v in variants:
        assert v['id'].replace('_','').isalnum()
        if v['kind']=='bank':
            v['checkpoint_sha256']=sha256(root/'references'/v['checkpoint'])
        else:
            assert v['kind']=='program'
            v['workspace_files']=files(root/v['workspace'])
            if v.get('external_entry'):
                v['entry_sha256']=sha256(root/v['external_entry'])
            else:assert (root/v['workspace']/v['entry']).is_file()
    declaration=dict(utc=utc(),passed=True,all_native_sessions_terminal=True,
        no_final_predictions_or_scores_created=True,protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'),
        cohorts=cohorts,variants=variants,
        selection_controls=dict(development_selected=bank['development_selected'],
            training_cv_selected=min(bank['models'],key=lambda x:(x['training_cv']['mean_bearing_mae_minutes'],
                config['fixed_bank'].index(x['name'])))['name']),
        numerical_rules='All four final choices and registered alternatives are fixed; no refits or changed thresholds after this seal.')
    save(root/'ALL_CHOICES_SEALED278.json',declaration)
    save(PROJECT/'results/feedback278/ALL_CHOICES_SEALED278.json',declaration)
    shutil.copy2(__file__,root/'code/evaluate_xjtu_feedback.py')
    print(json.dumps(dict(sealed_cohorts=len(cohorts),registered_variants=len(variants))),flush=True)


def prepare_development_queries(root):
    destination=root/'evaluation_inputs/development'
    if destination.exists():return
    destination.mkdir(parents=True)
    raw=dict(np.load(root/'public_data/datasets/XJTU/val.npz',allow_pickle=False))
    rows=[]
    for i,sid in enumerate(raw['sample_ids']):
        start,end=map(int,raw['history_offsets'][i:i+2])
        data={k:(v if k=='dataset' else v[i:i+1]) for k,v in raw.items()
              if k!='labels' and not k.startswith('history_')}
        for key in ['history_features','history_elapsed_minutes','history_channel_rms','history_channel_max_abs']:
            data[key]=raw[key][start:end]
        data['history_offsets']=np.asarray([0,end-start])
        path=destination/(str(sid)+'.npz');np.savez_compressed(path,**data)
        rows.append(dict(sample_id=str(sid),unit_id=int(raw['unit_ids'][i]),scenario_id=int(raw['scenario_ids'][i]),
            actual_rul=float(raw['labels'][i]),input_path=str(path.relative_to(root))))
    save(destination/'labels.json',rows)


def cold(root, variant, split, observation, query_id, output_root):
    out=output_root/variant['id']/split/query_id;out.mkdir(parents=True)
    app=root/'clients-cache/cold278'/output_root.name/variant['id']/split/query_id
    app.parent.mkdir(parents=True,exist_ok=True)
    if variant['kind']=='program':
        source=root/variant['workspace']
        assert files(source)==variant['workspace_files'],'Registered program changed'
        shutil.copytree(source,app,symlinks=True,ignore=shutil.ignore_patterns('.runtime','.git','__pycache__'))
    else:app.mkdir()
    for folder in ['.runtime/tmp','.runtime/mpl']:(app/folder).mkdir(parents=True,exist_ok=True)
    cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net',
         '--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    for a,b in [(PYTHON,'/opt/python312'),(RUNTIME/'deps','/opt/phm_python/extra'),
        (root/'public_tools','/tools'),(root/'references','/reference'),(observation,'/input/query.npz')]:
        cmd+=['--ro-bind',str(a),b]
    cmd+=['--bind',str(app),'/app','--bind',str(out),'/output',
          '--proc','/proc','--dev','/dev','--chdir','/app']
    py='/opt/python312/bin/python3.12'
    if variant['kind']=='bank':
        assert sha256(root/'references'/variant['checkpoint'])==variant['checkpoint_sha256']
        command=[py,'-B','/tools/real_xjtu_rul_model.py','--model','/reference/'+variant['checkpoint']]
    else:
        entry='/app/'+variant['entry']
        if variant.get('external_entry'):
            path=root/variant['external_entry'];assert sha256(path)==variant['entry_sha256']
            cmd+=['--ro-bind',str(path),'/replay/entry.py'];entry='/replay/entry.py'
        command=[py,'-B',entry]+variant.get('arguments',[])
    cmd+=['--remount-ro','/']+command+['--input','/input/query.npz','--output','/output/predictions.json']
    save(out/'command.json',cmd)
    env=clean_env(root);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',
        PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',
        TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (out/'stdout').open('wb') as stdout,(out/'stderr').open('wb') as stderr:
        try:
            code=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=180).returncode
        except subprocess.TimeoutExpired:code='timeout'
    result=dict(variant=variant['id'],split=split,query_id=query_id,exit_code=code,
        input_sha256=sha256(observation),fresh_query_process=True,other_query_inputs_mounted=False,
        final_labels_mounted=False)
    p=out/'predictions.json'
    if p.exists():result['prediction_sha256']=sha256(p)
    save(out/'STATUS.json',result)
    shutil.rmtree(app)
    return result


def numerical_metrics(predictions, gold):
    lookup={x['sample_id']:x for x in predictions}
    assert len(lookup)==len(predictions)==len(gold)
    assert set(lookup)=={x['sample_id'] for x in gold}
    rows=[]
    for g in gold:
        p=lookup[g['sample_id']]
        assert int(p['unit_id'])==g['unit_id'] and int(p['scenario_id'])==g['scenario_id']
        value=float(p['predicted_rul']);assert math.isfinite(value) and value>=0
        rows.append(dict(g,predicted_rul=value,error_minutes=value-g['actual_rul']))
    by_unit=[]
    for unit in sorted({x['unit_id'] for x in rows}):
        group=[x for x in rows if x['unit_id']==unit];e=[x['error_minutes'] for x in group]
        item=dict(unit_id=unit,scenario_id=group[0]['scenario_id'],queries=len(e),
            mae_minutes=sum(abs(x) for x in e)/len(e),rmse_minutes=math.sqrt(sum(x*x for x in e)/len(e)),
            bias_minutes=sum(e)/len(e),overestimate_minutes=sum(max(x,0) for x in e)/len(e),
            underestimate_minutes=sum(max(-x,0) for x in e)/len(e))
        if 'lifetime_minutes' in group[0]:
            item.update(native_bearing=group[0]['native_bearing'],lifetime_minutes=group[0]['lifetime_minutes'],
                normalized_mae=item['mae_minutes']/group[0]['lifetime_minutes'],
                terminal_damage_locations=group[0]['terminal_damage_locations'])
        by_unit.append(item)
    errors=[x['error_minutes'] for x in rows]
    # Independent vectorized checks do not reuse the pure Python reductions.
    vector=np.asarray([lookup[x['sample_id']]['predicted_rul']-x['actual_rul'] for x in gold],float)
    ids=np.asarray([x['unit_id'] for x in gold])
    mae=sum(x['mae_minutes'] for x in by_unit)/len(by_unit)
    alternate=float(np.mean([np.abs(vector[ids==u]).mean() for u in np.unique(ids)]))
    assert math.isclose(mae,alternate,rel_tol=1e-12,abs_tol=1e-9)
    rmse=math.sqrt(sum(x*x for x in errors)/len(errors))
    assert math.isclose(rmse,float(np.sqrt(np.mean(vector**2))),rel_tol=1e-12,abs_tol=1e-9)
    return dict(unit='minutes',bearings=len(by_unit),queries=len(rows),mean_bearing_mae_minutes=mae,
        pooled_mae_minutes=sum(abs(x) for x in errors)/len(errors),pooled_rmse_minutes=rmse,
        mean_bearing_rmse_minutes=sum(x['rmse_minutes'] for x in by_unit)/len(by_unit),
        per_bearing=by_unit,per_query=rows,independent_metric_check=True)


def predict_all(root):
    seal=json.loads((root/'ALL_CHOICES_SEALED278.json').read_text())
    assert seal['passed'] and sha256(root/'CAMPAIGN_CONFIG.json')==seal['protocol_sha256']
    assert not (root/'evaluation278').exists(),'No automatic final prediction rerun'
    prepare_development_queries(root)
    output=root/'evaluation278';output.mkdir()
    tasks=[]
    for variant in seal['variants']:
        for split,folder in [('development',root/'evaluation_inputs/development'),('final',root/'evaluation_data/queries')]:
            for path in sorted(folder.glob('*.npz')):tasks.append((variant,split,path,path.stem))
    completed=[]
    def work(task):return cold(root,*task,output_root=output)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(work,tasks):
            completed.append(row)
            save(root/'EVALUATION_PROGRESS278.json',dict(utc=utc(),status='cold_prediction',
                completed=len(completed),planned=len(tasks),failures=sum(x['exit_code']!=0 for x in completed),
                final_scores_read=False))
    receipt=dict(utc=utc(),all_cold_processes_terminal=True,rows=completed,
        final_labels_not_yet_scored=True,registered_choices_sha256=sha256(root/'ALL_CHOICES_SEALED278.json'))
    save(root/'ALL_PREDICTIONS_SEALED278.json',receipt)
    print(json.dumps(dict(cold_runs=len(tasks),failures=sum(x['exit_code']!=0 for x in completed))),flush=True)


def score(root):
    assert not (root/'NUMERICAL_RESULTS278.json').exists()
    receipt=json.loads((root/'ALL_PREDICTIONS_SEALED278.json').read_text())
    assert receipt['all_cold_processes_terminal'] and receipt['registered_choices_sha256']==sha256(root/'ALL_CHOICES_SEALED278.json')
    sealed=json.loads((root/'ALL_CHOICES_SEALED278.json').read_text())
    gold={'development':json.loads((root/'evaluation_inputs/development/labels.json').read_text()),
          'final':json.loads((root/'hidden/final_labels.json').read_text())}
    rows=[]
    for variant in sealed['variants']:
        scores={}
        for split in ['development','final']:
            records=[x for x in receipt['rows'] if x['variant']==variant['id'] and x['split']==split]
            try:
                predictions=[]
                for r in records:
                    assert r['exit_code']==0, 'Cold process failed'
                    path=root/'evaluation278'/variant['id']/split/r['query_id']/'predictions.json'
                    assert sha256(path)==r['prediction_sha256']
                    result=json.loads(path.read_text())
                    assert result['unit']=='minutes'
                    assert len(result['predictions'])==1 and result['predictions'][0]['sample_id']==r['query_id']
                    predictions.extend(result['predictions'])
                scores[split]=dict(valid=True,metrics=numerical_metrics(predictions,gold[split]))
            except (OSError,KeyError,ValueError,AssertionError,TypeError) as e:
                scores[split]=dict(valid=False,error=type(e).__name__+': '+str(e))
        rows.append(dict(variant=variant,results=scores))
    result=dict(utc=utc(),research_adaptation=True,original_benchmark_score=False,
        planned_native_sessions=4,physical_allocation_count=1,final_bearings=3,final_queries=15,
        cohorts=sealed['cohorts'],selection_controls=sealed['selection_controls'],variants=rows,
        new_mechanisms_not_inferred_automatically=True)
    save(root/'NUMERICAL_RESULTS278.json',result)
    save(PROJECT/'results/feedback278/NUMERICAL_RESULTS278.json',result)
    print(json.dumps([dict(variant=x['variant']['id'],scores={s:(v['metrics']['mean_bearing_mae_minutes'] if v['valid'] else v['error']) for s,v in x['results'].items()}) for x in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','predict','score']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--alternatives',type=Path);a=p.parse_args();root=a.run.resolve()
    if a.mode=='seal':seal(root,a.alternatives)
    elif a.mode=='predict':predict_all(root)
    else:score(root)
