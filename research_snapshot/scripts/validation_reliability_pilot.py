"""Real fixed-recipe mechanism pilot, separate from coding Agent evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from controlled_feedback import save,sha,utc
from program_revision_runtime import Runtime,load,subset,score
from validation_evidence_calibration import context,cv_weight,choose,decide

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261025T030800Z_source349')
OLD=Path('/data/yyf-zyq/PHM_20261010/20261025T012600Z_local347')
def read(path):return json.loads(Path(path).read_text())
def guard(root):
    for name,digest in read(root/'CODE_FREEZE349.json').items():assert sha(root/name)==digest,name
    if (root/'PREPARED349.json').exists():
        for name,digest in read(root/'PREPARED349.json')['bindings'].items():assert sha(root/name)==digest,name
def check_score(data,prediction):
    value=score(data,prediction);matrix=[[0]*4 for _ in range(4)]
    for group in sorted(set(data['groups'])):
        mask=data['groups']==group;truth=set(data['labels'][mask].tolist());assert len(truth)==1
        count=Counter(prediction[mask].tolist());label=min(count,key=lambda k:(-count[k],k))
        matrix[next(iter(truth))][label]+=1
    assert matrix==value['confusion']
    assert abs((1-sum(matrix[i][i] for i in range(4))/sum(map(sum,matrix)))-value['record_error'])<1e-12
    return value

def freeze(root):
    assert not (root/'CODE_FREEZE349.json').exists()
    assert not list(root.rglob('training_audit.json'))
    assert read(root/'EFFECTIVE_PROTOCOL349.json')['method']['pair_weight']=='Exactinverse-regret:validation_regret/(CV_regret+validation_regret);bothzero0.5'
    for name in ['validation_reliability_pilot.py','validation_evidence_calibration.py','program_revision_runtime.py','controlled_feedback.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    shutil.copytree(OLD/'common_tools',root/'common_tools')
    # Pure arithmetic checks,not fabricated PHM samples or experimental outcomes.
    assert choose('old','new',0)=='old' and choose('old','new',-.25)=='new'
    assert cv_weight(0,.25)==1 and cv_weight(.25,0)==0 and cv_weight(0,0)==.5
    assert choose('old','new',cv_weight(0,.25)*0+(1-cv_weight(0,.25))*(-.25))=='old'
    for path in [*list((root/'code').glob('*.py')),*list((root/'programs').rglob('*.py'))]:compile(path.read_text(),str(path),'exec')
    names=[p for directory in ['code','common_tools','programs'] for p in (root/directory).rglob('*') if p.is_file()]
    names += [root/n for n in ['PROTOCOL349.json','EFFECTIVE_PROTOCOL349.json','IMPLEMENTATION_AMENDMENT349.json','SOURCE_SPEC349.json','SOURCE_METADATA_SELECTION349.json','PRIOR_RAW_SHA349.json']]
    save(root/'CODE_FREEZE349.json',{str(p.relative_to(root)):sha(p) for p in names})
    save(root/'SOFTWARE_QUALIFICATION349.json',dict(utc=utc(),arithmetic_and_compile_passed=True,
        fixture_scope='Purearithmeticzero/tie/signeddecisionchecks,noPHMdataorperformanceclaim',new_fits=0,new_predictions=0,new_LLM_calls=0))
    print(json.dumps(dict(status='implementation_frozen_before_first_fit',files=len(names))))

def prepare(root):
    guard(root);assert not (root/'PREPARED349.json').exists()
    receipt=read(SOURCE/'ARCHIVE_VERIFIED.json');assert sha(SOURCE/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']
    q=read(SOURCE/'QUALIFICATION349.json');assert q['passed'] and q['whole_acquisitions']==28
    identities=read(SOURCE/'qualification/IDENTITIES349.json');raw={x['raw_sha256'] for x in identities}
    assert len(raw)==28 and not raw.intersection(read(root/'PRIOR_RAW_SHA349.json')['raw_sha256'])
    inputs=root/'inputs';inputs.mkdir();data={}
    for role in ['train','development','independent_check']:
        path=SOURCE/'qualification'/(role+'.npz');assert sha(path)==q['files'][path.name]['sha256']
        shutil.copy2(path,inputs/path.name);data[role]=load(inputs/path.name)
    conditions=sorted(set(data['train']['condition_id'].tolist()));assert len(conditions)==3
    assert len(set(data['train']['groups']))==12
    assert all(len(set(data[r]['groups']))==4 for r in ['development','independent_check'])
    contexts={};(inputs/'contexts').mkdir()
    for n in [1,2,3]:
        for chosen in combinations(conditions,n):
            name=context(chosen);path=inputs/'contexts'/(name+'.npz')
            values=subset(data['train'],np.isin(data['train']['condition_id'],chosen))
            assert set(values['labels'].tolist())==set(range(4))
            np.savez_compressed(path,**values);contexts[name]=dict(conditions=list(chosen),path=str(path),sha256=sha(path))
    save(root/'CONTEXT_INPUTS349.json',contexts)
    cfg=dict(input_root=str(inputs),common_tools=str(root/'common_tools'),fit_seconds=120,predict_seconds=45,program_seconds=1800)
    save(root/'ENGINE_CONFIG.json',cfg)
    save(root/'PREPARED349.json',dict(utc=utc(),source_archive=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],
        raw_records=28,raw_overlap_with203_prior_or_reserved=0,train_conditions=conditions,
        bindings={str(p.relative_to(root)):sha(p) for p in [root/'ENGINE_CONFIG.json',root/'CONTEXT_INPUTS349.json',*inputs.rglob('*.npz')]},
        final_observations_loaded=False,new_fits=0,new_predictions=0,new_LLM_calls=0))
    print(json.dumps(dict(status='real_inputs_qualified',training_contexts=len(contexts),prior_raw_overlap=0)))

def run_context(root,pid,chosen,data,deadline):
    directory=root/'programs'/pid/'contexts'/context(chosen)
    record=directory/'CONTEXT.json';source=root/'programs'/pid/'source'
    packet=read(root/'CONTEXT_INPUTS349.json')[context(chosen)]
    if record.exists():
        cached=read(record);assert cached['training_input_sha256']==packet['sha256']
        for name,digest in cached['model_artifacts'].items():assert sha(directory/name)==digest
        return cached
    directory.mkdir(parents=True,exist_ok=False);runtime=Runtime(root);runtime.deadline=deadline
    _,audit=runtime.command(source,None,Path(packet['path']),directory/'fit','fit')
    targets={'train_'+str(int(c)):subset(data['train'],data['train']['condition_id']==c)
             for c in sorted(set(data['train']['condition_id'])) if c not in chosen}
    targets.update(development=data['development'],independent_check=data['independent_check'])
    results={};vectors={}
    for target,observed in targets.items():
        prediction,_=runtime.predict(directory/'predictions'/target,source,directory/'fit/artifacts',observed)
        results[target]=check_score(observed,prediction);vectors[target]=prediction
    identity=None
    if len(chosen)==3:
        mask=data['development']['groups']==sorted(set(data['development']['groups']))[0]
        renamed,_=runtime.predict(directory/'identity',source,directory/'fit/artifacts',subset(data['development'],mask),rename=True)
        assert np.array_equal(renamed,vectors['development'][mask]);identity=True
    bindings={str(p.relative_to(directory)):sha(p) for p in (directory/'fit/artifacts').rglob('*') if p.is_file()}
    value=dict(utc=utc(),status='complete',training_conditions=chosen,training_input_sha256=packet['sha256'],
        scores=results,identity_invariance=identity,completed_estimator_fits=audit['completed_estimator_fits'],
        optimizer_steps=audit['optimizer_steps'],model_artifacts=bindings,final_data_read=False)
    save(record,value);return value

def preflight(root):
    guard(root);assert not (root/'PREFLIGHT349.json').exists()
    started=time.time();save(root/'PREFLIGHT_STARTED349.json',dict(utc=utc(),time=started,pid=os.getpid()))
    data={k:load(root/'inputs'/(k+'.npz')) for k in ['train','development','independent_check']}
    conditions=read(root/'PREPARED349.json')['train_conditions']
    value=run_context(root,'baseline',conditions,data,time.monotonic()+1800)
    assert value['identity_invariance'] and value['completed_estimator_fits']>0
    save(root/'PREFLIGHT349.json',dict(utc=utc(),status='passed',real_fit_context='baseline/full',
        fitted_on_training_only=True,all_program_predictions_external_and_label_isolated=True,
        predictions=9,fit_entries=1,counts_are_part_of_original42_and654=True,seconds=time.time()-started))
    print(json.dumps(dict(status='real_training_and_prediction_preflight_passed',fit_entries=1,prediction_processes=9)))

def legal(root):
    guard(root);assert read(root/'PREFLIGHT349.json')['status']=='passed'
    assert not (root/'LEGAL_STARTED349.json').exists()
    cfg=read(root/'EFFECTIVE_PROTOCOL349.json');started=time.monotonic()
    save(root/'LEGAL_STARTED349.json',dict(utc=utc(),pid=os.getpid(),new_LLM_calls=0,final_data_read=False))
    save(root/'HANDLE349.json',dict(utc=utc(),pid=os.getpid(),status='running',phase='legal_real_training_and_prediction'))
    data={k:load(root/'inputs'/(k+'.npz')) for k in ['train','development','independent_check']}
    conditions=read(root/'PREPARED349.json')['train_conditions']
    def one(program):
        pid=program['id'];begin=time.monotonic();remaining=1800
        if pid=='baseline':remaining-=time.time()-read(root/'PREFLIGHT_STARTED349.json')['time']
        deadline=min(begin+remaining,started+cfg['budget']['legal_wall_seconds'])
        rows={}
        try:
            for n in [3,2,1]:
                for chosen in combinations(conditions,n):
                    rows[context(chosen)]=run_context(root,pid,list(chosen),data,deadline)
                    save(root/'programs'/pid/'PROGRESS.json',dict(completed_contexts=list(rows),final_data_read=False))
            value=dict(status='complete',contexts=rows,seconds=time.monotonic()-begin)
        except Exception as error:value=dict(status='program_error',completed_contexts=rows,
            error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-begin)
        save(root/'programs'/pid/'LEGAL_RESULT.json',value);return pid,value
    values={}
    with ThreadPoolExecutor(max_workers=cfg['budget']['max_parallel_programs']) as pool:
        for future in as_completed([pool.submit(one,p) for p in cfg['programs']]):
            pid,value=future.result();values[pid]=value;save(root/'LEGAL_PROGRESS349.json',values)
    fits=[p for p in (root/'programs').rglob('training_audit.json') if read(p)['mode']=='fit']
    predicts=[p for p in (root/'programs').rglob('training_audit.json') if read(p)['mode']=='predict']
    assert len(fits)<=42 and len(predicts)<=558
    complete=all(v['status']=='complete' for v in values.values())
    if complete:assert len(fits)==42 and len(predicts)==558
    save(root/'LEGAL_RESULTS349.json',dict(utc=utc(),status='complete' if complete else 'incomplete',programs=values,
        actual_fit_entries=len(fits),completed_estimator_fits=sum(read(p)['completed_estimator_fits'] for p in fits),
        actual_prediction_processes=len(predicts),seconds=time.monotonic()-started,new_LLM_calls=0,final_data_read=False))
    save(root/'HANDLE349.json',dict(utc=utc(),pid=os.getpid(),status='terminal',phase='legal_finished',complete=complete))
    print(json.dumps(dict(status='complete' if complete else 'incomplete',fit_entries=len(fits),predictions=len(predicts))))

def seal(root):
    guard(root);assert not (root/'DECISIONS_SEALED349.json').exists()
    handle=read(root/'HANDLE349.json');assert handle['status']=='terminal' and not Path('/proc',str(handle['pid'])).exists()
    legal_data=read(root/'LEGAL_RESULTS349.json');assert legal_data['status']=='complete'
    cfg=read(root/'EFFECTIVE_PROTOCOL349.json');conditions=read(root/'PREPARED349.json')['train_conditions']
    table={p:v['contexts'] for p,v in legal_data['programs'].items()}
    choices=decide(table,cfg['pairs'],conditions)
    artifacts={str(p.relative_to(root)):sha(p) for p in (root/'programs').rglob('*') if p.is_file() and 'artifacts' in p.parts}
    save(root/'DECISIONS_SEALED349.json',dict(utc=utc(),decisions=choices,legal_result_sha256=sha(root/'LEGAL_RESULTS349.json'),
        frozen_code_sha256=sha(root/'CODE_FREEZE349.json'),artifacts=artifacts,final_data_read=False))
    save(PROJECT/'results/validation349/DECISIONS_SEALED349.json',read(root/'DECISIONS_SEALED349.json'))
    print(json.dumps(dict(status='all_choices_sealed',informative_disagreements=choices['informative_view_disagreements'],final_data_read=False)))

def evaluate(root):
    guard(root);assert not (root/'FINAL_STARTED349.json').exists()
    sealed=read(root/'DECISIONS_SEALED349.json');assert sealed['frozen_code_sha256']==sha(root/'CODE_FREEZE349.json')
    assert sealed['legal_result_sha256']==sha(root/'LEGAL_RESULTS349.json')
    for name,digest in sealed['artifacts'].items():assert sha(root/name)==digest
    cfg=read(root/'EFFECTIVE_PROTOCOL349.json');conditions=read(root/'PREPARED349.json')['train_conditions']
    path=SOURCE/'qualification/final.npz';assert sha(path)==read(SOURCE/'QUALIFICATION349.json')['files']['final.npz']['sha256']
    save(root/'FINAL_STARTED349.json',dict(utc=utc(),choices_sha256=sha(root/'DECISIONS_SEALED349.json'),
        final_input_sha256=sha(path),no_Actors_in_this_numerical_pilot=True))
    data=load(path);assert len(set(data['groups']))==8;started=time.monotonic()
    def one(program):
        pid=program['id'];runtime=Runtime(root);runtime.deadline=started+cfg['budget']['final_wall_seconds']
        source=root/'programs'/pid/'source';artifacts=root/'programs'/pid/'contexts'/context(conditions)/'fit/artifacts'
        folder=root/'evaluation'/pid
        try:
            a,aa=runtime.predict(folder/'forward',source,artifacts,data)
            b,bb=runtime.predict(folder/'repeat',source,artifacts,data)
            assert np.array_equal(a,b)
            result=dict(status='complete',**check_score(data,a),prediction_processes=len(aa+bb),cold_repeat_equal=True)
        except Exception as error:result=dict(status='prediction_error',error_type=type(error).__name__,error=str(error))
        save(folder/'RESULT.json',result);return pid,result
    results={}
    with ThreadPoolExecutor(max_workers=cfg['budget']['max_parallel_programs']) as pool:
        for future in as_completed([pool.submit(one,p) for p in cfg['programs']]):
            pid,value=future.result();results[pid]=value;save(root/'FINAL_PROGRESS349.json',results)
    complete=all(x['status']=='complete' for x in results.values());pair_rows=[];aggregate={}
    if complete:
        for row in sealed['decisions']['pairs']:
            old=results[row['incumbent']]['record_error'];oracle=min(old,results[row['candidate']]['record_error'])
            opportunity=old-oracle;methods={}
            for name,choice in row['methods'].items():
                error=results[choice['program']]['record_error']
                methods[name]=dict(**choice,record_error=error,selection_loss=error-oracle,harm=max(error-old,0),
                    gain=max(old-error,0),gain_capture=max(old-error,0)/opportunity if opportunity>0 else None)
            pair_rows.append(dict(id=row['id'],incumbent=row['incumbent'],candidate=row['candidate'],
                                  incumbent_error=old,pair_oracle_error=oracle,methods=methods))
        for name in pair_rows[0]['methods']:
            rows=[p['methods'][name] for p in pair_rows]
            gains=[x['gain_capture'] for x in rows if x['gain_capture'] is not None]
            aggregate[name]=dict(mean_record_error=sum(x['record_error'] for x in rows)/3,
                mean_selection_loss=sum(x['selection_loss'] for x in rows)/3,mean_harm=sum(x['harm'] for x in rows)/3,
                mean_gain_capture=sum(gains)/len(gains) if gains else None,improvement_pairs=len(gains))
    controls=cfg['method']['fixed_controls'];best=min(controls,key=lambda n:(aggregate[n]['mean_selection_loss'],aggregate[n]['mean_harm'],n)) if complete else None
    advance=False
    if complete:
        method=aggregate['pair_specific'];reference=aggregate[best]
        advance=(sealed['decisions']['informative_view_disagreements']>0 and
            all(method['mean_selection_loss']<aggregate[n]['mean_selection_loss']-1e-12 for n in controls+['shared_calibration']) and
            method['mean_harm']<=reference['mean_harm']+1e-12 and
            method['mean_gain_capture'] is not None and method['mean_gain_capture']>=reference['mean_gain_capture']-1e-12)
    guard(root)
    for name,digest in sealed['artifacts'].items():assert sha(root/name)==digest
    processes=list((root/'programs').rglob('PROCESS.json'))+list((root/'evaluation').rglob('PROCESS.json'))
    final_processes=list((root/'evaluation').rglob('PROCESS.json'))
    assert len(processes)<=42+654 and len(final_processes)<=96
    value=dict(utc=utc(),iteration='A349',status='complete' if complete else 'incomplete',programs=results,pairs=pair_rows,
        methods=aggregate,best_fixed_control_descriptive_only=best,advance_to_new_coding_trial=advance,
        informative_view_disagreements=sealed['decisions']['informative_view_disagreements'],
        all_choices_sealed_before_final=True,original_source_and_weights_unchanged=True,
        final_prediction_processes=len(final_processes),all_fit_and_predict_processes=len(processes),
        legal_cost=read(root/'LEGAL_RESULTS349.json'),new_LLM_calls=0,independent_confirmation_scenarios=0,
        physical_development_cohorts=1,physical_rigs=1,three_related_program_updates_not_three_scenarios=True,
        method_novelty_established=False,majority_goal_complete=False,seconds=time.monotonic()-started,
        do_not_retrain_or_tune_on_this_opened_final=True)
    save(root/'RESULT349.json',value);save(PROJECT/'results/validation349/RESULT349.json',value)
    print(json.dumps(dict(status=value['status'],advance_to_new_coding_trial=advance,methods=aggregate,goal_complete=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','prepare','preflight','legal','seal','evaluate']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.run.resolve())
