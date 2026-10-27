"""External annotation evaluation for the sealed Engine coding campaign.

Compare with the actual starting ExtraTrees bundle. The weaker MLP is a
secondary reference only. No physical causal/risk or engine-level claims.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from audit_forge_expansion_development import main as development_replay,pointwise
from evaluate_forge_coding import terminal, files, namespace
from real_engine_health_model import HEADS
from run_full_benchmark_campaign import clean_env


def read(path):
    return json.loads(Path(path).read_text())


def suffix(root):
    return read(root/'CAMPAIGN_CONFIG.json')['iteration'].removeprefix('A')


def starting(root):
    """Test the actual common starter, with fitting/development data unmounted."""
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases'];run=root/cases[0]['cohort']
    job,=read(run/'PLAN187.json')['jobs'];ws=run/job['workspace']
    digest=sha256(ws/'solution.py')
    assert all(sha256(root/c['cohort']/read(root/c['cohort']/'PLAN187.json')['jobs'][0]['workspace']/'solution.py')==digest for c in cases)
    dest=root/'audit/starting_program_gate';dest.mkdir(exist_ok=False)
    app=dest/'app';shutil.copytree(ws,app,symlinks=True);output=dest/'output';output.mkdir()
    mounted=next(v['source'] for v in read(Path(job['mounts']))['mounts'] if v['destination']=='/data')
    data=Path(mounted)/'datasets'/job['dataset'];dev=dict(np.load(data/'val.npz',allow_pickle=False))
    np.savez_compressed(output/'observations.npz',**{k:v for k,v in dev.items() if k not in ['labels','groups']})
    public=dest/'public/datasets'/job['dataset'];public.mkdir(parents=True);shutil.copy2(data/'dataset.json',public/'dataset.json')
    mounts=read(Path(job['mounts']))
    for v in mounts['mounts']:
        if v['destination']=='/data':v['source']=str(dest/'public')
    save(dest/'mounts.json',mounts);cold=dict(job,evaluation_mounts=str(dest/'mounts.json'))
    command=namespace(run,cold,app,output,['/opt/python312/bin/python3.12','-B','/app/solution.py','--input','/output/observations.npz','--output','/output/predictions.json'])
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',
        TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    save(dest/'command.json',command)
    with (dest/'stdout.txt').open('wb') as out,(dest/'stderr.txt').open('wb') as err:
        code=subprocess.run(command,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=180).returncode
    assert code==0,'Actual starter cold inference failed'
    rows=read(output/'predictions.json')['predictions'];by_id={v['sample_id']:v for v in rows}
    assert len(by_id)==len(rows)==len(dev['sample_ids']) and set(by_id)==set(dev['sample_ids'])
    actual=float(np.mean([by_id[str(i)]['predicted_label']==v for i,v in zip(dev['sample_ids'],dev['labels'])]))
    expected=read(root/'CAMPAIGN_CONFIG.json')['reference_development']['ExtraTrees']['joint_annotation_accuracy']
    assert abs(actual-expected)<1e-12 and sha256(ws/'solution.py')==digest
    result=dict(passed=True,actual_saved_starting_program=True,identical_starting_programs=len(cases),development_joint_accuracy=actual,
        development_rows=len(rows),fitting_and_development_arrays_unmounted=True,new_final_predictions=0,solution_sha256=digest)
    save(root/'STARTING_PROGRAM_QUALIFICATION316.json',result);return result


def development(root):
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases']
    # All subjects, including pending ones, must stop before any evaluation.
    for case in cases:
        terminal(root/case['cohort'])
    def one(case):
        run=root/case['cohort'];job,=terminal(run)
        frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
        assert files(run/job['workspace'])==frozen['files']
        normal=frozen['status']=='finished' and frozen['response_valid'] is True
        if normal:
            development_replay(run,job['id'])
            evidence=read(run/'audit/development_replay'/job['id']/'RESULT.json')
        else:
            evidence=dict(skipped=True,reason='No normal final submission',status=frozen['status'])
        save(run/'OBSERVABLE_REVIEW_COMPLETE.json',dict(
            utc=datetime.now(timezone.utc).isoformat(),reviewed_before_new_test_scores=True,
            actual_program_development_replay=evidence,annotation_only=True,
            not_a_post_submission_Agent_review=True,
            weak_MLP_not_actual_starting_candidate=True))
        return dict(case=case,normal=normal,development=evidence)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,cases))
    save(root/('DEVELOPMENT_AUDIT'+suffix(root)+'.json'),dict(rows=rows,final_scores_not_opened=True))
    return rows


def wait_development(root):
    """Wait for existing subjects, then audit development without test scores.

    This starts no subjects and performs no fitting or final-test inference.
    The human-readable reports still require Root review before final scoring.
    """
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases'];start=time.monotonic()
    while True:
        ready=True;observed=[]
        for case in cases:
            run=root/case['cohort'];plans=read(run/'PLAN187.json')['jobs']
            for job in plans:
                path=run/'primary'/job['id']/job['id']/'status.json'
                status=read(path) if path.exists() else {'status':'pending'}
                pid=status.get('cli_pid');stat=Path('/proc')/str(pid)/'stat'
                live=stat.exists() and stat.read_text().split(') ',1)[1][0] not in ['Z','X']
                observed.append(dict(cohort=case['cohort'],status=status['status'],cli_pid=pid,process_verified_live=live))
            try:terminal(run)
            except (AssertionError,FileNotFoundError):ready=False
            if not (run/'CODING_CHOICE_AUDIT.json').exists():ready=False
        save(root/'audit/ROOT_DEVELOPMENT_WATCH.json',dict(utc=datetime.now(timezone.utc).isoformat(),
            all_choices_terminal_and_sealed=ready,observations=observed,elapsed_seconds=time.monotonic()-start,
            new_Agent_subjects=0,new_fits=0,new_final_test_predictions=0))
        if ready:break
        if time.monotonic()-start>7200:raise TimeoutError('Bounded wait ended; inspect original subjects, do not restart them.')
        time.sleep(15)
    assert not list(root.glob('ANNOTATION_RESULTS*.json'))
    gate=root/('DEVELOPMENT_AUDIT'+suffix(root)+'.json')
    rows=read(gate)['rows'] if gate.exists() else development(root)
    eligible=[r for r in rows if r['normal'] and r['development']['passed']]
    def one(row):
        run=root/row['case']['cohort'];job,=terminal(run)
        output=run/'audit/pointwise_development'/job['id']/'RESULT.json'
        frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
        assert files(run/job['workspace'])==frozen['files']
        if output.exists():return read(output)
        pointwise(run,job['id'],root/'audit/POINTWISE_DEVELOPMENT_PROTOCOL.json')
        return read(output)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,eligible))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),normal_programs=len(eligible),
        pointwise_checks=[{k:v for k,v in r.items() if k!='rows'} for r in checks],
        all_program_reports_still_require_Root_review=True,final_scores_not_opened=True,
        new_fits=0,new_Agent_subjects=0)
    save(root/('POINTWISE_DEVELOPMENT_DRIVER'+suffix(root)+'.json'),result)
    return result


def component_bits(value):
    """Accept the source's 0/1 bits and equivalent health/fault names."""
    if not isinstance(value,dict) or set(value)!=set(HEADS):return None
    result=[]
    for head in HEADS:
        state=value[head]
        if type(state) in (int,float) and state in (0,1):result.append(int(state))
        elif isinstance(state,str) and state.strip().lower() in ['health','healthy','fault','faulty']:
            result.append(int(state.strip().lower() in ['fault','faulty']))
        else:return None
    return ''.join(map(str,result))


def timing_items(value):
    """Public supervision uses pipe strings; the starter uses token lists."""
    if value is None:return None
    values=[value] if isinstance(value,str) else value
    if not isinstance(values,list) or not all(isinstance(x,str) for x in values):return False
    return [part.strip() for item in values for part in item.split('|') if part.strip()]


def annotations(prediction,gold):
    rows=prediction['predictions'];assert len(rows)==len(gold)==1
    p=rows[0];g=gold[0];assert p['sample_id']==g['sample_id']
    if 'condition' in g:
        label=p['predicted_label'];assert isinstance(label,str) and len(label.split('|'))==2
        condition,probability=label.split('|')
        return dict(n=1,component_correct=int(condition==g['condition'])+int(probability==g['probability_bin']),component_n=2,
            joint_correct=int(label==g['label']),condition_scored=True,condition_correct=int(condition==g['condition']),
            probability_bin_scored=True,probability_bin_correct=int(probability==g['probability_bin']),
            condition_field_consistent=p.get('qualitative_condition')==condition,
            probability_bin_field_consistent=p.get('failure_probability_bin_annotation')==probability,
            calibrated_failure_probability_supplied=p.get('calibrated_failure_probability') is not None,
            physical_probability_calibration_validated=False)
    bits=p['predicted_label'];assert isinstance(bits,str) and len(bits)==5 and set(bits)<=set('01')
    components=p.get('component_health');faults=p.get('faulty_components')
    cause=p.get('timing_cause_annotation')
    parsed_cause=timing_items(cause)
    true_cause=g.get('timing_causes')
    causes_valid=parsed_cause is not False and (parsed_cause is not None or true_cause is None)
    causal=p.get('physical_causal_conclusion')
    return dict(n=1,component_correct=sum(a==b for a,b in zip(bits,g['label'])),
        component_n=5,joint_correct=int(bits==g['label']),
        component_health_consistent=component_bits(components)==bits,
        faulty_components_consistent=isinstance(faults,list) and len(faults)==len(set(faults)) and set(faults)=={h for h,b in zip(HEADS,bits) if b=='1'},
        state_present=isinstance(p.get('overall_health_state'),str),
        state_scored=g.get('health_state') is not None,
        state_correct=int(p.get('overall_health_state')==g['health_state']) if g.get('health_state') is not None else None,
        timing_scored=true_cause is not None,
        timing_correct=int(causes_valid and parsed_cause is not None and set(parsed_cause)==set(timing_items(true_cause))) if true_cause is not None else None,
        timing_field_valid=causes_valid,
        timing_field_representation='pipe_string' if isinstance(cause,str) else 'token_list' if isinstance(cause,list) else 'missing' if cause is None else 'unsupported',
        semantic_format_normalization=True,
        physical_causal_claim_present=causal is not None,
        physical_causal_claim_validated=False,
        predicted_bits=bits,actual_bits=g['label'])


def score(root):
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases']
    assert (root/('DEVELOPMENT_AUDIT'+suffix(root)+'.json')).exists()
    assert not (root/('ANNOTATION_RESULTS'+suffix(root)+'.json')).exists()
    for case in cases:
        run=root/case['cohort'];terminal(run)
        assert read(run/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
    def one(case):
        run=root/case['cohort'];job,=terminal(run)
        frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
        ws=run/job['workspace'];assert files(ws)==frozen['files']
        normal=frozen['status']=='finished' and frozen['response_valid'] is True
        row=dict(case=case,status=frozen['status'],formal_submission=normal)
        if not normal:
            row.update(final_scores_not_computed=True,reason='Do not score a starter as an Agent submission')
            return row
        dest=run/'evaluation312';dest.mkdir(exist_ok=False)
        app=dest/'app';shutil.copytree(ws,app,symlinks=True)
        for f in ['.runtime/tmp','.runtime/mpl']:(app/f).mkdir(parents=True,exist_ok=True)
        output=dest/'output';output.mkdir()
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
            PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
            MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        obs='/data/datasets/'+job['dataset']+'/test.npz'
        commands={
            'final':['/opt/python312/bin/python3.12','-B','/app/solution.py','--input',obs,'--output','/output/final.json'],
            'starting':['/opt/python312/bin/python3.12','-B','-c',
                "import sys;sys.path.insert(0,'/tools');from real_engine_health_model import infer;infer('/reference/engine_health.joblib',sys.argv[1],'/output/starting.json')",obs],
            'weak_mlp':['/opt/python312/bin/python3.12','-B','-c',
                "import sys;sys.path.insert(0,'/tools');from real_fault_model import infer;infer('/reference/model.pt',sys.argv[1],'/output/weak_mlp.json')",obs]}
        if job['dataset']=='ENGINE_QA_TREND':
            commands['starting']=['/opt/python312/bin/python3.12','-B','-c',
                "import sys,json;from pathlib import Path;sys.path.insert(0,'/tools');from frozen_classifier import infer;v=infer('/reference/ordinary_extra_trees.joblib',sys.argv[1],'/output/starting.json');[(r.update(qualitative_condition=r['predicted_label'].split('|')[0],failure_probability_bin_annotation=r['predicted_label'].split('|')[1],calibrated_failure_probability=None)) for r in v['predictions']];Path('/output/starting.json').write_text(json.dumps(v))",obs]
        gold=read(run/'hidden'/job['dataset']/'labels.json')
        for kind,command in commands.items():
            argv=namespace(run,job,app,output,command);save(dest/(kind+'_command.json'),argv)
            try:
                with (dest/(kind+'.stdout')).open('wb') as out,(dest/(kind+'.stderr')).open('wb') as err:
                    code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
                assert code==0,'Cold execution failed, exit='+str(code)
                payload=read(output/(kind+'.json'))
                if kind=='weak_mlp':
                    values=payload['predictions'];assert len(values)==1 and values[0]['sample_id']==gold[0]['sample_id']
                    b=values[0]['predicted_label'];assert isinstance(b,str)
                    if job['dataset']=='ENGINE_QA_TREND':
                        fields=b.split('|');assert len(fields)==2
                        metric=dict(component_correct=sum(a==v for a,v in zip(fields,gold[0]['label'].split('|'))),component_n=2,joint_correct=int(b==gold[0]['label']))
                    else:
                        assert len(b)==5 and set(b)<=set('01')
                        metric=dict(component_correct=sum(a==v for a,v in zip(b,gold[0]['label'])),component_n=5,joint_correct=int(b==gold[0]['label']))
                else:metric=annotations(payload,gold)
                row[kind]=dict(valid=True,metrics=metric,prediction_sha256=sha256(output/(kind+'.json')))
            except (AssertionError,ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired) as error:
                row[kind]=dict(valid=False,error_type=type(error).__name__,error=str(error))
        assert files(ws)==frozen['files'],'Evaluation mutated the frozen subject'
        save(dest/'SCORE.json',row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,cases))
    aggregate={}
    for kind in ['final','starting','weak_mlp']:
        valid=[r[kind]['metrics'] for r in rows if r.get(kind,{}).get('valid')]
        aggregate[kind]=dict(valid_source_records=len(valid),planned_original_directions=len(cases),
            component_correct=sum(r['component_correct'] for r in valid),component_n=sum(r['component_n'] for r in valid),
            joint_correct=sum(r['joint_correct'] for r in valid),joint_n=len(valid))
        for field in (['condition','probability_bin'] if cases[0]['dataset']=='ENGINE_QA_TREND' else ['state','timing']):
            scored=[r for r in valid if r.get(field+'_scored')]
            if kind!='weak_mlp':aggregate[kind].update({field+'_correct':sum(r[field+'_correct'] for r in scored),field+'_n':len(scored)})
    pairs=[r for r in rows if r.get('final',{}).get('valid') and r.get('starting',{}).get('valid')]
    result=dict(utc=datetime.now(timezone.utc).isoformat(),original_directions=len(cases),rows=rows,aggregate=aggregate,
        paired_final_minus_starting=dict(n=len(pairs),component_correct=sum(r['final']['metrics']['component_correct']-r['starting']['metrics']['component_correct'] for r in pairs),
            joint_correct=sum(r['final']['metrics']['joint_correct']-r['starting']['metrics']['joint_correct'] for r in pairs)),
        annotation_agreement_only=True,official_original_benchmark_score=False,
        independent_physical_engines_established=False,physical_causality_or_risk_validated=False,
        new_confirmed_hard_problems=0,automatic_mechanism_claim=False)
    save(root/('ANNOTATION_RESULTS'+suffix(root)+'.json'),result)
    return result


def repair_reference(root):
    """Replay only the unchanged available reference after a missing import.

    A317 retained the original reference checkpoint but its generic public
    tool copy omitted the Engine helper. Do not alter submissions or re-fit.
    """
    assert suffix(root)=='317'
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases'];path=root/'ANNOTATION_RESULTS317.json';result=read(path)
    dest=root/'audit/reference_import_repair';dest.mkdir(exist_ok=False)
    shutil.copy2(path,dest/'ORIGINAL_ANNOTATION_RESULTS317.json')
    parent=Path(read(root/cases[0]['cohort']/'audit/COMPLETION_LINEAGE317.json')['previous_run']).parent
    helper=parent/cases[0]['cohort']/'public_tools/real_engine_health_model.py'
    model=root/'source/shared_reference/engine_health.joblib'
    assert sha256(model)==sha256(parent/'source/shared_reference/engine_health.joblib')
    tools=dest/'tools';tools.mkdir();shutil.copy2(helper,tools/helper.name)
    def one(row):
        run=root/row['case']['cohort'];job,=terminal(run);frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
        assert files(run/job['workspace'])==frozen['files'] and row['formal_submission']
        evaluation=run/'evaluation312';assert "No module named 'real_engine_health_model'" in (evaluation/'starting.stderr').read_text()
        shutil.copy2(evaluation/'SCORE.json',dest/(run.name+'_ORIGINAL_SCORE.json'))
        mounts=read(Path(job.get('evaluation_mounts',job['mounts'])))
        ref=next(m['source'] for m in mounts['mounts'] if m['destination']=='/reference')
        assert sha256(Path(ref)/'engine_health.joblib')==sha256(model)
        mounts['mounts'].append(dict(source=str(tools),destination='/reference_evaluator'))
        mountfile=dest/(run.name+'_mounts.json');save(mountfile,mounts)
        cold=dict(job,evaluation_mounts=str(mountfile))
        output=evaluation/'output';old_prediction=sha256(output/'final.json')
        command=['/opt/python312/bin/python3.12','-B','-c',
            "import sys;sys.path.insert(0,'/reference_evaluator');from real_engine_health_model import infer;infer('/reference/engine_health.joblib',sys.argv[1],'/output/starting_repaired.json')",
            '/data/datasets/'+job['dataset']+'/test.npz']
        argv=namespace(run,cold,evaluation/'app',output,command);save(dest/(run.name+'_command.json'),argv)
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (dest/(run.name+'.stdout')).open('wb') as out,(dest/(run.name+'.stderr')).open('wb') as err:
            code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
        assert code==0
        pred=read(output/'starting_repaired.json');gold=read(run/'hidden'/job['dataset']/'labels.json')
        row['starting']=dict(valid=True,metrics=annotations(pred,gold),prediction_sha256=sha256(output/'starting_repaired.json'),evaluator_import_repaired=True)
        assert sha256(output/'final.json')==old_prediction and files(run/job['workspace'])==frozen['files']
        save(evaluation/'SCORE.json',row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,result['rows']))
    valid=[r['starting']['metrics'] for r in rows]
    aggregate=dict(valid_source_records=len(valid),planned_original_directions=len(cases),component_correct=sum(r['component_correct'] for r in valid),component_n=sum(r['component_n'] for r in valid),joint_correct=sum(r['joint_correct'] for r in valid),joint_n=len(valid))
    for field in ['state','timing']:
        scored=[r for r in valid if r.get(field+'_scored')];aggregate.update({field+'_correct':sum(r[field+'_correct'] for r in scored),field+'_n':len(scored)})
    result['rows']=rows;result['aggregate']['starting']=aggregate
    pairs=[r for r in rows if r.get('final',{}).get('valid')]
    result['paired_final_minus_starting']=dict(n=len(pairs),component_correct=sum(r['final']['metrics']['component_correct']-r['starting']['metrics']['component_correct'] for r in pairs),joint_correct=sum(r['final']['metrics']['joint_correct']-r['starting']['metrics']['joint_correct'] for r in pairs))
    correction=dict(utc=datetime.now(timezone.utc).isoformat(),reason='Checkpoint was available and unchanged in every completion /reference mount; the Root evaluator import helper was missing. Replay only that fixed reference using the original A312 helper in an evaluation-only mount.',
        original_error_preserved=True,reference_predictions_replayed=len(rows),new_Agent_predictions=0,new_fits=0,new_LLM_calls=0,
        unchanged_reference_checkpoint_sha256=sha256(model),original_helper_sha256=sha256(helper),all_Agent_outputs_and_workspaces_unchanged=True,not_an_Agent_failure=True)
    result['reference_import_correction']=correction;save(dest/'CORRECTION.json',correction);save(path,result)
    return correction


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['starting','development','wait-development','score','repair-reference'],required=True);args=p.parse_args()
    result={'starting':starting,'development':development,'wait-development':wait_development,'score':score,'repair-reference':repair_reference}[args.mode](args.run.resolve())
    print(json.dumps(result,ensure_ascii=False))
