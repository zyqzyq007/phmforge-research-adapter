"""Read-only developmental audit of actual program pools and validation views.

This does not train a forecaster, execute candidates, or evaluate new data.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
ARCHIVES = Path('/data/yyf-zyq/PHM_20261010')
SOURCES = [
    ('A337', '20261024T181000Z_freeprogram337r1', 'RESULTS337.json', 'all_results'),
    ('A342', '20261024T211000Z_r342l2', 'RESULT342.json', 'all_program_results'),
    ('A347', '20261025T012600Z_local347', 'RESULT347.json', 'all_program_results'),
]
RULES = ['keep','development','mean_cv','mean_cv_and_development',
         'minimax_cv_and_development','full_checkpoint_equal_condition',
         'minimax_all_legal_conditions']

def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def mapped(root, old):
    parts=Path(old).parts
    assert root.name in parts
    return root.joinpath(*parts[parts.index(root.name)+1:])

def freeze(root):
    root.mkdir(exist_ok=False)
    (root/'runtime/tmp').mkdir(parents=True)
    bindings=[]
    for cohort,name,result,_ in SOURCES:
        archive=ARCHIVES/name
        receipt=read(archive/'ARCHIVE_VERIFIED.json')
        manifest=archive/receipt['manifest_filename']
        assert receipt['source_cleanup']=='complete' and sha(manifest)==receipt['manifest_sha256']
        bindings.append(dict(cohort=cohort,archive=str(archive),manifest_sha256=sha(manifest),
                             main_result_sha256=sha(archive/result)))
    shutil.copy2(Path(__file__),root/'audit_program_evidence.py')
    protocol=dict(utc=utc(),iteration='A348',type='retrospective_developmental_metadata_audit',
        sources=bindings,script_sha256=sha(root/'audit_program_evidence.py'),rules=RULES,
        rule_definition='Error only. Ties keep p00 then ascending actual registration ID. mean_cv equally weights actual held-condition reports. dev is original whole-acquisition error. Combined mean uses mean_cv/dev equally. Minimax uses individual held-CV and physical dev conditions. Full-checkpoint rules require actual new-check coverage and use full-training dev/check physical conditions. Last minimax also uses individual CV conditions.',
        missingness='Keep all12 planned pools. Missing checks or original incomplete native submissions stay visible. No fabricated check reports or successful fallback submissions.',
        cohort_unit='Three complete training cohorts on one COMFAULDA rig. Four pools within each cohort are related. R343/R344/R345 old-weight replays are NOT extra training cohorts.',
        all_scores_already_opened_before_this_audit=True,outputs_are_not_confirmation=True,
        original_protocols_choices_and_results_immutable=True,
        budgets=dict(new_LLM_calls=0,new_fits=0,new_predictions=0,wall_seconds=120,live_disk_GiB=16),
        next_design_question='Do conflicting validation views and common candidate headroom support a bounded evidence-reliability forecast pilot? No forecast model or threshold is chosen here.',
        forbidden='No A338 performance, new waveform reads, model execution, weight fitting, rule optimization, or method superiority claims.')
    save(root/'PROTOCOL348.json',protocol)
    save(PROJECT/'results/evidence348/PROTOCOL348.json',protocol)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
         shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4',usage='Reference only; this audit uses the Python standard library.'),
         retain=['All source bindings,complete derived tables,negative results,protocol and code; original archives remain immutable.']))
    print(json.dumps(dict(status='frozen_before_derived_audit',run=str(root),new_LLM_calls=0,new_fits=0,new_predictions=0)))

def checked_metric(value):
    assert value.get('status','complete')=='complete'
    records=value['records']
    assert len({x['group'] for x in records})==len(records)
    matrix=[[0]*4 for _ in range(4)]
    for x in records: matrix[int(x['actual'])][int(x['predicted'])]+=1
    error=sum(int(x['actual']!=x['predicted']) for x in records)/len(records)
    assert abs(error-value['record_error'])<1e-12 and matrix==value['confusion']
    return error

def condition_errors(value):
    checked_metric(value)
    result={}
    for hz in sorted({float(x['condition_hz']) for x in value['records']}):
        rows=[x for x in value['records'] if float(x['condition_hz'])==hz]
        result[str(hz)]=sum(x['actual']!=x['predicted'] for x in rows)/len(rows)
    return result

def run(root):
    assert not (root/'STARTED348.json').exists()
    cfg=read(root/'PROTOCOL348.json')
    assert sha(root/'audit_program_evidence.py')==cfg['script_sha256']
    assert sha(Path(__file__))==cfg['script_sha256']
    save(root/'STARTED348.json',dict(utc=utc(),new_LLM_calls=0,new_fits=0,new_predictions=0))
    pools=[];programs={};metrics_verified=0;bindings=[]
    for (cohort,name,result,result_key),frozen in zip(SOURCES,cfg['sources']):
        archive=ARCHIVES/name;manifest_path=archive/'SOURCE_MANIFEST.json'
        assert sha(manifest_path)==frozen['manifest_sha256'] and sha(archive/result)==frozen['main_result_sha256']
        manifest={x['path']:x for x in read(manifest_path)['files']}
        def bound(path):
            relative=str(path.relative_to(archive));entry=manifest[relative]
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
            bindings.append(dict(cohort=cohort,path=relative,sha256=entry['sha256']))
            return read(path)
        main=bound(archive/result);scores=main[result_key]
        for session_index,session in enumerate(bound(archive/'SESSIONS337.json'),1):
            subject=mapped(archive,session['path']);state=bound(subject/'broker/STATE.json')
            if (subject/'BUILD_RESULT.json').exists():
                build=bound(subject/'BUILD_RESULT.json')
                completion_provenance='subject/BUILD_RESULT.json.completed'
            else:
                assert cohort=='A337'
                original=next(x for x in main['rows'] if x['id']==session['id'])
                build=dict(completed=original['end_to_end_completed'])
                completion_provenance=result+'.rows.end_to_end_completed (older archive schema)'
            probe=state.get('probe') or {}
            checks=probe.get('reports',{}) if probe.get('status')=='complete' else {}
            common=subject/'common_check/REPORT.json'
            check_source='builder_probe' if checks else 'none'
            if common.exists():
                checks=bound(common)['programs'];check_source='post_builder_common_check'
            members={}
            for ident,entry in sorted(state['programs'].items()):
                program=mapped(archive,entry['path']);report=bound(program/'REPORT.json')
                assert report['status']=='complete'
                code=program/'source/program.py';relative=str(code.relative_to(archive))
                assert sha(code)==manifest[relative]['sha256']
                score_prefix='pool_'+str(session_index) if cohort=='A342' else session['id']
                score_key='common_p00' if ident=='p00' else score_prefix+'__'+ident
                final=scores[score_key];final_error=checked_metric(final);metrics_verified+=1
                cv=[checked_metric(x) for x in report['cv']];metrics_verified+=len(cv)
                dev=checked_metric(report['development']);metrics_verified+=1
                dev_conditions=condition_errors(report['development'])
                check=checks.get(ident)
                if check is not None:
                    checked_metric(check);metrics_verified+=1
                full_conditions=list(dev_conditions.values())+(list(condition_errors(check).values()) if check else [])
                values=dict(keep=0 if ident=='p00' else None,development=dev,mean_cv=sum(cv)/len(cv),
                    mean_cv_and_development=(sum(cv)/len(cv)+dev)/2,
                    minimax_cv_and_development=max(cv+list(dev_conditions.values())),
                    full_checkpoint_equal_condition=sum(full_conditions)/len(full_conditions) if check else None,
                    minimax_all_legal_conditions=max(cv+full_conditions) if check else None)
                uid=cohort+'/'+score_key
                row=dict(program=ident,source_sha256=sha(code),final_error=final_error,cv_errors=cv,
                    development_error=dev,development_condition_errors=dev_conditions,
                    check_condition_errors=condition_errors(check) if check else None,
                    check_source=check_source,rule_values=values,
                    full_model_weight_bindings={k:v for k,v in report['artifacts'].items() if k.startswith('full_fit/')})
                programs[uid]=row;members[ident]=row
            oracle=min(x['final_error'] for x in members.values());selected={}
            for rule in RULES:
                eligible=[ident for ident,row in members.items() if row['rule_values'][rule] is not None]
                if not eligible: selected[rule]=dict(status='missing_actual_check_evidence');continue
                ident=min(eligible,key=lambda q:(members[q]['rule_values'][rule],q!='p00',q))
                error=members[ident]['final_error']
                selected[rule]=dict(status='scored',program=ident,final_error=error,own_pool_selection_loss=error-oracle,
                                    eligible_programs=len(eligible),actual_pool_programs=len(members))
            pools.append(dict(cohort=cohort,subject=session['id'],generator_arm=session['arm'],programs=list(members),
                original_native_completed=build['completed'],original_submission=state.get('submission'),
                completion_provenance=completion_provenance,
                submission_scope='builder_handoff_not_final_selector' if cohort=='A342' else 'original_native_choice_or_missing',
                strict_registry_deviation=cohort=='A347' and session['id']=='local_revision_2',
                oracle_error_descriptive_only=oracle,methods=selected))
    assert len(pools)==12
    cohorts={}
    for cohort,_,_,_ in SOURCES:
        subset=[p for p in pools if p['cohort']==cohort];table={}
        for rule in RULES:
            valid=[p['methods'][rule] for p in subset if p['methods'][rule]['status']=='scored']
            table[rule]=dict(planned_pools=len(subset),scored_pools=len(valid),
                mean_final_error=sum(p['final_error'] for p in valid)/len(subset) if len(valid)==len(subset) else None,
                mean_selection_loss=sum(p['own_pool_selection_loss'] for p in valid)/len(subset) if len(valid)==len(subset) else None)
        cohorts[cohort]=table
    value=dict(utc=utc(),status='complete',cohorts=cohorts,pools=pools,unique_program_instances=len(programs),
        actual_pool_memberships=sum(len(p['programs']) for p in pools),records_metrics_recounted=metrics_verified,
        training_cohorts=3,physical_source_clusters=1,new_independent_confirmation_scenarios=0,
        no_method_or_threshold_trained=True,no_rule_search=True,new_LLM_calls=0,new_fits=0,new_predictions=0,
        limits='Retrospective already-opened development material, only3 related training cohorts and different candidate banks. Builder failures and strict registry deviations retained. Full-check controls with missing evidence have no imputed cohort mean. No Agent superiority or causal mechanism inferred from ranking associations.')
    save(root/'PROGRAM_EVIDENCE_CORPUS348.json',dict(programs=programs,bindings=bindings))
    save(root/'RESULT348.json',value);save(PROJECT/'results/evidence348/RESULT348.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['pools','cohorts']}))

def submission(root):
    """Independently recount A396 sealed arrays, decisions and actual costs."""
    import math
    import numpy as np
    assert (root/'TERMINAL396.json').exists(), 'Never audit an unsealed/live pilot'
    handle=read(root/'HANDLE337.json')
    assert handle['status']!='running' and not Path('/proc'/Path(str(handle['pid']))).exists()
    result=read(root/'RESULT396.json');cfg=read(root/'PROTOCOL396.json')
    seal=read(root/'ALL_CHOICES_SEALED337.json')
    assert len(seal['episodes'])==4
    checks=0;bindings=0;problems=[]
    def equal(actual,expected,where):
        nonlocal checks
        checks+=1
        if isinstance(expected,float): ok=actual is not None and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12)
        else: ok=actual==expected
        if not ok: problems.append(dict(where=where,actual=actual,expected=expected))
    for relative,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():
        equal(sha(root/relative),digest,'code/'+relative);bindings+=1
    for binding in read(root/'SOURCE_BINDINGS396.json'):
        equal(sha(Path(binding['target'])),binding['sha256'],'source/'+binding['target']);bindings+=1
    source=Path(cfg['source_archive']);qualification=read(source/'QUALIFICATION346.json')
    final=source/'qualification/final.npz'
    equal(sha(final),qualification['files']['final.npz']['sha256'],'final_source')
    with np.load(final,allow_pickle=False) as z: data={k:z[k].tolist() for k in ['sample_ids','labels','groups','condition_id']}
    truth=dict(zip(data['sample_ids'],data['labels']));groups={}
    for sid,group in zip(data['sample_ids'],data['groups']): groups.setdefault(group,[]).append(sid)
    raw={}; predictions={}
    for name,value in result['all_program_results'].items():
        if value['status']!='complete': continue
        arrays=[]
        for replay in ['forward','cold_repeat']:
            path=root/'evaluation'/name/replay/'PREDICTIONS.npz'
            with np.load(path,allow_pickle=False) as z:
                ids=z['sample_ids'].tolist();pp=z['prediction'].tolist()
            equal(len(set(ids)),len(ids),name+'/'+replay+'/unique_ids')
            equal(ids,data['sample_ids'],name+'/'+replay+'/ids');arrays.append(dict(zip(ids,pp)))
        equal(arrays[0],arrays[1],name+'/cold_repeat');pred=arrays[0]
        equal(all(type(v) is int and 0<=v<4 for v in pred.values()),True,name+'/label_contract')
        records=[];matrix=[[0]*4 for _ in range(4)]
        for group,ids in sorted(groups.items()):
            actual={truth[sid] for sid in ids};equal(len(actual),1,name+'/'+group+'/truth')
            actual=next(iter(actual));counts=Counter(pred[sid] for sid in ids)
            chosen=min(counts,key=lambda k:(-counts[k],k));matrix[actual][chosen]+=1
            records.append(dict(group=group,actual=actual,predicted=chosen))
        errors=sum(r['actual']!=r['predicted'] for r in records)
        metrics=dict(record_error=errors/len(records),record_accuracy=1-errors/len(records),
                     window_error=sum(pred[sid]!=truth[sid] for sid in truth)/len(truth),confusion=matrix)
        metrics['macro_F1']=sum(2*matrix[i][i]/(sum(matrix[i])+sum(row[i] for row in matrix)) if sum(matrix[i])+sum(row[i] for row in matrix) else 0 for i in range(4))/4
        for key,v in metrics.items(): equal(value[key],v,name+'/'+key)
        equal([{k:r[k] for k in ['group','actual','predicted']} for r in value['records']],records,name+'/records')
        raw[name]=dict(metrics,records=records);predictions[name]=pred
    baseline=raw['common_p00'];best=min(v['record_error'] for v in raw.values());opportunity=baseline['record_error']-best
    equal(result['baseline_error'],baseline['record_error'],'baseline');equal(result['common_union_best_error'],best,'common_best')
    all_scored=len(raw)==len(result['all_program_results']);equal(result['all_actual_candidates_scored'],all_scored,'all_scored')
    usage=[];commands=[];reviews=[];training=[];actual_choices={}
    for episode in result['episodes']:
        name=episode['id'];subject=root/'subjects'/name;state=read(subject/'broker/STATE.json')
        native=read(subject/'primary'/name/'status.json');sealed=next(e for e in seal['episodes'] if e['id']==name)
        assert native['status']!='running' and not Path('/proc'/Path(str(native['cli_pid']))).exists()
        equal(episode['completed'],sealed['completed'],name+'/completion')
        equal(state.get('submission'),sealed.get('submission'),name+'/seal')
        for ident,entry in state['programs'].items():
            program=Path(entry['path']);report=read(program/'REPORT.json')
            equal(entry['report'],report,name+'/'+ident+'/report')
            for rel,digest in report['artifacts'].items():equal(sha(program/rel),digest,name+'/'+ident+'/'+rel);bindings+=1
        ids=list(state['programs']); reports={k:v['report'] for k,v in state['programs'].items()}
        mean_cv=lambda k:sum(z['record_error'] for z in reports[k]['cv'])/len(reports[k]['cv'])
        dev=lambda k:reports[k]['development']['record_error']
        choose=lambda f,eligible=ids:min(eligible,key=lambda k:(f(k),k!='p00',k))
        fixed=dict(keep='p00',development=choose(dev),mean_cv=choose(mean_cv),equal_CV_development=choose(lambda k:(mean_cv(k)+dev(k))/2))
        packet=state.get('probe') or {};checked=packet.get('reports',{}) if packet.get('status')=='complete' else {}
        eligible=[k for k in ids if k in checked and checked[k]['status']=='complete']
        if eligible:
            fixed['full_checkpoint_validation']=choose(lambda k:(dev(k)+checked[k]['record_error'])/2,eligible)
            risk=lambda k:[z['record_error'] for z in reports[k]['cv']]+[dev(k),checked[k]['record_error']]
            fixed['minimax_all_legal']=min(eligible,key=lambda k:(max(risk(k)),sum(risk(k))/len(risk(k)),k!='p00',k))
        if state.get('submission'):fixed['actor']=state['submission']['program_id']
        for method,ident in fixed.items():
            observed=episode['methods'][method];equal(observed['program_id'],ident,name+'/'+method+'/choice')
            key='common_p00' if ident=='p00' else name+'__'+ident
            if key not in raw: continue
            value=raw[key];old={r['group']:r for r in baseline['records']};rows=value['records'];n=len(rows)
            harm=sum(old[r['group']]['actual']==old[r['group']]['predicted'] and r['actual']!=r['predicted'] for r in rows)/n
            gain=sum(old[r['group']]['actual']!=old[r['group']]['predicted'] and r['actual']==r['predicted'] for r in rows)/n
            expected=dict(record_error=value['record_error'],harm=max(0.,value['record_error']-baseline['record_error']),gain=baseline['record_error']-value['record_error'],record_harm=harm,record_gain=gain,common_union_selection_loss=value['record_error']-best,common_union_gain_capture=max(0.,baseline['record_error']-value['record_error'])/opportunity if opportunity>0 else None)
            for metric,v in expected.items():equal(observed[metric],v,name+'/'+method+'/'+metric)
            if method=='actor':actual_choices[name]=expected
        submission=state.get('submission') or {};review_id=submission.get('review_id');required=read(subject/'BROKER_CONFIG.json')['submission_review_required']
        if required and submission:
            equal(bool(review_id),True,name+'/required_receipt')
            if review_id:
                receipt=read(subject/'broker/submission_reviews'/(review_id+'.json'))
                payload=dict(program_reports={i:sha(Path(v['path'])/'REPORT.json') for i,v in state['programs'].items()},probe=state.get('probe'),attempts=state['attempts'])
                signature=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                equal(receipt['bank_signature'],signature,name+'/current_bank');equal(receipt['program_id'],submission['program_id'],name+'/current_choice')
                for pair in receipt['comparisons']:
                    left,right=pair['incumbent'],pair['challenger']
                    viewmap={'development/full_checkpoint':(reports[left]['development'],reports[right]['development'])}
                    for a,b in zip(reports[left]['cv'],reports[right]['cv']):viewmap['CV/fold_checkpoint/'+str(int(a['condition_hz']))]=(a,b)
                    if left in checked and right in checked:viewmap['reusable_check/full_checkpoint']=(checked[left],checked[right])
                    for view in pair['views']:
                        aa,bb=viewmap[view['view']];aa={r['group']:r for r in aa['records']};bb={r['group']:r for r in bb['records']}
                        equal(set(aa)==set(bb),True,name+'/paired_groups')
                        hp=sum(a['actual']==a['predicted'] and bb[g]['actual']!=bb[g]['predicted'] for g,a in aa.items())
                        gp=sum(a['actual']!=a['predicted'] and bb[g]['actual']==bb[g]['predicted'] for g,a in aa.items())
                        equal(view['harms'],hp,name+'/receipt_harms');equal(view['repairs'],gp,name+'/receipt_repairs')
        reviews.append(dict(subject=name,required=required,submitted=bool(submission),acknowledged=bool(review_id),receipts=len(list((subject/'broker/submission_reviews').glob('*.json')))))
        gateway=subject/'model_gateway'/name;gs=read(gateway/'status.json');equal(gs['active'],False,name+'/gateway_inactive')
        rr=[]
        for p in sorted(gateway.glob('*_receipt.json')):
            receipt=read(p);rr.append(receipt);prefix=f"{receipt['sequence']:03d}"
            equal(sha(gateway/(prefix+'_request.json')),receipt['request_sha256'],name+'/'+prefix+'/request')
            if receipt.get('response_sha256'):equal(sha(gateway/(prefix+'_response.sse')),receipt['response_sha256'],name+'/'+prefix+'/response')
            equal(receipt['model_requested'],'deepseek-flash',name+'/'+prefix+'/model')
        counts=dict(requests=len(list(gateway.glob('*_request.json'))),receipts=len(rr),input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) for r in rr),output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) for r in rr))
        for k in ['requests','input_tokens','output_tokens']:equal(counts[k],gs[k],name+'/gateway_'+k)
        usage.append(dict(subject=name,status=native['status'],**counts,http_statuses=dict(Counter(r.get('upstream_status') for r in rr)),returned_model_ids=sorted({m for r in rr for m in r.get('returned_model_ids',[])}),input_cap_overshoot=max(0,counts['input_tokens']-cfg['episode_input_tokens']),missing_receipts=counts['requests']-counts['receipts']))
        for line in (subject/'primary'/name/'events.jsonl').read_text().splitlines():
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')=='item.completed' and item.get('type')=='command_execution':commands.append(dict(subject=name,command=item['command'],exit_code=item.get('exit_code')))
    for folder,label in [(root/'subjects','scientific'),(root/'qualification_subject','qualification'),(root/'evaluation','external_evaluation')]:
        aa=[read(p) for p in folder.rglob('training_audit.json')];ff=[r for r in aa if r.get('mode')=='fit'];pp=[r for r in aa if r.get('mode')=='predict']
        training.append(dict(scope=label,fit_entries=len(ff),completed_estimator_fits=sum(r['completed_estimator_fits'] for r in ff),optimizer_steps=sum(r['optimizer_steps'] for r in ff),prediction_processes=len(pp),prediction_fits=sum(r['completed_estimator_fits'] for r in pp),prediction_optimizer_steps=sum(r['optimizer_steps'] for r in pp)))
    joint={}
    for arm in cfg['arms']:
        ee=[e for e in result['episodes'] if e['arm']==arm];valid=all(e['completed'] and e['id'] in actual_choices for e in ee)
        joint[arm]=dict(valid=valid,planned=len(ee))
        for field in ['record_error','record_harm','record_gain','common_union_selection_loss','common_union_gain_capture']:
            label='mean_error' if field=='record_error' else 'mean_'+field
            joint[arm][label]=sum(actual_choices[e['id']][field] for e in ee)/len(ee) if valid and (field!='common_union_gain_capture' or opportunity>0) else None
        for k,v in joint[arm].items():equal(result['joint_comparison'][arm][k],v,'joint/'+arm+'/'+k)
    a,b=joint['ordinary_coding'],joint['submission_review']
    passed=bool(all_scored and a['valid'] and b['valid'] and b['mean_error']<a['mean_error'] and b['mean_error']<baseline['record_error'] and b['mean_record_harm']<=a['mean_record_harm'] and b['mean_record_gain']>=a['mean_record_gain'])
    equal(result['joint_pilot_gate_passed'],passed,'pilot_gate')
    output=dict(utc=utc(),passed=not problems,problems=problems,checks=checks,artifact_bindings=bindings,raw_programs_recounted=len(raw),cold_arrays_recounted=2*len(raw),joint_gate_passed=passed,joint_comparison=joint,usage=usage,usage_totals={k:sum(v[k] for v in usage) for k in ['requests','receipts','input_tokens','output_tokens']},training=training,review_receipts=reviews,command_count=len(commands),new_LLM_calls=0,new_fits=0,new_predictions=0,trace_compliance='Command inventory saved;executed local fitting and evidence-use require separate interpretation,not inferred from declared registrations.',scope='Previously opened pilot;no independent confirmation or novelty claim.')
    for filename,value in [('COMPLETE_AUDIT396.json',output),('COMMAND_INVENTORY396.json',commands)]:
        assert not (root/filename).exists();save(root/filename,value)
    print(json.dumps({k:v for k,v in output.items() if k not in ['usage','review_receipts']},ensure_ascii=False))


def submission_trace(root):
    """Audit realized intervention and closing calls after all native seals."""
    assert read(root/'HANDLE337.json')['status']=='terminal'
    assert not Path('/proc',str(read(root/'HANDLE337.json')['pid'])).exists()
    cfg=read(root/'PROTOCOL396.json');rows=[]
    for p in sorted((root/'subjects').iterdir()):
        state=read(p/'broker/STATE.json');submission=state['submission'];calls=[]
        for line in (p/'broker/ACCESS.jsonl').read_text().splitlines():
            raw=json.loads(line);request=json.loads(raw['request'])
            response=json.loads(raw['response']) if isinstance(raw['response'],str) else raw['response']
            result=response.get('result') or {}
            calls.append(dict(request_id=raw['request_id'],started_utc=raw['started_utc'],ended_utc=raw['ended_utc'],action=request['action'],program_id=request.get('program_id'),ok=response.get('ok'),status=result.get('status'),error=response.get('error') or result.get('error'),review_id=request.get('review_id')))
        reviews=[v for v in calls if v['action']=='review_update' and v['ok']]
        closing=[read(f) for f in (p/'model_gateway'/p.name).glob('*_receipt.json') if datetime.fromisoformat(read(f)['started_utc'])>datetime.fromisoformat(submission['utc'])]
        chosen=reviews[-1]['program_id'] if reviews else None
        registrations=[v for v in calls if v['action']=='register_program']
        later=[v for v in registrations if reviews and v['started_utc']>reviews[-1]['ended_utc']]
        rows.append(dict(subject=p.name,policy_requires_acknowledgement=read(p/'BROKER_CONFIG.json')['submission_review_required'],final_program=submission['program_id'],explicit_review_calls=len(reviews),submit_triggered_review_required=sum(v['status']=='review_required' for v in calls),candidate_at_last_explicit_review=chosen,choice_changed_after_last_explicit_review=chosen is not None and chosen!=submission['program_id'],registrations_after_last_explicit_review=len(later),actual_acknowledgement=bool(submission.get('review_id')),register_tool_requests=len(registrations),successful_programs=len(state['programs'])-1,registration_failures=[v for v in registrations if not v['ok'] or v['status']!='complete'],post_seal_requests=len(closing),post_seal_input_tokens=sum((v.get('usage') or {}).get('input_tokens',0) for v in closing),post_seal_output_tokens=sum((v.get('usage') or {}).get('output_tokens',0) for v in closing),calls=calls))
    commands=read(root/'NATIVE_COMMANDS396.json')
    output=dict(utc=utc(),rows=rows,scientific_outcomes_used=False,final_evaluation_complete_at_audit=(root/'TERMINAL396.json').exists(),
        realized_contrast=dict(all_four_voluntarily_requested_review=all(r['explicit_review_calls']>0 for r in rows),submit_triggered_review_required=sum(r['submit_triggered_review_required'] for r in rows),changed_choices_after_explicit_review=sum(r['choice_changed_after_last_explicit_review'] for r in rows),new_registrations_after_explicit_review=sum(r['registrations_after_last_explicit_review'] for r in rows)),
        protocol_accounting=dict(declared_post_submission_model_calls=cfg['post_submission_model_calls'],actual_same_session_post_seal_requests=sum(r['post_seal_requests'] for r in rows),literal_zero_claim_supported=all(r['post_seal_requests']==0 for r in rows),no_extra_controller_review_session=True,closing_requests_remain_in_original_total_budgets=True,correction='The frozen zero-post-submission field is literally inaccurate: native clients made closing/reporting requests after immutable tool submission. Do not change that field or omit costs; no choices or evaluations were returned to these requests.'),
        command_audit=dict(commands=len(commands),inventory_sha256=sha(root/'NATIVE_COMMANDS396.json'),observed_unregistered_estimator_fit_calls=0,observed_unregistered_optimizer_steps=0,interpretation='Manual review of command inventory and invoked feature-extraction/analyze code found data summaries and feature screening,not additional classifier/optimizer fitting. This is a trace-bounded finding,not a proof of the absence of any conceivable computation.',supervised_feature_screening=[dict(subject='submission_review_1',command_index=17,scope='Class means and cross-condition correlation/spread on permitted training data guide feature design;not a separately trained predictor. Adaptively reused CV is not independent certification.')]),
        mechanism_limit='Both ordinary runs voluntarily used the same challenge;all four kept the reviewed choice. A final-score difference cannot by itself establish an effect of the mandatory submit-time intervention.',new_LLM_calls=0,new_fits=0,new_predictions=0)
    target=root/'TRACE_AUDIT396.json';assert not target.exists();save(target,output)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},ensure_ascii=False))


def check_only_control(root):
    """Post-hoc falsification using only already scored, real validation rows."""
    assert read(root/'COMPLETE_AUDIT396.json')['passed']
    archive=Path(read(PROJECT/'results/evidence348/ARCHIVE_VERIFIED.json')['archive_path'])
    receipt=read(archive/'ARCHIVE_VERIFIED.json');manifest=archive/'SOURCE_MANIFEST.json'
    assert sha(manifest)==receipt['manifest_sha256']
    index={r['path']:r for r in read(manifest)['files']}
    def bound(relative):
        path=archive/relative;assert sha(path)==index[relative]['sha256'];return read(path)
    corpus=bound('recovery2/PROGRAM_EVIDENCE_CORPUS348.json')['programs']
    old=bound('recovery2/RESULT348.json');pools=[]
    for pool in old['pools']:
        members={}
        for ident in pool['programs']:
            prefix='pool_'+str(1+next(i for i,p in enumerate([z for z in old['pools'] if z['cohort']=='A342']) if p['subject']==pool['subject'])) if pool['cohort']=='A342' else pool['subject']
            key=pool['cohort']+'/'+('common_p00' if ident=='p00' else prefix+'__'+ident)
            row=corpus[key];views=row['check_condition_errors']
            members[ident]=dict(check_error=sum(views.values())/len(views) if views else None,final_error=row['final_error'])
        pools.append(dict(batch=pool['cohort'],subject=pool['subject'],original_native_completed=pool['original_native_completed'],strict_registry_deviation=pool['strict_registry_deviation'],members=members))
    current=read(root/'RESULT396.json')
    for episode in current['episodes']:
        state=read(root/'subjects'/episode['id']/'broker/STATE.json');checks=(state.get('probe') or {}).get('reports',{});members={}
        for ident in state['programs']:
            key='common_p00' if ident=='p00' else episode['id']+'__'+ident
            members[ident]=dict(check_error=checks.get(ident,{}).get('record_error'),final_error=current['all_program_results'][key].get('record_error'))
        pools.append(dict(batch='A396',subject=episode['id'],original_native_completed=episode['completed'],strict_registry_deviation=False,members=members))
    for pool in pools:
        members=pool['members'];eligible=[k for k,v in members.items() if v['check_error'] is not None]
        pool['check_coverage_complete']=len(eligible)==len(members)
        if not eligible:pool['status']='missing_check_evidence';continue
        chosen=min(eligible,key=lambda k:(members[k]['check_error'],k!='p00',k))
        oracle=min(v['final_error'] for v in members.values());error=members[chosen]['final_error']
        pool.update(status='scored_posthoc',selected=chosen,final_error=error,own_pool_selection_loss=error-oracle,misses_better_existing_program=error>oracle)
    value=dict(utc=utc(),posthoc=True,rule='Choose lowest actual reusable-check error;tie keeps p00 then ascending program ID. No fitted weights or tuned threshold.',rows=pools,
        available_pools=sum(p['status']=='scored_posthoc' for p in pools),missing_pools=sum(p['status']=='missing_check_evidence' for p in pools),
        misses_better_existing_program=sum(p.get('misses_better_existing_program',False) for p in pools),
        historical_corpus_manifest_sha256=receipt['manifest_sha256'],batch_count=4,distinct_training_cohorts=3,physical_source_clusters=1,
        limits='A396 reuses A347 data. Four batches and16 related pools are not16 independent tasks. Historical native failures and a registry deviation remain. This simple control is post-hoc and cannot establish prospective superiority.',
        new_LLM_calls=0,new_fits=0,new_predictions=0,confirmation_data_read=False)
    path=root/'SIMPLE_CHECK_CONTROL396.json';assert not path.exists();save(path,value)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'},ensure_ascii=False))


def ensemble_freeze(root):
    """Freeze strong ensemble controls, before aggregating opened predictions."""
    root.mkdir(exist_ok=False)
    (root/'runtime/tmp').mkdir(parents=True)
    sources=SOURCES+[('A396','20261025T215000Z_submit396','RESULT396.json','all_program_results')]
    bindings=[]
    for cohort,name,result,key in sources:
        archive=ARCHIVES/name;receipt=read(archive/'ARCHIVE_VERIFIED.json')
        assert receipt['source_cleanup']=='complete'
        assert sha(archive/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']
        bindings.append(dict(cohort=cohort,archive=str(archive),result=result,result_key=key,
                             manifest_sha256=receipt['manifest_sha256']))
    shutil.copy2(Path(__file__),root/'audit_program_evidence.py')
    cfg=dict(iteration='A397',utc=utc(),scope='Retrospective fixed strong-control audit;all outcomes previously opened',
        script_sha256=sha(root/'audit_program_evidence.py'),sources=bindings,
        hypothesis='Complementary errors among all actual complete programs may make aggregation less fragile than selecting a single program.',
        rules=['keep','development','mean_cv','mean_cv_and_development','uniform_vote',
               'greedy_development','greedy_cv','greedy_equal_cv_dev'],
        ensemble_unit='Each member first uses its original nine-window whole-acquisition majority. Then combine the resulting class labels. No probability outputs are invented.',
        tie_rule='Among maximum vote labels prefer p00 prediction if tied, otherwise smallest class integer.',
        greedy='Start empty;20 forward additions with replacement;minimize whole-acquisition error on the named legal view. Equal CV/dev averages the two view losses. Addition ties prefer p00 then registration ID. Keep earliest iteration attaining smallest validation loss. Final scores never used in these choices.',
        cohort_units='16 related pools in4 batches;A347/A396 share one cohort, hence3 training cohorts and1 physical rig. No new confirmation.',
        missingness='All actual fully registered programs included;native noncompletion and registry deviations retained;no outcome-based pool filtering.',
        advancement='A control is worth prospective qualification only if cohort-mean error is below every frozen single-selector control, harmful-change loss no higher, and beneficial-change amount no lower, with strict error improvements in at least2 of3 cohorts. This is only a developmental screen, not a method-success or novelty gate.',
        prior_art=dict(url='https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf',
            title='Ensemble Selection from Libraries of Models, ICML2004;author corrected version',
            scope='Fixed hard-vote control inspired by forward ensemble selection;not a faithful probability implementation or novel method.'),
        budgets=dict(new_LLM_calls=0,new_PHM_fits=0,new_PHM_forward_calls=0,wall_seconds=120,live_disk_MiB=128),
        forbidden='No new waveform/model execution,target-dependent weights,subset search,unopened A338/HUST finals,or superiority claims from historical pools.')
    save(root/'PROTOCOL397.json',cfg)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4',usage='Read-only reference;standard library only'),
        retain=['Frozen code/protocol,all pool rows and source bindings,negative controls,independent arithmetic audit;source archives immutable']))
    save(PROJECT/'results/ensemble397/PROTOCOL397.json',cfg)
    print(json.dumps(dict(status='frozen',run=str(root),rules=cfg['rules'])))


def ensemble_run(root):
    """Aggregate immutable actual record predictions;never execute a PHM model."""
    from fractions import Fraction
    import time
    start=time.monotonic();cfg=read(root/'PROTOCOL397.json')
    assert sha(Path(__file__))==cfg['script_sha256'] and not (root/'STARTED397.json').exists()
    save(root/'STARTED397.json',dict(utc=utc(),action='read_only_cached_prediction_control'))
    bindings=[];pools=[];metric_checks=0
    def view(metrics):
        nonlocal metric_checks
        checked_metric(metrics);metric_checks+=1
        return {x['group']:(int(x['actual']),int(x['predicted'])) for x in metrics['records']}
    def error(pred,truth):
        assert set(pred)==set(truth)
        return Fraction(sum(pred[k]!=truth[k] for k in truth),len(truth))
    def votes(weights,members,field):
        maps={p:members[p][field] for p in weights};base=members['p00'][field];out={}
        assert all(set(v)==set(base) for v in maps.values())
        for group in base:
            counts=Counter()
            for p,w in weights.items(): counts[maps[p][group][1]]+=w
            top=max(counts.values());tied=[v for v,n in counts.items() if n==top]
            out[group]=base[group][1] if base[group][1] in tied else min(tied)
        return out
    def loss(weights,members,scope):
        fields=['cv','dev'] if scope=='equal_cv_dev' else [scope]
        return sum((error(votes(weights,members,f),{g:v[0] for g,v in members['p00'][f].items()}) for f in fields),Fraction())/len(fields)
    for source in cfg['sources']:
        archive=Path(source['archive']);manifest=read(archive/'SOURCE_MANIFEST.json')
        assert sha(archive/'SOURCE_MANIFEST.json')==source['manifest_sha256']
        index={x['path']:x for x in manifest['files']}
        def bound(path):
            rel=str(path.relative_to(archive));entry=index[rel]
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
            bindings.append(dict(archive=str(archive),path=rel,sha256=entry['sha256']))
            return read(path)
        main=bound(archive/source['result']);results=main[source['result_key']]
        sessions=bound(archive/'SESSIONS337.json')
        for number,session in enumerate(sessions,1):
            subject=mapped(archive,session['path']);state=bound(subject/'broker/STATE.json');members={}
            prefix='pool_'+str(number) if source['cohort']=='A342' else session['id']
            for ident,entry in sorted(state['programs'].items()):
                report=bound(mapped(archive,entry['path'])/'REPORT.json');assert report['status']=='complete'
                key='common_p00' if ident=='p00' else prefix+'__'+ident
                cv={}
                for fold in report['cv']:
                    vv=view(fold);assert not set(cv)&set(vv);cv.update(vv)
                members[ident]=dict(cv=cv,dev=view(report['development']),final=view(results[key]))
            for field in ['cv','dev','final']:
                base=members['p00'][field]
                assert all({g:v[0] for g,v in m[field].items()}=={g:v[0] for g,v in base.items()} for m in members.values())
            methods={};selection_traces={}
            methods['keep']={'p00':1}
            for rule,scope in [('development','dev'),('mean_cv','cv'),('mean_cv_and_development','equal_cv_dev')]:
                chosen=min(members,key=lambda p:(loss({p:1},members,scope),p!='p00',p))
                methods[rule]={chosen:1}
            methods['uniform_vote']={p:1 for p in members}
            for rule,scope in [('greedy_development','dev'),('greedy_cv','cv'),('greedy_equal_cv_dev','equal_cv_dev')]:
                weights=Counter();best=None;trace=[]
                for step in range(20):
                    options={p:dict(weights+Counter({p:1})) for p in members}
                    chosen=min(members,key=lambda p:(loss(options[p],members,scope),p!='p00',p))
                    weights=Counter(options[chosen]);score=loss(weights,members,scope)
                    trace.append(dict(step=step+1,added=chosen,loss_fraction=str(score),weights=dict(weights)))
                    if best is None or score<best[0]: best=(score,dict(weights))
                methods[rule]=best[1];selection_traces[rule]=trace
            # Selection weights above use cv/dev only;finals are consumed only below.
            final_truth={g:v[0] for g,v in members['p00']['final'].items()};baseline={g:v[1] for g,v in members['p00']['final'].items()}
            scores={};oracle=min(error({g:v[1] for g,v in m['final'].items()},final_truth) for m in members.values())
            for rule,weights in methods.items():
                pred=votes(weights,members,'final');err=error(pred,final_truth);n=len(final_truth)
                scores[rule]=dict(weights=weights,predictions=pred,error=float(err),error_fraction=str(err),
                    harmful_change=sum(baseline[g]==final_truth[g] and pred[g]!=final_truth[g] for g in final_truth)/n,
                    beneficial_change=sum(baseline[g]!=final_truth[g] and pred[g]==final_truth[g] for g in final_truth)/n,
                    signed_gap_to_best_single=float(err-oracle),unique_models_per_record=len(weights))
            original=next((x for x in main.get('episodes',main.get('rows',[])) if x['id']==session['id']),{})
            completed=original.get('completed',original.get('end_to_end_completed'))
            if completed is None and (subject/'BUILD_RESULT.json').exists():completed=bound(subject/'BUILD_RESULT.json')['completed']
            pools.append(dict(batch=source['cohort'],cohort='A347' if source['cohort']=='A396' else source['cohort'],subject=session['id'],
                original_native_completed=completed,strict_registry_deviation=source['cohort']=='A347' and session['id']=='local_revision_2',
                original_submission=state.get('submission'),original_scope='generator_handoff_not_final_selector' if source['cohort']=='A342' else 'native_choice',
                members=members,truth=final_truth,best_single_error=float(oracle),scores=scores,selection_traces=selection_traces))
    assert len(pools)==16
    cohorts={};metrics=['error','harmful_change','beneficial_change','signed_gap_to_best_single']
    for cohort in sorted({p['cohort'] for p in pools}):
        rows=[p for p in pools if p['cohort']==cohort]
        cohorts[cohort]={rule:{k:sum(p['scores'][rule][k] for p in rows)/len(rows) for k in metrics} for rule in cfg['rules']}
    summary={rule:{k:sum(v[rule][k] for v in cohorts.values())/len(cohorts) for k in metrics} for rule in cfg['rules']}
    controls=['keep','development','mean_cv','mean_cv_and_development'];gate={}
    for rule in cfg['rules'][4:]:
        comparisons={control:dict(strictly_better_cohorts=sum(v[rule]['error']<v[control]['error']-1e-12 for v in cohorts.values()),
            mean_error_lower=summary[rule]['error']<summary[control]['error']-1e-12,
            harm_not_higher=summary[rule]['harmful_change']<=summary[control]['harmful_change']+1e-12,
            gains_not_lower=summary[rule]['beneficial_change']>=summary[control]['beneficial_change']-1e-12) for control in controls}
        gate[rule]=dict(comparisons=comparisons,passed=all(v['strictly_better_cohorts']>=2 and v['mean_error_lower'] and v['harm_not_higher'] and v['gains_not_lower'] for v in comparisons.values()))
    output=dict(utc=utc(),status='complete',scope=cfg['scope'],pools=pools,cohorts=cohorts,summary=summary,gate=gate,
        source_metrics_recounted=metric_checks,source_bindings=bindings,elapsed_seconds=time.monotonic()-start,
        batch_count=4,related_pools=16,training_cohorts=3,physical_rigs=1,new_LLM_calls=0,new_PHM_fits=0,new_PHM_forward_calls=0,
        confirmation_data_opened=False,method_novelty_claim=False,goal_complete=False,
        limits='Record-level algebraic replay,not new end-to-end execution;probabilities absent,so hard-vote controls only. Pools/models are dependent;all data already opened. Greedy controls optimize legal labels and are ordinary baselines. More unique members increase deployment inference cost.')
    save(root/'RESULT397.json',output);save(PROJECT/'results/ensemble397/RESULT397.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['pools','source_bindings']},ensure_ascii=False))


def ensemble_audit(root):
    """Independent NumPy tally and exact greedy-optimum replay on cached labels."""
    import math
    from fractions import Fraction
    import numpy as np
    cfg=read(root/'PROTOCOL397.json');result=read(root/'RESULT397.json');checks=0;failures=[]
    def check(a,b,where):
        nonlocal checks
        checks+=1
        ok=math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) if isinstance(b,float) else a==b
        if not ok: failures.append(dict(where=where,actual=a,expected=b))
    for b in result['source_bindings']:
        check(sha(Path(b['archive'])/b['path']),b['sha256'],'source/'+b['path'])
    check(sha(root/'audit_program_evidence.py'),cfg['script_sha256'],'frozen_evaluator')
    actual_comparisons=[];outcomes=[]
    for pool in result['pools']:
        uid=pool['batch']+'/'+pool['subject'];members=pool['members'];names=sorted(members);base_index=names.index('p00');views={}
        for field in ['cv','dev','final']:
            groups=sorted(members['p00'][field]);truth=np.array([members['p00'][field][g][0] for g in groups])
            matrix=np.array([[members[name][field][g][1] for g in groups] for name in names])
            views[field]=(groups,truth,matrix)
        def tally(weights,field):
            groups,truth,matrix=views[field]
            ww=np.array([weights.get(p,0) for p in names])
            counts=np.stack([((matrix==label)*ww[:,None]).sum(axis=0) for label in range(4)])
            pred=counts.argmax(axis=0);parent=matrix[base_index];idx=np.arange(len(groups))
            tie=(counts[parent,idx]==counts.max(axis=0));pred[tie]=parent[tie]
            return dict(zip(groups,pred.tolist())),Fraction(int((pred!=truth).sum()),len(groups))
        def legal(weights,scope):
            fields=['cv','dev'] if scope=='equal_cv_dev' else [scope]
            return sum((tally(weights,field)[1] for field in fields),Fraction())/len(fields)
        for rule,scope in [('development','dev'),('mean_cv','cv'),('mean_cv_and_development','equal_cv_dev')]:
            best=min(names,key=lambda p:(legal({p:1},scope),p!='p00',p))
            check(pool['scores'][rule]['weights'],{best:1},uid+'/'+rule+'/selection')
        check(pool['scores']['uniform_vote']['weights'],{p:1 for p in names},uid+'/all_actual_members')
        for rule,scope in [('greedy_development','dev'),('greedy_cv','cv'),('greedy_equal_cv_dev','equal_cv_dev')]:
            weights={};best=None
            for i,step in enumerate(pool['selection_traces'][rule]):
                choices={p:{q:weights.get(q,0)+(q==p) for q in names if weights.get(q,0)+(q==p)>0} for p in names}
                ident=min(names,key=lambda p:(legal(choices[p],scope),p!='p00',p));weights=choices[ident];score=legal(weights,scope)
                check(step['added'],ident,uid+'/'+rule+'/'+str(i)+'/addition')
                check(step['weights'],weights,uid+'/'+rule+'/'+str(i)+'/weights')
                check(step['loss_fraction'],str(score),uid+'/'+rule+'/'+str(i)+'/loss')
                if best is None or score<best[0]:best=(score,dict(weights))
            check(len(pool['selection_traces'][rule]),20,uid+'/'+rule+'/steps')
            check(pool['scores'][rule]['weights'],best[1],uid+'/'+rule+'/earliest_best')
        base={g:v[1] for g,v in members['p00']['final'].items()};truth=pool['truth'];n=len(truth);row={}
        oracle=min(tally({p:1},'final')[1] for p in names)
        for rule,score in pool['scores'].items():
            pred,err=tally(score['weights'],'final')
            check(score['predictions'],pred,uid+'/'+rule+'/predictions');check(score['error'],float(err),uid+'/'+rule+'/error')
            bad=sum(base[g]==truth[g] and pred[g]!=truth[g] for g in truth)/n
            good=sum(base[g]!=truth[g] and pred[g]==truth[g] for g in truth)/n
            values=dict(error=float(err),harmful_change=bad,beneficial_change=good,signed_gap_to_best_single=float(err-oracle))
            for k,v in values.items():check(score[k],v,uid+'/'+rule+'/'+k)
            check(float(err),float(tally({'p00':1},'final')[1])+bad-good,uid+'/'+rule+'/harm_gain_identity')
            row[rule]=values
        outcomes.append(dict(cohort=pool['cohort'],values=row))
        sub=pool.get('original_submission') or {};selected=sub.get('program_id') or sub.get('selected_program')
        if selected in names and pool['original_scope']=='native_choice' and pool['original_native_completed']:
            native=float(tally({selected:1},'final')[1]);vote=row['uniform_vote']['error']
            actual_comparisons.append(dict(batch=pool['batch'],subject=pool['subject'],selected=selected,
                native_error=native,vote_error=vote,vote_minus_native=vote-native,
                strict_registry_deviation=pool['strict_registry_deviation']))
    summaries={}
    for rule in cfg['rules']:
        summaries[rule]={}
        for metric in ['error','harmful_change','beneficial_change','signed_gap_to_best_single']:
            means=[]
            for cohort in sorted({x['cohort'] for x in outcomes}):
                vv=[x['values'][rule][metric] for x in outcomes if x['cohort']==cohort];mean=sum(vv)/len(vv)
                check(result['cohorts'][cohort][rule][metric],mean,cohort+'/'+rule+'/'+metric);means.append(mean)
            summaries[rule][metric]=sum(means)/len(means)
            check(result['summary'][rule][metric],summaries[rule][metric],rule+'/'+metric+'/summary')
    for rule,gate in result['gate'].items():
        passed=True
        for control in ['keep','development','mean_cv','mean_cv_and_development']:
            vv=dict(strictly_better_cohorts=sum(result['cohorts'][c][rule]['error']<result['cohorts'][c][control]['error']-1e-12 for c in result['cohorts']),
                mean_error_lower=summaries[rule]['error']<summaries[control]['error']-1e-12,
                harm_not_higher=summaries[rule]['harmful_change']<=summaries[control]['harmful_change']+1e-12,
                gains_not_lower=summaries[rule]['beneficial_change']>=summaries[control]['beneficial_change']-1e-12)
            check(gate['comparisons'][control],vv,rule+'/'+control+'/gate')
            passed=passed and vv['strictly_better_cohorts']>=2 and vv['mean_error_lower'] and vv['harm_not_higher'] and vv['gains_not_lower']
        check(gate['passed'],bool(passed),rule+'/gate')
    output=dict(utc=utc(),passed=not failures,checks=checks,failures=failures,source_bindings=len(result['source_bindings']),
        cached_greedy_rounds_recounted=16*3*20,actual_completed_native_comparisons=actual_comparisons,
        native_comparison_limit='Posthoc same-bank aggregation;not a fresh equal-budget Agent comparison. Incomplete episodes and generator handoffs absent from this descriptive paired table,retained in main16-pool denominator. Registry deviation remains flagged.',
        new_LLM_calls=0,new_PHM_fits=0,new_PHM_forward_calls=0,unopened_confirmation_read=False)
    save(root/'INDEPENDENT_AUDIT397.json',output);save(PROJECT/'results/ensemble397/INDEPENDENT_AUDIT397.json',output)
    print(json.dumps(output,ensure_ascii=False))
    assert not failures


def ensemble_native_context(root):
    """Add the old A342 final selectors;do not confuse builder handoffs with them."""
    result=read(root/'RESULT397.json');cfg=read(root/'PROTOCOL397.json')
    assert read(root/'INDEPENDENT_AUDIT397.json')['passed']
    source=next(s for s in cfg['sources'] if s['cohort']=='A342');archive=Path(source['archive'])
    manifest=read(archive/'SOURCE_MANIFEST.json');entry=next(x for x in manifest['files'] if x['path']==source['result'])
    assert sha(archive/source['result'])==entry['sha256']
    main=read(archive/source['result']);pools=[p for p in result['pools'] if p['batch']=='A342'];rows=[]
    for selector in main['selectors']:
        for decision in selector['rows']:
            pool=pools[int(decision['case'].split('_')[1])-1];selected=decision['selected']
            records=pool['members'][selected]['final'];n=len(records)
            actual=sum(a!=p for a,p in records.values())/n
            assert abs(actual-decision['record_error'])<1e-12
            rows.append(dict(selector=selector['id'],arm=selector['condition'],completed=selector['completed'],
                pool=decision['case'],selected=selected,native_error=actual,vote_error=pool['scores']['uniform_vote']['error']))
    summary={}
    for selector in sorted({r['selector'] for r in rows}):
        rr=[r for r in rows if r['selector']==selector]
        summary[selector]=dict(mean_native_error=sum(r['native_error'] for r in rr)/len(rr),mean_vote_error=sum(r['vote_error'] for r in rr)/len(rr),
            wins=sum(r['vote_error']<r['native_error'] for r in rr),ties=sum(r['vote_error']==r['native_error'] for r in rr),losses=sum(r['vote_error']>r['native_error'] for r in rr))
    output=dict(utc=utc(),rows=rows,summary=summary,source_sha256=entry['sha256'],
        limits='Historical fixed-bank final selectors,not four new independent experiments. Each sees all4related banks;repeats give identical choices. No end-to-end fresh method comparison.',
        maximum_unique_member_count=max(len(p['members']) for p in result['pools']),minimum_unique_member_count=min(len(p['members']) for p in result['pools']),
        budgets_note='Voting uses2to5modelpredictions per record versus1selected model. Future experiments must measure full wrapper time and allow identical tools/cost cap. No cost superiority is claimed.',
        new_LLM_calls=0,new_PHM_fits=0,new_PHM_forward_calls=0)
    path=root/'NATIVE_CONTEXT397.json';assert not path.exists();save(path,output);save(PROJECT/'results/ensemble397/NATIVE_CONTEXT397.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},ensure_ascii=False))


def portfolio(root):
    """Recount A399 cold outputs and chain costs independently of its scorer."""
    import numpy as np
    result=read(root/'RESULT399.json');cfg=read(root/'PROTOCOL399.json');seal=read(root/'ALL_CHOICES_SEALED337.json')
    assert len(seal['episodes'])==4 and read(root/'HANDLE337.json')['status']=='terminal'
    for relative,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/relative)==digest
    source=Path(cfg['source_archive']);raw=source/'qualification/final.npz'
    assert sha(raw)==read(source/'QUALIFICATION346.json')['files']['final.npz']['sha256']
    with np.load(raw,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    checks=[];recount={}
    def check(actual,expected,label):
        assert actual==expected,(label,actual,expected)
        checks.append(label)
    for name,value in result['all_program_results'].items():
        if value['status']!='complete':continue
        folder=root/'evaluation'/name
        with np.load(folder/'forward/PREDICTIONS.npz',allow_pickle=False) as f:ids=f['sample_ids'];pred=f['prediction']
        with np.load(folder/'cold_repeat/PREDICTIONS.npz',allow_pickle=False) as f:
            check(np.array_equal(ids,f['sample_ids']) and np.array_equal(pred,f['prediction']),True,name+'/coldrepeat')
        check(np.array_equal(ids,data['sample_ids']),True,name+'/ids');rows={};matrix=[[0]*4 for _ in range(4)]
        for group in sorted(set(data['groups'].tolist())):
            indices=[i for i,g in enumerate(data['groups']) if g==group];votes=Counter(int(pred[i]) for i in indices)
            selected=min(votes,key=lambda k:(-votes[k],k));truth={int(data['labels'][i]) for i in indices};assert len(truth)==1
            actual=truth.pop();rows[group]=dict(actual=actual,predicted=selected);matrix[actual][selected]+=1
        recount[name]=rows
        check(value['record_error'],sum(v['actual']!=v['predicted'] for v in rows.values())/len(rows),name+'/recorderror')
        check(value['confusion'],matrix,name+'/confusion')
        for row in value['records']:check((row['actual'],row['predicted']),(rows[row['group']]['actual'],rows[row['group']]['predicted']),name+'/'+row['group'])
        actual_processes=list(folder.rglob('PROCESS.json'))
        check(value['actual_prediction_processes'],len(actual_processes),name+'/processcount')
        for p in actual_processes:
            process=read(p);audit=read(p.with_name('training_audit.json'))
            check((process['returncode'],audit['mode'],audit['completed_estimator_fits'],audit['optimizer_steps']),(0,'predict',0,0),str(p.relative_to(root)))
    base=recount['common_p00'];best=min(sum(r['actual']!=r['predicted'] for r in rr.values())/len(rr) for rr in recount.values())
    check(result['common_union_best_error'],best,'commonunionbest')
    for episode in result['episodes']:
        lookup=lambda k:recount['common_p00' if k=='p00' else episode['id']+'__'+k]
        for label,v in episode['methods'].items():
            if v['evaluation_status']!='complete':continue
            rr=lookup(v['program_id']);n=len(rr)
            check(v['record_harm'],sum(base[g]['actual']==base[g]['predicted'] and x['actual']!=x['predicted'] for g,x in rr.items())/n,episode['id']+'/'+label+'/harm')
            check(v['record_gain'],sum(base[g]['actual']!=base[g]['predicted'] and x['actual']==x['predicted'] for g,x in rr.items())/n,episode['id']+'/'+label+'/gain')
            check(v['common_union_selection_loss'],v['record_error']-best,episode['id']+'/'+label+'/regret')
        vote=episode['same_bank_equal_vote']
        if vote['status']=='complete':
            for row in vote['records']:
                g=row['group'];counts=Counter(lookup(k)[g]['predicted'] for k in vote['base_programs']);tied={k for k,v in counts.items() if v==max(counts.values())}
                anchor=lookup(vote['anchor'])[g]['predicted'];expected=anchor if anchor in tied else min(tied)
                check(row['predicted'],expected,episode['id']+'/vote/'+g)
            check(vote['record_error'],sum(r['actual']!=r['predicted'] for r in vote['records'])/len(vote['records']),episode['id']+'/vote_error')
    for arm,value in result['primary'].items():
        rows=[e for e in result['episodes'] if e['arm']==arm]
        valid=all(e['completed'] and e['methods'].get('actor',{}).get('evaluation_status')=='complete' for e in rows)
        check(value['valid'],valid,arm+'/valid')
        if valid:
            for key in ['record_error','record_harm','record_gain','common_union_selection_loss']:
                check(value[key],sum(e['methods']['actor'][key] for e in rows)/len(rows),arm+'/'+key)
    a,b=result['primary']['ordinary'],result['primary']['portfolio']
    gate=all(v['status']=='complete' for v in result['all_program_results'].values()) and a['valid'] and b['valid']
    if gate:gate=b['record_error']<a['record_error'] and b['record_error']<result['baseline_error'] and b['record_harm']<=a['record_harm'] and b['record_gain']>=a['record_gain']
    check(result['joint_pilot_gate_passed'],bool(gate),'joint_gate')
    check(result['context_attribution_gate_passed'],bool(gate and a['same_bank_vote_error'] is not None and b['record_error']<a['same_bank_vote_error']),'attribution_gate')
    receipts=[read(p) for pattern in ['subjects/*/model_gateway/*/*_receipt.json','chains/*/model_gateway/*/*_receipt.json'] for p in root.glob(pattern)]
    requests=[p for pattern in ['subjects/*/model_gateway/*/*_request.json','chains/*/model_gateway/*/*_request.json'] for p in root.glob(pattern)]
    check(result['cost']['actual_requests'],len(requests),'requests');check(result['cost']['receipts'],len(receipts),'receipts')
    for field,key in [('input_tokens','input_tokens'),('output_tokens','output_tokens')]:check(result['cost'][field],sum(r.get('usage',{}).get(key,0) for r in receipts),field)
    stage_paths=list(root.glob('subjects/*/STAGE_RESULT399.json'));stages=[read(p) for p in stage_paths]
    chain_costs=[]
    for repeat in [1,2]:
        usage=read(root/'chains'/('portfolio_'+str(repeat))/'model_gateway/builders/status.json')
        selector=read(root/'subjects'/('selector_'+str(repeat))/'MODEL_GATEWAY_CONFIG.json')
        check(selector['max_input_tokens'],min(1000000,max(0,4000000-usage['input_tokens'])),'selector_residual_'+str(repeat))
        total=[r for pattern in ['chains/portfolio_'+str(repeat)+'/model_gateway/*/*_receipt.json','subjects/selector_'+str(repeat)+'/model_gateway/*/*_receipt.json'] for p in root.glob(pattern) for r in [read(p)]]
        check(len(total)<=60,True,'chain_requestcap_'+str(repeat));check(sum(r.get('usage',{}).get('output_tokens',0) for r in total)<=131072,True,'chain_outputcap_'+str(repeat))
        check(usage['peak_inflight']<=1,True,'serial_shared_builder_accounting_'+str(repeat))
        chain_costs.append(dict(repeat=repeat,requests=len(total),input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in total),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in total)))
    output=dict(utc=utc(),passed=True,checks=len(checks),check_names=checks,cold_programs_recounted=len(recount),
        ended_native_stages=len(stages),normal_native_stages=sum(s['completed'] for s in stages),incomplete_native_stages=sum(not s['completed'] for s in stages),
        chain_costs=chain_costs,qualification_new_fits=0,qualification_prediction_processes=48,
        final_confirmation_data_read=False,scope='Recounts this previously opened developmental cohort only;not independent method confirmation.',new_LLM_calls=0,new_PHM_fits=0,new_predictions=0)
    save(root/'INDEPENDENT_AUDIT399.json',output);save(PROJECT/'results/portfolio399/INDEPENDENT_AUDIT399.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='check_names'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run','submission','submission_trace','check_only_control','ensemble_freeze','ensemble_run','ensemble_audit','ensemble_native_context','portfolio']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.run.resolve())
