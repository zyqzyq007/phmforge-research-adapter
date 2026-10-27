"""Join sealed A/C choices to identical archived real predictions and compare A/B/C.

No fitting or model inference; old final predictions are hash verified and
independently scored. The researcher's prior access is explicitly retained.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics

from controlled_feedback import save, sha, utc
from evaluate_competing_hypotheses import independent_score


def read(path):return json.loads(path.read_text())


def evaluate(root):
    import numpy as np
    config=read(root/'CAMPAIGN_CONFIG.json');source=Path(config['reuse_source'])
    suffix=config['iteration'][1:]
    seal=read(root/'ALL_CHOICES_SEALED.json');assert len(seal['episodes'])==config['episodes']
    assert seal['protocol_sha256']==sha(root/'CAMPAIGN_CONFIG.json')
    assert not (root/('FINAL_RESULTS'+suffix+'.json')).exists(),'Do not repeat the sealed comparison'
    assert config['researcher_previously_knows_final_scores'] and not config['confirmation_experiment']
    changed=[r['path'] for r in seal['files'] if sha(root/r['path'])!=r['sha256']]
    assert not changed,'Changed sealed files: '+str(changed)
    manifest=source/'SOURCE_RECOVERY_MANIFEST.json'
    assert sha(manifest)==config['reuse_source_manifest_sha256']
    index={r['path']:r['sha256'] for r in read(manifest)['files']}
    verified=[]
    def bound(relative):
        path=source/relative;assert sha(path)==index[relative]
        verified.append(dict(path=relative,sha256=index[relative]));return path
    old=read(bound('FINAL_RESULTS325.json'));old_seal=read(bound('ALL_CHOICES_SEALED.json'))
    folder=root/'evaluation';folder.mkdir(exist_ok=False)
    save(folder/'START.json',dict(utc=utc(),new_choices_previously_sealed=True,researcher_previously_knows_final_scores=True,
        new_fits=0,new_model_forwards=0,reused_final_inference_source=str(source/'evaluation')))
    scores={}
    for block in config['blocks']:
        b=block['id'];scores[b]={}
        with np.load(bound('hidden/'+b+'/test.npz'),allow_pickle=False) as z:truth=dict(z)
        for candidate in config['candidate_ids']:
            p1=read(bound('evaluation/'+b+'/'+candidate+'/predictions1.json'))
            p2=read(bound('evaluation/'+b+'/'+candidate+'/predictions2.json'))
            assert p1==p2 and p1['new_fits']==0 and not p1['input_labels_present']
            result=independent_score(truth,p1['predictions'])
            original=old['candidate_scores'][b][candidate]
            assert all(result[k]==original[k] for k in result)
            scores[b][candidate]=result
        save(folder/(b+'_independent_scores.json'),scores[b])
    outcomes=[]
    for e in seal['episodes']:
        path=root/'episodes'/e['id'];bank=scores[e['block']];chosen=bank[e['selected']]
        cost=read(path/'model_gateway'/e['id']/'status.json')
        state=read(path/'broker/STATE.json') if (path/'broker/STATE.json').exists() else dict(checks={})
        receipts=[read(p) for p in (path/'model_gateway').glob('*/*_receipt.json')]
        assert len(receipts)==cost['requests'] and not cost['active'] and not cost['inflight']
        assert sum(r.get('usage',{}).get('input_tokens',0) for r in receipts)==cost['input_tokens']
        assert sum(r.get('usage',{}).get('output_tokens',0) for r in receipts)==cost['output_tokens']
        limits=config['episode_budget']
        assert cost['requests']<=limits['model_requests'] and cost['output_tokens']<=limits['output_tokens']
        error=chosen['record_error'];old_error=bank['P00']['record_error'];oracle=min(x['record_error'] for x in bank.values())
        elapsed=e.get('elapsed_seconds')
        elapsed_source='episode_controller'
        if elapsed is None:
            # A controller-level error may only preserve terminal stage clocks.
            # Their union is wall time (parallel branches are not summed).
            elapsed_source='first_stage_start_to_last_stage_end'
            elapsed=(max(datetime.fromisoformat(s['ended_utc']) for s in e['stages'])-
                     min(datetime.fromisoformat(s['started_utc']) for s in e['stages'])).total_seconds() if e['stages'] else 0.0
        access=[json.loads(s) for s in (path/'broker/ACCESS.jsonl').read_text().splitlines()] if (path/'broker/ACCESS.jsonl').exists() else []
        failures=[x for x in access if not x['response']['ok']]
        plan_times=[x['ended_utc'] for x in access if x['response']['ok'] and json.loads(x['request'])['action']=='plan']
        checks=[x for x in access if x['response']['ok'] and json.loads(x['request'])['action']=='check' and json.loads(x['request'])['candidate_id'] not in ['P00','P11B']]
        ordering=all(plan_times and x['started_utc']>=min(plan_times) for x in checks)
        assert ordering,'Intermediate check preceded registered plan'
        if e['arm']=='C' and e['completed']:
            handoff=read(path/'audit/INDEPENDENT_BRANCH_HANDOFFS.json')
            for h in handoff['files']:
                assert sha(path/'branches'/h['branch']/'app'/h['file'])==h['sha256']
                assert sha(path/'app/inputs/branches'/h['branch']/h['file'])==h['sha256']
        outcomes.append(dict(id=e['id'],block=e['block'],arm=e['arm'],repeat=e['repeat'],source_campaign=config['iteration'],
            completed=e['completed'],fallback=e['fallback'],selected=e['selected'],
            failure_reason=e.get('error'),failure_type=e.get('error_type'),
            proposed=(e.get('choice') or {}).get('proposed'),gate_reason=(e.get('choice') or {}).get('gate_reason'),
            record_error=error,record_correct=chosen['record_correct'],record_total=chosen['record_total'],
            selection_regret=error-oracle,harm=max(error-old_error,0),gain=max(old_error-error,0),opportunity=old_error-oracle,
            window_accuracy=chosen['window_accuracy'],window_macro_f1=chosen['window_macro_f1'],
            model_requests=cost['requests'],input_tokens=cost['input_tokens'],output_tokens=cost['output_tokens'],
            input_budget_overrun=max(0,cost['input_tokens']-limits['input_tokens']),peak_model_inflight=cost['peak_inflight'],
            returned_model_ids=cost['returned_model_ids'],provider_status_counts=dict(Counter(str(r.get('upstream_status')) for r in receipts)),
            provider_accounting_errors=[r for r in receipts if r.get('accounting_error')],
            provider_usage_missing=sum(not {'input_tokens','output_tokens'}<=r.get('usage',{}).keys() for r in receipts),
            token_cost_complete=all({'input_tokens','output_tokens'}<=r.get('usage',{}).keys() for r in receipts),
            output_token_upper_bound=cost['output_tokens']+sum(r['effective_max_output_tokens'] for r in receipts if 'output_tokens' not in r.get('usage',{})),
            elapsed_seconds=elapsed,elapsed_source=elapsed_source,distinct_checks=len(state['checks']),
            check_seconds=sum(x['elapsed_seconds'] for x in state['checks'].values()),
            check_replays=sum(x['actual_cold_models_replayed'] for x in state['checks'].values()),
            stages=e['stages'],tool_rejections=failures,intermediate_checks_after_registered_plan=ordering))
    old_details={e['id']:e for e in old_seal['episodes']}
    for e in old['episodes']:
        detail=old_details[e['id']]
        outcomes.append(dict(e,source_campaign='A325',elapsed_seconds=detail['elapsed_seconds'],
            proposed=detail['choice']['proposed'],gate_reason=detail['choice']['gate_reason'],stages=detail['stages']))
    engineering_attempts=[];parent_source=None
    if config.get('engineering_recovery'):
        parent=Path(config['comparison_parent_run'])
        if not parent.exists():parent=Path(config['comparison_parent_archive'])
        previous=read(parent/'FINAL_RESULTS326.json')
        parent_seal=read(parent/'ALL_CHOICES_SEALED.json')
        assert previous['protocol_sha256']==parent_seal['protocol_sha256']==sha(parent/'CAMPAIGN_CONFIG.json')
        prior_A=[e for e in previous['episodes'] if e['arm']=='A'];assert len(prior_A)==4
        for e in prior_A:
            assert e['record_error']==scores[e['block']][e['selected']]['record_error']
        outcomes.extend(prior_A)
        engineering_attempts=[e for e in previous['episodes'] if e['arm']=='C']
        assert len(engineering_attempts)==4
        parent_source=dict(run=str(parent),final_sha256=sha(parent/'FINAL_RESULTS326.json'),
            choices_sha256=sha(parent/'ALL_CHOICES_SEALED.json'),invalidation=read(parent/'audit/C_HARNESS_INVALIDATION.json'))
    summary={}
    for arm in ['ordinary','A','B','C']:
        rows=[x for x in outcomes if x['arm']==arm];assert len(rows)==4
        blocks=sorted(set(x['block'] for x in rows))
        def mean(metric):return statistics.mean(statistics.mean(x[metric] for x in rows if x['block']==b) for b in blocks)
        metrics={k:mean(k) for k in ['record_error','selection_regret','harm','gain','opportunity','window_accuracy','window_macro_f1',
            'model_requests','input_tokens','output_tokens','elapsed_seconds','check_seconds','distinct_checks']}
        summary[arm]=dict(metrics,record_accuracy=1-metrics['record_error'],episodes=4,
            completed=sum(x['completed'] for x in rows),fallbacks=sum(x['fallback'] for x in rows),
            choices=dict(Counter(x['selected'] for x in rows)),harmful_updates=sum(x['harm']>1e-12 for x in rows),
            improvement_capture=sum(x['gain'] for x in rows)/sum(x['opportunity'] for x in rows) if sum(x['opportunity'] for x in rows)>0 else None,
            stage_sessions=sum(len(x['stages']) for x in rows),
            finished_stage_sessions=sum(s['status']=='finished' for x in rows for s in x['stages']),
            total_requests=sum(x['model_requests'] for x in rows),total_input_tokens=sum(x['input_tokens'] for x in rows),
            total_output_tokens=sum(x['output_tokens'] for x in rows))
        normal=[x for x in rows if x['completed'] and not x['fallback']]
        summary[arm]['normal_choice_count']=len(normal)
        summary[arm]['normal_choice_mean_record_accuracy']=(1-statistics.mean(x['record_error'] for x in normal)) if normal else None
        summary[arm]['token_cost_complete']=all(x.get('token_cost_complete',True) for x in rows)
        summary[arm]['provider_usage_missing']=sum(x.get('provider_usage_missing',0) for x in rows)
        summary[arm]['complete_comparison_available']=len(normal)==4
        summary[arm]['system_accuracy_includes_predeclared_fallbacks']=bool(summary[arm]['fallbacks'])
    result=dict(iteration=config['iteration'],completed_utc=utc(),protocol_sha256=seal['protocol_sha256'],
        new_choices_sealed_before_score_join=True,researcher_previously_knows_final_scores=True,confirmation_experiment=False,
        all_reused_predictions_match_original=True,new_PHM_fits=0,reused_real_PHM_fits=60,
        new_final_model_forwards=0,reused_actual_final_cold_processes=20,independent_motor_count=1,
        unique_final_records=40,physical_blocks=2,new_episodes=config['episodes'],historical_comparator_episodes=len(outcomes)-config['episodes'],
        candidate_scores=scores,episodes=outcomes,summary=summary,fixed_controls=old['fixed_controls'],
        engineering_attempts_retained=engineering_attempts,parent_comparison_source=parent_source,
        verified_original_files=verified,
        limitations=['Previously opened data: exploratory architecture comparison only',
            'One laboratory motor, two related acquisition blocks; repeats are decisions, not new devices',
            'Fixed-candidate program selection, not unrestricted code improvement',
            'No candidate improves on P00 final performance: improvement capture is unavailable',
            'Architecture packages and different stage caps, not single-mechanism ablation',
            'A/C max2 simultaneous episodes vs historical ordinary/B max4; wall time descriptive'])
    save(root/('FINAL_RESULTS'+suffix+'.json'),result)
    out=Path(config['project'])/config['result_directory'];save(out/('FINAL_RESULTS'+suffix+'.json'),result)
    save(root/'audit/POST_EVALUATION_INTEGRITY.json',dict(passed=True,checked_sealed_files=len(seal['files']),
        changed_files=[],all_reused_predictions_match_original=True,new_final_model_forwards=0,
        all_new_choices_sealed_before_score_join=True,verified_original_files=len(verified)))
    print(json.dumps(dict(summary=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run.resolve())
