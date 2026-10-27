"""Read-only audit of strict validation disagreement versus uninformative ties."""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import datetime
import hashlib
import json
import math

root=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fraction(value):
    assert math.isfinite(value) and 0<=value<=1
    out=F(value).limit_denominator(100000)
    assert abs(float(out)-value)<1e-12
    return out
def mean(values):return sum(values,F(0))/len(values)
def sign(value):return (value>0)-(value<0)
def category(cv,val):
    if cv==val==0:return 'both_tie'
    if val==0:return 'CV_only'
    if cv==0:return 'validation_only'
    return 'strict_conflict' if sign(cv)!=sign(val) else 'same_direction'

cfg=read(root/'PROTOCOL351.json')
assert sha(Path(__file__))==cfg['script_sha256']
assert not (root/'STARTED351.json').exists()
(root/'STARTED351.json').write_text(json.dumps(dict(utc=utc(),new_LLM_calls=0,new_fits=0,new_predictions=0))+'\n')
archive=Path(cfg['source_archive']);receipt=read(archive/'ARCHIVE_VERIFIED.json')
assert sha(archive/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']==cfg['manifest_sha256']
manifest={v['path']:v for v in read(archive/'SOURCE_MANIFEST.json')['files']}
for name,digest in cfg['input_bindings'].items():
    path=archive/name
    assert sha(path)==digest==manifest[name]['sha256'] and path.stat().st_size==manifest[name]['bytes']
payload=read(archive/'recovery2/PROGRAM_EVIDENCE_CORPUS348.json');corpus=payload['programs']
bound_index={(x['cohort'],x['path']):x['sha256'] for x in payload['bindings']}
source_archives={x['cohort']:Path(x['archive']) for x in cfg['original_sources']}
for source in cfg['original_sources']:
    assert sha(Path(source['archive'])/'SOURCE_MANIFEST.json')==source['manifest_sha256']
bound_cache={}
def bound(cohort,name):
    key=(cohort,name)
    if key not in bound_cache:
        path=source_archives[cohort]/name;assert sha(path)==bound_index[key]
        bound_cache[key]=read(path)
    return bound_cache[key]
def actual_condition_errors(value):
    rows=value['records'];assert len({x['group'] for x in rows})==len(rows)
    loss=F(sum(x['actual']!=x['predicted'] for x in rows),len(rows))
    assert loss==fraction(value['record_error'])
    values=[]
    for hz in sorted({x['condition_hz'] for x in rows}):
        selected=[x for x in rows if x['condition_hz']==hz]
        values.append(F(sum(x['actual']!=x['predicted'] for x in selected),len(selected)))
    return values
old=read(archive/'recovery2/RESULT348.json');assert len(corpus)==36 and len(old['pools'])==12
assert category(F(-1,4),F(0))=='CV_only' and category(F(-1,4),F(1,4))=='strict_conflict'
assert category(F(0),F(0))=='both_tie'
rows=[];pool_summaries=[];cohort_index=Counter();verified=0
for pool in old['pools']:
    cohort=pool['cohort'];cohort_index[cohort]+=1;members={}
    primary=pool['original_native_completed'] and not pool['strict_registry_deviation']
    candidates=[name for c,name in bound_index if c==cohort and name.endswith('/'+pool['subject']+'/broker/STATE.json')]
    assert len(candidates)==1
    state=bound(cohort,candidates[0]);probe=state.get('probe') or {}
    checks=probe.get('reports',{}) if probe.get('status')=='complete' else {}
    common=candidates[0].removesuffix('/broker/STATE.json')+'/common_check/REPORT.json'
    if (cohort,common) in bound_index:checks=bound(cohort,common)['programs']
    for ident in pool['programs']:
        prefix='pool_'+str(cohort_index[cohort]) if cohort=='A342' else pool['subject']
        uid=cohort+'/'+('common_p00' if ident=='p00' else prefix+'__'+ident)
        value=corpus[uid];assert value['program']==ident
        cv=mean([fraction(x) for x in value['cv_errors']])
        assert abs(float(cv)-value['rule_values']['mean_cv'])<1e-12
        check=checks.get(ident);val=None
        if check is not None:
            val=mean([fraction(x) for x in value['development_condition_errors'].values()]+actual_condition_errors(check))
        if ident!='p00':
            original=value['rule_values']['full_checkpoint_equal_condition']
            assert (val is None and original is None) or (val is not None and abs(float(val)-original)<1e-12)
        members[ident]=dict(CV=cv,validation=val,final=fraction(value['final_error']))
        verified+=1
    local=[]
    for incumbent,candidate in combinations(sorted(members),2):
        a,b=members[incumbent],members[candidate];dc=b['CV']-a['CV']
        dv=b['validation']-a['validation'] if a['validation'] is not None and b['validation'] is not None else None
        row=dict(cohort=cohort,pool=pool['subject'],incumbent=incumbent,candidate=candidate,
            strict_completed_pool=primary,incumbent_is_common_start=incumbent=='p00',
            adjacent_registration_ids=int(candidate[1:])-int(incumbent[1:])==1,
            not_claimed_to_be_parent_child=True,CV_delta=float(dc),validation_delta=float(dv) if dv is not None else None,
            incumbent_final_error=float(a['final']),candidate_final_error=float(b['final']),
            category=category(dc,dv) if dv is not None else 'missing_full_validation')
        if dv is not None:
            decisions=dict(keep=False,CV=dc<0,full_validation=dv<0,equal_fusion=dc+dv<0,
                nonconflicting_gain=dc<=0 and dv<=0 and (dc<0 or dv<0))
            oracle=min(a['final'],b['final']);methods={}
            for method,accept in decisions.items():
                loss=b['final'] if accept else a['final']
                methods[method]=dict(program=candidate if accept else incumbent,
                    final_error=float(loss),selection_loss=float(loss-oracle))
            row['methods']=methods
            row['CV_versus_validation_final_difference']=float(abs(
                fraction(methods['CV']['final_error'])-fraction(methods['full_validation']['final_error'])))
        rows.append(row);local.append(row)
    pool_summaries.append(dict(cohort=cohort,pool=pool['subject'],strict_completed_pool=primary,
        original_native_completed=pool['original_native_completed'],strict_registry_deviation=pool['strict_registry_deviation'],
        program_count=len(members),pair_count=len(local),categories=dict(Counter(x['category'] for x in local))))
assert verified==45
def summary(values):
    strict=[r for r in values if r['category']=='strict_conflict']
    return dict(pairs=len(values),categories=dict(Counter(r['category'] for r in values)),
        pools_with_strict_conflict=len({(r['cohort'],r['pool']) for r in strict}),
        strict_conflict_pairs_with_nonzero_final_difference=sum(r['CV_versus_validation_final_difference']>0 for r in strict))
primary=[r for r in rows if r['strict_completed_pool']]
result=dict(utc=utc(),status='complete',input_memberships_verified=verified,unique_program_instances=36,
    pool_specific_source_metadata_verified=len(bound_cache),A348_pool_results_unchanged=True,
    planned_pools=12,strict_completed_pools=sum(p['strict_completed_pool'] for p in pool_summaries),
    all_pairs_summary=summary(rows),strict_completed_summary=summary(primary),pools=pool_summaries,pairs=rows,
    strict_conflict_cases=[r for r in primary if r['category']=='strict_conflict'],
    already_opened_development_only=True,all_pairs_not_natural_update_frequency=True,
    three_related_training_cohorts_one_physical_rig=True,primary_method_superiority_test=False,
    new_LLM_calls=0,new_fits=0,new_predictions=0,independent_confirmation_scenarios=0,majority_goal_complete=False)
(root/'RESULT351.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','input_memberships_verified','planned_pools','strict_completed_pools','all_pairs_summary','strict_completed_summary']}))
