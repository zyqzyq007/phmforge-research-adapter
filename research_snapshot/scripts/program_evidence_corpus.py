"""Read SHA-bound, pool-specific evidence from the already opened A348 corpus."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):return sum(values,F(0))/len(values)
def fraction(value):
    result=F(value).limit_denominator(100000)
    assert 0<=result<=1 and abs(float(result)-value)<1e-12
    return result
def checked_error(value):
    rows=value['records'];assert len({r['group'] for r in rows})==len(rows)
    result=F(sum(r['actual']!=r['predicted'] for r in rows),len(rows))
    assert result==fraction(value['record_error'])
    return result
def condition_errors(value):
    checked_error(value);out=[]
    for hz in sorted({r['condition_hz'] for r in value['records']}):
        records=[r for r in value['records'] if r['condition_hz']==hz]
        out.append(F(sum(r['actual']!=r['predicted'] for r in records),len(records)))
    return out

def load_pools(cfg):
    archive=Path(cfg['evidence_archive']);assert sha(archive/'SOURCE_MANIFEST.json')==cfg['manifest_sha256']
    for name,digest in cfg['input_bindings'].items():assert sha(archive/name)==digest
    payload=read(archive/'recovery2/PROGRAM_EVIDENCE_CORPUS348.json');corpus=payload['programs']
    old=read(archive/'recovery2/RESULT348.json');bound_index={(x['cohort'],x['path']):x['sha256'] for x in payload['bindings']}
    sources={x['cohort']:Path(x['archive']) for x in read(archive/'PROTOCOL348.json')['sources']}
    for x in read(archive/'PROTOCOL348.json')['sources']:assert sha(Path(x['archive'])/'SOURCE_MANIFEST.json')==x['manifest_sha256']
    cache={}
    def bound(cohort,name):
        key=(cohort,name)
        if key not in cache:
            p=sources[cohort]/name;assert sha(p)==bound_index[key]
            cache[key]=read(p)
        return cache[key]
    pools=[];counter=Counter();memberships=0
    for pool in old['pools']:
        cohort=pool['cohort'];counter[cohort]+=1
        state_name=next(n for c,n in bound_index if c==cohort and n.endswith('/'+pool['subject']+'/broker/STATE.json'))
        state=bound(cohort,state_name);probe=state.get('probe') or {}
        checks=probe.get('reports',{}) if probe.get('status')=='complete' else {}
        common=state_name.removesuffix('/broker/STATE.json')+'/common_check/REPORT.json'
        if (cohort,common) in bound_index:checks=bound(cohort,common)['programs']
        members={}
        for ident in pool['programs']:
            original=Path(state['programs'][ident]['path']);parts=original.parts
            relative=Path(*parts[parts.index(sources[cohort].name)+1:])
            report=bound(cohort,str(relative/'REPORT.json'))
            cv=mean([checked_error(v) for v in report['cv']]);dev=condition_errors(report['development'])
            check=condition_errors(checks[ident]) if ident in checks else None
            prefix='pool_'+str(counter[cohort]) if cohort=='A342' else pool['subject']
            uid=cohort+'/'+('common_p00' if ident=='p00' else prefix+'__'+ident)
            assert cv==fraction(corpus[uid]['rule_values']['mean_cv'])
            members[ident]=dict(CV=cv,development=mean(dev),check=mean(check) if check is not None else None,
                full_validation=mean(dev+check) if check is not None else None,final=fraction(corpus[uid]['final_error']))
            memberships+=1
        pools.append(dict(cohort=cohort,pool=pool['subject'],primary=pool['original_native_completed'] and not pool['strict_registry_deviation'],
            native_completed=pool['original_native_completed'],registry_deviation=pool['strict_registry_deviation'],
            original_submission=pool['original_submission'],submission_scope=pool['submission_scope'],members=members))
    assert len(pools)==12 and memberships==45
    return pools,dict(source_metadata_files_verified=len(cache),program_memberships_verified=memberships,unique_program_instances=len(corpus))
