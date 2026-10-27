"""Independent stdlib recount; does not call the scientific scorer or models."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    result=read(root/'RESULT389.json')
    config=read(root/'CAMPAIGN_CONFIG.json')
    for item in read(root/'ALL_CHOICES_SEALED.json')['files']:
        assert digest(root/item['path'])==item['sha256']
    for item in read(root/'PREPARATION389.json')['public_files']:
        assert digest(root/'public'/item['path'])==item['sha256']
    source=Path(config['source_archives']['historical'])
    assert digest(source/'SOURCE_MANIFEST.json')==config['source_manifests']['historical']
    manifest={x['path']:x for x in read(source/'SOURCE_MANIFEST.json')['files']}
    for name,sha in result['source_bindings'].items():
        assert digest(source/name)==sha==manifest[name]['sha256']
    contrast={r['name']:r for r in read(source/'audit/REGISTERED_RECORD_CONTRASTS274.json')['contrasts']}
    # Identities come from the frozen public definitions, not score signs.
    definitions=[('change_a','a_review_full_vs_base','base','full'),
                 ('change_b','b_full_vs_base','base','full'),
                 ('change_c','c_compound_added','healthy_ir_only','full'),
                 ('change_d','d_all_family_prior_vs_chosen','prior_family57','chosen_loadfamily42')]
    deltas={}
    count=0
    for cid,key,old,new in definitions:
        row=contrast[key]
        right=sum(r['fixed'] for r in row['changed_records'])
        wrong=sum(r['broken'] for r in row['changed_records'])
        assert (right,wrong)==(row['fixed'],row['broken'])
        if (row['before'],row['after'])==(old,new):change=wrong-right
        else:
            assert (row['before'],row['after'])==(new,old)
            change=right-wrong
        deltas[cid]=F(change,3865)
        assert abs(float(deltas[cid])-result['historical_effects'][cid]['relative_after_error'])<1e-14
        count+=1

    def summary(choices):
        rows=[]
        for choice in choices:
            c=choice['case'];loss=F(0) if choice['selected']=='before' else deltas[c]
            optimum=min(F(0),deltas[c]);benefit=max(F(0),-loss)
            rows.append(dict(case=c,selected=choice['selected'],relative_window_error=loss,
                             selection_loss=loss-optimum,harm=max(F(0),loss),gain=benefit,
                             gain_capture=benefit/(-optimum) if optimum<0 else None))
        assert len(rows)==4 and {r['case'] for r in rows}==set(deltas)
        avg={f'mean_{key}':sum(r[key] for r in rows)/4 for key in ['relative_window_error','selection_loss','harm','gain']}
        opportunities=[r['gain_capture'] for r in rows if r['gain_capture'] is not None]
        avg['mean_gain_capture']=sum(opportunities)/len(opportunities)
        return rows,avg

    numeric=0
    def verify(expected,actual):
        nonlocal numeric
        for k,v in expected.items():
            assert abs(float(v)-actual[k])<1e-13,(k,v,actual[k])
            numeric+=1

    for name,choices in read(root/'FIXED_CHOICES.json')['choices'].items():
        rows,avg=summary(choices);verify(avg,result['fixed'][name])
    ordinary=[];structured=[]
    outcomes={e['id']:e for e in result['episodes']}
    for e in read(root/'AGENT_CHOICES.json')['episodes']:
        assert e['completed']==outcomes[e['id']]['completed']
        if e['choice'] is not None:
            rows,avg=summary(e['choice']['choices']);verify(avg,outcomes[e['id']]['scores'])
            if e['completed']:(ordinary if e['condition']=='ordinary' else structured).append(avg)
    complete=len(ordinary)==len(structured)==2
    primary=False
    if complete:
        old={k:sum(r[k] for r in ordinary)/2 for k in ordinary[0]}
        new={k:sum(r[k] for r in structured)/2 for k in structured[0]}
        verify(old,result['arms']['ordinary']['aggregate']);verify(new,result['arms']['structured']['aggregate'])
        fixed=[summary(c)[1]['mean_selection_loss'] for c in read(root/'FIXED_CHOICES.json')['choices'].values()]
        primary=new['mean_selection_loss']<old['mean_selection_loss'] and new['mean_selection_loss']<min(fixed) and new['mean_harm']<=old['mean_harm'] and new['mean_gain_capture']>=old['mean_gain_capture']
    assert primary==result['primary_qualification_passed']
    receipts=[read(p) for p in (root/'episodes').glob('*/model_gateway/*/*_receipt.json')]
    model_ids=sorted({m for r in receipts for m in r.get('returned_model_ids',[])})
    assert model_ids==['deepseek-flash']
    gates=read(root/'audit/ALL_NATIVE_GATES.json');assert gates['passed'] and gates['episodes']==4
    for ep in (root/'episodes').iterdir():
        assert read(ep/'audit/ISOLATION.json')['exit_code']==0
        assert read(ep/'audit/transport/SUBMISSION_CHAIN.json')['exit_code']==0
    cost=dict(requests=len(receipts),upstream_statuses=dict(Counter(r['upstream_status'] for r in receipts)),returned_models=model_ids,
              input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in receipts),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in receipts),
              missing_usage=sum('usage' not in r for r in receipts),normal_completed=len(ordinary)+len(structured),choices_delivered=sum(e['choice_delivered'] for e in result['episodes']),
              planned_native_sessions=4,new_PHM_fits=0,new_PHM_predictions=0)
    audit=dict(passed=True,oriented_contrasts=count,numerical_aggregate_checks=numeric,all4native_isolation_and_submission_gates=True,
               primary_qualification_passed=primary,all_sessions_normal=complete,independent_implementation=True,independent_dataset=False,
               no_new_fitting_or_model_calls_in_recount=True,command_trace_fit_compliance_still_requires_review=True,cost=cost)
    (root/'RECOUNT389.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'COST389.json').write_text(json.dumps(cost,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run.resolve())
