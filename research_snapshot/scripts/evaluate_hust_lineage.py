"""Post-seal scoring of A442; no model execution or held-out22 access."""
import argparse
import difflib
import json
from pathlib import Path
import numpy as np
from controlled_feedback import save, sha, utc

read = lambda p: json.loads(Path(p).read_text())


def prediction(folder, ids):
    with np.load(folder/'PREDICTIONS.npz', allow_pickle=False) as z:
        assert np.array_equal(z['sample_ids'], ids)
        values = z['prediction'].copy()
    direct = np.asarray([read(folder/('prediction'+str(i))/'predictions.json')['rul'][0]
                         for i in range(len(ids))])
    np.testing.assert_array_equal(values, direct)
    assert values.shape == (8,) and np.isfinite(values).all() and (values >= 0).all()
    return values


def activation(run, state, root):
    third = run/'programs/p03'
    if not all(p in state['programs'] for p in ['p01', 'p02', 'p03']):
        return dict(formal_activation=False, reason='First three new registrations did not all complete')
    claim = read(third/'CLAIM.json')
    receipts = [read(p) for p in (run/'broker/lineage_receipts').glob('*.json')]
    receipts = [r for r in receipts if state['programs']['p02']['report']['utc'] < r['utc'] < claim['utc']]
    if not receipts:
        return dict(formal_activation=False, reason='No prospective hint after two completions and before third claim')
    receipt = max(receipts, key=lambda r:r['utc'])
    parent = receipt['recommended_parent']
    text = claim['claim']
    formal = (receipt['alternative_available'] and ('parent='+parent) in text
              and receipt['utc'] in text)
    parent_source = Path(state['programs'][parent]['path'])/'source/program.py'
    successor = third/'source/program.py'
    diff = ''.join(difflib.unified_diff(parent_source.read_text().splitlines(True),
                                      successor.read_text().splitlines(True), fromfile=parent, tofile='p03'))
    target = root/'audit'/(run.name+'_parent_to_p03.diff')
    target.write_text(diff)
    return dict(formal_activation=bool(formal and diff), parent=parent, receipt=receipt,
                third_claim=claim, source_changed=bool(diff), diff_file=str(target.relative_to(root)),
                substantive_parent_refinement='requires_manual_source_review')


def cost(root):
    requests = list(root.glob('subjects/*/model_gateway/*/*_request.json'))
    receipts = []
    for p in root.glob('subjects/*/model_gateway/*/*_receipt.json'):
        r=read(p); prefix=str(r['sequence']).zfill(3)
        assert sha(p.parent/(prefix+'_request.json')) == r['request_sha256']
        if r.get('response_sha256'):
            assert sha(p.parent/(prefix+'_response.sse')) == r['response_sha256']
        assert r['model_requested']=='deepseek-flash'
        receipts.append(r)
    processes = [read(p) for p in root.glob('subjects/*/programs/*/**/PROCESS.json')]
    audits = [read(p) for p in root.glob('subjects/*/programs/*/**/training_audit.json')]
    return dict(actual_requests=len(requests), receipts=len(receipts),
                fit_entries_with_process_record=sum(p.get('mode')=='fit' for p in processes),
                fit_entries_with_audit=sum(a.get('mode')=='fit' for a in audits),
                prediction_processes=sum(p.get('mode')=='predict' for p in processes),
                qualification_prediction_processes=72,
                completed_estimator_fits=sum(a['completed_estimator_fits'] for a in audits),
                optimizer_steps=sum(a['optimizer_steps'] for a in audits),
                input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in receipts),
                output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in receipts),
                returned_model_ids=sorted({m for r in receipts for m in r.get('returned_model_ids',[])}))


def evaluate(root):
    assert read(root/'TERMINAL442.json')['episodes']==4
    seal=read(root/'ALL_CHOICES_SEALED442.json'); assert len(seal['results'])==4
    assert not (root/'RESULT442.json').exists(), 'No repeated outcome-driven evaluation'
    assert read(root/'PARENT_QUALIFICATION442.json')['passed']
    cfg=read(root/'PROTOCOL442.json'); source=Path(cfg['source_archive'])
    assert sha(source/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256']
    truth=source/'private_inputs/discovery_truth.npz'
    index={e['path']:e for e in read(source/'SOURCE_MANIFEST.json')['files']}
    assert sha(truth)==index['private_inputs/discovery_truth.npz']['sha256']
    with np.load(truth, allow_pickle=False) as z:
        actual,ids=z['y'].copy(),z['sample_ids'].copy()
    assert actual.shape==(8,) and len(set(ids))==8
    initial_hashes={sha(p) for p in (root/'reference').glob('*/source/program.py')}
    rows=[]
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);state=read(run/'broker/STATE.json');programs={}
        build=read(run/'BUILD_RESULT.json')
        assert build==next(r for r in seal['results'] if r['id']==row['id'])
        for ident,item in state['programs'].items():
            folder=Path(item['path']);report=read(folder/'REPORT.json')
            assert report==item['report'] and report['status']=='complete'
            for relative,digest in report['artifacts'].items(): assert sha(folder/relative)==digest
            is_new=not report.get('initial_models_reused',False)
            where=folder/'development' if is_new else root/'qualification/parents'/ident
            values=prediction(where,ids);errors=np.abs(values-actual)
            cv=[c['MAE'] for f in report['cv'] for c in f['cells']];assert len(cv)==47
            programs[ident]=dict(MAE=float(errors.mean()), CV=float(np.mean(cv)),
                predictions=values.tolist(), absolute_errors=errors.tolist(),
                new_registration=is_new, new_source_sha=sha(folder/'source/program.py') not in initial_hashes)
        choice=state.get('submission');selected=choice['program_id'] if choice else None
        base=np.asarray(programs['p00']['absolute_errors'])
        if choice:
            assert sha(Path(state['programs'][selected]['path'])/'REPORT.json')==choice['program_report_sha256']
            errors=np.asarray(programs[selected]['absolute_errors']);delta=errors-base
            selected_metrics=dict(MAE=float(errors.mean()),harm=float(np.maximum(delta,0).mean()),
                                  gain=float(np.maximum(-delta,0).mean()))
        else: selected_metrics=None
        cv=min(programs,key=lambda p:(programs[p]['CV'],p!='p00',list(programs).index(p)))
        rows.append(dict(id=row['id'],arm=row['arm'],completed=build['completed'],native_status=build.get('native',{}).get('status'),
            registration_attempts=state['attempts'],selected=selected,selected_metrics=selected_metrics,
            programs=programs,fixed_CV_choice=cv,fixed_CV_MAE=programs[cv]['MAE'],
            activation=activation(run,state,root),submission_not_counted_as_normal_completion=not build['completed']))
    arms={}
    for arm in cfg['arms']:
        rr=[r for r in rows if r['arm']==arm]
        complete=all(r['completed'] and r['selected_metrics'] is not None for r in rr)
        arms[arm]=dict(completed=sum(r['completed'] for r in rr),planned=2,
            means={k:float(np.mean([r['selected_metrics'][k] for r in rr])) for k in ['MAE','harm','gain']} if complete else None)
    common_best=min(p['MAE'] for p in rows[0]['programs'].values() if not p['new_registration'])
    generations=[dict(episode=r['id'],program=p,MAE=v['MAE']) for r in rows if r['arm']=='lineage'
                 for p,v in r['programs'].items() if v['new_registration'] and v['new_source_sha'] and v['MAE']<common_best]
    o,m=arms['ordinary']['means'],arms['lineage']['means']
    numerical=bool(o and m and m['MAE']<o['MAE'] and m['harm']<=o['harm'] and m['gain']>=o['gain'])
    formal=all(r['activation']['formal_activation'] for r in rows if r['arm']=='lineage')
    result=dict(utc=utc(),rows=rows,arms=arms,cost=cost(root),common_initial_best_MAE=common_best,
        new_method_programs_beating_all_initial=generations,numerical_primary_gate=numerical,
        both_method_formal_activation=formal,generation_gate=bool(generations),
        primary_passed=False,primary_status='pending_source_and_command_review' if numerical and formal and generations else 'failed_frozen_gate',
        cells=8,source_clusters=1,exploratory_only=True,independent_confirmation_scenarios=0,
        majority_complete=False,final22_read=False,metric_recount_passed=True,
        own_pool_oracle_not_used_for_superiority=True)
    save(root/'RESULT442.json',result)
    print(json.dumps({k:result[k] for k in ['arms','cost','numerical_primary_gate','both_method_formal_activation','generation_gate','primary_status']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    evaluate(p.parse_args().run.resolve())
