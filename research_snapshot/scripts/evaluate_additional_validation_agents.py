"""Post-seal A418 numerical evaluation and independent provenance checks."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

from controlled_feedback import save, sha, utc


def read(p):
    return json.loads(Path(p).read_text())


def main(root):
    started=time.monotonic()
    cfg=read(root/'PROTOCOL418.json')
    assert sha(root/'PROTOCOL418.json')==read(root/'PROTOCOL_FREEZE418.json')['protocol_sha256']
    assert read(root/'NATIVE_EXECUTION418.json')['terminal']
    assert all(x['exit_code']==0 for x in read(root/'NATIVE_EXECUTION418.json')['rows'])
    checks=0
    seals={}
    for acq in [7,8]:
        child=root/f'cal{acq}'
        assert read(child/'CONTROLLER418.json')['status']=='terminal'
        seal=read(child/'ALL_CHOICES_SEALED.json')
        for row in seal['files']:
            assert sha(child/row['path'])==row['sha256'];checks+=1
        for relative,digest in read(child/'PREPARATION418.json')['public_files'].items():
            assert sha(child/'public'/relative)==digest;checks+=1
        for p in child.glob('episodes/*/primary/*/status.json'):
            assert read(p)['status']!='running';checks+=1
        seals[acq]=seal
    save(root/'ALL_CHOICES_SEALED418.json',dict(utc=utc(),children={str(k):sha(root/f'cal{k}/ALL_CHOICES_SEALED.json') for k in seals},
        checks=checks,current_native_final_feedback_started=False,researcher_prior_final_exposure=True))
    source=Path(cfg['prediction_archive']);manifest=read(source/'SOURCE_MANIFEST.json')
    original,=[x for x in manifest['files'] if x['path']=='RESULT417.json']
    cached=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/results/rotor417/RESULT417.json')
    assert sha(cached)==original['sha256'];checks+=1
    data=read(cached);save(root/'evaluation/SOURCE_RESULT417.json',data)
    final={}
    for held in [7,8]:
        final[held]={}
        for cohort,ident in cfg['candidate_map'].items():
            group_losses=[];confusion=[[0]*5 for _ in range(5)]
            for load in [3.5,4.0]:
                row,=[x for x in data['rows'] if (x['acquisition'],x['load'],x['cohort'],x['name'])==(held,load,cohort,'complete')]
                losses=[Fraction(r['errors'],r['windows']) for r in row['records']]
                assert sum(losses)/len(losses)==Fraction(row['record_mean_error']);checks+=1
                group_losses+=losses
                confusion=[[confusion[i][j]+row['confusion'][i][j] for j in range(5)] for i in range(5)]
            risk=sum(group_losses)/len(group_losses)
            final[held][ident]=dict(error=risk,cohort=cohort,
                accuracy=sum(confusion[i][i] for i in range(5))/sum(map(sum,confusion)),
                record_count=len(group_losses))
    evaluated=[];episode_rows=[];cost_rows=[];syntax_calls=[]
    for acq in [7,8]:
        child=root/f'cal{acq}';held=15-acq
        case,=read(child/'public/cases.json')
        rows=read(child/'public/rows/rotor_program_portfolio.json')
        groups={r['group'] for r in rows['gold']}
        new_risk={};old_risk={}
        for ident in case['candidate_ids']:
            pred={p['sample_id']:int(p['predicted_label']) for p in rows['candidates'][ident]['predictions']}
            values=[]
            for group in groups:
                gold=[g for g in rows['gold'] if g['group']==group]
                values.append(Fraction(sum(pred[g['sample_id']]!=g['label'] for g in gold),len(gold)))
            new_risk[ident]=sum(values)/len(values)
            index=case['candidate_ids'].index(ident)
            assert abs(float(new_risk[ident])-sum(case['calibration_group_losses'][index])/len(groups))<1e-12;checks+=1
            old_risk[ident]=case['historical_validation_mean_record_errors'][index]
        def stat(incumbent,selected):
            initial=final[held][incumbent]['error'];selected_loss=final[held][selected]['error']
            oracle=min(x['error'] for x in final[held].values())
            return dict(error=float(selected_loss),initial_error=float(initial),selection_loss=float(selected_loss-oracle),
                harm=float(max(selected_loss-initial,0)),gain=float(max(initial-selected_loss,0)),
                available_gain=float(initial-oracle),selected=selected,incumbent=incumbent)
        fixed=read(child/'FIXED_CHOICES.json')['choices']
        for repeat in [1,2]:
            ep=child/'episodes'/f'ordinary_{repeat}'
            item=read(ep/'EPISODE_RESULT.json')
            completed=item.get('completed',False)
            choices=item.get('choice',{}).get('choices',[]) if item.get('choice') else []
            indexed={row['incumbent']:row for row in choices}
            episode_rows.append(dict(calibration=acq,repeat=repeat,completed=completed,
                status=item.get('status'),has_sealed_choice=bool(choices),seconds=item['seconds']))
            for old in case['incumbent_ids']:
                chosen=indexed[old]['selected'] if choices else old
                evaluated.append(dict(calibration=acq,held_acquisition=held,repeat=repeat,method='ordinary',
                    completed=completed,fallback=not bool(choices),**stat(old,chosen)))
            for reference in fixed:
                old=reference['incumbent']
                for method,selected in reference['selected'].items():
                    if method!='keep':
                        values={i:float(new_risk[i]) if method=='additional_calibration' else old_risk[i] if method=='historical_development' else (float(new_risk[i])+old_risk[i])/2 if method=='mean_old_and_new' else max(float(new_risk[i]),old_risk[i]) for i in case['candidate_ids']}
                        lowest=min(values.values());ties=[i for i,v in values.items() if abs(v-lowest)<1e-12]
                        expected=old if old in ties else min(ties)
                        assert selected==expected;checks+=1
                    evaluated.append(dict(calibration=acq,held_acquisition=held,repeat=repeat,method=method,
                        completed=True,fallback=False,**stat(old,selected)))
            g=read(ep/'model_gateway'/ep.name/'status.json')
            assert not g.get('inflight');assert g.get('credential_in_agent') is False
            receipts=[]
            for path in sorted((ep/'model_gateway'/ep.name).glob('*_receipt.json')):
                receipt=read(path);prefix=path.name.split('_')[0]
                assert sha(path.parent/(prefix+'_request.json'))==receipt['request_sha256'];checks+=1
                if receipt.get('response_sha256'):
                    assert sha(path.parent/(prefix+'_response.sse'))==receipt['response_sha256'];checks+=1
                assert receipt['model_requested']=='deepseek-flash'
                if receipt.get('returned_model_ids'):assert receipt['returned_model_ids']==['deepseek-flash']
                receipts.append(receipt)
            cost_rows.append(dict(calibration=acq,repeat=repeat,requests=g['requests'],receipts=len(receipts),
                input_tokens=g['input_tokens'],output_tokens=g['output_tokens'],
                upstream_statuses=dict(Counter(str(x.get('upstream_status')) for x in receipts)),returned_model_ids=g['returned_model_ids']))
            for path in (ep/'app').rglob('*.py'):
                try:tree=ast.parse(path.read_text())
                except (SyntaxError,UnicodeError):continue
                calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','fit_transform','partial_fit','step']]
                if calls:syntax_calls.append(dict(calibration=acq,repeat=repeat,path=str(path.relative_to(ep)),calls=calls))
    aggregates={}
    for method in ['ordinary',*cfg['fixed_controls']]:
        rows=[r for r in evaluated if r['method']==method]
        gains=sum(x['gain'] for x in rows);available=sum(x['available_gain'] for x in rows)
        aggregates[method]=dict(decisions=len(rows),mean_error=sum(x['error'] for x in rows)/len(rows),
            mean_selection_loss=sum(x['selection_loss'] for x in rows)/len(rows),
            mean_harm=sum(x['harm'] for x in rows)/len(rows),gain_capture=gains/available if available else None)
    residual=[]
    for acq in [7,8]:
        gaps=[]
        for repeat in [1,2]:
            def loss(method):
                q=[x for x in evaluated if x['calibration']==acq and x['repeat']==repeat and x['method']==method]
                return sum(x['error'] for x in q)/len(q)
            gaps.append(loss('ordinary')-min(loss(m) for m in cfg['fixed_controls']))
        completed=all(x['completed'] for x in episode_rows if x['calibration']==acq)
        residual.append(dict(calibration=acq,gaps_to_best_fixed=gaps,completed=completed,
            residual_gate=completed and all(x>=.05 for x in gaps)))
    result=dict(utc=utc(),episodes=episode_rows,decisions=evaluated,aggregate=aggregates,
        residual_difficulty=residual,residual_difficulty_gate=any(x['residual_gate'] for x in residual),
        cost=cost_rows,new_supervised_fits_registered=0,new_model_forwards=0,
        cached_final_source_sha256=original['sha256'],independent_sources=1,
        formal_confirmation=False,new_method_superiority_claimed=False)
    save(root/'RESULT418.json',result)
    save(root/'AUDIT418.json',dict(passed=True,checks=checks,
        generated_python_fit_call_candidates=syntax_calls,execution_command_audit_still_required=True,
        primary_metrics_reconstructed_from_record_counts=True,all_choices_sealed_before_current_score_join=True,
        source_already_opened_development=True,seconds=time.monotonic()-started))
    print(json.dumps(dict(aggregate=aggregates,episodes=episode_rows,residual=residual,cost=cost_rows,
        checks=checks,generated_fit_call_candidates=syntax_calls),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
