"""Score sealed same-information acceptance choices; preserve incomplete attempts."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from controlled_feedback import save, sha, utc


def read(path):
    return json.loads(Path(path).read_text())


def weight(selected):
    if selected in ['P00', 'P01']:
        return float(selected == 'P01')
    assert selected.startswith('mix:')
    value = float(selected[4:])
    assert math.isfinite(value) and 0 <= value <= 1
    return value


def evaluate(root):
    assert (root/'TERMINAL375.json').exists() and not (root/'RESULT375.json').exists()
    seal = read(root/'ALL_CHOICES_SEALED375.json')
    for path, digest in seal['files'].items():
        assert sha(root/path) == digest
    records, costs = [], []
    for target in read(root/'EVALUATION_TARGETS375.json'):
        child = root/'case_runs'/('q'+str(target['held']))
        episodes = {r['id']:r for r in read(child/'AGENT_CHOICES.json')['episodes']} if (child/'AGENT_CHOICES.json').exists() else {}
        b, c, y, cap = target['parent'], target['candidate'], target['actual'], target['cap']
        oracle = min(max(y, min(b,c)), max(b,c))
        old_error, oracle_error = abs(b-y), abs(oracle-y)
        opportunity = old_error-oracle_error
        for repeat in [1,2]:
            ident = 'ordinary_'+str(repeat)
            episode = episodes.get(ident, {})
            valid = bool(episode.get('completed') and not episode.get('fallback'))
            if valid:
                rows = episode['choice']['choices']
                assert len(rows)==1 and rows[0]['case']=='hust_revision' and rows[0]['incumbent']=='P00'
                alpha = weight(rows[0]['selected'])
            else:
                alpha = 0.
            weights = dict(ordinary=alpha, paired_partial_cap=min(alpha,cap),
                paired_binary_cap=alpha if alpha <= cap+1e-12 else 0., fixed_partial_cap=cap)
            outcome = {}
            for method, a in weights.items():
                prediction = b+a*(c-b)
                error = abs(prediction-y)
                harm, gain = max(0.,error-old_error), max(0.,old_error-error)
                outcome[method] = dict(alpha=a,prediction=prediction,MAE=error,regret=error-oracle_error,
                    harm=harm,gain=gain,gain_capture=gain/opportunity if opportunity>1e-9 else None)
            records.append(dict(held=target['held'],repeat=repeat,valid_submission=valid,
                fallback_scored_only_for_accounting=not valid,actual=y,parent_error=old_error,
                continuous_segment_oracle_error=oracle_error,opportunity=opportunity,methods=outcome))
            run = child/'episodes'/ident
            status_path = run/'model_gateway'/ident/'status.json'
            status = read(status_path) if status_path.exists() else {}
            receipts = [read(p) for p in (run/'model_gateway').glob('*/*_receipt.json')]
            costs.append(dict(held=target['held'],repeat=repeat,valid_submission=valid,
                requests=status.get('requests',0),input_tokens=status.get('input_tokens',0),output_tokens=status.get('output_tokens',0),
                complete_receipts=len(receipts),returned_model_ids=status.get('returned_model_ids',[]),
                missing_success_usage=sum(r.get('upstream_status')==200 and not r.get('usage') for r in receipts),
                upstream_statuses=[r.get('upstream_status') for r in receipts]))
    aggregate = {}
    for method in records[0]['methods']:
        aggregate[method] = {key:float(np.mean([r['methods'][method][key] for r in records if r['methods'][method][key] is not None]))
            for key in ['MAE','regret','harm','gain','gain_capture']}
    complete = all(r['valid_submission'] for r in records)
    ordinary, shield = aggregate['ordinary'], aggregate['paired_partial_cap']
    gate = (shield['MAE'] < ordinary['MAE']-1e-9 and shield['regret'] < ordinary['regret']-1e-9 and
            shield['harm'] <= ordinary['harm']+1e-9 and shield['gain_capture'] >= ordinary['gain_capture']-1e-9) if complete else None
    cell_diffs = [float(np.mean([r['methods']['paired_partial_cap']['MAE']-r['methods']['ordinary']['MAE'] for r in records if r['held']==i])) for i in range(8)]
    result = dict(utc=utc(),records=records,aggregate=aggregate,costs=costs,all16_valid=complete,
        valid_submissions=sum(r['valid_submission'] for r in records),paired_joint_exploratory_gate=gate,
        paired_cell_wins=sum(v<-1e-9 for v in cell_diffs),paired_cell_ties=sum(abs(v)<=1e-9 for v in cell_diffs),paired_cell_losses=sum(v>1e-9 for v in cell_diffs),
        study_role='Same-information baseline qualification on one opened cohort; constrained continuous revision acceptance, not free coding or independent confirmation.',
        planned_native_sessions=16,new_native_sessions=sum(c['requests']>0 for c in costs),
        new_requests=sum(c['requests'] for c in costs),new_input_tokens=sum(c['input_tokens'] for c in costs),
        new_output_tokens=sum(c['output_tokens'] for c in costs),new_controller_PHM_fits=0,new_controller_PHM_predictions=0,
        actor_training_compliance_requires_trace_review=True,independent_confirmation_scenarios=0,
        novelty_confirmed=False,final22_opened=False,A33872_opened=False,majority_complete=False)
    save(root/'RESULT375.json',result)
    print(json.dumps({k:result[k] for k in ['all16_valid','valid_submissions','aggregate','paired_joint_exploratory_gate','new_requests']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    evaluate(args.run.resolve())
