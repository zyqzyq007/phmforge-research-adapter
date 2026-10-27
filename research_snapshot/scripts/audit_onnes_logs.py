"""Recompute Onnes classification and detection separately from frozen public logs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ['agent_eval_turns.jsonl', 'agent_eval_fewshot_turns.jsonl',
             'gemini_zero_shot_turns.jsonl', 'gemini_enhanced_turns.jsonl']
    reported = {'agent_eval_turns.jsonl':0.685, 'agent_eval_fewshot_turns.jsonl':0.99,
                'gemini_zero_shot_turns.jsonl':0.715, 'gemini_enhanced_turns.jsonl':0.995}
    records, errors = [], []
    for name in files:
        path = args.sources / name
        grouped = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped[row['scenario']].append(row)
        assert len(grouped) == 200
        cases = []
        for scenario, rows in sorted(grouped.items()):
            assert len(rows) == 5 and len({x['role'] for x in rows}) == 5
            for key in ['seed','truth','agent_pred','ml_pred']:
                assert len({x[key] for x in rows}) == 1
            roles = {x['role']:x['reply'] for x in rows}
            supervisor = roles['supervisor']
            diag = roles['diagnostician']
            a = rows[0]
            case = dict(log=name, scenario=scenario, seed=a['seed'], truth=a['truth'],
                        scored_agent_prediction=a['agent_pred'], ml_prediction=a['ml_pred'],
                        raw_supervisor_class=supervisor.get('fault_class'),
                        raw_fault_detected=bool(supervisor.get('fault_detected')),
                        raw_confidence=supervisor.get('confidence'),
                        diagnostician_error=diag.get('_error'),
                        diagnostic_votes=diag.get('_sc_votes'),
                        guardian_approved=roles['guardian'].get('approved'),
                        final_action=supervisor.get('final_action'))
            cases.append(case)
            if a['agent_pred'] != a['truth']:
                errors.append(dict(case, role_replies=roles))
        correct = sum(x['scored_agent_prediction'] == x['truth'] for x in cases)
        accuracy = correct / len(cases)
        assert abs(accuracy-reported[name]) < 1e-12
        detection = {}
        for scope in ['raw_supervisor_flag', 'normalized_class_non_normal']:
            tp = fp = tn = fn = 0
            for a in cases:
                truth = a['truth'] != 'normal'
                pred = a['raw_fault_detected'] if scope == 'raw_supervisor_flag' else a['scored_agent_prediction'] != 'normal'
                tp += truth and pred
                fp += not truth and pred
                tn += not truth and not pred
                fn += truth and not pred
            detection[scope] = dict(tp=tp, fp=fp, tn=tn, fn=fn,
                                    accuracy=(tp+tn)/len(cases),
                                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        records.append(dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            scenarios=len(cases), role_rows=sum(len(v) for v in grouped.values()),
                            classification_correct=correct, classification_accuracy=accuracy,
                            diagnostician_error_scenarios=sum(bool(x['diagnostician_error']) for x in cases),
                            detection=detection,
                            wrong_with_diagnostician_error=sum(x['scored_agent_prediction']!=x['truth'] and bool(x['diagnostician_error']) for x in cases),
                            wrong_without_diagnostician_error=sum(x['scored_agent_prediction']!=x['truth'] and not x['diagnostician_error'] for x in cases)))
        (args.output / (name.replace('.jsonl', '_case_summary.json'))).write_text(json.dumps(cases,indent=2)+'\n')
    result = dict(kind='public_trajectory_reanalysis_not_model_or_simulation_rerun',
                  records=records,
                  interpretation='Keep interface errors, normalized labels, fault detection, diagnosis and safe actions separate.',
                  models='Model names are author-reported metadata; service requests are not present in these JSONL logs.',
                  limits=['synthetic cryogenic episodes, not independent real hardware faults',
                          'no original prompt or complete sensor window in released turn logs',
                          'classification and detection are different endpoints'])
    (args.output / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output / 'error_case_details.json').write_text(json.dumps(errors,indent=2)+'\n')
    print(json.dumps([dict(file=r['file'],accuracy=r['classification_accuracy'],
                           errors_with_api=r['wrong_with_diagnostician_error'],
                           errors_without_api=r['wrong_without_diagnostician_error'],
                           raw_detection=r['detection']['raw_supervisor_flag']) for r in records],indent=2))


if __name__ == '__main__':
    main()
