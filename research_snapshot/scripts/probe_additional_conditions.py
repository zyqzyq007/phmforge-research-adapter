"""Fixed-program, real-observation qualification for an additional condition check.

No Agent run, new training, synthetic signal, or new final-set inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_free_program_comparison import independent_score
from program_revision_runtime import Runtime, load, score

NUMBER = '406'


def read(path):
    return json.loads(Path(path).read_text())


def verify(root):
    for relative, digest in read(root / f'CODE_FREEZE{NUMBER}.json').items():
        assert sha(root / relative) == digest, relative
    cfg = read(root / f'PROTOCOL{NUMBER}.json')
    for row in cfg['source_bindings']:
        assert sha(Path(row['path'])) == row['sha256'], row['path']
    return cfg


def summarize(cfg, rows):
    if len(rows) != len(cfg['models']) * len(cfg['conditions']) or any(r['status'] != 'complete' for r in rows):
        return dict(complete=False, qualification_gate=False)
    errors = {name: sum(Fraction(r['errors'], r['records_count']) for r in rows if r['program'] == name) / len(cfg['conditions'])
              for name in cfg['models']}
    decisions = []
    for pair in cfg['comparisons']:
        old, new = pair['incumbent'], pair['challenger']
        selected = new if errors[new] < errors[old] else old
        decisions.append(dict(**pair, selected=selected,
            check_old_error=str(errors[old]), check_new_error=str(errors[new]),
            cached_final_selected_error=pair['cached_final_errors'][selected],
            incumbent_tie_rule=True))
    primary = [r for r in decisions if r['role'] == 'primary_failure']
    positive = [r for r in decisions if r['role'] == 'positive_success_control']
    gate = all(r['selected'] == r['challenger'] for r in primary) and all(r['selected'] == r['incumbent'] for r in positive)
    result = dict(complete=True, mean_check_errors={k: str(v) for k,v in errors.items()},
                comparisons=decisions, qualification_gate=gate,
                gate_scope=cfg.get('gate_scope', 'Both A404 ranking reversals corrected while preserving ordinary successful selection; one related cohort, tool-information pilot only'))
    if cfg.get('banks'):
        result['whole_bank_checks'] = {}
        for name, bank in cfg['banks'].items():
            anchor = bank['actor'] or bank['baseline']
            selected = min(bank['members'], key=lambda key: (errors[key], key != anchor, key))
            result['whole_bank_checks'][name] = dict(selected=selected,
                extra_check_error=str(errors[selected]), original_actor=bank['actor'],
                original_actor_complete=bank['actor'] is not None,
                cached_old_final_selected_error=bank['cached_final_errors'][selected],
                all_generated_candidates_retained=True)
    return result


def run(root):
    cfg = verify(root)
    assert not (root / f'STARTED{NUMBER}.json').exists(), 'Inspect existing handle; never restart a started scientific run'
    save(root / f'STARTED{NUMBER}.json', dict(utc=utc(), status='running', protocol_sha256=sha(root/f'PROTOCOL{NUMBER}.json')))
    save(root / f'HANDLE{NUMBER}.json', dict(pid=os.getpid(), status='running', phase='fixed_weight_new_conditions'))
    started = time.monotonic(); deadline = started + cfg['total_seconds']
    data = {role: load(Path(cfg['source_archive'])/'qualification'/(role+'.npz')) for role in cfg['conditions']}
    assert sum(len(set(d['groups'])) for d in data.values()) == 16
    rows = []

    def one(name, role):
        folder = root/'predictions'/role/name; folder.mkdir(parents=True)
        path = Path(cfg['models'][name]['path'])
        engine = Runtime(root); engine.deadline = min(deadline,time.monotonic()+cfg['per_job_seconds'])
        try:
            prediction, audit1 = engine.predict_registered(folder/'forward',path,data[role])
            repeat, audit2 = engine.predict_registered(folder/'cold_repeat',path,data[role])
            assert np.array_equal(prediction,repeat)
            assert all(a['completed_estimator_fits']==a['optimizer_steps']==0 for a in audit1+audit2)
            observed = score(data[role],prediction); independent = independent_score(data[role],prediction)
            assert observed['confusion']==independent['confusion'] and observed['record_error']==independent['record_error']
            result = dict(**observed,status='complete',records_count=len(independent['records']),
                          errors=sum(r['actual']!=r['predicted'] for r in independent['records']),
                          actual_prediction_processes=len(audit1+audit2),cold_repeat_equal=True,
                          independent_record_vote_passed=True)
        except Exception as error:
            result = dict(status='prediction_error',error_type=type(error).__name__,error=str(error))
        result.update(program=name,condition=role)
        save(folder/'RESULT.json',result)
        return result

    try:
        with ThreadPoolExecutor(max_workers=cfg['parallel']) as executor:
            futures = [executor.submit(one,name,role) for name in cfg['models'] for role in cfg['conditions']]
            for future in as_completed(futures):
                rows.append(future.result())
                save(root/f'PROGRESS{NUMBER}.json',dict(utc=utc(),completed_jobs=len(rows),planned_jobs=len(futures),
                    errors=sum(r['status']!='complete' for r in rows),rows=rows))
        # Current qualification choices are determined only from extra-check results.
        summary = summarize(cfg,rows)
        save(root/f'CHECK_SELECTIONS{NUMBER}.json',dict(utc=utc(),summary=summary,
             no_final_observations_or_labels_loaded=True,cached_final_scores_known_to_researcher=True))
        result = dict(utc=utc(),status='complete' if summary['complete'] else 'incomplete',seconds=time.monotonic()-started,
            rows=rows,summary=summary,actual_prediction_processes=sum(r.get('actual_prediction_processes',0) for r in rows),
            actual_started_prediction_processes=sum(1 for p in (root/'predictions').rglob('PROCESS.json')),
            new_LLM_calls=0,new_fits=0,new_final_forwards=0,physical_rigs=1,independent_confirmation_scenarios=0,
            read_only_existing_weights=True,real_author_records=16,not_a_new_agent_or_method_trial=True,
            confirmation_panel_opened=False,goal_complete=False)
        save(root/f'RESULT{NUMBER}.json',result)
        save(root/f'HANDLE{NUMBER}.json',dict(utc=utc(),status='terminal',result_status=result['status']))
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],prediction_processes=result['actual_prediction_processes'],summary=summary)))
    except BaseException as error:
        save(root/f'HANDLE{NUMBER}.json',dict(utc=utc(),status='terminal_error',error_type=type(error).__name__,error=str(error)))
        raise


def audit(root):
    cfg = verify(root); result = read(root/f'RESULT{NUMBER}.json'); checks = 0
    assert read(root/f'HANDLE{NUMBER}.json')['status']=='terminal'
    observed = {}
    for row in result['rows']:
        assert row['status']=='complete'
        data = load(Path(cfg['source_archive'])/'qualification'/(row['condition']+'.npz'))
        folder = root/'predictions'/row['condition']/row['program']
        a = load(folder/'forward/PREDICTIONS.npz'); b = load(folder/'cold_repeat/PREDICTIONS.npz')
        assert np.array_equal(a['sample_ids'],data['sample_ids']) and np.array_equal(a['prediction'],b['prediction']);checks+=2
        # Recount from emitted cold window predictions, separate from stored metrics.
        recounted = independent_score(data,a['prediction'])
        expected = [(r['group'],r['actual'],r['predicted']) for r in recounted['records']]
        actual = [(r['group'],r['actual'],r['predicted']) for r in row['records']]
        assert expected==actual and recounted['confusion']==row['confusion'];checks+=2
        n=len(expected);bad=sum(y!=q for _,y,q in expected)
        assert (n,bad)==(row['records_count'],row['errors']);checks+=1
        observed[row['program'],row['condition']]=Fraction(bad,n)
        composition = Runtime(root).verified_members(Path(cfg['models'][row['program']]['path']))
        for kind in ['forward','cold_repeat']:
            for index, group in enumerate(sorted(set(data['groups']))):
                input_file = (folder/kind/f'record_{index}/observations.npz' if composition
                              else folder/kind/f'input{index}.npz')
                obs = load(input_file);assert 'labels' not in obs
                mask=data['groups']==group
                for k,v in obs.items():assert np.array_equal(v,data[k][mask]);checks+=1
                logs = ((folder/kind/f'record_{index}').glob('*/training_audit.json') if composition
                        else [folder/kind/f'prediction{index}/training_audit.json'])
                for log_path in logs:
                    log=read(log_path)
                    assert log['completed_estimator_fits']==log['optimizer_steps']==0;checks+=1
                if composition:
                    from collections import Counter
                    manifest, members = composition
                    votes = {}
                    for name in members:
                        values = read(folder/kind/f'record_{index}'/name/'predictions.json')['predictions']
                        counts = Counter(v['predicted_label'] for v in values)
                        votes[name] = min(counts, key=lambda label: (-counts[label], label))
                    counts = Counter(votes.values()); tied = [k for k,v in counts.items() if v == max(counts.values())]
                    selected = votes[manifest['tie_anchor']] if votes[manifest['tie_anchor']] in tied else min(tied)
                    assert all(int(y) == selected for y in a['prediction'][mask]);checks+=1
    means={name:sum(observed[name,c] for c in cfg['conditions'])/len(cfg['conditions']) for name in cfg['models']}
    assert {k:str(v) for k,v in means.items()}==result['summary']['mean_check_errors'];checks+=1
    for pair in result['summary']['comparisons']:
        selected=pair['challenger'] if means[pair['challenger']]<means[pair['incumbent']] else pair['incumbent']
        assert selected==pair['selected'];checks+=1
    primary=[p for p in result['summary']['comparisons'] if p['role']=='primary_failure']
    positive=[p for p in result['summary']['comparisons'] if p['role']=='positive_success_control']
    assert result['summary']['qualification_gate']==(all(p['selected']==p['challenger'] for p in primary) and all(p['selected']==p['incumbent'] for p in positive));checks+=1
    assert result['actual_prediction_processes']==result['actual_started_prediction_processes']==cfg['maximum_prediction_processes'];checks+=1
    if cfg.get('banks'):
        for name, bank in cfg['banks'].items():
            anchor = bank['actor'] or bank['baseline']
            chosen = min(bank['members'], key=lambda key: (means[key], key != anchor, key))
            actual = result['summary']['whole_bank_checks'][name]
            assert chosen == actual['selected'] and str(means[chosen]) == actual['extra_check_error'];checks+=2
    save(root/f'INDEPENDENT_AUDIT{NUMBER}.json',dict(utc=utc(),passed=True,checks=checks,
        source_result_sha256=sha(root/f'RESULT{NUMBER}.json'),cold_arrays_recounted=True,
        every_worker_input_matches_real_observations=True,labels_absent_in_workers=True,
        new_LLM_calls=0,new_fits=0,new_forwards=0))
    print(json.dumps(dict(passed=True,checks=checks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit']);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--iteration', default='406', choices=['406','428'])
    args=parser.parse_args();NUMBER=args.iteration;globals()[args.action](args.run.resolve())
