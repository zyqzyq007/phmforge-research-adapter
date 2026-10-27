"""Prospective complete-budget lineage contrast with outcome-unopened A444 controls.

Reuse the qualified execution engine. Its 442/444 filenames are compatibility
names; PROTOCOL445.id and this run's unique root identify the new experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
from controlled_feedback import save, sha, utc
from run_hust_lineage_comparison import METHOD
import run_native_budget_qualification as qualified
from evaluate_hust_lineage import activation, cost, prediction

read = lambda p: json.loads(Path(p).read_text())


def extract_bound(archive, destination, predicate):
    index = {x['path']: x for x in read(archive/'SOURCE_MANIFEST.json')['files']}
    wanted = {p for p in index if predicate(p)}
    copied = {}
    with tarfile.open(archive/'RUN_CONTENTS.tar.gz', 'r|gz') as bundle:
        for member in bundle:
            if member.name not in wanted:
                continue
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
            raw = bundle.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == index[member.name]['sha256']
            target = destination/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            copied[member.name] = index[member.name]['sha256']
    assert set(copied) == wanted and copied
    return copied


def setup(root):
    cfg = read(root/'PROTOCOL445.json')
    assert read(root/'PROTOCOL444.json') == cfg and cfg['arms'] == ['lineage']
    archive = Path(cfg['ordinary_archive'])
    assert sha(archive/'SOURCE_MANIFEST.json') == cfg['ordinary_manifest_sha256']
    # This is the exact qualified setup: same data, tools, initial programs,
    # native CLI and per-episode caps. No new preparation fits or predictions.
    qualified.setup(root)
    keep_roots = {'PROTOCOL444.json', 'PROMPTS444.json', 'ALL_CHOICES_SEALED444.json',
                  'QUALIFICATION_RESULT444.json', 'audit/MANUAL_REVIEW444.json',
                  'audit/COST_RECOUNT444.json'}
    def ordinary_file(p):
        if p in keep_roots:
            return True
        if not p.startswith('subjects/ordinary_'):
            return False
        return (p.endswith('/broker/STATE.json') or p.endswith('/BUILD_RESULT.json')
                or ('/programs/' in p and (p.endswith('/REPORT.json')
                    or '/source/' in p or '/development/' in p
                    and (p.endswith('/PREDICTIONS.npz') or p.endswith('/predictions.json')))))
    bindings = extract_bound(archive, root/'reused_ordinary', ordinary_file)
    assert read(root/'reused_ordinary/audit/MANUAL_REVIEW444.json')['passed']
    baseline_prompts = read(root/'reused_ordinary/PROMPTS444.json')
    for row in read(root/'SESSIONS337.json'):
        prompt = (Path(row['path'])/'plans/prompt.txt').read_text()
        assert prompt.count(METHOD) == 1
        assert prompt.replace(METHOD, '') == baseline_prompts['ordinary_'+str(row['repeat'])]
    for key in ['episode_seconds', 'episode_requests', 'episode_input_tokens',
                'episode_output_tokens', 'output_per_request', 'max_registered_programs',
                'cli_sha256', 'model', 'fit_seconds', 'program_seconds', 'predict_seconds']:
        assert cfg[key] == read(root/'reused_ordinary/PROTOCOL444.json')[key]
    parents = Path(cfg['initial_parent_qualification_reused_from'])
    assert sha(parents/'SOURCE_MANIFEST.json') == cfg['initial_parent_qualification_manifest']
    parent_bindings = extract_bound(parents, root, lambda p:
        p.startswith('qualification/parents/') and
        (p.endswith('/PREDICTIONS.npz') or p.endswith('/predictions.json')))
    assert len([p for p in parent_bindings if p.endswith('/PREDICTIONS.npz')]) == 9
    save(root/'REUSED_BINDINGS445.json', dict(ordinary=bindings, parents=parent_bindings))
    controller_files = [Path(__file__).name, 'evaluate_hust_lineage.py', 'audit_program_trial.py']
    for filename in controller_files:
        shutil.copyfile(Path(__file__).with_name(filename), root/'code'/filename)
    freeze = dict(read(root/'LAUNCH_FREEZE444.json'))
    for relative in ['PROTOCOL445.json', 'REUSED_BINDINGS445.json'] + ['code/'+p for p in controller_files]:
        freeze[relative] = sha(root/relative)
    save(root/'LAUNCH_FREEZE445.json', freeze)
    save(root/'PREPARATION445.json', dict(utc=utc(), passed=True,
        exact_common_prompt_and_caps=True, new_model_calls=0, new_fits=0,
        new_prediction_processes=0, external_outcomes_read=False,
        reused_ordinary_episodes=2, nonconcurrent_control=True))
    print(json.dumps(dict(prepared=True, same_common_prompt=True,
        reused_ordinary=2, new_method_episodes=2, new_preparation_fits=0)), flush=True)


def evaluate(root):
    assert read(root/'HANDLE444.json')['state'] == 'terminal'
    assert read(root/'audit/EXECUTION_AUDIT444.json')['passed']
    assert not (root/'RESULT445.json').exists()
    cfg = read(root/'PROTOCOL445.json')
    archive = Path(cfg['ordinary_archive'])
    old_index = {x['path']: x for x in read(archive/'SOURCE_MANIFEST.json')['files']}
    for relative, digest in read(root/'REUSED_BINDINGS445.json')['ordinary'].items():
        assert sha(root/'reused_ordinary'/relative) == digest
    source = Path(cfg['source_archive'])
    assert sha(source/'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    source_index = {x['path']: x for x in read(source/'SOURCE_MANIFEST.json')['files']}
    truth = source/'private_inputs/discovery_truth.npz'
    assert sha(truth) == source_index['private_inputs/discovery_truth.npz']['sha256']
    with np.load(truth, allow_pickle=False) as z:
        actual, ids = z['y'].copy(), z['sample_ids'].copy()
    assert actual.shape == (8,) and len(set(ids)) == 8
    rows = []
    for arm in ['ordinary', 'lineage']:
        base_root = root/'reused_ordinary' if arm == 'ordinary' else root
        for repeat in [1, 2]:
            run = base_root/'subjects'/(arm+'_'+str(repeat))
            state, build = read(run/'broker/STATE.json'), read(run/'BUILD_RESULT.json')
            sealed = read(base_root/'ALL_CHOICES_SEALED444.json')['results']
            assert build == next(x for x in sealed if x['id'] == run.name)
            programs = {}
            for ident, item in state['programs'].items():
                report = item['report']
                fresh = not report.get('initial_models_reused', False)
                folder = run/'programs'/ident if fresh else root/'reference'/ident
                assert read(folder/'REPORT.json') == report
                for relative, digest in report['artifacts'].items():
                    if arm == 'ordinary' and fresh:
                        key = str((folder/relative).relative_to(base_root))
                        assert old_index[key]['sha256'] == digest
                    else:
                        assert sha(folder/relative) == digest
                where = folder/'development' if fresh else root/'qualification/parents'/ident
                values = prediction(where, ids)
                errors = np.abs(values-actual)
                cv = [c['MAE'] for fold in report['cv'] for c in fold['cells']]
                assert len(cv) == 47
                programs[ident] = dict(MAE=float(errors.mean()), CV=float(np.mean(cv)),
                    predictions=values.tolist(), errors=errors.tolist(), new_registration=fresh)
            submission = state.get('submission')
            selected = submission['program_id'] if submission else None
            if submission:
                folder = (run/'programs'/selected if programs[selected]['new_registration']
                          else root/'reference'/selected)
                assert sha(folder/'REPORT.json') == submission['program_report_sha256']
            baseline = np.asarray(programs['p00']['errors'])
            def metrics(values):
                errors = np.abs(np.asarray(values)-actual)
                delta = errors-baseline
                return dict(MAE=float(errors.mean()), harm=float(np.maximum(delta, 0).mean()),
                            gain=float(np.maximum(-delta, 0).mean()), errors=errors.tolist())
            cv_choice = min(programs, key=lambda p: (programs[p]['CV'], p != 'p00', list(programs).index(p)))
            values = np.asarray([p['predictions'] for p in programs.values()])
            rows.append(dict(id=run.name, arm=arm, completed=build['completed'],
                selected=selected, metrics=metrics(programs[selected]['predictions']) if selected else None,
                programs=programs, registration_attempts=state['attempts'],
                fixed_CV_choice=cv_choice, fixed_CV_metrics=metrics(programs[cv_choice]['predictions']),
                equal_mean_control=metrics(values.mean(0)), equal_median_control=metrics(np.median(values, 0)),
                activation=activation(run, state, root) if arm == 'lineage' else None))
    means = {}
    for arm in ['ordinary', 'lineage']:
        rr = [r for r in rows if r['arm'] == arm]
        means[arm] = ({k: float(np.mean([r['metrics'][k] for r in rr]))
                       for k in ['MAE', 'harm', 'gain']}
                      if all(r['completed'] and r['metrics'] is not None for r in rr) else None)
    ordinary, method = means['ordinary'], means['lineage']
    numeric = bool(ordinary and method and method['MAE'] < ordinary['MAE']
                   and method['harm'] <= ordinary['harm'] and method['gain'] >= ordinary['gain'])
    formal = all(r['activation']['formal_activation'] for r in rows if r['arm'] == 'lineage')
    qualifying = []
    for r in rows:
        if r['arm'] != 'lineage' or not r['activation']['formal_activation']:
            continue
        programs = r['programs']; parent = r['activation']['parent']
        best_initial = min(v['MAE'] for v in programs.values() if not v['new_registration'])
        if programs['p03']['MAE'] < min(best_initial, programs[parent]['MAE']):
            qualifying.append(dict(episode=r['id'], parent=parent, successor='p03'))
    counted = cost(root)
    counted.update(qualification_prediction_processes=0, parent_qualification_reused=72,
                   historical_control_episodes_recounted=False)
    result = dict(utc=utc(), rows=rows, means=means, cost_new_only=counted,
        numerical_gate=numeric, both_formal_activation=formal,
        post_intervention_improvements=qualifying, numerical_and_formal_gate=bool(numeric and formal and qualifying),
        primary_passed=False, manual_source_review_pending=True, source_clusters=1,
        physical_queries=8, reused_ordinary_episodes=2, nonconcurrent_control=True,
        exploratory_only=True, independent_confirmations=0, final22_read=False,
        majority_goal_complete=False, own_pool_oracle_not_compared=True)
    save(root/'RESULT445.json', result)
    print(json.dumps({k: result[k] for k in ['means', 'cost_new_only', 'numerical_gate',
          'both_formal_activation', 'post_intervention_improvements']}), flush=True)


def run(root):
    assert not (root/'HANDLE445.json').exists()
    for relative, digest in read(root/'LAUNCH_FREEZE445.json').items():
        assert sha(root/relative) == digest
    save(root/'HANDLE445.json', dict(pid=os.getpid(), state='running', utc=utc()))
    try:
        qualified.run(root)
        qualified.audit(root)
        evaluate(root)
        save(root/'TERMINAL445.json', dict(utc=utc(), exit_code=0, manual_review_pending=True))
    except BaseException as error:
        save(root/'TERMINAL445.json', dict(utc=utc(), exit_code=1,
            error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        state = read(root/'HANDLE445.json')
        state.update(state='terminal', utc=utc())
        save(root/'HANDLE445.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['setup', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run.resolve())
