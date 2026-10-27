"""Same-information native Agent qualification for the frozen partial-update controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from controlled_feedback import save, sha, utc
from native_validation_comparison import closure

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261025T104200Z_hust367')
PROFILE = Path('/data/yyf-zyq/PHM_20261010/20261025T121800Z_profiles371')
CAPACITY = Path('/data/yyf-zyq/PHM_20261010/20261025T130111Z_capacity373')
PARTIAL = Path('/data/yyf-zyq/PHM_20261010/20261025T133000Z_partial374')
PRIOR = Path('/data/yyf-zyq/PHM_20261010/20261024T143100Z_m10extension329')
read = lambda p: json.loads(Path(p).read_text())

PROMPT = '''Review one actual RUL program revision for a measured battery at cycle500. You are an ordinary coding Agent and may run Python to inspect the real evidence. Read /data/contract.json, /data/cases.json and call phmforge evidence. This is the acceptance stage for one already trained parent and one already trained correction: choose any convex combination, not a new feature/model search. P00 retains the parent, P01 accepts the full correction; mix:ALPHA accepts any finite ALPHA in [0,1], with prediction parent + ALPHA*(candidate-parent). You need not use a grid, change the parent, or follow any prescribed uncertainty rule.

Historical completed cells, genuine leave-one-cell-out predictions, one fixed split and a min-max error-adjusted interval are available. The interval has only seven calibration units, no supplied confidence guarantee, and can be wrong. Source residual training used cross-fitted parent predictions while historical development used the full parent; this fitting-context difference is disclosed. Current RUL and unfinished/future capacity are unavailable. Other leave-one-out tasks and other Agents are isolated; do not access their files, the network, other model APIs or spawn agents. No new PHM fitting is needed or permitted, but numerical/statistical analysis of all supplied evidence is allowed.

Minimize unseen absolute RUL error, retain useful improvements and avoid harmful replacement. Compare the candidate with the parent using the legal evidence; do not assume that conservative, CV-optimal, or full updates are correct. Explain actual checks and limitations. Submit exactly one row {case,incumbent,selected,reason} through phmforge select, with case=hust_revision and incumbent=P00. selected is P00, P01, or mix:ALPHA. Save selection.json and report.md. Tools are also available via python3 /tools/program_update_tools.py client --request followed by JSON. Selection is immutable; no evaluation feedback is provided. All outcomes are scored externally only after every run seals.

Budget:1200seconds,30model requests,2000000input tokens,131072output tokens total,32768output per request,2numeric threads. Use /app/.runtime for scratch, never /tmp. An actual tool submission is required; prose is not a result.
'''


def environment(root):
    from run_full_benchmark_campaign import clean_env
    env = clean_env(root)
    env.update(PYTHONPATH=str(DEPS)+':'+str(root/'code'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def prepare(root):
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    cli = read(root/'audit/CLI_SOURCE375.json')
    assert cli['tag'] == 'rust-v0.164.0' and sha(Path(cli['cli'])) == cli['cli_sha256']
    sources = {}
    for key, folder in [('source', SOURCE), ('profiles', PROFILE), ('capacity', CAPACITY), ('partial', PARTIAL)]:
        receipt = read(folder/'ARCHIVE_VERIFIED.json')
        assert sha(folder/'SOURCE_MANIFEST.json') == receipt['manifest_sha256']
        sources[key] = dict(path=str(folder), manifest_sha256=receipt['manifest_sha256'])
    names = closure(['run_partial_update_baseline', 'screen_program_updates', 'program_update_tools',
                     'run_validation_transfer', 'qualify_native_cli_mcp'])
    for name in names:
        shutil.copy2(PROJECT/'scripts'/(name+'.py'), root/'code'/(name+'.py'))
    for folder in ['shared_tools', 'scripts']:
        (root/folder).mkdir()
    for name in ['program_update_tools.py', 'controlled_feedback.py', 'prepare_fault_extensions.py',
                 'acquire_forge_pdmb_sources244.py', 'isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name, root/'shared_tools'/name)
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py', root/'scripts/isolated_model_gateway.py')
    for name in ['base_cli_config.toml', 'model_catalog187.json', 'retention_policy.json']:
        shutil.copy2(PRIOR/name, root/name)
    config = dict(id='A375', frozen_utc=utc(), sources=sources, model='deepseek-flash',
        cli=cli['cli'], cli_sha256=cli['cli_sha256'], cli_version='0.164.0',
        purpose='Qualify an ordinary native Agent with the same evidence and continuous acceptance action space as the frozen conventional controls. No new method promotion or novelty claim.',
        episode_seconds=1200, episode_requests=30, episode_input_tokens=2000000,
        episode_output_tokens=131072, output_per_request=32768, numerical_threads=2,
        repeats=2, case_count=8, maximum_parallel_agents=2, parallel_case_workers=2,
        maximum_total_concurrent_agents=4, maximum_new_requests=480, maximum_new_input_tokens=32000000,
        maximum_new_output_tokens=2097152, maximum_disk_bytes=4294967296,
        preparation_file='PREPARATION375.json', qualify_submission_chain=True,
        qualification_selected='mix:0.5',
        reset_qualification_transport=True, researcher_previously_knows_final_scores=True,
        new_PHM_fit_budget=0, new_PHM_prediction_budget=0,
        actual_training_reused='A373 128 and A37464 trained models, same qualified cached predictions; do not recount training.',
        isolation='Each held cell has separate public data and fresh native sessions. Its target label is absent; other7 historical labels allowed. No cross-episode memory, files or network.',
        same_information='All current controls and ordinary subjects can use all source47 labels, other7 historical development labels, genuine inner forecasts, original code and query forecasts.',
        paired_policy='Cap the actual ordinary-chosen ALPHA at the frozen A374 maximum interval-supported ALPHA; no extra model requests. Also report binary cap and independent fixed A374 controls.',
        action_space='All alpha in[0,1] between the SAME parent and candidate, for ordinary and shield; conditional one-revision acceptance only, not unrestricted coding or a global PHM optimum.',
        primary='Mean cell MAE, regret against best point on the common continuous segment, mean harmful-update loss, and mean gain/opportunity across positive-opportunity cells.',
        aggregation='Average the two repeats within cell then average all8. This is one correlated opened development cohort, not8 independent scenarios.',
        gate='Exploratory paired shield requires lower mean MAE/regret, no greater harm and no lower gain-capture than ordinary, with all16 submissions valid. This is not goal completion.',
        failure_policy='Retain all timeouts/invalid/missing submissions. No outcome-based recovery; fallback keep separately reported and never called a successful Agent.',
        fixed_thresholds_unchanged=True, all8_previously_used_for_design=True,
        independent_confirmation_scenarios=0, final22_opened=False, A33872_opened=False, novelty_confirmed=False,
        admission_rationale='This is missing strong-baseline qualification, not admission of R13 as an innovative method. A374 numeric-gate failure is preserved; no thresholds, intervals, recipes or outcome criteria are retuned.')
    save(root/'CAMPAIGN_CONFIG.json', config)
    save(root/'PROMPT_FREEZE.json', dict(ordinary=PROMPT))
    save(root/'CODE_FREEZE.json', {p.name:sha(p) for p in (root/'code').glob('*.py')})
    with np.load(SOURCE/'private_inputs/training_queries_private.npz', allow_pickle=False) as z:
        source_data = {k:z[k] for k in ['x','y','groups','past_cap','rates']}
    with np.load(SOURCE/'private_inputs/discovery_truth.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in ['x','y','past_cap','rates','age','cycle']}
    with np.load(PROFILE/'PROFILES371.npz', allow_pickle=False) as z:
        cv, full, program_names = z['cv_prediction'], z['development_prediction'], z['programs'].tolist()
    decisions = read(PARTIAL/'DECISIONS374.json')['rows']
    fits = read(PARTIAL/'FIT_EVENTS374.json')
    targets = []
    for held, decision in enumerate(decisions):
        assert decision['held'] == held
        child = root/'case_runs'/('q'+str(held)); child.mkdir()
        for folder in ['code','shared_tools','public/rows','scripts','audit','outputs','runtime/tmp','runtime/mpl']:
            (child/folder).mkdir(parents=True)
        for folder in ['code', 'shared_tools']:
            for p in (root/folder).glob('*.py'):
                shutil.copy2(p, child/folder/p.name)
        shutil.copy2(root/'scripts/isolated_model_gateway.py', child/'scripts/isolated_model_gateway.py')
        for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json','PROMPT_FREEZE.json','CODE_FREEZE.json']:
            shutil.copy2(root/name, child/name)
        allowed = [i for i in range(8) if i != held]
        inner = [e for e in fits if e['held'] == held and e['kind'] == 'loo']
        assert {e['probe'] for e in inner} == set(allowed)
        records = [dict(id='historical_'+str(e['probe']), actual_RUL=float(data['y'][e['probe']]),
            parent_prediction=e['parent_predictions'][0], candidate_prediction=e['warm_predictions'][0],
            current_query_candidate_prediction=e['warm_predictions'][1],
            parent_program=e['parent'], fitting_historical_cells=['historical_'+str(i) for i in e['fitting_development_cells']],
            actual_model_sha256=e['model_sha256']) for e in inner]
        case = dict(id='hust_revision', candidate_ids=['P00','P01'], incumbent_ids=['P00'], continuous_update=True,
            original_incumbent='P00', metric='Absolute RUL error in cycles',
            candidate_descriptions=['Actually trained complete parent', 'Actually trained parent plus residual Ridge correction'],
            group_losses=[[abs(r['actual_RUL']-r[k]) for r in records] for k in ['parent_prediction','candidate_prediction']],
            groups=[dict(id=r['id'], unit='whole historical cell at cycle500') for r in records],
            query_predictions=dict(P00=decision['base_prediction'], P01=decision['candidate_prediction']),
            prediction_interval=dict(lower=decision['interval_lower'], upper=decision['interval_upper'], calibration_cells=7,
                definition='min/max of inner query forecasts minus/plus their own held-cell absolute errors', confidence_guarantee=None),
            numerical_data='/data/rows/arrays.npz', recipe_sources=['/data/rows/parent_program.py','/data/rows/correction_recipe.py'])
        split = next(e for e in fits if e['held'] == held and e['kind'] == 'split')
        packet = dict(case=case, inner_predictions=records,
            split_validation=[dict(id='historical_'+str(i),actual_RUL=float(data['y'][i]),
                parent_prediction=split['parent_predictions'][k],candidate_prediction=split['warm_predictions'][k])
                for k,i in enumerate(split['query_development_cells'][:-1])],
            split_query_predictions=dict(parent=split['parent_predictions'][-1], candidate=split['warm_predictions'][-1]),
            history_ids=['historical_'+str(i) for i in allowed], program_columns=program_names,
            training_context='source47 parent forecasts are cross-fitted; full historical7 parent forecasts use all47 source cells. Inner correction fits exclude the inner validation and current query cell.')
        np.savez_compressed(child/'public/rows/arrays.npz',
            **{'source_'+k:v for k,v in source_data.items()}, source_cv_predictions=cv,
            historical_indices=np.asarray(allowed), historical_x=data['x'][allowed], historical_y=data['y'][allowed],
            historical_full_predictions=full[allowed], historical_past_cap=data['past_cap'][allowed], historical_rates=data['rates'][allowed],
            query_x=data['x'][held:held+1], query_past_cap=data['past_cap'][held:held+1], query_rates=data['rates'][held:held+1],
            query_full_predictions=full[held:held+1], query_age=data['age'][held:held+1], query_cycle=data['cycle'][held:held+1])
        save(child/'public/cases.json', [case]); save(child/'public/rows/hust_revision.json', packet)
        shutil.copy2(SOURCE/'subjects/ordinary_2/programs/p01/source/program.py', child/'public/rows/parent_program.py')
        shutil.copy2(CAPACITY/'code/audit_program_update_capacity.py', child/'public/rows/correction_recipe.py')
        save(child/'public/contract.json', dict(task='RUL revision acceptance on measured HUST-LFP cells; source-native remaining cycles including current unfinished cycle.',
            action='P00, P01, or mix:ALPHA for any finite ALPHA in[0,1]; no new model fitting. Mixture is evaluated on actual immutable parent/candidate forecasts.',
            information='47 labeled source cells,7 historical labeled cells, one current query without its target. Other task views are absent.',
            current_capacity_and_future_targets_unavailable=True, physical_EOL_independently_confirmed=False,
            source='Measured LFP/graphite 30C; original native RUL convention; historical source model/prediction hashes retained.',
            inputs='173 documented past curve/capacity/age/rate features; past_cap contains only9 already completed discharge capacities.',
            code='Original program source is provided for inspection; referenced host archives are not mounted. Cached real forecasts are the permitted immutable candidate interface.',
            uncertainty='Only7 calibration cells; no claimed confidence, conditional safety or deployment-shift guarantee.'))
        save(child/'FIXED_CHOICES.json', dict(weights=decision['weights'], fixed_predictions=decision['predictions'], final_scores_joined=False))
        save(child/'CAMPAIGN_CONFIG.json', dict(config, case_index=held))
        with np.load(child/'public/rows/arrays.npz', allow_pickle=False) as z:
            assert held not in z['historical_indices'] and np.array_equal(z['historical_y'], data['y'][allowed])
            assert 'query_y' not in z.files and len(z['historical_y']) == 7
        save(child/'PREPARATION375.json', dict(passed=True, held_label_absent=True, historical_indices=allowed,
            file_hashes={str(p.relative_to(child/'public')):sha(p) for p in (child/'public').rglob('*') if p.is_file()},
            original_cold_checks_reused=True, new_fits=0, new_predictions=0))
        targets.append(dict(held=held, actual=float(data['y'][held]), parent=decision['base_prediction'],
            candidate=decision['candidate_prediction'], cap=decision['weights']['interval_partial_maxsafe']))
    save(root/'EVALUATION_TARGETS375.json', targets)
    save(root/'PREPARATION375.json', dict(passed=True, isolated_cases=8, planned_native_sessions=16,
        new_PHM_fits=0, new_PHM_predictions=0, source_bindings=sources))
    out=PROJECT/'results/native375';out.mkdir(exist_ok=True)
    save(out/'PROTOCOL375.json', config)
    print(json.dumps(dict(prepared=True, cases=8, planned_native_sessions=16, new_LLM_calls=0)))


def child(root):
    import screen_program_updates as runner
    runner.OUT = root/'outputs'
    runner.agents(root)


def agents(root):
    assert read(root/'PREPARATION375.json')['passed'] and not (root/'HANDLE375.json').exists()
    config = read(root/'CAMPAIGN_CONFIG.json')
    save(root/'HANDLE375.json', dict(controller_pid=os.getpid(), utc=utc(), state='running'))
    rows=[]
    def run_one(folder):
        with (root/'logs'/(folder.name+'.log')).open('w') as log:
            p=subprocess.run([str(PYTHON),'-B',str(root/'code/run_partial_update_baseline.py'),'child','--run',str(folder)],
                env=environment(root), stdout=log, stderr=subprocess.STDOUT)
        return dict(case=folder.name, returncode=p.returncode)
    with ThreadPoolExecutor(max_workers=config['parallel_case_workers']) as pool:
        for f in as_completed([pool.submit(run_one,c) for c in sorted((root/'case_runs').iterdir())]):
            rows.append(f.result());save(root/'PROGRESS375.json',dict(utc=utc(),cases=rows))
    files=[p for c in sorted((root/'case_runs').iterdir()) for p in [c/'AGENT_CHOICES.json',c/'ALL_CHOICES_SEALED.json'] if p.exists()]
    save(root/'ALL_CHOICES_SEALED375.json',dict(utc=utc(),cases=rows,files={str(p.relative_to(root)):sha(p) for p in files},
        researcher_previously_knows_outcomes=True,new_agent_scores_joined=False))
    save(root/'TERMINAL375.json',dict(utc=utc(),exit_code=0 if all(r['returncode']==0 for r in rows) else 1,cases=rows))
    print(json.dumps(dict(terminal=True,cases=rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','child','agents']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.run.resolve())
