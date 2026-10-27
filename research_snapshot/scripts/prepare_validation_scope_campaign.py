"""Prepare a bounded cross-task validation-scope pilot; makes no model calls."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


ARCH = Path('/data/yyf-zyq/PHM_20261010')
MILL = ARCH / '20261010T185641Z_crossfamily09'
BATT = ARCH / '20261018T003500Z_a153'
IMP = ARCH / '20261017T_background129/imprl129'
ADMIT = ARCH / '20261017T235800Z_admit152/source/admit-bench'
TOOL = ARCH / '20261018T075630Z_admit166/workspace/tools'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run):
    assert not (run / 'PLAN185.json').exists()
    work = run / 'workspaces'
    work.mkdir()
    (run / 'score_data').mkdir()
    # Freeze sampling independently of future labels; the old CALB EOL-fraction
    # cutoff is deliberately not reused. Missing/censored cases are retained.
    write(run / 'SAMPLING185.json', {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'milling': ['milling_material1', 'milling_material2'],
        'battery': [{'temperature_C': 25, 'sorted_index': 0, 'cycle': 100},
                    {'temperature_C': 35, 'sorted_index': 0, 'cycle': 200}],
        'battery_selection': 'Cell names only; fixed cycle independent of EOL. Both targets excluded from every reference bank.',
        'imprl': ['c3_k2_campaign', 'c5_k4_campaign'],
        'admit': [{'cartridge': 'distillation', 'case': 'D03'},
                  {'cartridge': 'cstr', 'case': 'C06'}],
        'max_primary_sessions': 8, 'subject_seconds': 3600,
        'historical_inputs': 'Milling and IMP scenarios previously measured by researchers; new sessions are outcome-blind, not researcher-unseen confirmation.'})
    jobs = []

    def register(job_id, family, prompt, case_ids, readonly, metadata=None):
        ws = work / job_id
        write(ws / 'readonly_inputs.json', readonly)
        write(ws / 'task_budget.json', {'total_seconds': 3600, 'no_neural_or_RL_training': True})
        prompt_file = run / 'plans' / (job_id + '.txt')
        prompt_file.write_text(prompt)
        jobs.append({'id': job_id, 'family': family, 'condition': 'prospective_natural_task',
                     'repeat': 1, 'workspace': str(ws.relative_to(run)),
                     'prompt_file': str(prompt_file.relative_to(run)),
                     'answer_mode': 'text', 'case_ids': case_ids, **(metadata or {})})

    data_names = ['data_dictionary.json', 'features.csv', 'prediction_schema.json',
                  'query_inventory.csv', 'sensors.npz', 'test.csv', 'train.csv']
    for material in [1, 2]:
        job = f'milling185_material{material}'
        ws = work / job
        ws.mkdir()
        old = f'milling_material{material}_r1'
        for name in data_names:
            shutil.copy2(MILL / 'tasks' / old / name, ws / name)
        prompt = (MILL / 'tasks' / (old + '_prompt.txt')).read_text()
        prompt = prompt.replace('1200 second', '3600 second')
        register(job, 'milling', prompt, [f'milling_material{material}'], data_names + ['task_budget.json'],
                 {'origin': 'researcher task on original NASA/BEST measurements, not an official Agent benchmark'})

    sys.path.insert(0, str(BATT / 'pydeps'))
    spec = importlib.util.spec_from_file_location('original_calb_parser', BATT / 'BatteryBench/src/preprocess/augment_calb_soh.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    capacities = mod.load_summary_capacity(str(BATT / 'overall_CALB_cycling_data.xlsx'))
    selected = [(sorted(k for k in capacities if k.split('_')[1] == str(temp))[0], temp, cycle)
                for temp, cycle in [(25, 100), (35, 200)]]
    excluded = {x[0] for x in selected}
    write(run / 'score_data/BATTERY_SELECTION185.json', {'selected_before_eol_read': selected, 'selection_uses_names_only': True})
    gold = {}
    for number, (cell, temp, cycle) in enumerate(selected, 1):
        job = f'battery185_case{number:02}'
        ws = work / job
        ws.mkdir()
        refs = ws / 'reference_trajectories'
        refs.mkdir()
        series = capacities[cell]
        if len(series) < cycle:
            raise ValueError('Prespecified fixed cutoff unavailable: ' + cell)
        first = float(series[0])
        with (ws / 'target_history.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['cycle', 'discharge_capacity_Ah', 'soh'])
            writer.writerows([i + 1, float(x), float(x) / first] for i, x in enumerate(series[:cycle]))
        write(ws / 'target_metadata.json', {'case_id': job, 'cell_temperature_C': temp,
              'current_cycle': cycle, 'soh_definition': 'capacity divided by cycle1 capacity',
              'eol_definition': 'first cycle with SOH strictly below0.90',
              'target_future_in_workspace': False, 'decision_cycle_selected_independently_of_eol': True})
        rows = []
        for i, name in enumerate(sorted(set(capacities) - excluded), 1):
            values = capacities[name]
            norm = float(values[0])
            eol = next((j + 1 for j, x in enumerate(values) if float(x) / norm < .9), None)
            rid = f'ref_{i:02}'
            with (refs / (rid + '.csv')).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['cycle', 'discharge_capacity_Ah', 'soh'])
                writer.writerows([j + 1, float(x), float(x) / norm] for j, x in enumerate(values))
            rows.append({'reference_id': rid, 'file': f'reference_trajectories/{rid}.csv',
                         'cell_temperature_C': int(name.split('_')[1]), 'observed_cycles': len(values),
                         'eol_cycle_at_0.90': eol})
        with (ws / 'reference_manifest.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # Targets remain exclusively in controller data, outside every namespace.
        eol = next((j + 1 for j, x in enumerate(series) if float(x) / first < .9), None)
        gold[job] = {'source_cell': cell, 'current_cycle': cycle, 'eol_cycle': eol,
                     'rul_cycles': None if eol is None else max(0, eol - cycle),
                     'current_soh': float(series[cycle - 1]) / first, 'observed_final_cycle': len(series)}
        prompt = (BATT / 'lane153/PROMPT153.txt').read_text()
        prompt = prompt.replace('a bounded 90% prediction interval for RUL.',
                                'a 90% prediction interval for RUL if defensible; otherwise null with explanation.')
        prompt += ('\nYour total budget is3600seconds. Build/select and execute a predictor using the allowed data; '
                   'save runnable analysis.py, validation.json documenting actual checks and their limits, '
                   'report.md and predictions.json. Do not train neural/RL models. '
                   f'predictions.json must contain case_id="{job}", predicted_eol_cycle, predicted_rul_cycles, '
                   'current_soh, interval_90pct_low_cycles, interval_90pct_high_cycles and rationale. '
                   'Use finite nonnegative predictions, or null when unsupported. End with the same JSON object.\n')
        register(job, 'battery', prompt, [job], ['target_history.csv', 'target_metadata.json',
                 'reference_manifest.csv', 'reference_trajectories', 'task_budget.json'],
                 {'origin': 'researcher causal prefix adapter using original CALB trajectories and BatteryBench EOL definition; not official split'})
    write(run / 'score_data/BATTERY_GOLD185.json', gold)

    for setting in ['c3_k2_campaign', 'c5_k4_campaign']:
        job = 'imprl185_' + setting
        ws = work / job
        ws.mkdir()
        shutil.copytree(IMP / 'workspaces' / setting / 'inputs', ws / 'inputs')
        shutil.copy2(ws / 'inputs/starter_policy.py', ws / 'policy.py')
        task = (ws / 'inputs/TASK.md').read_text()
        task = task.replace('For simulation/parameter choice only seeds1and2 are allowed.',
                            'For simulation/parameter choice development seeds1through64 are allowed.')
        task = task.replace('your2dev simulations', 'your development simulations')
        task = task.replace('hidden seeds/IDs/results', 'final evaluation seeds/IDs/results')
        task = task.replace('Two independent coding sessions have1200s each', 'Your coding session has3600s')
        task = task.replace('Final evaluation has8pre-frozen simulated repetitions per configuration',
                            'Final evaluation has64pre-frozen simulated repetitions per configuration')
        (ws / 'inputs/TASK.md').write_text(task)
        evaluator = ws / 'inputs/runtime/evaluate129.py'
        code = evaluator.read_text()
        assert 'for i in[1,2]' in code
        evaluator.write_text(code.replace('for i in[1,2]', 'for i in range(1,9)'))
        prompt = ('Solve the full original30-step joint inspection/repair task in inputs/TASK.md and inputs/config.json. '
                  'Inspect the source, assess costs and observation semantics, develop and test a defensible self-contained '
                  'policy.py plus rationale.md. Use development seeds1through64 only; default test helper runs1through8. '
                  'Use the unchanged original model and standard methods; no newRL/neuraltraining. '
                  'Your total coding budget is3600seconds. Save actual checks, comparisons and limitations in validation.json.\n')
        register(job, 'imprl', prompt, [setting], ['inputs', 'task_budget.json'],
                 {'setting': setting, 'origin': 'centralized code-policy adapter over original IMP-MARL dynamics/reward; not pretrained MARL reproduction'})

    sys.path.insert(0, str(ADMIT))
    from admitbench.cartridge import load_cartridge
    from admitbench.prompts import SEALED_MARKERS, render_system, render_user
    for cartridge, case_id in [('distillation', 'D03'), ('cstr', 'C06')]:
        job = 'admit185_' + case_id.lower()
        ws = work / job
        ws.mkdir()
        tools = ws / 'tools'
        shutil.copytree(TOOL, tools, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        adapter = tools / 'verify.py'
        code = adapter.read_text()
        code = code.replace("if name != 'distillation':", "if name not in ('distillation','cstr'):")
        code = code.replace("manifest={'world': 'column'}", "manifest={'world': 'column' if name=='distillation' else 'cstr'}")
        adapter.write_text(code)
        cart = load_cartridge(cartridge)
        case = cart.case(case_id)
        context = cart.context_for(case)
        text = ('## Official ADMITBench system context\n\n' + render_system(cart, style='narrative') +
                '\n\n## Official ADMITBench episode input\n\n' +
                render_user(cart, case, cart.evidence_log_for(case), context, style='narrative'))
        assert not [m for m in SEALED_MARKERS if m in text]
        text += ('\n\nThe existing published consequence checker is available as tools/verify.py. '
                 'Its tool docstring and source specify its interface. You may write a request JSON with action, plant_state, '
                 f'cartridge="{cartridge}",params,horizon_s=1200 and run python3 tools/verify.py request.json. '
                 'Use the original task requirements and submit the original action-record response format. '
                 'Total budget3600seconds. Save any development checks you perform; no hidden feedback is supplied.\n')
        register(job, 'admit', text, [case_id], ['tools', 'task_budget.json'],
                 {'cartridge': cartridge, 'original_case': case_id,
                  'origin': 'original ADMITBench prompt/gates with a case-free adapter to its published consequence function'})

    write(run / 'PLAN185.json', {'jobs': jobs})
    for family in ['milling', 'battery', 'imprl', 'admit']:
        write(run / 'plans' / (family + '.json'), {'jobs': [j for j in jobs if j['family'] == family]})
        mounts = []
        if family == 'imprl':
            mounts = [{'source': str(IMP / 'source/imp_marl_original'), 'destination': '/opt/imp_source'},
                      {'source': str(ARCH / '20261016T220500Z_inventory127/infra_runtime127/deps'), 'destination': '/opt/numdeps'}]
        write(run / 'plans' / (family + '_mounts.json'), {'mounts': mounts})
    # Plans in plans/ must refer to their own parent, not the run root.
    for family in ['milling', 'battery', 'imprl', 'admit']:
        p = run / 'plans' / (family + '.json')
        obj = json.loads(p.read_text())
        for job in obj['jobs']:
            job['workspace'] = '../' + job['workspace']
            job['prompt_file'] = Path(job['prompt_file']).name
        write(p, obj)
    shutil.copy2(ARCH / '20261018T075630Z_admit166/cli_config166.toml', run / 'cli_config185.toml')
    write(run / 'PREPARED185.json', {'prepared_utc': datetime.now(timezone.utc).isoformat(),
          'primary_sessions_prepared': len(jobs), 'model_calls': 0,
          'families': {f: sum(j['family'] == f for j in jobs) for f in ['milling', 'battery', 'imprl', 'admit']},
          'files': [{'path': str(p.relative_to(run)), 'sha256': sha(p), 'bytes': p.stat().st_size}
                    for p in sorted(work.rglob('*')) if p.is_file()]})
    print(json.dumps({'run': str(run), 'prepared_jobs': len(jobs), 'model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    prepare(parser.parse_args().run.resolve())
