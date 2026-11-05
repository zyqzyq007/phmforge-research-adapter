"""Map historical questions to explicitly scoped source-based numerical subsets."""
import json
from pathlib import Path
import sys

from .data import resource_json, write_json
from .runner import _ready, evaluate, train_reference, training_fingerprint, workspace


def inventory(task=None, include_additional=False, include_substitutions=False, include_unavailable=False,
              include_alternatives=False):
    base = resource_json('source55_inventory.json')['tasks']
    additional = resource_json('source20_inventory.json')['tasks']
    alternatives = {row['task_id']: row for row in resource_json('research_alternatives.json')['tasks']}

    def selected(row):
        if not include_alternatives or row['task_id'] not in alternatives:
            return row
        alternative = alternatives[row['task_id']]
        return {**row, **alternative, 'original_status': row['status'],
                'original_unavailable_reason': row['blocked_reason'],
                'limitations': row['limitations'] + alternative['limitations']}

    if task:
        found = [row for row in base + additional if row['task_id'] == task]
        if len(found) != 1:
            raise ValueError(f'Unknown historical task ID: {task}')
        return selected(found[0])
    tasks = list(base)
    if include_additional or include_substitutions or include_unavailable or include_alternatives:
        tasks.extend(row for row in additional if row['status'] == 'related_source_partial_adaptation')
    if include_substitutions or include_unavailable or include_alternatives:
        tasks.extend(row for row in additional if row['status'] == 'source_substitution')
    if include_unavailable or include_alternatives:
        tasks.extend(selected(row) for row in additional if row['status'] == 'unavailable')
    return {'historical_task_entries': len(tasks), 'original_task_entries': 75,
            'base_entries': 55, 'additional_related_partial': 7,
            'additional_explicit_substitutions': 9, 'additional_unavailable': 4,
            'explicit_research_alternatives_selected': 4 if include_alternatives else 0,
            'interpretation': 'Numeric adaptations are partial; related or substituted sources and shared fits do not create complete or independent original-task results.',
            'tasks': [{key: row.get(key) for key in ('task_id', 'dataset', 'prepare_dataset', 'numeric_scope', 'status', 'source_relation', 'adapter_id', 'original_target_status')}
                      for row in tasks]}


def case_data(task, data_root, research_alternative=False):
    item = inventory(task, include_alternatives=research_alternative)
    if item['status'] == 'unavailable':
        raise ValueError(f"Task {task} is unavailable: {item['blocked_reason']}")
    root = Path(data_root).resolve()
    is_engine = item['prepare_dataset'] in ('engine-qa', 'engine-qa-alternatives')
    case_id = item.get('case_id', task)
    if is_engine and (root / 'cases' / case_id / 'READY.json').is_file():
        root = root / 'cases' / case_id
    root, ready = _ready(root)
    if ready['dataset'] != item['dataset']:
        raise ValueError(f"{task} requires {item['dataset']}; the supplied root contains {ready['dataset']}")
    if is_engine and ready['task_id'] != case_id:
        raise ValueError('The engine case root must match the requested task identity')
    return item, root


def create_workspace(task, data_root, out, research_alternative=False):
    item, root = case_data(task, data_root, research_alternative)
    result = workspace(root, out)
    dest = Path(out).resolve()
    with (dest / 'TASK.md').open('a', encoding='utf-8') as f:
        f.write('\n## Original question and adaptation boundary\n\n')
        if item.get('adapted_question'):
            f.write('This explicitly selected experiment replaces or restricts the original source/target. '
                    'The original target remains unavailable.\n\n')
            f.write('Current experiment question: ' + item['adapted_question'] + '\n\n')
            f.write('Historical question for provenance only; its template values are not the experiment observations:\n\n')
        f.write(item['original_question'] + '\n\n')
        f.write('The runnable numeric part is: ' + item['numeric_scope'] + '.\n')
        if item.get('source_relation') in ('substitution', 'related_dataset'):
            f.write('The actual source is ' + item['dataset'] + ', a different dataset from the original question. '
                    'Do not present it as the original dataset.\n')
        for limit in item['limitations']:
            f.write('- ' + limit + '\n')
        f.write('\nDo not invent unknown costs, risk, legal compliance or unsupported source labels.\n')
    meta = json.loads((dest / 'WORKSPACE.json').read_text())
    meta.update(original_task_id=task, source_numeric_scope=item['numeric_scope'],
                source_relation=item.get('source_relation', 'existing_partial_adaptation'),
                adapter_id=item.get('adapter_id'), original_target_status=item.get('original_target_status'),
                limitations=item['limitations'], complete_original_task_claimed=False)
    write_json(dest / 'WORKSPACE.json', meta)
    return {**result, 'original_task_id': task, 'complete_original_task_claimed': False}


def _rank_latest_prefixes(predictions, observations):
    """Rank at most one observed prefix per unit; do not imply physical risk."""
    time_keys = ('time_sec', 'elapsed_minutes', 'elapsed_acquisitions')
    time_key = next((key for key in time_keys if key in observations), None)
    if time_key is None:
        raise ValueError('No observed elapsed-time field for selecting one prefix per unit')
    metadata = {str(sid): (int(unit), float(elapsed)) for sid, unit, elapsed in zip(
        observations['sample_ids'], observations['unit_ids'], observations[time_key])}
    latest = {}
    for row in predictions:
        sid = str(row['sample_id'])
        unit, elapsed = metadata[sid]
        if int(row['unit_id']) != unit:
            raise ValueError('Prediction and observation have different unit identities')
        if unit not in latest or (elapsed, sid) > (latest[unit][0], latest[unit][1]):
            latest[unit] = (elapsed, sid, float(row['predicted_rul']))
    ranked = sorted(({'unit_id': unit, 'observed_elapsed': values[0], 'predicted_rul': values[2]}
                     for unit, values in latest.items()), key=lambda row: (row['predicted_rul'], row['unit_id']))
    return {'available_units': len(ranked), 'top_five': ranked[:5] if len(ranked) >= 5 else None,
            'ranked_available_units': ranked,
            'interpretation': 'One latest available prefix per unit, which may represent different stages; lower predicted RUL is not validated physical risk.'}


def run(task, data_root, out, epochs=10, timeout=300, submission=None, research_alternative=False):
    item, root = case_data(task, data_root, research_alternative)
    dest = Path(out).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if root.is_relative_to(dest) or dest.is_relative_to(root):
        raise ValueError('Run directory must be separate from the private prepared data root')
    dest.mkdir(parents=True)
    try:
        if submission:
            model = Path(submission).resolve()
            manifest = json.loads((model / 'SUBMISSION.json').read_text())
            if manifest['dataset'] != item['dataset']:
                raise ValueError('Reused submission was trained for a different dataset')
            if manifest.get('training_fingerprint') != training_fingerprint(root):
                raise ValueError('Reused reference does not match the current training/development data and configuration; train a new reference.')
            fit = {'status': 'reused', 'submission': str(model), 'trained_again': False}
        else:
            model = dest / 'submission'
            fit = train_reference(root, model, epochs)
        scored = evaluate(root, model, dest / 'evaluation', timeout)
        summary = {'status': 'completed', 'historical_task_id': task, 'dataset': item['dataset'],
                   'numeric_scope': item['numeric_scope'], 'complete_original_task_claimed': False,
                   'source_relation': item.get('source_relation', 'existing_partial_adaptation'),
                   'adapter_id': item.get('adapter_id'), 'original_target_status': item.get('original_target_status'),
                   'adapted_question': item.get('adapted_question'),
                   'original_question': item['original_question'], 'limitations': item['limitations'],
                   'training': fit, 'evaluation': scored,
                   'interpretation': 'Source-based numerical research adaptation; a completed execution is not a correct or complete original-task answer.'}
        if task.startswith('pdm_cost_benefit_'):
            summary['unsupported_original_fields'] = {'costs': None, 'optimal_maintenance_threshold': None,
                'optimized_schedule': None, 'reason': 'Required costs and action-dependent decision model are unavailable.'}
        if task.startswith('pdm_safety_'):
            summary['unsupported_original_fields'] = {'validated_physical_risk': None, 'legal_compliance': None,
                'safety_policy': None, 'reason': 'No validated physical-risk or legal-compliance evidence is available.'}
        if task in ('pdm_rul_001', 'pdm_safety_001'):
            predictions = json.loads((dest / 'evaluation/predictions.json').read_text())['predictions']
            ranked = sorted(predictions, key=lambda row: float(row['predicted_rul']))
            if task == 'pdm_rul_001':
                summary['lowest_predicted_rul_top_10'] = [{'unit_id': row['unit_id'], 'predicted_rul': row['predicted_rul']}
                                                           for row in ranked[:10]]
            else:
                summary['predicted_rul_below_20_cycles'] = [row['unit_id'] for row in ranked if float(row['predicted_rul']) < 20]
                summary['threshold_interpretation'] = 'Question-specified rule applied to predicted RUL, not a validated safety assessment.'
        if task in ('pdm_rul_002', 'pdm_rul_011', 'pdm_rul_012', 'pdm_rul_013', 'pdm_rul_014', 'pdm_rul_015'):
            import numpy as np
            predictions = json.loads((dest / 'evaluation/predictions.json').read_text())['predictions']
            with np.load(root / 'holdout/test.npz', allow_pickle=False) as observations:
                summary['low_predicted_rul_ranking'] = _rank_latest_prefixes(predictions, observations)
        write_json(dest / 'SCENARIO.json', summary)
        return {'status': 'completed', 'historical_task_id': task, 'summary': str(dest / 'SCENARIO.json'),
                'evaluation': scored, 'complete_original_task_claimed': False}
    except Exception as error:
        write_json(dest / 'SCENARIO.json', {'status': 'failed', 'historical_task_id': task,
                   'error_type': type(error).__name__, 'error': str(error)})
        raise


def _prepare_source(name, dest, source_map, workers, extractor):
    """Prepare exactly the requested public source; never substitute another source silently."""
    source = Path(source_map[name]).expanduser() if name in source_map else None
    if name in ('xjtu-rul', 'xjtu-terminal') and source is None:
        other = 'xjtu-terminal' if name == 'xjtu-rul' else 'xjtu-rul'
        if other in source_map:
            source = Path(source_map[other]).expanduser()
        elif (dest.parent / other / 'raw').is_dir():
            source = dest.parent / other / 'raw'
    if name in ('engine-qa', 'engine-qa-alternatives'):
        from .engine import prepare_engine
        cached = dest.parent / 'engine-qa/raw'
        if name == 'engine-qa-alternatives' and source is None and all((cached / file).is_file() for file in ('train_qa.jsonl', 'test_qa.jsonl')):
            source = cached
        return prepare_engine(dest, source, workers, include_alternatives=name == 'engine-qa-alternatives')
    if name == 'snu-planetary':
        from .snu import prepare_snu
        return prepare_snu(dest, source)
    if name == 'mfpt':
        from .data import prepare_mfpt
        return prepare_mfpt(dest, source, workers)
    if name == 'ldm-rul':
        from .ldm import prepare_ldm
        return prepare_ldm(dest, source, workers)
    if name == 'ims-endpoint':
        from .ims import prepare_ims
        return prepare_ims(dest, source, extractor)
    if name.startswith('cmapss-'):
        from .data import prepare_cmapss
        return prepare_cmapss(dest, source, name.split('-', 1)[1])
    if name == 'femto-rul':
        from .femto import prepare_femto
        pdf = Path(source_map['femto-challenge-pdf']).expanduser() if 'femto-challenge-pdf' in source_map else None
        return prepare_femto(dest, source, pdf)
    if name in ('xjtu-rul', 'xjtu-terminal'):
        from .xjtu import prepare_xjtu, prepare_xjtu_terminal
        return (prepare_xjtu if name == 'xjtu-rul' else prepare_xjtu_terminal)(dest, source, workers, extractor)
    from .pdmb_fault import prepare_pdmb_fault
    return prepare_pdmb_fault(dest, name, source)


def run_all(data_home, out, epochs=10, timeout=600, prepare_missing=False,
            source_map=None, workers=4, extractor=None, selected_tasks=None,
            include_additional=False, include_substitutions=False, include_alternatives=False):
    """Run selected entries, reusing a fit for the same source and preserving failures."""
    home, dest = Path(data_home).resolve(), Path(out).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if dest.is_relative_to(home) or home.is_relative_to(dest):
        raise ValueError('Batch run directory must be separate from the data home')
    source_map = source_map or {}
    tasks = inventory(include_additional=include_additional, include_substitutions=include_substitutions,
                      include_alternatives=include_alternatives)['tasks']
    if selected_tasks:
        selected = set(selected_tasks)
        all_tasks = inventory(include_unavailable=True, include_alternatives=include_alternatives)['tasks']
        known = {item['task_id'] for item in all_tasks}
        if selected - known:
            raise ValueError(f'Unknown task IDs: {sorted(selected - known)}')
        unavailable = [item['task_id'] for item in all_tasks if item['task_id'] in selected and item['status'] == 'unavailable']
        if unavailable:
            raise ValueError(f'Task IDs unavailable for source-based execution: {unavailable}')
        tasks = [item for item in all_tasks if item['task_id'] in selected]
    home.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True)
    models, source_errors, results = {}, {}, []
    for item in tasks:
        ident, dataset, source_name = item['task_id'], item['dataset'], item['prepare_dataset']
        source_root = home / source_name
        try:
            if source_name in source_errors:
                raise RuntimeError('Source preparation previously failed: ' + source_errors[source_name])
            if not (source_root / 'READY.json').is_file():
                if not prepare_missing:
                    raise FileNotFoundError(f'Prepare {source_name} at {source_root}, or pass --prepare-missing')
                try:
                    _prepare_source(source_name, source_root, source_map, workers, extractor)
                except Exception as error:
                    source_errors[source_name] = f'{type(error).__name__}: {error}'
                    raise
            value = run(ident, source_root, dest / 'tasks' / ident, epochs, timeout, models.get(dataset),
                        research_alternative=include_alternatives)
            if dataset not in models:
                models[dataset] = dest / 'tasks' / ident / 'submission'
            results.append({'task_id': ident, 'dataset': dataset, 'source_relation': item['source_relation'], 'status': 'completed',
                            'adapter_id': item.get('adapter_id'), 'original_target_status': item.get('original_target_status'),
                            'score': value['evaluation']['score']})
        except Exception as error:
            results.append({'task_id': ident, 'dataset': dataset, 'status': 'failed',
                            'error_type': type(error).__name__, 'error': str(error)})
        state = {'status': 'running', 'requested': len(tasks),
                 'completed': sum(x['status'] == 'completed' for x in results),
                 'failed': sum(x['status'] == 'failed' for x in results), 'results': results,
                 'source_preparation_errors': source_errors, 'fitted_datasets': sorted(models),
                 'research_alternatives_selected': sum(row['status'] == 'explicit_research_alternative' for row in tasks),
                 'original_tasks_completed_claimed': False,
                 'interpretation': 'Shared-data numeric adaptations; complete original tasks and independent fits are not claimed.'}
        write_json(dest / 'BATCH.json', state)
        print(f"Scenario {len(results)}/{len(tasks)}: {ident}: {results[-1]['status']}", file=sys.stderr, flush=True)
    state['status'] = 'completed' if state['failed'] == 0 else 'completed_with_failures'
    write_json(dest / 'BATCH.json', state)
    return {'status': state['status'], 'requested': state['requested'], 'completed': state['completed'],
            'failed': state['failed'], 'report': str(dest / 'BATCH.json')}
