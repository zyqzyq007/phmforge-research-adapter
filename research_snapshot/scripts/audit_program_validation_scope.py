"""Audit retained program/validation evidence without executing model code."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def matrix_metrics(matrix):
    size = len(matrix)
    if not size or any(not isinstance(row, list) or len(row) != size for row in matrix):
        raise ValueError('Expected square confusion matrix')
    if any(not isinstance(x, (int, float)) or x < 0 for row in matrix for x in row):
        raise ValueError('Expected nonnegative numeric counts')
    truth = [sum(row) for row in matrix]
    predicted = [sum(row[i] for row in matrix) for i in range(size)]
    f1 = [2 * matrix[i][i] / (truth[i] + predicted[i])
          if truth[i] + predicted[i] else 0.0 for i in range(size)]
    active = [i for i in range(size) if truth[i] + predicted[i]]
    total = sum(truth)
    return dict(rows=total, truth_support=truth, prediction_support=predicted,
                per_class_f1=f1, fixed_label_macro_f1=sum(f1) / size,
                observed_union_macro_f1=sum(f1[i] for i in active) / len(active) if active else 0.0,
                observed_union_labels=active,
                accuracy=sum(matrix[i][i] for i in range(size)) / total if total else None)


def walk_matrices(value, location='$'):
    if isinstance(value, dict):
        matrix = value.get('confusion_matrix')
        if isinstance(matrix, list) and matrix and all(isinstance(row, list) for row in matrix):
            reduction = matrix_metrics(matrix)
            reported = value.get('macro_f1')
            reduction.update(location=location, reported_macro_f1=reported)
            if isinstance(reported, (int, float)):
                reduction['reported_matches_fixed_labels'] = abs(reported - reduction['fixed_label_macro_f1']) < 1e-12
                reduction['reported_matches_observed_union'] = abs(reported - reduction['observed_union_macro_f1']) < 1e-12
            yield reduction
        for key, child in value.items():
            yield from walk_matrices(child, location + '.' + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk_matrices(child, location + '[' + str(i) + ']')


def code_scope(path):
    text = path.read_text()
    tree = ast.parse(text)
    excerpts = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if any(name in ('TRAINING_FAMILIES', 'selected', 'fit_mask', 'family_rows', 'test_records', 'cv') for name in names):
                excerpts.append(dict(line=node.lineno, names=names, source=ast.get_source_segment(text, node)))
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                functions=[node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
                selection_and_fold_excerpts=excerpts)


def training_scope(root, cases):
    """Reduce source IDs/labels and already recorded predictions, never signals."""
    import numpy as np
    config = json.loads((root / 'PROTOCOL301.json').read_text())
    archive = Path(config['source_archive'])
    source, = (archive / 'd/public_data').glob('*/datasets/HUST/train.npz')
    relative = str(source.relative_to(archive))
    entry = next(row for row in json.loads((archive / 'SOURCE_MANIFEST.json').read_text())['files']
                 if row['path'] == relative)
    h = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    assert source.stat().st_size == entry['bytes'] and h.hexdigest() == entry['sha256']
    keys = ['labels', 'groups', 'native_record_mean', 'native_record_scale',
            'shaft_frequency_hz', 'bearing_family', 'nominal_load_watts']
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in keys}
    signatures = np.asarray(['|'.join((f'{float(mean):.17g}', f'{float(scale):.17g}',
        f'{float(speed):.17g}', str(family), str(int(load)))) for mean, scale, speed, family, load in zip(
            arrays['native_record_mean'], arrays['native_record_scale'], arrays['shaft_frequency_hz'],
            arrays['bearing_family'], arrays['nominal_load_watts'])])
    _, row_to_record = np.unique(signatures, return_inverse=True)
    first = np.asarray([np.flatnonzero(row_to_record == i)[0] for i in range(row_to_record.max() + 1)])
    labels = arrays['labels']
    family = arrays['bearing_family'].astype(str)
    record_family = family[first]
    record_labels = labels[first]
    assert np.array_equal(record_labels[row_to_record], labels)
    for i in range(len(first)):
        assert len(np.unique(arrays['groups'][row_to_record == i])) == 1
    selected = np.isin(record_family, ['6205', '6206'])
    counts = []
    primary = next(case for case in cases if case['cohort'] == 'd')
    validation = json.loads((root / 'inputs' / primary['workspace'] / 'validation.json').read_text())
    matrices = validation['training_family_holdout_metrics']
    for held in ('6204', '6205', '6206'):
        actual = record_family != held
        faithful = actual & selected
        held_rows = family == held
        supports = np.bincount(labels[held_rows], minlength=7).tolist()
        matrix = matrices[held]['confusion_matrix']
        assert supports == [sum(row) for row in matrix]
        counts.append(dict(held_family=held, evaluated_rows=int(held_rows.sum()),
            evaluated_records=int(np.sum(record_family == held)), observed_support=supports,
            actual_CV_fit_records=int(actual.sum()), actual_CV_fit_families=sorted(set(record_family[actual])),
            final_policy_CV_fit_records=int(faithful.sum()), final_policy_CV_fit_families=sorted(set(record_family[faithful])),
            extra_fit_record_count=int(np.sum(actual & ~selected))))
    review = next(case for case in cases if case['cohort'] == 'd_review')
    research_path = root / 'inputs' / review['workspace'] / '.runtime/research_results.json'
    candidates = json.loads(research_path.read_text())
    comparator, = [row for row in candidates if row.get('view') == 'order_gz' and
                   row.get('model') == 'lda_auto' and row.get('kind') == 'multiclass']
    comparisons = []
    for i, held in enumerate(('6204', '6205', '6206')):
        selected_records = np.flatnonzero(record_family == held)
        predictions = np.asarray(comparator['cv_preds'][held])
        assert len(predictions) == len(selected_records)
        group_matrix = [[0] * 7 for _ in range(7)]
        for truth, prediction in zip(record_labels[selected_records], predictions):
            group_matrix[int(truth)][int(prediction)] += 1
        group_metrics = matrix_metrics(group_matrix)
        assert abs(group_metrics['fixed_label_macro_f1'] - comparator['cv_folds'][i]) < 1e-12
        mapping = dict(zip(selected_records.tolist(), predictions.tolist()))
        rows = np.flatnonzero(family == held)
        window_matrix = [[0] * 7 for _ in range(7)]
        for row in rows:
            window_matrix[int(labels[row])][int(mapping[int(row_to_record[row])])] += 1
        window_metrics = matrix_metrics(window_matrix)
        comparisons.append(dict(held_family=held, recorded_alternative_record_metrics=group_metrics,
            same_recorded_predictions_window_metrics=window_metrics,
            alternative_window_matrix_equals_primary_report=window_matrix == matrices[held]['confusion_matrix'],
            alternative_window_matrix=window_matrix))
    result = dict(source_path=str(source), source_sha256=entry['sha256'],
        fields_read=keys, waveforms_or_features_read=False, records=len(first), rows=len(labels),
        final_selected_records=int(selected.sum()), final_selected_rows=int(np.isin(family, ['6205', '6206']).sum()),
        folds=counts, all_original_families_still_evaluated=True,
        CV_fitting_rule_differs_in=sum(row['extra_fit_record_count'] > 0 for row in counts),
        all_family_alternative_recorded_prediction_comparison=comparisons,
        full_matched_filtered_CV_not_computed=True, new_fits=0, new_inference=0,
        interpretation='Evaluation rows did not narrow. Two CV fits reinclude excluded6204. Numerical predictions here were already recorded,not generated by this audit.')
    save(root / 'outputs/TRAINING_SCOPE_COUNTS301.json', result)
    return result


def main(root):
    binding = json.loads((root / 'audit/SOURCE_BINDINGS301.json').read_text())
    for row in binding['files']:
        path = root / 'inputs' / row['path']
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    cases = json.loads((root / 'audit/CASE_INDEX301.json').read_text())
    result = []
    all_matrices = []
    for case in cases:
        workspace = root / 'inputs' / case['workspace']
        validation = json.loads((workspace / 'validation.json').read_text())
        matrices = list(walk_matrices(validation))
        for path in sorted(workspace.rglob('*.json')):
            if path.name == 'validation.json':
                continue
            extra = json.loads(path.read_text())
            matrices.extend(walk_matrices(extra, str(path.relative_to(workspace))))
        for row in matrices:
            row['cohort'] = case['cohort']
        all_matrices += matrices
        commands = []
        for line_number, raw in enumerate((root / 'inputs' / case['events']).read_text().splitlines(), 1):
            event = json.loads(raw)
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                commands.append(dict(event_line=line_number, item_id=item['id'],
                    command=item['command'], output=item.get('aggregated_output', ''),
                    exit_code=item.get('exit_code'), status=item.get('status')))
        save(root / 'outputs/commands' / (case['cohort'] + '.json'), commands)
        scope = [code_scope(path) for path in sorted(workspace.rglob('*.py'))]
        report_lines = (workspace / 'report.md').read_text().splitlines()
        warnings = [dict(line=i, text=line) for i, line in enumerate(report_lines, 1)
                    if any(word in line.lower() for word in ('uncertain', 'guarantee', 'limitation',
                        'pending', 'not accessed', 'excluded', 'small', 'seven-group'))]
        result.append(dict(cohort=case['cohort'], validation_matrix_count=len(matrices),
            completed_commands=len(commands), nonzero_commands=sum(x['exit_code'] != 0 for x in commands),
            code_scope=scope, report_scope_or_warning_lines=warnings,
            validation_top_level_keys=list(validation)))
    reported = [x for x in all_matrices if x.get('reported_macro_f1') is not None]
    unmatched = [x for x in reported if not x['reported_matches_fixed_labels'] and not x['reported_matches_observed_union']]
    output = dict(utc=datetime.now(timezone.utc).isoformat(), evidence_files_verified=len(binding['files']),
        source_archive=binding['source_archive'], source_manifest_sha256=binding['source_manifest_sha256'],
        cases=result, matrix_reductions=all_matrices, matrices_with_reported_F1=len(reported),
        reported_F1_matching_neither_convention=unmatched,
        numeric_checks=len(all_matrices), additional_estimator_fits=0, additional_model_inference=0,
        new_LLM_calls=0, new_independent_cases=0,
        interpretation='Exploratory retained-evidence audit. Different F1 conventions are not automatically arithmetic errors. Static code mismatches do not identify harm causality.')
    output['training_scope_audit'] = training_scope(root, cases)
    save(root / 'outputs/PROGRAM_SCOPE_AUDIT301.json', output)
    print(json.dumps(dict(evidence_files_verified=len(binding['files']), cases=len(cases),
        existing_confusion_matrices=len(all_matrices), matrices_with_reported_F1=len(reported),
        reported_matching_neither=len(unmatched), new_fits=0, new_model_inference=0), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
