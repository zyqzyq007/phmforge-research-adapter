"""Read-only synthesis of registered PHM program comparisons (A324).

No model execution, training, threshold search, or new final-set scoring.
The selected historical cases describe existence and mechanisms, not prevalence.
"""
import argparse
import collections
import csv
import hashlib
import json
import tarfile
from pathlib import Path


PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
ARCHIVE = Path('/data/yyf-zyq/PHM_20261010')
SOURCES = {
    'motor': ('results/selection270/NUMERICAL_RESULTS270.json', '20261021T174126Z_choice270'),
    'hust': ('results/validation274/NUMERICAL_RESULTS274.json', '20261022T023131Z_validation274'),
    'hust_judgment': ('results/validation274/ADJUDICATION274.json', '20261022T023131Z_validation274'),
    'rotor276': ('results/pipeline276/NUMERICAL_RESULTS276.json', '20261022T052053Z_pipeline276'),
    'rotor305': ('results/rotor305/NUMERICAL_RESULTS305.json', '20261023T052522Z_rotor305'),
    'cwru': ('results/expansion307/NUMERICAL_RESULTS307.json', '20261023T131916Z_expansion307'),
    'cwru_review': ('results/review311/RESULTS311.json', '20261023T145500Z_review311'),
    'mechanism': ('results/mechanism323/MECHANISM323.json', '20261024T035900Z_mechanism323'),
    'gap': ('results/gap321/NUMERICAL_SUMMARY321.json', '20261023T231827Z_gap321_recovered'),
    'review281': ('results/review281/NUMERICAL_RESULTS281.json', '20261022T130911Z_review281'),
    'review284': ('results/evidence284/NUMERICAL_RESULTS281.json', '20261022T161303Z_evidence284'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def one(rows, **fields):
    selected = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    assert len(selected) == 1, (fields, len(selected))
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not (output == Path('/tmp') or Path('/tmp') in output.parents)
    output.mkdir(parents=True, exist_ok=True)
    data, bindings = {}, []
    for key, (relative, run) in SOURCES.items():
        project_file = PROJECT / relative
        digest = sha(project_file)
        archive = ARCHIVE / run
        manifest_name = 'SOURCE_MANIFEST_RECOVERY3.json' if key == 'gap' else 'SOURCE_MANIFEST.json'
        manifest = archive / manifest_name
        entries = json.loads(manifest.read_text())['files']
        matches = [row for row in entries if row['sha256'] == digest]
        assert matches, ('source_not_in_archive_manifest', key, relative)
        archive_file = archive / matches[0]['path']
        member = None
        if archive_file.exists():
            content = archive_file.read_bytes()
        else:
            member = matches[0]['path']
            archive_file = archive / 'RUN_CONTENTS.tar.gz'
            content = None
            with tarfile.open(archive_file, 'r|gz') as stream:
                for entry in stream:
                    if entry.name.lstrip('./') == member:
                        content = stream.extractfile(entry).read()
                        break
            assert content is not None, ('missing_archive_member', member)
        assert hashlib.sha256(content).hexdigest() == digest
        data[key] = json.loads(content)
        bindings.append({'id': key, 'project': str(project_file), 'archive': str(archive_file),
                         'member': member, 'sha256': digest, 'bytes': len(content),
                         'manifest': str(manifest), 'manifest_sha256': sha(manifest)})

    cases = []

    def add(case_id, source, baseline, selected, n, units, metric, comparator, review, pointer):
        loss = (baseline - selected) * 100 if metric == 'accuracy' else (selected / baseline - 1) * 100
        cases.append(dict(case_id=case_id, source_family=source, comparator_kind=comparator,
                          baseline=baseline, selected=selected, metric=metric,
                          deterioration=loss,
                          deterioration_unit='percentage_points' if metric == 'accuracy' else 'relative_percent',
                          final_windows_or_queries=n, final_reported_groups_or_bearings=units,
                          review_status=review, source_pointer=pointer,
                          representative_selected_case=True, population_prevalence_inference=False))

    motor = one(data['motor']['cases'], case='motor_a')
    svc = one(motor['controls'], name='svc_c1')
    add('A270-A', 'ElectricMotorVibrations', svc['metrics']['accuracy'],
        motor['stages']['review']['final']['metrics']['accuracy'], 338, 6, 'accuracy',
        'provided_and_inspected_candidate', 'ordinary_review_preserved_loss', 'motor:/cases/0')
    hust_d = one(data['hust']['cases'], case='hust_geometry_r2')
    old_d = one(hust_d['registered_agent_candidates'], candidate='prior_family57')
    add('A274-d', 'HUST', old_d['metrics']['accuracy'],
        hust_d['stages']['review']['final']['metrics']['accuracy'], 3865, 7, 'accuracy',
        'actual_prior_recipe_reconstructed_before_final_scoring', 'ordinary_review_preserved_loss', 'hust:/cases/3')
    for case, candidate, label in [('ordinary_r1', 'subset_2', 'A276-A'),
                                    ('ordinary_r2', 'component_et_base_sqrt', 'A276-C')]:
        row = one(data['rotor276']['cases'], case=case)
        old = one(row['registered_agent_candidates'], candidate=candidate)
        add(label, 'RotorBrokenBar', old['metrics']['accuracy'],
            row['stages']['initial']['final']['metrics']['accuracy'], 737, 10, 'accuracy',
            'actually_fitted_available_component', 'fixed_candidate_review_preserved_loss_A284',
            'rotor276:case=' + case)
    a0 = one(data['rotor305']['rows'], cohort='a', stage='initial')
    a1 = one(data['rotor305']['rows'], cohort='a_review', stage='review')
    add('A305-a', 'RotorBrokenBar', a0['final_accuracy'], a1['final_accuracy'], 1468, 20,
        'accuracy', 'previous_formal_submission', 'ordinary_review_introduced_loss', 'rotor305:/rows/0,4')
    cwru = one(data['cwru']['rows'], task_id='pdm_cost_benefit_002')
    add('A307/A311', 'CWRU', cwru['reference']['metrics']['accuracy'], cwru['final']['metrics']['accuracy'],
        177, 3, 'accuracy', 'starting_fitted_reference', 'outcome_selected_ordinary_review_preserved_loss',
        'cwru:/rows/0;cwru_review:/score')
    xjtu = data['mechanism']['xjtu278_b']['variants']
    add('A278-B/A281', 'XJTU', xjtu['bank_knn5']['final']['mean_bearing_mae_minutes'],
        xjtu['agent_b']['final']['mean_bearing_mae_minutes'], 15, 3, 'mae_minutes',
        'provided_and_inspected_candidate', 'fixed_candidate_review_preserved_loss',
        'mechanism:/xjtu278_b/variants')
    ldm = data['mechanism']['ldm321']['final_variants']
    add('A321-l1', 'LDM2024', ldm['fixed_hgb100']['mean_bearing_mae'], ldm['agent_l1']['mean_bearing_mae'],
        15, 5, 'mae_remaining_acquisitions', 'starting_fitted_reference', 'no_postsubmission_review',
        'mechanism:/ldm321/final_variants')
    component = data['hust_judgment']['cases']['c']
    add('A274-c', 'HUST', component['without_compound_accuracy'], component['full_accuracy'],
        3865, 7, 'accuracy', 'researcher_preregistered_same_weight_component_ablation',
        'ordinary_review_preserved_harmful_component', 'hust_judgment:/cases/c')

    # Entire paired cohorts from these three batches, not only adverse revisions.
    pairs = []
    for key, batch in [('motor', 'A270'), ('hust', 'A274')]:
        for row in data[key]['cases']:
            before = row['stages']['initial']['final']['metrics']['accuracy']
            after = row['stages']['review']['final']['metrics']['accuracy']
            pairs.append({'batch': batch, 'trajectory': row['case'], 'before': before, 'after': after})
    for c in 'abcd':
        before = one(data['rotor305']['rows'], cohort=c, stage='initial')['final_accuracy']
        after = one(data['rotor305']['rows'], cohort=c + '_review', stage='review')['final_accuracy']
        pairs.append({'batch': 'A305', 'trajectory': c, 'before': before, 'after': after})
    for row in pairs:
        row['delta_percentage_points'] = 100 * (row['after'] - row['before'])
        row['direction'] = 'unchanged' if abs(row['after'] - row['before']) < 1e-12 else ('improved' if row['after'] > row['before'] else 'worse')
    counts = dict(collections.Counter(row['direction'] for row in pairs))
    assert counts == {'unchanged': 6, 'improved': 3, 'worse': 1}, counts
    assert len(set(row['source_family'] for row in cases)) == 6
    assert round(next(row for row in cases if row['case_id'] == 'A278-B/A281')['deterioration'], 2) == 36.54
    assert round(next(row for row in cases if row['case_id'] == 'A321-l1')['deterioration'], 2) == 31.45
    result = {'scope': 'retrospective selected evidence, not a prevalence survey',
              'new_agent_sessions': 0, 'new_fits': 0, 'new_model_forward_calls': 0,
              'new_test_scores': 0, 'new_independent_cases': 0,
              'source_families_in_selected_examples': 6, 'cases': cases,
              'complete_pairs_in_three_selected_batches': pairs, 'pair_directions': counts,
              'pairs_not_all_historical_reviews': True,
              'note': 'Related windows, repeated allocations and reviews are not independent devices. HUST ablation is not an actual rejected submission.'}
    for name, obj in [('EVIDENCE324.json', result), ('SOURCE_BINDINGS324.json', bindings)]:
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    with (output / 'cases324.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (output / 'review_pairs324.csv').write_text('batch,trajectory,before,after,direction\n' + ''.join(
        f"{r['batch']},{r['trajectory']},{r['before']},{r['after']},{r['direction']}\n" for r in pairs))
    print(json.dumps({'cases': len(cases), 'source_families': 6, 'paired_trajectories': len(pairs),
                      'directions': counts, 'archive_bound_numeric_sources': len(bindings),
                      'new_models_or_scores': 0, 'output': str(output)}))


if __name__ == '__main__':
    main()
