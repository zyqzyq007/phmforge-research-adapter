"""Freeze data-preserving observation-entry variants of verified LiU-ICE runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    run, source = args.run.resolve(), args.source.resolve()
    inventory = {x['path']: x for x in json.loads((source / 'SOURCE_MANIFEST.json').read_text())['files']}
    def verified(relative):
        p = source / relative
        assert sha(p) == inventory[relative]['sha256']
        return p
    records = json.loads(verified('controller_only/source_records.json').read_text())
    raw_zip = verified('sources/snapshot02/LiU_trainingdata.zip')
    groups = {
        'A': [x for x in records if x['source_member'].startswith('ftp75city2') and (x['source_member'].endswith('_085.csv') or x['class_code'] == 0)],
        'B': [x for x in records if x['source_member'].startswith('ftp75city2') and (x['source_member'].endswith('_115.csv') or x['class_code'] == 0)],
        'C': [x for x in records if x['partition'] == 'EngineMagnitude_queries'],
    }
    assert [len(groups[k]) for k in ['A', 'B', 'C']] == [4, 4, 8]
    save(run / 'entry_selection84.json', {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'rule': 'All FTP sensor085+nominal; all FTP sensor115+nominal; all existing heldoutWLTP095/105,4mm andsecondnominal. No selection by current prediction/error.',
        'groups': {k: [x['source_member'] for x in v] for k, v in groups.items()},
        'observation_starts_source_seconds': [0.0, 200.0], 'decision_source_seconds': 540.0,
        'scope': 'AI-defined observation-start stress workflows on real source recordings; not actual recorded reboot, newphysicaldevice, originalAgentbenchmark or globallyunseenrecordconfirmation.',
        'source_zip_sha256': sha(raw_zip), 'physical_engines': 1,
    })
    oldwork = source / 'tasks/EngineTransfer_r1'
    dictionary = json.loads(verified('tasks/EngineTransfer_r1/data_dictionary.json').read_text())
    dictionary['source'] += ' A84 preserves exact numeric CSV lines and absolute time; each query view ends at its single current decision.'
    dictionary['query_observation'] = 'A query may start partway through an operating episode. Its first available sample has unknown health; source_time/CSV.time remain absolute source seconds. No query prehistory before its declared observation_start is available. Labeled external healthy and faulty references remain available. Do not infer a query onset from reference onset times.'
    dictionary['query_files']['queries.zip'] = 'Original header plus unmodified source rows between declared observation start and current time; future rows absent. No resampling or timestamp rebasing.'
    dictionary['causal_information_boundary'] = 'For each independent record use its available past through the one current checkpoint plus labeled external references. Do not infer health from the first row, identifiers or reference onset; any same-record adaptation must justify its prerequisites. Do not fit/update using other query records or inferred query labels.'
    dictionary['split'] += ' Records reused from historical discovery; views differ only in available initial past. Separate source-cohort and paired-view denominators.'
    prompt = '''Build and execute a PHM fault diagnosis workflow for the supplied real engine-air-path records. Read data_dictionary.json and queries.json. Implement a defensible CPU-feasible diagnosis method, validate it using the permitted labeled reference records, and predict every current query. The system is dynamically coupled; inspect empirical evidence and references before selecting rules. All fitting, selection, preprocessing, thresholds and adaptation must respect the declared information boundary. Use complete-reference-record validation and explain what supports the delivered final algorithm. A declared train+validation refit after selection is allowed. IDs and filenames are not predictors. Report uncertainty and limitations; unknown or abstention is permitted when justified.
You have1800seconds total including reasoning/code, CPU only,four numerical threads. Installed Python3,numpy,scipy,pandas,scikit-learn,matplotlib. No network, other model API, credentials, model downloads, new packages or large neural training. Preserve all source input bytes, fix numerical seeds, implement early and save completed outputs.
Deliver analysis.py: python3 analysis.py from this workspace must reconstruct predictions.json and validation.json using only provided inputs without credentials/network; save required helper source and report.md. predictions.json is {"results":[...]} ordered exactly as queries.json. Each entry has case_id,decision_kind="diagnosis",class_code integer0..5 (null only for documented abstention),and evidence from actual computations. validation.json records reference populations, metrics, preprocessing/model choices, delivered algorithm and limitations. Correct diagnostic labels do not certify physical causality, unique model parameters or maintenance safety.
Final CLI answer must be the small metadata JSON matching the supplied schema, with one workflow entry: case_id="WORKFLOW_ID",analysis_code="analysis.py",predictions_file="predictions.json",validation_file="validation.json",report_file="report.md". Paths are actual relative files and workflowID differs from query IDs.
'''
    schema = json.loads(verified('tasks/EngineTransfer_r1_schema.json').read_text())
    (run / 'tasks').mkdir(exist_ok=True)
    (run / 'hidden').mkdir(exist_ok=True)
    jobs, inputs, gold, checks = [], [], [], []
    with zipfile.ZipFile(raw_zip) as archive:
        for group, selected in groups.items():
            for start in [0.0, 200.0]:
                name = f'entry_{group}_{int(start)}'
                work = run / 'tasks' / name
                work.mkdir()
                fixed_names = ['train.zip', 'validation.zip', 'reference_labels.json', 'structural_model_reference.py', 'published_method_reference.txt', 'organizer_example_reference.txt']
                for n in fixed_names:
                    shutil.copy2(verified('tasks/EngineTransfer_r1/' + n), work / n)
                queries = []
                with zipfile.ZipFile(work / 'queries.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as target:
                    for record in sorted(selected, key=lambda r: r['source_member']):
                        member = record['source_member']
                        original = archive.read(member)
                        frame = pd.read_csv(io.BytesIO(original), float_precision='round_trip')
                        lo = int(np.searchsorted(frame.time.to_numpy(), start, side='left'))
                        hi = int(np.searchsorted(frame.time.to_numpy(), 540.0, side='right') - 1)
                        assert hi >= lo and frame.time.iloc[hi] > 539.0
                        lines = original.splitlines(keepends=True)
                        view = lines[0] + b''.join(lines[lo + 1:hi + 2])
                        sid = 'series_' + hashlib.sha256(('entry84:' + member).encode()).hexdigest()[:16]
                        cid = 'case_' + hashlib.sha256(('entry84:' + member + ':' + str(start)).encode()).hexdigest()[:16]
                        filename = sid + '.csv'
                        target.writestr(filename, view)
                        current = float(frame.time.iloc[hi])
                        q = {'case_id': cid, 'record_id': sid, 'file': filename, 'sample_index': hi - lo,
                             'time_seconds': current, 'observation_start_seconds': float(frame.time.iloc[lo]),
                             'driving_regime': record['driving_regime']}
                        queries.append(q)
                        decoded = pd.read_csv(io.BytesIO(view), float_precision='round_trip')
                        assert np.array_equal(decoded.to_numpy(), frame.iloc[lo:hi + 1].to_numpy())
                        checks.append({'job_id': name, 'record_id': sid, 'rows': len(decoded), 'source_rows_start_end': [lo, hi],
                                       'original_member_sha256': hashlib.sha256(original).hexdigest(), 'view_sha256': hashlib.sha256(view).hexdigest(),
                                       'values_and_absolute_timestamps_exact': True, 'future_rows_absent': True})
                        gold.append({'job_id': name, 'case_id': cid, 'record_id': sid, 'source_member': member,
                                     'class_code': record['class_code'], 'source_time': current,
                                     'observed_start': float(frame.time.iloc[lo]), 'label_basis': 'Original nominal/fault source labels, persistent beyond declared source injection; not inferred from signal.'})
                save(work / 'queries.json', queries)
                save(work / 'data_dictionary.json', dictionary)
                save(work / 'readonly_inputs.json', fixed_names + ['queries.zip', 'queries.json', 'data_dictionary.json'])
                (work / 'AGENTS.md').write_text('All temporary files and caches must stay in workspace/.runtime, never /tmp. Source inputs are read-only. Use installed scientific Python. No network or other model APIs.\n')
                (run / 'tasks' / (name + '_prompt.txt')).write_text(prompt.replace('WORKFLOW_ID', name))
                save(run / 'tasks' / (name + '_schema.json'), schema)
                jobs.append({'id': name, 'condition': f'observation_entry_{int(start)}', 'repeat': 1, 'workspace': 'tasks/' + name,
                             'prompt_file': 'tasks/' + name + '_prompt.txt', 'schema_file': 'tasks/' + name + '_schema.json', 'case_ids': [name]})
                for p in sorted(work.iterdir()):
                    if p.is_file():
                        inputs.append({'job_id': name, 'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    save(run / 'entry_plan84.json', {'jobs': jobs})
    save(run / 'entry_input_manifest84.json', inputs)
    save(run / 'hidden/entry_gold84.json', gold)
    save(run / 'entry_value_parity84.json', checks)
    save(run / 'entry_preparation_receipt84.json', {'flows': len(jobs), 'source_cohorts': 3, 'paired_views': 2,
                                                  'planned_predictions': len(gold), 'distinct_source_query_records': len({x['record_id'] for x in gold}),
                                                  'input_files': len(inputs), 'all_view_checks': len(checks), 'qualifier': 'No model calls yet; originalsourceSHA verified,exactclock/valueviews andreferencesavailable. Same-info reference qualification still pending.'})
    print(json.dumps({'prospective_flows': len(jobs), 'prediction_views': len(gold), 'source_query_records': len({x['record_id'] for x in gold}), 'input_files': len(inputs)}))


if __name__ == '__main__':
    main()
