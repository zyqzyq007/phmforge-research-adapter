"""Prepare fresh, label-isolated C-MAPSS endpoint workflows from NASA ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source-zip', required=True, type=Path)
    parser.add_argument('--template', required=True, type=Path)
    args = parser.parse_args()
    run, template = args.run.resolve(), args.template.resolve()
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    jobs, inventory, gold, counts = [], [], [], []
    with zipfile.ZipFile(args.source_zip) as source:
        assert source.testzip() is None
        for subset, minimum in [('FD002', 1), ('FD003', 21), ('FD004', 1)]:
            job_id = 'cmapss_' + subset
            case_id = subset + '_unseen_endpoints'
            work = run / 'tasks' / job_id
            work.mkdir(parents=True, exist_ok=False)
            train_bytes = source.read('train_' + subset + '.txt')
            test_bytes = source.read('test_' + subset + '.txt')
            label_bytes = source.read('RUL_' + subset + '.txt')
            train = pd.read_csv(io.BytesIO(train_bytes), sep=r'\s+', header=None, names=cols)
            test = pd.read_csv(io.BytesIO(test_bytes), sep=r'\s+', header=None, names=cols)
            labels = [float(x) for x in label_bytes.split()]
            assert test.unit.nunique() == len(labels)
            test = test[test.unit >= minimum].copy()
            queries = [{'case_id': f'{subset}_test_{int(u):03d}', 'unit': int(u),
                        'endpoint_cycle': int(g.cycle.max())} for u, g in test.groupby('unit')]
            for q in queries:
                gold.append(dict(job_id=job_id, subset=subset, **q, true_rul=labels[q['unit'] - 1]))
            n = len(queries)
            (work / 'train.txt').write_bytes(train_bytes)
            test.to_csv(work / 'test.csv', index=False)
            (work / 'original_readme.txt').write_bytes(source.read('readme.txt'))
            dictionary = json.loads((template / 'data_dictionary.json').read_text())
            dictionary.update(source=f'Official NASA C-MAPSS original ZIP; {subset}',
                training=f'{train.unit.nunique()} complete engine trajectories ending at simulator failure',
                test=f'{n} original test prefixes, original unit IDs >= {minimum}; no test future or RUL labels',
                query_manifest=queries,
                operating_conditions=1 if subset == 'FD003' else 6,
                fault_modes=1 if subset == 'FD002' else 2,
                task_origin='AI-authored analysis workflow on the original NASA prediction task, not a native Agent benchmark',
                prior_exposure='These test endpoints have not previously been submitted as Agent RUL prediction tasks; historical label-availability audits and related subset workflows exist.')
            save(work / 'data_dictionary.json', dictionary)
            schema = json.loads((template / 'prediction_schema.json').read_text())
            schema['properties']['results'].update(minItems=n, maxItems=n)
            schema['properties']['results']['items']['properties']['case_id']['enum'] = [q['case_id'] for q in queries]
            save(work / 'prediction_schema.json', schema)
            save(work / 'task_budget.json', {
                'budget_seconds': 1200,
                'instruction': 'Total wall-clock budget from start of this conversation, including reasoning and tools. Use computationally reasonable methods and save completed artifacts before the deadline.',
                'cpu_threads': 4, 'model_artifact_limit_MiB': 64,
                'training_scope': 'CPU task predictors may be fitted on the supplied training trajectories; no LLM or RL weight training.'})
            readonly = ['train.txt', 'test.csv', 'original_readme.txt', 'data_dictionary.json', 'prediction_schema.json', 'task_budget.json']
            save(work / 'readonly_inputs.json', readonly)
            (work / 'AGENTS.md').write_text('Keep all temporary files and caches in this workspace/.runtime; never use /tmp. Inputs are read-only. No network or other model APIs. Use installed scientific Python libraries.\n')
            prompt = f'''Read task_budget.json and plan to complete the analysis within the total 1200-second wall-clock budget, including reasoning and code execution.

Analyze the official NASA C-MAPSS {subset} data in this workspace. Read data_dictionary.json and original_readme.txt. Estimate uncapped remaining operating cycles at the endpoint of every test.csv engine ({n} units). The complete legal training trajectories are train.txt. Test future observations and exact RUL labels are unavailable.

Use Python3 and the installed scientific libraries to build/select an appropriate predictor and validate it using only the allowed training data. Report point predictions and 90% uncertainty intervals when defensible, with their interpretation, assumptions, and limits. If a supported finite interval or point estimate is unavailable, use null and explain the limitation; do not fabricate guarantees. No internet or dependency installation is available. Use computationally reasonable methods within this run; keep saved model artifacts below64MiB.

Write predictions.json matching prediction_schema.json (all {n} unique case IDs), a runnable analysis source file, validation.json with method selection/validation/calibration information, and report.md describing the conclusions and limitations. Finish executing your analysis and save the deliverables. The final reply is ONLY the small metadata JSON matching the supplied final schema: results contains one entry with case_id exactly {case_id} and artifact_file predictions.json; analysis_code, validation_file, report_file are actual relative paths. Do not repeat scientific rows in the final reply. No gold feedback is supplied.
'''
            prompt_path = run / 'tasks' / (job_id + '_prompt.txt')
            prompt_path.write_text(prompt)
            final_schema = json.loads(template.with_name(template.name + '_schema.json').read_text())
            final_schema['properties']['results']['items']['properties']['case_id']['enum'] = [case_id]
            save(run / 'tasks' / (job_id + '_schema.json'), final_schema)
            jobs.append({'id': job_id, 'condition': 'fresh_original_prediction_endpoints', 'repeat': 1,
                         'workspace': 'tasks/' + job_id, 'prompt_file': 'tasks/' + job_id + '_prompt.txt',
                         'schema_file': 'tasks/' + job_id + '_schema.json', 'case_ids': [case_id]})
            counts.append({'job_id': job_id, 'subset': subset, 'fit_engines': int(train.unit.nunique()), 'test_engines': n})
            for f in sorted(work.iterdir()):
                inventory.append({'job_id': job_id, 'path': str(f.relative_to(work)), 'bytes': f.stat().st_size,
                                  'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
    save(run / 'plan_cmapss91.json', {'jobs': jobs})
    save(run / 'hidden/cmapss_gold91.json', gold)
    save(run / 'cmapss_inputs91.json', {'source_zip': str(args.source_zip),
        'source_sha256': hashlib.sha256(args.source_zip.read_bytes()).hexdigest(), 'crc_checked': True,
        'counts': counts, 'files': inventory, 'agent_workflow_count': 3, 'upstream_source_families': 1})
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
