"""Inventory complete, version-specific PHMForge and AssetOpsBench contracts.

Source-only census: this does not invoke an Agent or infer task correctness.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    forge = run / 'source/forge/ReActXen/src/reactxen/demo/intent_implementation_demo'
    asset = run / 'source/asset'
    source = forge / 'scenarios/phm_scenarios.json'
    scenarios = json.loads(source.read_text())['pdm_scenarios']
    forge_rows = []
    for task in scenarios:
        dataset = str(task.get('dataset', ''))
        has_inline = 'ENGINE SIGNAL DATA SUMMARY' in task['input_question']
        forge_rows.append(dict(
            task_id=task['task_id'], category=task['classification_type'],
            dataset=dataset, question=task['input_question'],
            question_sha256=hashlib.sha256(task['input_question'].encode()).hexdigest(),
            required_tools=task.get('required_tools', []),
            inline_sensor_summary=has_inline,
            numerical_prediction_caveat=(
                'Published predictor reads RUL truth and adds fixed noise; fault predictor '
                'generates simulated labels. These are orchestration tools, not trained '
                'physical PHM predictors.'
                if task['classification_type'] in ['RUL Prediction', 'Fault Classification'] else None),
            status='pending_tool_and_data_preflight',
            source_path=str(source), source_sha256=sha(source)))
    profile_path = asset / 'benchmarks/scenario_suite/all.yaml'
    profile = yaml.safe_load(profile_path.read_text())
    scenario_root = asset / 'src/couchdb/scenarios_data'
    asset_rows = []
    for category, ids in profile.items():
        for task_id in ids:
            folder = scenario_root / ('scenario_' + str(task_id))
            needed = {k: folder / v for k, v in dict(
                question='question.txt', manifest='manifest.json', gold='groundtruth.txt').items()}
            missing = [k for k, p in needed.items() if not p.is_file()]
            row = dict(task_id=str(task_id), category=category,
                       missing=missing, paths={k: str(p) for k, p in needed.items()},
                       hashes={k: sha(p) for k, p in needed.items()},
                       status='unavailable_missing_published_scenario_files' if missing
                       else 'pending_original_tool_and_data_preflight')
            if not missing:
                row['question'] = needed['question'].read_text()
            asset_rows.append(row)
    # These official HF tasks are a different contract from the current main's
    # scenario_<id> suite. Identical numeric IDs must never be merged.
    legacy_path = run / 'source/asset_all_utterance.jsonl'
    legacy = [json.loads(line) for line in legacy_path.read_text().splitlines() if line.strip()]
    legacy_rows = [dict(task_id=str(t['id']), category=t.get('type'), question=t['text'],
                        status='separate_official_HF_contract_not_interchangeable_with_main_suite')
                   for t in legacy]
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        classification='source_census_not_Agent_experiment',
        model_calls=0,
        phmforge=dict(repository='DeveloperMindset123/PHMForge-A-Scenario-Driven-Agentic-Benchmark-for-Industrial-Asset-Lifecycle-Maintenance',
                      commit='28430199f7b8ccba924d49e06c2d59b761846038',
                      tasks=len(forge_rows), category_counts=dict(Counter(r['category'] for r in forge_rows)),
                      rows=forge_rows,
                      paper_v3_gap='Paper v3 describes99/39 and a battery server; published main has75/22. '
                                   'The24 battery tasks and third server were not found in main or claude/modest-gould. '
                                   'Do not invent IDs or claim this is a99-task reproduction.'),
        assetops_main=dict(repository='IBM/AssetOpsBench',commit='e844f6468c6c68b328022e7ca8e31da140842cc7',
                           profile=str(profile_path), profile_sha256=sha(profile_path),
                           tasks=len(asset_rows),
                           category_counts={k: len(v) for k, v in profile.items()},
                           scenarios_with_question_manifest_gold=sum(not r['missing'] for r in asset_rows),
                           unavailable=sum(bool(r['missing']) for r in asset_rows), rows=asset_rows),
        assetops_official_HF=dict(tasks=len(legacy_rows), sha256=sha(legacy_path), rows=legacy_rows,
                                  caveat='152question metadata cannot supply232current scenario-specific databases and gold. '
                                         'Do not relabel HF IoT task1 as current WOSR scenario1.'),
        scope='No missing task is a PHM Agent failure. Source defects and evaluation limitations remain separate.'
    )
    out = run / 'TASK_INVENTORY187.json'
    if out.exists():
        raise FileExistsError('Refusing to overwrite frozen inventory: ' + str(out))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(forge_tasks=len(forge_rows), asset_main_tasks=len(asset_rows),
                          asset_main_scenario_files_present=sum(not r['missing'] for r in asset_rows),
                          asset_main_unavailable=sum(bool(r['missing']) for r in asset_rows),
                          asset_HF_separate_tasks=len(legacy_rows), output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
