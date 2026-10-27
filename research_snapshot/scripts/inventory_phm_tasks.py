"""Inventory cached original PHM tasks without running or rewriting them."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def smartgrid_inventory(archive):
    seen = {}
    records = []
    prior = {x['scenario_id'] for x in json.loads((archive / 'inventory80.json').read_text())}
    for folder in ('smartgrid_snapshots80', 'smartgrid_extra80'):
        base = archive / folder
        manifest = base / 'manifest.json'
        if not manifest.exists():
            continue
        for item in json.loads(manifest.read_text()):
            source_path = item.get('original_path', '')
            if (not source_path.startswith('data/scenarios/') or
                    not source_path.endswith('.json') or item.get('status') != 'saved'):
                continue
            path = base / item['file']
            raw = path.read_bytes()
            sha = digest(raw)
            if sha != item['sha256']:
                raise ValueError('Cached source SHA256 mismatch: ' + str(path))
            git_sha = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
            if item.get('git_blob_sha1') and git_sha != item['git_blob_sha1']:
                raise ValueError('Cached Git blob mismatch: ' + str(path))
            task = json.loads(raw)
            task_id = task['id']
            if task_id in seen:
                if seen[task_id] != sha:
                    raise ValueError('Different cached content for task: ' + task_id)
                continue
            seen[task_id] = sha
            prompt = task['text']
            records.append({
                'id': task_id, 'type': task.get('type'), 'category': task.get('category'),
                'difficulty': task.get('difficulty'), 'source_path': source_path,
                'source_archive_file': str(path), 'source_url': item['url'],
                'source_sha256': sha, 'prompt_sha256': digest(prompt.encode()),
                'prompt': prompt, 'expected_tools': task.get('expected_tools', []),
                'ground_truth_fields': sorted(task.get('ground_truth', {})),
                'prior_A80_primary_task': task_id in prior,
                'postsubmission_adversarial': task_id.startswith('SGT-') and
                    int(task_id.split('-')[1]) >= 51,
                'static_source_verified': True, 'new_task_runtime_smoke': False,
                'independent_physical_truth': False,
                'data_world': 'one synthetic transformer world; assigned source labels',
                'performance_admission': 'pending exact task/tool/scoring smoke',
            })
    return sorted(records, key=lambda x: x['id'])


def assetops_inventory(archive):
    records = []
    for rel, pool in (
        ('sources/assetops_hf/data/scenarios/all_utterance.jsonl', 'general_legacy'),
        ('sources/assetops_hf/data/task/phm_utterance.jsonl', 'PHM_legacy'),
    ):
        path = archive / rel
        raw = path.read_bytes()
        for line in raw.splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            prompt = task['text']
            records.append({
                'id': task['id'], 'pool': pool, 'category': task.get('category'),
                'type': task.get('type'), 'deterministic_declared': task.get('deterministic'),
                'source_archive_file': str(path), 'source_file_sha256': digest(raw),
                'prompt_sha256': digest(prompt.encode()), 'prompt': prompt,
                'record_fields': sorted(task),
                'runtime_binding': 'unqualified; current Mini ID mismatch retained from A78',
                'performance_admission': 'pending matched public source branch and runtime',
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smartgrid-archive', type=Path, required=True)
    parser.add_argument('--assetops-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path('/tmp')):
        parser.error('Task work under /tmp is forbidden')
    args.output.mkdir(parents=True, exist_ok=False)
    sg = smartgrid_inventory(args.smartgrid_archive)
    aob = assetops_inventory(args.assetops_archive)
    for name, value in [('smartgrid_tasks.json', sg), ('assetops_tasks.json', aob)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'new_LLM_calls': 0,
        'smartgrid_original_tasks': len(sg),
        'smartgrid_source_sha256_verified': len(sg),
        'smartgrid_not_run_in_A80': sum(not x['prior_A80_primary_task'] for x in sg),
        'smartgrid_categories': dict(Counter(x['type'] for x in sg)),
        'assetops_legacy_tasks': len(aob),
        'assetops_legacy_pools': dict(Counter(x['pool'] for x in aob)),
        'new_performance_qualified_tasks': 0, 'new_Agent_failures': 0, 'E3': 0,
        'scope': 'Original cached task inventory, not runtime or performance validation',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
