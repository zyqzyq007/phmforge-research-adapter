"""Score a complete frozen pass with unchanged author evaluators.

Original benchmark scores are not physical PHM correctness certificates.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    plan = run / 'PLAN187_LAUNCH.json'
    if not plan.exists():
        plan = run / 'PLAN187.json'
    jobs = json.loads(plan.read_text())['jobs']
    if (run / 'RESULT187.json').exists():
        raise FileExistsError('Refusing to score completed pass a second time')
    controller = json.loads((run / 'CONTROLLER_STATUS187.json').read_text())
    if controller.get('status') != 'all_selected_subjects_terminal_no_scoring':
        raise RuntimeError('Full controller is not terminal')
    statuses = {}
    freeze = []
    for job in jobs:
        folder = run / 'primary' / job['id'] / job['id']
        status = json.loads((folder / 'status.json').read_text())
        if status['status'] == 'running':
            raise RuntimeError('A subject is still running')
        statuses[job['id']] = status
        for path in [folder / 'status.json', folder / 'response.txt',
                     run / 'traces' / job['id'] / 'calls.jsonl']:
            if path.is_file():
                freeze.append(dict(path=str(path), sha256=sha(path)))
    # The source's evaluator is extracted verbatim to avoid importing its
    # WatsonX framework and credential-loading code. No condition is changed.
    source = run / 'source/forge/ReActXen/src/reactxen/demo/intent_implementation_demo'
    evaluator = source / 'benchmark_pass1.py'
    tree = ast.parse(evaluator.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate_pass')
    namespace = dict(re=re)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(evaluator), 'exec'), namespace)
    scenarios = {x['task_id']: x for x in json.loads((source / 'scenarios/phm_scenarios.json').read_text())['pdm_scenarios']}
    freeze += [dict(path=str(evaluator), sha256=sha(evaluator))]
    (run / 'PRE_SCORE_ARTIFACT_FREEZE187.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), files=freeze,
        subjects=len(jobs), no_scores_returned_to_subjects=True), indent=2) + '\n')
    rows, asset_pairs = [], []
    for job in jobs:
        status = statuses[job['id']]
        path = run / 'primary' / job['id'] / job['id'] / 'response.txt'
        answer = path.read_text() if path.is_file() else ''
        row = dict(job=job['id'], family=job['family'], task_id=job['task_id'],
                   category=job['category'], status=status['status'],
                   response_present=bool(answer.strip()), usage=status.get('usage'),
                   elapsed_seconds=status.get('elapsed_seconds'), answer_sha256=sha(path) if path.exists() else None)
        if job['family'] == 'forge':
            passed, reason = namespace['evaluate_pass'](scenarios[job['task_id']], answer)
            row.update(original_scorer='benchmark_pass1.evaluate_pass',
                       original_passed=passed, original_reason=reason,
                       physical_PHM_correctness='not established by this keyword/heuristic evaluator')
        else:
            root = run / 'source/asset/src/couchdb/scenarios_data' / ('scenario_' + job['task_id'])
            asset_pairs.append(dict(job=job['id'], task_id=job['task_id'], answer=answer,
                                    question=(root / 'question.txt').read_text(),
                                    gold=(root / 'groundtruth.txt').read_text()))
        rows.append(row)
    # The author package requires Python>=3.12; run its unchanged deterministic
    # scorer in the pinned Python3.13 runtime, without network or LLM credentials.
    worker = '''import json,sys
from evaluation.models import Scenario
from evaluation.scorers.static_json import StaticJsonScorer
rows=json.load(sys.stdin); scorer=StaticJsonScorer()
out=[]
for row in rows:
 scenario=Scenario(id=row['task_id'],text=row['question'],expected_answer=row['gold'])
 result=scorer(scenario,row['answer'],'')
 out.append({'job':row['job'],'score':result.model_dump(mode='json')})
print(json.dumps(out))
'''
    command = ['docker', 'run', '--rm', '-i', '--network', 'none',
               '-v', str(run / 'runtime/asset_deps') + ':/deps:ro',
               '-v', str(run / 'source/asset/src/evaluation') + ':/eval_src/evaluation:ro',
               '-e', 'PYTHONPATH=/deps:/eval_src', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-v', str(run / 'runtime/tmp') + ':/runtime_tmp',
               '-e', 'TMPDIR=/runtime_tmp', '-e', 'TMP=/runtime_tmp', '-e', 'TEMP=/runtime_tmp',
               'python:3.13-slim', 'python', '-c', worker]
    result = subprocess.run(command, input=json.dumps(asset_pairs), capture_output=True,
                            text=True, timeout=120)
    (run / 'ASSET_SCORE.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Original AssetOps scorer failed: ' + result.stderr[-1500:])
    asset_scores = {x['job']: x['score'] for x in json.loads(result.stdout)}
    for row in rows:
        if row['family'] == 'asset':
            score = asset_scores[row['job']]
            row.update(original_scorer='static_json', original_passed=score['passed'], original_score=score)
    output = dict(utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                  subjects=len(rows), terminal_counts=dict(Counter(x['status'] for x in rows)),
                  phmforge_public=dict(tasks=75, original_passed=sum(x.get('original_passed', False) for x in rows if x['family']=='forge')),
                  asset_main=dict(profile_total=232, executed_public=3, missing_source=229,
                                  original_passed=sum(x.get('original_passed', False) for x in rows if x['family']=='asset')),
                  full_232_completed=False, paper_v3_99_reproduced=False,
                  common_mechanism_confirmed=False, scoped_problem_confirmed=False,
                  interpretation='Original evaluator outcomes only; trace audit and source/data validity are required. No mechanism is automatically confirmed.')
    (run / 'RESULT187.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
