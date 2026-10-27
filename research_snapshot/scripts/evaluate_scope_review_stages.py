"""Exploratory, all-repeat proposal/final comparison; no scientific editing."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os
import signal
import math


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(root):
    protocol = read(root / 'audit/stage_comparison/PROTOCOL.json')
    work = root / 'audit/stage_comparison'
    assert sha(Path(__file__)) == protocol['script_sha256']
    assert not (work / 'STARTED.json').exists(), 'No implicit replay'
    sources = [Path(p) for p in protocol['runs']]
    for source in sources:
        assert (source / 'TERMINAL392.json').exists() and (source / 'RESULT392.json').exists()
        assert sha(source / 'PREDICTIONS392.json') == read(source / 'PREDICTIONS_SEAL392.json')['sha256']
        for p in source.glob('episodes/*/*/model_gateway/*/status.json'):
            state = read(p); assert not state['active'] and not state['inflight']
    save(work / 'STARTED.json', dict(unix=time.time(), source_runs_all_terminal=True))
    query = {r['index']: r for r in read(root / 'EVALUATION_INPUTS392.json')}
    cache = {}
    for source in sources:
        for name in ['healthy_reference.npz', 'reference_geometry.json']:
            assert sha(source / 'public/data' / name) == sha(root / 'public/data' / name)
        for p in read(source / 'PREDICTIONS392.json'):
            if p['prediction'] is not None:
                pred = p['prediction']
                cache[pred['source_sha256'], pred['query_sha256']] = dict(prediction=pred, source=str(source / 'PREDICTIONS392.json'))
    plans, outputs, jobs = [], [], []
    for source in sources:
        for name in ['edit_review_1', 'edit_review_2']:
            code = source / 'episodes' / name / 'propose/app/solution.py'
            final = source / 'sealed' / name / 'solution.py'
            plan = dict(run=source.name, episode=name, proposal_sha256=sha(code), final_sha256=sha(final),
                        proposal_path=str(code), final_path=str(final), final_valid=next(r for r in read(source / 'ALL_SUBMISSIONS_SEALED392.json')['results'] if r['id'] == name)['valid'])
            plans.append(plan)
            for case in range(7):
                row = dict(run=source.name, episode=name, case=case)
                key = sha(code), query[case]['sha256']
                if key in cache:
                    outputs.append(dict(row, reused=True, **cache[key]))
                else:
                    jobs.append((row, code))
    save(work / 'INPUTS_FROZEN.json', dict(plans=plans, jobs=len(jobs), reused=len(outputs), all_four_method_proposals_included=True))
    assert len(jobs) + len(outputs) == 28 and len(jobs) <= protocol['maximum_new_predictions']
    def predict(job):
        row, code = job
        target = work / row['run'] / row['episode'] / str(row['case'])
        for d in ['app/.runtime/tmp', 'app/.runtime/mpl']:
            (target / d).mkdir(parents=True, exist_ok=True)
        template_dir = root / 'evaluation' / row['episode'] / str(row['case'])
        command = read(template_dir / 'COMMAND.json')
        old_code = str(root / 'sealed' / row['episode'] / 'solution.py')
        assert command.count(old_code) == 1 and command.count(str(template_dir / 'app')) == 1
        command = [str(code) if x == old_code else str(target / 'app') if x == str(template_dir / 'app') else x for x in command]
        save(target / 'COMMAND.json', command)
        start = time.monotonic()
        with (target / 'stdout.log').open('w') as out, (target / 'stderr.log').open('w') as err:
            p = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try: status = p.wait(timeout=135)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); status = p.wait()
        pred = read(target / 'app/PREDICTION.json') if (target / 'app/PREDICTION.json').exists() else None
        if pred is not None:
            assert pred['source_sha256'] == sha(code) and pred['query_sha256'] == query[row['case']]['sha256']
        result = dict(row, prediction=pred, reused=False, exit_code=status, seconds=time.monotonic()-start)
        save(target / 'RESULT.json', result)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(predict, job) for job in jobs]):
            outputs.append(f.result())
            save(work / 'PREDICTIONS.json', outputs)
    if not jobs: save(work / 'PREDICTIONS.json', outputs)
    save(work / 'PREDICTIONS_SEAL.json', dict(sha256=sha(work / 'PREDICTIONS.json'), unix=time.time(), no_source_edits=True))
    # These are already-open development targets, bound to the earlier sealed audit.
    comparisons = []
    for plan in plans:
        original = read(Path(protocol['runs'][0] if plan['run'] == sources[0].name else protocol['runs'][1]) / 'RESULT392.json')
        final_rows = {r['case']: r for r in original['rows'] if r['program'] == plan['episode']}
        rows = []
        for case in range(7):
            output = next(r for r in outputs if (r['run'], r['episode'], r['case']) == (plan['run'], plan['episode'], case))
            pred = output['prediction']
            before = None if pred is None else math.dist(pred['location_m'], final_rows[case]['target_m']) * 1000
            after = final_rows[case]['error_mm']
            rows.append(dict(case=case, proposal_error_mm=before, final_error_mm=after,
                             change_mm=None if before is None or after is None else after-before))
        complete = all(r['change_mm'] is not None for r in rows)
        comparisons.append(dict(**plan, rows=rows,
                                proposal_MAE_mm=sum(r['proposal_error_mm'] for r in rows)/7 if complete else None,
                                final_MAE_mm=sum(r['final_error_mm'] for r in rows)/7 if complete else None,
                                mean_change_mm=sum(r['change_mm'] for r in rows)/7 if complete else None,
                                interpretation='Diagnostic artifact comparison. Budget-truncated stages remain invalid; no randomized causal claim about review.'))
    result = dict(comparisons=comparisons, actual_new_predictions=len(jobs), reused_predictions=28-len(jobs),
                  successful_predictions=sum(r['prediction'] is not None for r in outputs),
                  numpy_lstsq_calls=sum(r['prediction']['numerical_calls']['numpy_lstsq'] for r in outputs if not r['reused'] and r['prediction'] is not None),
                  new_LLM_calls=0, primary_results_unchanged=True, independent_confirmation_scenarios=0)
    save(work / 'RESULT.json', result)
    print(json.dumps({**{k:v for k,v in result.items() if k != 'comparisons'}, 'comparisons':[{k:v for k,v in x.items() if k not in ['rows', 'proposal_path', 'final_path', 'interpretation']} for x in comparisons]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)
