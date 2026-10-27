"""Audit classification review outputs without fitting or reading final scores.

The trace inventory is a screen for subsequent interpretation, not a proof of
all possible behavior. Original events and every flagged command are retained.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from run_validation_transfer import utc


def read(path):
    return json.loads(path.read_text())


def aligned(rows, ids):
    mapping = {row['sample_id']: row['predicted_label'] for row in rows}
    assert len(mapping) == len(rows) == len(ids)
    assert set(mapping) == set(ids)
    result = np.array([mapping[sid] for sid in ids])
    assert np.isin(result, np.arange(5)).all()
    return result


def formula(y, p):
    matrix = np.array([[int(((y == a) & (p == b)).sum())
                       for b in range(5)] for a in range(5)])
    actual, predicted = matrix.sum(1), matrix.sum(0)
    f1 = [2 * matrix[c, c] / (actual[c] + predicted[c])
          if actual[c] + predicted[c] else 0.0 for c in range(5)]
    return dict(n=len(y), accuracy=float(np.trace(matrix) / len(y)),
                macro_f1=float(np.mean(f1)), confusion_matrix=matrix.tolist())


def main(root):
    config = read(root / 'CAMPAIGN_CONFIG.json')
    assert all(c['family'] == 'classification' for c in config['cases'])
    summaries, traces, comparisons, selector, execution = [], [], [], [], []
    pattern = re.compile(r'\.fit\s*\(|\.backward\s*\(|\.step\s*\(|'
                         r'(?:python\S*|python3\s+-B)\s+(?:train_|.*_eval\.py)|'
                         r'https?://|/hidden/|final_labels|curl |wget ')
    for case in config['cases']:
        run = root / case['id']
        job = read(run / 'PLAN187.json')['jobs'][0]
        workspace = run / job['workspace']
        output = run / 'primary' / job['id'] / job['id']
        status = read(output / 'status.json')
        assert status['status'] == 'finished' and status.get('response_valid')
        rows = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        items = [row['item'] for row in rows if row.get('type') == 'item.completed']
        commands = [row for row in items if row.get('type') == 'command_execution']
        mcp = [row for row in items if row.get('type') == 'mcp_tool_call']
        frozen = read(run / 'INITIAL_FREEZE187_FIFO_FILES.json')['files']
        before = {row['path']: row['sha256'] for row in frozen}
        new_sources, new_models = [], []
        for path in workspace.rglob('*'):
            if not path.is_file() or sha256(path) == before.get(str(path)):
                continue
            if path.suffix in {'.joblib', '.pt', '.pth', '.pkl', '.pickle'}:
                new_models.append(str(path.relative_to(workspace)))
            if path.suffix == '.py':
                tree = ast.parse(path.read_text())
                calls = Counter(ast.unparse(node.func) for node in ast.walk(tree)
                                if isinstance(node, ast.Call))
                new_sources.append(dict(path=str(path.relative_to(workspace)), sha256=sha256(path),
                                        calls=dict(calls), fitting_calls=[name for name in calls
                                        if re.search(r'(?:^|\.)(fit|fit_transform|partial_fit|backward|step)$', name)]))
        trace = dict(case=case['id'], commands=commands, mcp=mcp, new_sources=new_sources,
                     new_serialized_models=new_models,
                     flagged_commands=[row for row in commands if pattern.search(row['command'])])
        traces.append(trace)
        with np.load(run / 'public_data/datasets' / case['dataset'] / 'val.npz', allow_pickle=False) as data:
            ids = data['sample_ids'].astype(str).tolist()
            y = data['labels']
        qualification = read(run / 'FIXED_CANDIDATE_REPLAY281.json')
        catalog = read(workspace / 'candidates.json')['candidates']
        qindex = {entry['candidate']: entry for entry in qualification['candidates']}
        assert set(qindex) == {entry['id'] for entry in catalog}
        # Examine all newly written JSON prediction artifacts. Same predictions
        # may legitimately be saved more than once; bind matching file hashes.
        predictions = []
        for path in workspace.rglob('*.json'):
            if 'historical_validation' in path.parts or path.stat().st_size > 2_000_000:
                continue
            if sha256(path) == before.get(str(path)):
                continue
            value = read(path)
            if isinstance(value, dict) and isinstance(value.get('predictions'), list):
                try:
                    vector = aligned(value['predictions'], ids)
                except (AssertionError, KeyError, TypeError):
                    continue
                predictions.append((path, vector))
        casechecks = []
        for candidate in catalog:
            cid = candidate['id']; qualified = qindex[cid]
            expected = aligned(qualified['predictions'], ids)
            numerical = formula(y, expected)
            assert all(abs(numerical[k] - qualified['development'][k]) < 1e-12
                       for k in ['accuracy', 'macro_f1'])
            matching = [dict(path=str(p.relative_to(workspace)), sha256=sha256(p))
                        for p, vector in predictions if np.array_equal(vector, expected)]
            assert matching, (case['id'], cid, 'No matching new subject prediction artifact')
            casechecks.append(dict(candidate=cid, independently_counted=numerical,
                                   matching_subject_predictions=matching, passed=True))
        comparisons.append(dict(case=case['id'], candidate_count=len(catalog), checks=casechecks,
                                scope='Exact predictions matched; duplicate candidate programs may share vectors.'))
        # Same deterministic rule preregistered for the earlier review. This does
        # not use new subject choices or final scores.
        best = max(catalog, key=lambda entry: qindex[entry['id']]['development']['macro_f1'])
        selector.append(dict(case=case['id'], candidate=best['id'], metric='macro_f1',
                             development=qindex[best['id']]['development']['macro_f1'],
                             qualification_sha256=sha256(run / 'FIXED_CANDIDATE_REPLAY281.json')))
        receipts = [read(p) for p in sorted((run / 'model_gateway' / job['id']).glob('*_receipt.json'))]
        gateway = read(run / 'model_gateway' / job['id'] / 'status.json')
        assert len(receipts) == gateway['requests'] and not gateway['inflight']
        execution.append(dict(case=case['id'], status=status['status'], exit_code=status.get('exit_code'),
                              elapsed_seconds=status['elapsed_seconds'], requests=len(receipts),
                              upstream_statuses=dict(Counter(str(r.get('upstream_status')) for r in receipts)),
                              requested_models=sorted({r['model_requested'] for r in receipts}),
                              returned_models=sorted({m for r in receipts for m in r.get('returned_model_ids', [])}),
                              usage=status['usage'], started_utc=status['started_utc'], ended_utc=status['ended_utc']))
        summaries.append(dict(case=case['id'], command_count=len(commands),
                              failed_commands=[r['id'] for r in commands if r.get('exit_code') != 0],
                              mcp_calls=len(mcp), new_source_count=len(new_sources),
                              fitting_calls=[r for r in new_sources if r['fitting_calls']],
                              new_serialized_models=new_models,
                              flagged_commands=[r['id'] for r in trace['flagged_commands']],
                              selection=read(workspace / 'review_selection.json')))
    save(root / 'audit/TRACE_COMMAND_SCREEN281.json', dict(utc=utc(), cases=traces))
    save(root / 'audit/TRACE_SCREEN_SUMMARY284.json', dict(utc=utc(), cases=summaries))
    save(root / 'audit/SUBJECT_DEVELOPMENT_PREDICTION_CHECK281.json',
         dict(utc=utc(), cases=comparisons, candidate_checks=sum(x['candidate_count'] for x in comparisons),
              new_fits=0, new_inference=0, final_scores_consulted=False))
    save(root / 'audit/FROZEN_DEVELOPMENT_SELECTOR281.json',
         dict(utc=utc(), rule='Best development macro-F1; exact ties use earliest catalog entry (original first).',
              cases=selector, models_refitted=0, additional_inference=0,
              new_subject_choices_consulted=False, prior_final_scores_known_to_researcher=True,
              scope='Descriptive control; no causal intervention.'))
    save(root / 'EXECUTION_SUMMARY281.json', dict(utc=utc(), cases=execution, native_sessions=len(execution),
         normal_completions=sum(x['status'] == 'finished' for x in execution),
         total_requests=sum(x['requests'] for x in execution),
         total_input_tokens=sum(x['usage']['input_tokens'] for x in execution),
         total_output_tokens=sum(x['usage']['output_tokens'] for x in execution), new_fits=0))
    shutil.copy2(__file__, root / 'code' / Path(__file__).name)
    print(json.dumps(dict(cases=len(summaries), candidate_checks=sum(x['candidate_count'] for x in comparisons),
                          requests=sum(x['requests'] for x in execution))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
