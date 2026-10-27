"""Source-grounded feature-layout intervention using the existing isolated runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np

from controlled_feedback import save, sha, utc
from hust_program_runtime import Runtime
from program_revision_runtime import load


def main(root):
    cfg = json.loads((root/'PROTOCOL467.json').read_text())
    arms = json.loads((root/'LAYOUT_INTERVENTIONS467.json').read_text())
    freeze = json.loads((root/'LAUNCH_FREEZE467.json').read_text())
    for relative, digest in freeze.items():
        assert sha(root/relative) == digest, relative
    assert not (root/'HANDLE467.json').exists(), 'Do not automatically repeat an invocation'
    started = time.monotonic()
    save(root/'HANDLE467.json', dict(pid=os.getpid(), utc=utc(), state='running'))
    try:
        runtime = Runtime(root)
        queries = load(runtime.inputs/'training_queries_private.npz')
        normalized = queries['nn'].transpose(0, 3, 2, 1)
        reconstructed = np.stack([normalized.mean(axis=2), normalized.std(axis=2),
                                  normalized[:, :, 0], normalized[:, :, -1]], axis=-1).reshape(len(normalized), -1)
        np.testing.assert_allclose(reconstructed, queries['x'][:, :160], rtol=0, atol=1e-7)
        qualified = []
        for arm in arms:
            old = Path(arm['old_source'])
            new = Path(arm['new_program'])/'source'
            expected = (old/arm['changed_relative']).read_text()
            for patch in arm['patches']:
                assert expected.count(patch['before']) == 1
                expected = expected.replace(patch['before'], patch['after'])
            assert (new/arm['changed_relative']).read_text() == expected
            for p in old.rglob('*'):
                if p.is_file() and str(p.relative_to(old)) != arm['changed_relative']:
                    assert sha(p) == sha(new/p.relative_to(old))
            # These inspected modules contain imports and function definitions only.
            # Evaluate feature construction on legal training queries; no fitting.
            modules = [runpy.run_path(str(p/arm['changed_relative']), run_name='layout_qualification') for p in [old, new]]
            if arm['case'] == 'regression':
                outputs = [m['causal_features'](queries['x']) for m in modules]
                allowed = [(173, 206), (23, 48)]
            else:
                outputs = [(m['featurize'](queries['x'], queries['nn'], queries['age'], queries['rates']),) for m in modules]
                allowed = [(190, 226)]
            for before, after, (lo, hi) in zip(*outputs, allowed):
                assert before.shape == after.shape and np.isfinite(after).all()
                np.testing.assert_array_equal(before[:, :lo], after[:, :lo])
                np.testing.assert_array_equal(before[:, hi:], after[:, hi:])
                assert np.any(before[:, lo:hi] != after[:, lo:hi])
            qualified.append(dict(case=arm['case'], shapes=[list(x.shape) for x in outputs[0]], changed_columns=allowed))
        save(root/'QUALIFICATION467.json', dict(utc=utc(), passed=True, training_cells=len(queries['sample_ids']),
             raw_layout_max_error=float(np.max(abs(reconstructed-queries['x'][:, :160]))), rows=qualified))

        observations = load(runtime.inputs/'development_observations.npz')
        def replay(arm):
            prediction, audits = Runtime(root).predict(root/'controls'/arm['case'], Path(arm['old_source']),
                                                      Path(arm['old_full_artifacts']), observations)
            mapping = dict(zip(arm['old_unit_ids'], arm['old_predictions']))
            expected = np.array([mapping[str(g)] for g in observations['groups']])
            np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-5)
            return dict(case=arm['case'], passed=True, queries=len(prediction), new_fits=sum(a['completed_estimator_fits'] for a in audits))
        with ThreadPoolExecutor(max_workers=2) as pool:
            control = list(pool.map(replay, arms))
        save(root/'CONTROL_REPLAY467.json', dict(utc=utc(), passed=True, rows=control))
        print(json.dumps(dict(phase='qualified', controls=len(control), cold_predictions=16)), flush=True)

        completed = []
        def fit(arm):
            try:
                report = Runtime(root).fit_program(Path(arm['new_program']))
                return dict(case=arm['case'], complete=True, report_sha256=sha(Path(arm['new_program'])/'REPORT.json'),
                            fit_entries=report['fit_entry_executions'])
            except Exception as error:
                return dict(case=arm['case'], complete=False, error_type=type(error).__name__, error=str(error))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(fit, arm) for arm in arms]):
                completed.append(future.result())
                save(root/'PROGRESS467.json', dict(utc=utc(), rows=completed, seconds=time.monotonic()-started))
                print(json.dumps(completed[-1]), flush=True)
        choices = []
        if all(row['complete'] for row in completed):
            for arm in arms:
                old = {c['cell']:c['MAE'] for fold in arm['old_report']['cv'] for c in fold['cells']}
                report = json.loads((Path(arm['new_program'])/'REPORT.json').read_text())
                new = {c['cell']:c['MAE'] for fold in report['cv'] for c in fold['cells']}
                assert set(old) == set(new) and len(old) == 47
                delta = np.asarray([new[k]-old[k] for k in sorted(old)])
                mean = float(delta.mean()); se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
                choices.append(dict(case=arm['case'], old_CV=float(np.mean(list(old.values()))),
                    new_CV=float(np.mean(list(new.values()))), paired_delta=mean, conditional_SE=se,
                    CV_choice='aligned' if mean < -1e-10 else 'original',
                    paired1SE_choice='aligned' if mean < -se-1e-10 else 'original'))
        seal = dict(utc=utc(), rows=completed, choices=choices, new_target_outcomes_read=False,
                    artifacts={str(p.relative_to(root)):sha(p) for p in (root/'programs').rglob('*')
                               if p.is_file() and (p.name in ['REPORT.json','PREDICTIONS.npz'] or 'artifacts' in p.parts)})
        save(root/'ALL_CHOICES_SEALED467.json', seal)
        if not all(row['complete'] for row in completed):
            raise RuntimeError('An intervention failed; retain incomplete comparison')

        source = Path(cfg['truth_archive']); member = cfg['truth_member']
        entry = next(e for e in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files'] if e['path'] == member)
        assert sha(source/member) == entry['sha256']
        truth = load(source/member)
        assert np.array_equal(truth['sample_ids'], observations['sample_ids'])
        save(root/'TRUTH_BINDING467.json', dict(archive=str(source), member=member, sha256=entry['sha256'],
             utc=utc(), choices_seal_sha256=sha(root/'ALL_CHOICES_SEALED467.json')))
        rows = []
        for arm in arms:
            original = load(root/'controls'/arm['case']/'PREDICTIONS.npz')
            aligned = load(Path(arm['new_program'])/'development/PREDICTIONS.npz')
            assert np.array_equal(original['sample_ids'], aligned['sample_ids'])
            old_error = abs(original['prediction']-truth['y']); new_error = abs(aligned['prediction']-truth['y'])
            delta = new_error-old_error
            rows.append(dict(case=arm['case'], origin=arm['origin'], original_MAE=float(old_error.mean()),
                aligned_MAE=float(new_error.mean()), delta=float(delta.mean()), harm=float(np.maximum(delta,0).mean()),
                gain=float(np.maximum(-delta,0).mean()), better_cells=int((delta < -1e-8).sum()),
                worse_cells=int((delta > 1e-8).sum()), original_errors=old_error.tolist(), aligned_errors=new_error.tolist(),
                sample_ids=truth['sample_ids'].tolist(), predictions=aligned['prediction'].tolist()))
        audits=[json.loads(p.read_text()) for p in root.glob('programs/**/training_audit.json')]
        processes=[json.loads(p.read_text()) for p in root.glob('programs/**/PROCESS.json')]
        result=dict(id='A467', utc=utc(), rows=rows, choices=choices,
                    mechanism_gate=all(r['delta'] < -1e-8 if r['case']=='regression' else r['delta'] <= 1e-8 for r in rows),
                    new_fit_entries=sum(p['mode']=='fit' for p in processes),
                    completed_estimator_fits=sum(a['completed_estimator_fits'] for a in audits),
                    optimizer_steps=sum(a['optimizer_steps'] for a in audits),
                    cold_predictions=sum(p['mode']=='predict' for p in processes)+16,
                    seconds=time.monotonic()-started, new_LLM_calls=0, new_native_stages=0,
                    source_clusters=1, query_cells=8, independent_confirmation=0, goal_achieved=False,
                    scope='Researcher-controlled feature intervention on two related historical submissions, including the successful counterexample. Not an Agent method victory.')
        save(root/'RESULT467.json',result)
        print(json.dumps(dict(mechanism_gate=result['mechanism_gate'], rows=rows, choices=choices,
                             fits=result['new_fit_entries'], seconds=result['seconds'])), flush=True)
    finally:
        save(root/'HANDLE467.json',dict(pid=os.getpid(), utc=utc(), state='terminal', seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
