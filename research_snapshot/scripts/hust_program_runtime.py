"""Whole-cell regression fits and isolated single-query, past-only prediction."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from controlled_feedback import save, sha, utc
from program_revision_runtime import Runtime as ClassificationRuntime, load, subset


OBSERVABLE = ['x', 'nn', 'past_cap', 'age', 'rates', 'cycle', 'groups', 'sample_ids']


def score(data, prediction):
    assert prediction.shape == data['y'].shape and np.isfinite(prediction).all()
    rows = [dict(cell=str(g), MAE=float(np.mean(np.abs(prediction[data['groups'] == g] - data['y'][data['groups'] == g]))))
            for g in np.unique(data['groups'])]
    return dict(mean_cell_MAE=float(np.mean([r['MAE'] for r in rows])), cells=rows)


class Runtime(ClassificationRuntime):
    def command(self, source, artifacts, input_file, output, mode):
        views = self.config.get('reference_views')
        if not views:
            return super().command(source, artifacts, input_file, output, mode)
        context = next((p.name for p in [output, *output.parents]
                        if p.name in ['fold_0', 'fold_1', 'fold_2']), 'full')
        view = Path(views[context])
        binding = json.loads((view / 'CONTEXT.json').read_text())
        assert binding['context'] == context, 'Reference checkpoint context mismatch'
        return super().command(source, artifacts, input_file, output, mode,
                               extra_readonly_mounts=[(view, '/reference')])

    def predict(self, folder, source, artifacts, data, rename=False):
        folder.mkdir(parents=True, exist_ok=False)
        def one(i):
            # Only this query is mounted. No later query, current capacity or
            # RUL target is present, including at an earlier query of this cell.
            obs = {k: data[k][i:i+1].copy() for k in OBSERVABLE}
            if rename:
                obs['groups'][:] = 'opaque'
                obs['sample_ids'][:] = 'opaque'
            path = folder / ('input' + str(i) + '.npz')
            np.savez_compressed(path, **obs)
            out = folder / ('prediction' + str(i))
            _, audit = self.command(source, artifacts, path, out, 'predict')
            result = json.loads((out / 'predictions.json').read_text())
            assert set(result) == {'rul'} and len(result['rul']) == 1
            p = result['rul'][0]
            assert isinstance(p, (int, float)) and np.isfinite(p) and p >= 0
            return float(p), audit
        with ThreadPoolExecutor(max_workers=self.config['prediction_workers']) as pool:
            results = list(pool.map(one, range(len(data['sample_ids']))))
        pred = np.asarray([r[0] for r in results])
        np.savez_compressed(folder / 'PREDICTIONS.npz', sample_ids=data['sample_ids'], prediction=pred)
        return pred, [r[1] for r in results]

    def fit_program(self, program_dir):
        self.deadline = time.monotonic() + self.config['program_seconds']
        source = program_dir / 'source'
        train, dev = load(self.inputs / 'train.npz'), load(self.inputs / 'development_observations.npz')
        queries = load(self.inputs / 'training_queries_private.npz')
        jobs = json.loads(Path(self.config['reference_jobs']).read_text())
        folds, audits = [], []
        for name, job in jobs.items():
            if name == 'full':
                continue
            folder = program_dir / name
            folder.mkdir()
            fit = subset(train, np.isin(train['groups'], job['fitting_cells']))
            valid = subset(queries, np.isin(queries['groups'], job['heldout_cells']))
            assert not set(fit['groups']) & set(valid['groups'])
            path = folder / 'train.npz'
            np.savez_compressed(path, **fit)
            _, audit = self.command(source, None, path, folder / 'fit', 'fit')
            audits.append(audit)
            pred, online = self.predict(folder / 'check', source, folder / 'fit/artifacts', valid)
            folds.append(dict(fold=name, **score(valid, pred)))
            audits.extend(online)
        full = program_dir / 'full_fit'
        _, audit = self.command(source, None, self.inputs / 'train.npz', full, 'fit')
        audits.append(audit)
        pred, online = self.predict(program_dir / 'development', source, full / 'artifacts', dev)
        audits.extend(online)
        renamed, online = self.predict(program_dir / 'identity_check', source, full / 'artifacts', dev, rename=True)
        audits.extend(online)
        np.testing.assert_allclose(pred, renamed, rtol=1e-7, atol=1e-5)
        artifacts = {str(p.relative_to(program_dir)): sha(p) for p in program_dir.rglob('*')
                     if p.is_file() and ('artifacts' in p.parts or 'source' in p.parts)}
        assert not any(p.is_symlink() for p in program_dir.rglob('*'))
        report = dict(status='complete', program_id=program_dir.name,
            cv=folds, development_execution_passed=True, fit_entry_executions=4,
            confirmed_estimator_fits=sum(a['completed_estimator_fits'] for a in audits),
            actual_optimizer_steps=sum(a['optimizer_steps'] for a in audits),
            nonzero_sampled_optimizer_steps=sum(a['nonzero_sampled_optimizer_steps'] for a in audits),
            training_only=True, whole_cell_split=True, identity_invariance_passed=True,
            artifacts=artifacts, final_data_read=False, utc=utc())
        save(program_dir / 'REPORT.json', report)
        return report
