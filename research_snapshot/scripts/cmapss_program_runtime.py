"""Whole-program RUL fitting and isolated, genuinely truncated development checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from controlled_feedback import save, sha, utc
from program_revision_runtime import Runtime as BaseRuntime, load


def scores(truth, predictions, engines):
    error = np.abs(np.asarray(truth) - np.asarray(predictions))
    groups = [dict(engine=int(g), MAE=float(error[np.asarray(engines) == g].mean()))
              for g in sorted(set(engines))]
    return dict(MAE=float(error.mean()), engine_mean_MAE=float(np.mean([r['MAE'] for r in groups])),
                engine_errors=groups, count=len(error))


class Runtime(BaseRuntime):
    def predict_queries(self, folder, source, artifacts, data, prefixes=False, rename=False):
        """Every child receives only one engine through its specified observation time."""
        folder.mkdir(parents=True, exist_ok=False)
        values, columns = data['values'], data['columns']
        tasks = []
        for value in sorted(set(values[:, 0].astype(int))):
            engine = int(value)
            frame = values[values[:, 0] == engine]
            assert np.all(np.diff(frame[:, 1]) > 0)
            cuts = [max(1, int(len(frame)*f)) for f in self.config['prefix_fractions']] if prefixes else [len(frame)]
            for cut in cuts:
                tasks.append((engine, frame[:cut].copy(), len(frame), cut))
        def one(item):
            engine, frame, length, cut = item
            key = f'e{engine}_c{cut}'
            if rename:
                frame[:, 0] = 999
            input_file = folder / (key+'.npz')
            np.savez_compressed(input_file, values=frame, columns=columns)
            out = folder / key
            _, audit = self.command(source, artifacts, input_file, out, 'predict')
            row = json.loads((out/'predictions.json').read_text())
            pred = np.asarray(row['rul'], dtype=float)
            assert pred.shape == (cut,) and np.isfinite(pred).all() and np.all(pred >= 0)
            np.savez_compressed(out/'rows.npz', cycle=frame[:, 1], prediction=pred)
            if prefixes:
                return dict(engine=engine, cycle=int(frame[-1, 1]), cut=cut,
                            prediction=float(pred[-1]), truth=float(length-cut), audit=audit)
            return dict(engine=engine, cycle=frame[:, 1].astype(int).tolist(),
                        prediction=pred.tolist(), truth=(length-frame[:, 1]).tolist(), audit=audit)
        with ThreadPoolExecutor(max_workers=self.config['prediction_workers']) as pool:
            rows = list(pool.map(one, tasks))
        save(folder/'PREDICTIONS.json', rows)
        return rows

    def development(self, program_dir):
        if not hasattr(self, 'deadline'):
            self.deadline = time.monotonic()+self.config['program_seconds']
        data = load(self.inputs/'development.npz')
        rows = self.predict_queries(program_dir/'development', program_dir/'source',
                                    program_dir/'full_fit/artifacts', data)
        truth = [v for r in rows for v in r['truth']]
        pred = [v for r in rows for v in r['prediction']]
        engines = [r['engine'] for r in rows for _ in r['prediction']]
        return dict(**scores(truth, pred, engines), capped125_MAE=float(np.mean(np.abs(np.minimum(truth, 125)-pred))),
                    scope='Retrospective full-trajectory row validation; not prospective certification.')

    def fit_program(self, program_dir):
        self.deadline = time.monotonic()+self.config['program_seconds']
        _, audit = self.command(program_dir/'source', None, self.inputs/'fit_input.npz', program_dir/'full_fit', 'fit')
        report = dict(status='complete', program_id=program_dir.name,
                      development=self.development(program_dir), fit_entry_executions=1,
                      confirmed_estimator_fits=audit['completed_estimator_fits'],
                      actual_optimizer_steps=audit['optimizer_steps'],
                      nonzero_sampled_optimizer_steps=audit['nonzero_sampled_optimizer_steps'],
                      training_label_policy='fit train engines only; development labels permitted for model selection',
                      final_data_read=False, utc=utc())
        assert not any(p.is_symlink() for p in program_dir.rglob('*'))
        report['artifacts'] = {str(p.relative_to(program_dir)):sha(p) for p in program_dir.rglob('*')
                               if p.is_file() and ('source' in p.parts or 'artifacts' in p.parts)}
        save(program_dir/'REPORT.json', report)
        return report

    def check(self, program_dir, output):
        self.deadline = time.monotonic()+self.config['program_seconds']
        rows = self.predict_queries(output, program_dir/'source', program_dir/'full_fit/artifacts',
                                    load(self.inputs/'development.npz'), prefixes=True)
        full = json.loads((program_dir/'development/PREDICTIONS.json').read_text())
        lookup = {r['engine']:r for r in full}
        for r in rows:
            r['full_context_prediction'] = lookup[r['engine']]['prediction'][r['cut']-1]
            r['absolute_context_change'] = abs(r['prediction']-r['full_context_prediction'])
        result = dict(program_id=program_dir.name, **scores([r['truth'] for r in rows],
                      [r['prediction'] for r in rows], [r['engine'] for r in rows]),
                      max_context_change=max(r['absolute_context_change'] for r in rows),
                      observations_only_through_query=True, adaptive_development_not_independent_test=True,
                      rows=[{k:v for k,v in r.items() if k!='audit'} for r in rows],
                      code_and_weights_sha256=sha(program_dir/'REPORT.json'), utc=utc())
        save(output/'CHECK.json', result)
        return result
