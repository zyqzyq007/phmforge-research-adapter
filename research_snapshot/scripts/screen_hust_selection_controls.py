"""Opened developmental evidence only: conventional uncertainty controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from controlled_feedback import save, sha, utc


def screen(root):
    cfg = json.loads((root / 'PROTOCOL369.json').read_text())
    source = Path(cfg['source_archive'])
    assert sha(source / 'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    result = json.loads((source / 'RESULT367.json').read_text())
    output = []
    for row in result['rows']:
        run = source / 'subjects' / row['id']
        state = json.loads((run / 'broker/STATE.json').read_text())
        errors = {}
        for ident in state['programs']:
            report = source / 'reference' / ident / 'REPORT.json' if ident.startswith('ref_') or ident == 'p00' else run / 'programs' / ident / 'REPORT.json'
            d = json.loads(report.read_text())
            cells = {c['cell']: c['MAE'] for fold in d['cv'] for c in fold['cells']}
            assert len(cells) == 47
            errors[ident] = cells
        ids = sorted(next(iter(errors.values())))
        matrix = {i: np.asarray([d[k] for k in ids]) for i, d in errors.items()}
        order = ['p00']+[i for i in errors if i != 'p00']
        best = min(order, key=lambda i: (float(np.mean(matrix[i])), order.index(i)))
        selections = {'CV_minimum': best, 'keep': 'p00'}
        evidence = {}
        for multiple in [1, 2]:
            eligible = []
            for i in order:
                diff = matrix[i]-matrix[best]
                mean, se = float(np.mean(diff)), float(np.std(diff, ddof=1)/np.sqrt(len(diff)))
                if mean <= multiple*se+1e-10:
                    eligible.append(i)
                evidence[i] = dict(mean_paired_difference=mean, conditional_standard_error=se)
            selections['paired_'+str(multiple)+'SE_earliest'] = eligible[0]
        threshold = float(np.mean(matrix[best])+np.std(matrix[best], ddof=1)/np.sqrt(len(ids)))
        selections['unpaired_1SE_earliest'] = next(i for i in order if np.mean(matrix[i]) <= threshold+1e-10)
        scored = {name: dict(program_id=i, MAE=row['all_programs'][i]['mean_cell_MAE'],
                    absolute_errors=row['all_programs'][i]['absolute_errors']) for name, i in selections.items()}
        output.append(dict(id=row['id'], controls=scored, legal_training_evidence=evidence, tie_order=order))
    aggregate = {k: float(np.mean([row['controls'][k]['MAE'] for row in output])) for k in output[0]['controls']}
    save(root / 'RESULT369.json', dict(utc=utc(), rows=output, mean_across_ordinary_repeats=aggregate,
        physical_discovery_cells=8, source_clusters=1, correlated_candidate_pools=2,
        inference='Conditionalstandarderrorof47cachedOOFerrors;doesnotincludeadaptivecandidateconstruction/modeltraininguncertainty;noguaranteeofcurrenttargetranking.',
        new_LLM_calls=0, new_fits=0, new_predictions=0, exploratory_only=True,
        new_method_or_novelty=False, independent_confirmation_scenarios=0, majority_complete=False))
    save(root / 'TERMINAL369.json', dict(exit_code=0, utc=utc()))
    print(json.dumps(aggregate))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    screen(p.parse_args().run.resolve())
