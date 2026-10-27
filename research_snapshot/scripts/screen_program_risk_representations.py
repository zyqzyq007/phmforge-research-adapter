"""One frozen representation ablation; old physical results are a regression check."""
import argparse
import json
import os
from pathlib import Path
from controlled_feedback import save, sha, utc
from screen_bound_program_risk import run


def study(root):
    read = lambda p: json.loads(Path(p).read_text())
    cfg = read(root/'PROTOCOL372.json')
    assert not (root/'HANDLE372.json').exists(), 'No implicit study restart'
    for name, digest in read(root/'CODE_FREEZE372.json').items():
        assert sha(root/'code'/name) == digest
    profile = Path(cfg['profile_archive'])
    assert read(profile/'ARCHIVE_VERIFIED.json')['source_cleanup'] == 'complete'
    assert sha(profile/'SOURCE_MANIFEST.json') == cfg['profile_manifest_sha256']
    assert read(profile/'QUALIFICATION371.json')['passed']
    save(root/'HANDLE372.json', dict(pid=os.getpid(), utc=utc(), state='running'))
    studies = {}
    for space in cfg['feature_spaces']:
        branch = root/'branches'/space
        run(branch)
        studies[space] = read(branch/'RESULT370.json')
        save(root/'PROGRESS372.json', dict(completed_spaces=list(studies), utc=utc()))
    old = read(Path(cfg['physical_baseline_archive'])/'RESULT370.json')
    for name, values in old['aggregate'].items():
        for metric, value in values.items():
            assert abs(studies['physical']['aggregate'][name][metric]-value) < 1e-9
    controls = [(space, name, metrics) for space, d in studies.items()
                for name, metrics in d['aggregate'].items() if not name.startswith('R12')]
    best_space, best_name, best = min(controls, key=lambda row: row[2]['absolute_error'])
    decisions = []
    for space in cfg['scientific_feature_spaces']:
        for name, values in studies[space]['aggregate'].items():
            if not name.startswith('R12'):
                continue
            passed = (values['absolute_error'] < best['absolute_error']-1e-9 and
                      values['cell_harm'] <= best['cell_harm']+1e-9 and
                      values['cell_gain'] >= best['cell_gain']-1e-9)
            decisions.append(dict(feature_space=space, correction=name, **values, advance=passed))
    result = dict(utc=utc(), corrected_risk_prototypes=decisions,
        all_aggregates={space: d['aggregate'] for space, d in studies.items()},
        best_fixed_control=dict(feature_space=best_space, name=best_name, **best),
        physical_regression_check_passed=True, new_PHM_fits=0, new_PHM_predictions=0, new_LLM_calls=0,
        scientific_multioutput_risk_function_fits=96, verification_risk_function_fits=32,
        scientific_kernel_linear_system_fits=72, verification_kernel_linear_system_fits=24,
        single_factor='Evidence representation;programs/stage/partitions/kernelrule/lambdaandallriskformsunchanged.',
        primary_feature_space='prediction_change', secondary_not_for_unseen_claim=cfg['scientific_feature_spaces'][1:],
        independent_confirmation_scenarios=0, ordinary_Agent_comparison=False, novelty_confirmed=False,
        all_development_outcomes_already_opened=True, final22_opened=False, majority_complete=False)
    save(root/'RESULT372.json', result)
    save(root/'TERMINAL372.json', dict(exit_code=0, utc=utc()))
    print(json.dumps(dict(best_fixed_control=result['best_fixed_control'],
        prototypes=[{k: v for k, v in row.items() if k in ['feature_space', 'correction', 'absolute_error', 'advance']}
                    for row in decisions], physical_regression_check_passed=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    study(p.parse_args().run.resolve())
