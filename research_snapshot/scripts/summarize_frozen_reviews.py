"""Summarize configured fixed reviews; never refit or select a new comparator."""
import argparse
import json
from pathlib import Path
import shutil


def main(root):
    read = lambda p: json.loads(p.read_text())
    config = read(root / 'CAMPAIGN_CONFIG.json')
    data = read(root / 'NUMERICAL_RESULTS281.json')
    execution = read(root / 'EXECUTION_SUMMARY281.json')
    number = config.get('iteration_number', 281)
    destination = Path(__file__).resolve().parents[1] / config['result_directory']
    comparisons, controls = [], []
    known = {'rotor_a': 'subset_2', 'rotor_c': 'component_et_base_sqrt', 'xjtu_b': 'bank_knn5'}
    for row in data['cases']:
        field = row['primary_metric']; current = row['reviewed'][field]
        fixed = row['deterministic_development_selector']
        direction = 1 if row['family'] == 'classification' else -1
        delta = direction * (current - fixed['metrics'][field])
        controls.append(dict(case=row['case'], candidate=fixed['candidate'], agent_improvement=delta,
                             outcome='better' if delta > 1e-9 else 'equal' if abs(delta) < 1e-9 else 'worse'))
        if row['case'] not in known:
            continue
        alternative = next(c for c in row['all_frozen_candidates'] if c['candidate'] == known[row['case']])
        if row['family'] == 'classification':
            accuracy_loss = alternative['metrics']['accuracy'] - row['reviewed']['accuracy']
            loss = dict(macro_f1=alternative['metrics']['macro_f1'] - current,
                        accuracy_points=100 * accuracy_loss,
                        extra_errors=round(accuracy_loss * sum(map(sum, row['reviewed']['confusion_matrix']))))
            major = loss['macro_f1'] >= .05 and accuracy_loss >= 0
        else:
            baseline = alternative['metrics'][field]
            loss = dict(extra_mae_minutes=current - baseline, relative_increase=(current - baseline) / baseline)
            major = loss['extra_mae_minutes'] >= 10 and loss['relative_increase'] >= .2
        comparisons.append(dict(case=row['case'], predeclared_comparator=known[row['case']],
                                reviewed_candidate=row['selected'], major_loss_persists=major, loss=loss))
    result = dict(utc=data['utc'], iteration=number, native_sessions=execution['native_sessions'],
                  normal_completions=execution['normal_completions'], requests=execution['total_requests'],
                  outcomes=data['counts'], new_fits=0, new_final_inference=0,
                  new_independent_physical_cases=0, physical_allocations=data['physical_allocations'],
                  comparator_cases=comparisons, development_selector_comparison=controls,
                  prior_major_losses_retained=sum(c['major_loss_persists'] for c in comparisons),
                  goal_status='active', all_outcomes_retained=True,
                  unconfirmed=['Repeated-feedback causality', 'Agent specificity', 'Population frequency',
                               'Unrestricted retraining difficulty'])
    for directory in [root, destination]:
        (directory / f'ADJUDICATION{number}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(root / 'EXECUTION_SUMMARY281.json', destination / 'EXECUTION_SUMMARY281.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    families = sorted({r['family'] for r in data['cases']})
    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 4), squeeze=False, layout='constrained')
    for ax, family in zip(axes[0], families):
        rows = [r for r in data['cases'] if r['family'] == family]
        field = 'accuracy' if family == 'classification' else 'mean_bearing_mae_minutes'
        scale = 100 if family == 'classification' else 1
        x = np.arange(len(rows)); a = [r['reviewed'][field] * scale for r in rows]
        b = [r['deterministic_development_selector']['metrics'][field] * scale for r in rows]
        ax.bar_label(ax.bar(x - .18, a, .36, label='Agent after review', color='#226c95'), fmt='%.1f', padding=3)
        ax.bar_label(ax.bar(x + .18, b, .36, label='Best development metric', color='#d79543'), fmt='%.1f', padding=3)
        marks = [(i, next(c for c in r['all_frozen_candidates'] if c['candidate'] == known[r['case']])['metrics'][field] * scale)
                 for i, r in enumerate(rows) if r['case'] in known]
        if marks:
            ax.scatter(*zip(*marks), marker='D', color='#8b3258', label='Predeclared existing component', zorder=4)
        ax.set_xticks(x, [r['case'] for r in rows]); ax.set_ylim(0, max(a + b + [v for _, v in marks]) * 1.2)
        ax.set_ylabel('Accuracy (%)' if family == 'classification' else 'MAE (minutes)')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        ax.legend(loc='lower left', fontsize=8)
    fig.suptitle(f'A{number}: frozen candidates; historical final predictions; no new fitting')
    figures = root / 'figures'; figures.mkdir(exist_ok=True)
    for extension in ['png', 'svg']:
        path = figures / f'review{number}_comparison.{extension}'
        fig.savefig(path, dpi=170); shutil.copy2(path, destination / path.name)
    plt.close(fig)
    shutil.copy2(__file__, root / 'code' / Path(__file__).name)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
