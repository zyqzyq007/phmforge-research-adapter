"""Plot descriptive A325 results; no additional experimental computation."""
import argparse
import json
from pathlib import Path
import statistics


def plot(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(source.read_text())
    candidates = ['P00', 'P10', 'P01', 'P11', 'P11B']
    rows = data['candidates']
    means = {k: [100 * statistics.mean(r[k] for r in rows if r['candidate'] == c)
                 for c in candidates] for k in ['cv_error', 'final_error']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    x = np.arange(len(candidates))
    for offset, metric, color, label in [(-.18, 'cv_error', '#3679b8', 'Legal grouped CV'),
                                        (.18, 'final_error', '#ce5c45', 'Sealed final evaluation')]:
        bars = axes[0].bar(x + offset, means[metric], .35, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
    axes[0].set_xticks(x, candidates)
    axes[0].set_title('Candidate ranking across observed and unseen loads')
    axes[0].set_xlabel('All candidates: development record error = 0%')
    axes[0].legend(frameon=False, loc='upper left', fontsize=9)
    names = ['Keep P00', 'Dev / CV /\nfixed grid', 'Ordinary\nAgent', 'B: three\nroles']
    values = [data['fixed']['keep_P00']['record_error'],
              data['fixed']['complete_fixed_grid']['record_error'],
              data['arms']['ordinary']['record_error'], data['arms']['B']['record_error']]
    # Collapsing these three controls is valid only if their actual choices agree.
    assert data['fixed']['development_best']['choices'] == data['fixed']['CV_best']['choices']
    assert data['fixed']['development_best']['choices'] == data['fixed']['complete_fixed_grid']['choices']
    bars = axes[1].bar(names, np.asarray(values) * 100, color=['#678b74', '#888888', '#3679b8', '#9366a8'], width=.6)
    axes[1].bar_label(bars, fmt='%.1f', fontsize=10, padding=3)
    axes[1].set_title('No improvement from B in this pilot')
    axes[1].set_xlabel('8 selection episodes; 40 distinct final records shared')
    for ax in axes:
        ax.set_ylabel('Whole-record classification error (%)')
        ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('A325: fixed-candidate selection, one motor and two acquisition blocks', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
