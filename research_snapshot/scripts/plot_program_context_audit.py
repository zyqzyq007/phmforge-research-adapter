"""Plot already-scored context audit JSON;no model/data execution."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(root):
    result = json.loads((root / 'CONTEXT_RESULTS.json').read_text())
    assert result['status'] == 'complete'
    roles = sorted({r['role'] for r in result['rows']}, key=lambda x: int(x.split('_')[-1]))
    models = ['common_p00', 'full_features', 'selected_repair']
    names = ['Original p00', 'Full features p01', 'Selected repair p04']
    colors = ['#8090a0', '#d38b39', '#247f91']
    lookup = {(r['role'], r['program']): r for r in result['rows']}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={'width_ratios': [1.7, 1]})
    x = np.arange(len(roles))
    for j, (model, name, color) in enumerate(zip(models, names, colors)):
        values = [100 * lookup[role, model]['record_error'] for role in roles]
        axes[0].bar(x + (j - 1) * .24, values, width=.23, color=color, label=name)
        for k, value in enumerate(values):
            axes[0].text(x[k] + (j - 1) * .24, value + 1.8, str(int(value)), ha='center', fontsize=8)
    axes[0].set_xticks(x, [r.split('_')[-1] for r in roles])
    axes[0].set_xlabel('New nominal operating condition (Hz)')
    axes[0].set_ylabel('Whole-acquisition error (%)')
    axes[0].set_ylim(0, 101)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[0].grid(axis='y', alpha=.2)
    axes[0].set_axisbelow(True)
    means = [100 * result['summary']['mean_record_errors'][m] for m in models]
    axes[1].barh(names, means, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 42)
    axes[1].set_xlabel('Equal-context mean error (%)')
    for j, value in enumerate(means):
        axes[1].text(value + .7, j, f'{value:.2f}%', va='center', fontsize=10)
    axes[1].grid(axis='x', alpha=.2)
    axes[1].set_axisbelow(True)
    fig.suptitle('A343 fixed-weight falsification: no overall p01 advantage', fontsize=13)
    fig.text(.5, .015, '16 new measured acquisitions; one rig. Fixed models, zero new fits/LLM. Not a method comparison.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .055, 1, .94))
    for suffix in ['png', 'svg']:
        fig.savefig(root / ('CONTEXT_RESULTS.' + suffix), dpi=170)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    main(p.parse_args().run)
