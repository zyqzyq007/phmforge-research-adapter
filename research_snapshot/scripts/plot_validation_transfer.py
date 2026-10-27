"""Plot the complete fixed A267 candidate comparisons from external scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main(source, output):
    data = json.loads(source.read_text())
    palette = {'reference': '#8996A6', 'submitted': '#D87C46', 'candidate': '#4676B4'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1.5, 1.2, 1]})
    motor = data['classification']['ELECTRICMOTORVIBRATIONS']['stages']['review']
    pu = data['classification']['PADERBORN']['stages']['review']
    rows = [
        [('Reference', motor['reference']['metrics']['accuracy'] * 100, 'reference'),
         ('Submitted + reviewed', motor['final']['metrics']['accuracy'] * 100, 'submitted')]
        + [(x['name'], x['metrics']['accuracy'] * 100, 'candidate') for x in data['motor_fixed_comparators']],
        [('Reference', pu['reference']['metrics']['accuracy'] * 100, 'reference'),
         ('Submitted after review', pu['final']['metrics']['accuracy'] * 100, 'submitted'),
         ('Restored old code', data['paderborn_fixed_prior']['metrics']['accuracy'] * 100, 'candidate')],
        [('Reference', data['rul']['review']['results']['reference']['consistent10']['mae'], 'reference'),
         ('Submitted after review', data['rul']['review']['results']['final']['consistent10']['mae'], 'submitted')],
    ]
    for index, (ax, values) in enumerate(zip(axes, rows)):
        ax.barh(range(len(values)), [x[1] for x in values], color=[palette[x[2]] for x in values], height=.62)
        ax.set_yticks(range(len(values)), [x[0] for x in values], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 112 if index < 2 else 4200)
        for i, (_, value, _) in enumerate(values):
            ax.text(value + (1 if index < 2 else 45), i, f'{value:.2f}' if index < 2 else f'{value:.1f}', va='center', fontsize=9)
        ax.set_xlabel('Test accuracy (%) - higher is better' if index < 2 else 'RUL MAE (seconds) - lower is better', fontsize=9)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.18)
    axes[0].set_title('Motor: 6 recording groups\nAll five prespecified saved candidates', loc='left', fontsize=11)
    axes[1].set_title('Paderborn: 6 bearing groups\nOld code refitted on matching data', loc='left', fontsize=11)
    axes[2].set_title('FEMTO: 10 consistent bearings\nSuccessful selection', loc='left', fontsize=11)
    fig.suptitle('A267 | Real model fitting, frozen choices, independent external scores', fontsize=14, y=.98)
    fig.legend(handles=[Patch(color=palette[k], label=v) for k, v in
        [('reference', 'Supplied reference'), ('submitted', 'Agent submission'), ('candidate', 'Fixed candidate / code restoration')]],
        loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Exploratory known-source study: 6 sessions, 442 model requests; 4 normal completions, 2 initial budget stops. Windows are correlated.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .91), w_pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
