"""Plot only completed train/CV/development feasibility; never open final data."""
import argparse
import json
from pathlib import Path
import shutil


def main(root, destination):
    data = json.loads((root/'FEASIBILITY283.json').read_text())
    assert data['new_LLM_calls']==0 and data['final_scores_opened'] is False
    rows=data['candidates'];destination.mkdir(parents=True,exist_ok=True)
    for name in ['FEASIBILITY283.json','FITTING_SETTINGS.json']:
        shutil.copy2(root/name,destination/name)
    for name in ['PREPARE_COMPLETE283.json','NATIVE_METADATA_CORRECTION283.json','CLOCK_ORIGIN_DIAGNOSTIC283.json']:
        shutil.copy2(root/'audit'/name,destination/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    complete=[row for row in rows if row['status']=='complete']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');x=np.arange(len(complete))
    for offset,field,label,color in [(-.25,'train','Training','#8aa8ba'),(0,'cv','Leave-condition-out CV','#226c95'),(.25,'development','Development','#d79543')]:
        bars=ax.bar(x+offset,[row[field]['macro_f1'] for row in complete],.25,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    ax.set_xticks(x,[row['recipe']['model']+'\n'+row['recipe']['features'] for row in complete])
    ax.set_ylabel('Macro-F1');ax.set_ylim(0,1.16);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left',fontsize=9)
    ax.set_title('Measured G1 stages: 24 actual fits; final split remains unscored')
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path=folder/f'feasibility283.{ext}';fig.savefig(path,dpi=170);shutil.copy2(path,destination/path.name)
    plt.close(fig);shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(actual_fits=data['real_fits_completed'],admission=data['admission_passed'],new_Agent_cases=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    main(args.run.resolve(),args.destination.resolve())
