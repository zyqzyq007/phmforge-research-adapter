"""Standalone descriptive figures from the complete A328 result table."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


LABELS = {
    'keep':'始终保留', 'development':'开发最优', 'cv':'CV最优',
    'ordinary_1':'普通Agent 1', 'ordinary_2':'普通Agent 2',
    'M01_group_mean':'M01 组均值', 'M02_worst_group':'M02 最差组',
    'M03_tail_risk':'M03 尾部风险', 'M04_dispersion':'M04 波动惩罚',
    'M05_group_regret':'M05 组内相对损失', 'M06_paired_bootstrap':'M06 配对重采样',
    'M07_paired_sign':'M07 符号检验', 'M08_delete_group':'M08 删组稳定性',
    'M09_pareto_update':'M09 逐组不退步', 'M10_broad_support':'M10 收益覆盖',
}


def main(source, output):
    result=json.loads(source.read_text());summary=result['summaries']
    font=next((f.fname for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name),None)
    if font:plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    names=['ordinary_1','ordinary_2','keep','development','cv',*[k for k in summary if k.startswith('M')]]
    colors=['#2670a8' if n.startswith('ordinary') else '#7f8c99' if not n.startswith('M') else '#157a6e' for n in names]
    y=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    keys=[('original_anchors','normalized_regret','主面板：归一化选择损失 ↓'),
          ('all_anchors','gain_capture','全部起点：有效改进利用率 ↑')]
    for ax,(panel,key,title) in zip(axes,keys):
        values=[summary[n][panel][key] for n in names]
        bars=ax.barh(y,[0 if v is None else v for v in values],color=colors,height=.65)
        for bar,value in zip(bars,values):
            ax.text(bar.get_width()+.014,bar.get_y()+bar.get_height()/2,'N/A' if value is None else f'{value:.3f}',va='center',fontsize=9)
        ax.set_title(title,pad=12);ax.set_xlim(0,1.13);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,[LABELS[n] for n in names]);axes[0].invert_yaxis()
    fig.suptitle('十种可执行更新规则与普通 coding Agent：历史真实候选库初筛',fontsize=14)
    fig.text(.02,.015,'3个来源／9个相关候选池。右图88个起点为同一数据上的反事实决策，不是88个独立实验。已开分历史数据，仅作开发证据。',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.045,1,.96));output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=170,bbox_inches='tight');fig.savefig(output.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
