"""Plot descriptive shared-candidate comparison; repeated choices are not devices."""
import argparse
import json
from pathlib import Path


def plot(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    data=json.loads(source.read_text());rows=data['summary']
    font=FontProperties(fname='/home/jupyter-yyf-zyq/.fonts/NotoSansCJKsc-Regular.otf')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    arms=['ordinary','A','B','C'];labels=['普通Agent','A：双角色','B：三角色','C：并行分支']
    colors=['#8091a5','#117b96','#7953a4','#cf752d']
    fig,axs=plt.subplots(1,3,figsize=(12.4,4.2))
    for ax,key,title,unit in zip(axs,['record_accuracy','selection_regret','model_requests'],
        ['完成提交：最终准确率','完成提交：程序选择损失','每次尝试的模型请求'],['%','个百分点','次']):
        values=[rows[a][key]*(100 if key!='model_requests' else 1) for a in arms]
        plotted=[v if key=='model_requests' or rows[a]['completed']==4 else 0 for a,v in zip(arms,values)]
        bars=ax.bar(range(4),plotted,color=colors,width=.62)
        for bar,arm in zip(bars,arms):
            if rows[arm]['fallbacks']:bar.set_hatch('///');bar.set_alpha(.55)
        ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_title(title,pad=12);ax.set_ylabel(unit)
        ax.set_ylim(0,max(105 if key!='model_requests' else 5,max(values)*1.18))
        for i,(a,v) in enumerate(zip(arms,values)):
            if key!='model_requests' and rows[a]['completed']<4:
                ax.text(i,12,f"仅{rows[a]['completed']}/4正常\n未作完整比较",ha='center',fontsize=9,color='#666666')
            else:ax.text(i,v+1.2,f'{v:.1f}',ha='center',fontsize=10)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if key=='record_accuracy':
            reference=100*(1-sum(x['record_error'] for x in data['fixed_controls'] if x['method']=='keep_P00')/2)
            ax.axhline(reference,color='#535d63',linestyle='--',linewidth=1)
            ax.text(3.48,reference-4,'保留P00',ha='right',fontsize=9)
    fig.suptitle('同一真实候选库上的A／B／C比较',fontsize=15)
    fig.text(.5,.015,'1台电机 · 40条最终记录。A有1次漏做登记检查；C无有效选择。完整分母与回退见报告；仅作探索性比较。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.92));output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg']:fig.savefig(output.with_suffix(suffix),dpi=180,bbox_inches='tight')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();plot(a.source,a.output)
