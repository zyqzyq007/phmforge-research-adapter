"""Report all six fixed reviews and their counterevidence without refitting."""
import argparse
import json
from pathlib import Path
import shutil

from acquire_forge_pdmb_sources244 import save
from run_validation_transfer import utc

PROJECT=Path(__file__).resolve().parents[1]


def main(root):
    data=json.loads((root/'NUMERICAL_RESULTS281.json').read_text())
    execution=json.loads((root/'EXECUTION_SUMMARY281.json').read_text())
    rows={x['case']:x for x in data['cases']}
    comparisons=[]
    for case,candidate in [('rotor_a','subset_2'),('rotor_c','component_et_base_sqrt'),('xjtu_b','bank_knn5')]:
        row=rows[case];alt=next(x for x in row['all_frozen_candidates'] if x['candidate']==candidate)
        if row['family']=='classification':
            loss=dict(accuracy_points=(alt['metrics']['accuracy']-row['reviewed']['accuracy'])*100,
                macro_f1=alt['metrics']['macro_f1']-row['reviewed']['macro_f1'],
                extra_errors=round((alt['metrics']['accuracy']-row['reviewed']['accuracy'])*737))
        else:
            a=row['reviewed']['mean_bearing_mae_minutes'];b=alt['metrics']['mean_bearing_mae_minutes']
            loss=dict(extra_mae_minutes=a-b,relative_increase=(a-b)/b)
        comparisons.append(dict(case=case,original_preexisting_comparator=candidate,retained_candidate='original',
            prior_major_loss_persists=True,loss=loss,new_physical_case=False))
    control=[]
    for row in data['cases']:
        field=row['primary_metric'];a=row['reviewed'][field];b=row['deterministic_development_selector']['metrics'][field]
        delta=a-b if row['family']=='classification' else b-a
        control.append(dict(case=row['case'],fixed_choice=row['deterministic_development_selector']['candidate'],
            primary_metric=field,agent_improvement_over_development_greedy=delta,
            interpretation='better' if delta>1e-9 else ('equal' if abs(delta)<1e-9 else 'worse')))
    adjudication=dict(utc=utc(),native_sessions=6,normal_completions=6,model_requests=execution['total_requests'],
        new_fits=0,new_final_inference=0,final_prediction_reuse='Verified original cold predictions of unchanged model artifacts, independently re-scored after six choices were sealed.',
        repairs=0,unchanged=6,worse=0,incomplete=0,protocol_deviations=0,
        previous_major_losses_followed_up=3,previous_major_losses_retained=3,new_major_failure_trajectories=0,
        previously_selected_sample_not_population=True,new_independent_physical_cases=0,
        comparator_cases=comparisons,development_selector_comparison=control,
        conclusions={'supported':'Three previously observed material selection losses persist through one outcome-blind fixed-candidate review with actual tool checks.',
          'counterevidence':'All six reviews disclosed uncertainty; four retained choices outperform greedy development-only choice, two tie. This does not support a simple no-checking or training-score-only explanation.',
          'unproven':['Repeated-feedback causal effect','Agent-specific overfitting mechanism','Population frequency','Failure of unrestricted retraining/review'],
          'alternative_explanations':['Few development records/bearings','Cross-condition/lifetime extrapolation','Candidate ranking uncertainty','Missing component CV evidence in the fixed review']},
        goal_status='active',next_action='Do not add fits/seeds on opened final data. Qualify an unused measured allocation and preregister equal candidate space/fit budget with feedback-timing intervention; retain negative results.')
    save(root/'ADJUDICATION281.json',adjudication)
    save(PROJECT/'results/review281/ADJUDICATION281.json',adjudication)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,subset,field,title in [(axes[0],list(data['cases'][:4]),'accuracy','Rotor: accuracy (higher is better)'),
                                 (axes[1],list(data['cases'][4:]),'mean_bearing_mae_minutes','XJTU: bearing MAE, minutes (lower is better)')]:
        x=np.arange(len(subset));a=[r['reviewed'][field] for r in subset];b=[r['deterministic_development_selector']['metrics'][field] for r in subset]
        if field=='accuracy':a=[v*100 for v in a];b=[v*100 for v in b]
        bars=ax.bar(x-.18,a,.36,label='Agent after review',color='#226c95')
        others=ax.bar(x+.18,b,.36,label='Best development metric',color='#d79543')
        ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3);ax.bar_label(others,fmt='%.1f',fontsize=9,padding=3)
        ax.set_xticks(x,[r['case'].split('_')[-1].upper() for r in subset]);ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max(a+b)*1.18);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('All six reviews retained the original model; no new fitting',fontsize=12)
    figure=root/'figures';figure.mkdir(exist_ok=True)
    fig.savefig(figure/'review281_counterevidence.png',dpi=170)
    fig.savefig(figure/'review281_counterevidence.svg')
    plt.close(fig)
    shutil.copy2(figure/'review281_counterevidence.png',PROJECT/'results/review281/review281_counterevidence.png')

    table=[]
    for row in data['cases']:
        if row['family']=='classification':
            result=f"accuracy {100*row['reviewed']['accuracy']:.2f}%；F1 {row['reviewed']['macro_f1']:.4f}"
            fixed=f"accuracy {100*row['deterministic_development_selector']['metrics']['accuracy']:.2f}%"
        else:
            result=f"MAE {row['reviewed']['mean_bearing_mae_minutes']:.2f} 分钟"
            fixed=f"MAE {row['deterministic_development_selector']['metrics']['mean_bearing_mae_minutes']:.2f} 分钟"
        table.append(f"| {row['case']} | 保留原程序 | {result} | {fixed} |")
    report='''# A281：真实候选复核完成，三例已知损失仍保留

六例均正常交付，全部选择保留原程序。转子A/C和寿命B此前已确认的重大选择损失，在一次不含最终反馈的固定候选复核后仍存在。**这是真实的复核残余，但不是重复反馈造成过拟合的因果证明，也不是新增三个独立失败设备。**

使用真实测量数据与此前真实拟合的权重，六个原生 Codex CLI 0.163.0 / deepseek-flash 会话共226请求，全部HTTP200；约1900万输入、25.59万输出token。模型阶段约12分钟、最多六例并发。本轮0新增拟合、0新增最终推理：先逐文件绑定旧训练制品，完成95候选开发重放；再封存六份新选择，以同一冻结候选原有隔离冷预测独立重算最终分。不是模拟预测，不是官方未修改PHMForge成绩。

| 复核对象 | 决定 | 复核后的最终结果（与原结果相同） | 直接选开发指标最优候选的最终结果 |
|---|---|---|---|
'''+ '\n'.join(table)+'''

原有重大损失保持同一比较对象：转子A所选集成72.05%，已有RF组件93.89%，净多错161/737窗；转子C所选集成86.70%，已有ET组件96.88%，净多错75/737窗；寿命B所选组合MAE450.18分钟，已检查过的KNN329.71分钟，增加120.48分钟/36.54%。它们均已在原轮开分前登记，不根据这次复核重新选择有利比较对象。全95候选结果保留，测试最优只作诊断，不能当部署选择依据。

**反证必须一起解释。** 转子B/D没有重大可用候选损失；寿命D相对KNN的7.83%差距未达原20%且10分钟的重大门槛，并优于两份旧完整程序。更关键的是，转子B/C、寿命B/D的Agent选择均优于“直接按开发集最优选择”，另外两例相同。Agent确实查看了分组CV、复算预测，并说明少量开发记录和外推风险；不能据此写成“不会验证”“只看训练分”或“盲目自信”。

![复核与直接按开发集选模的比较](../results/review281/review281_counterevidence.png)

当前更准确的问题是：**在开发实体少、需要跨工况或寿命尺度外推时，Agent即使认真复核现有证据，仍可能保留一个可运行、开发表现更好、但实际损失明显更大的PHM程序。** 数据覆盖不足、候选排序不确定性和普通模型泛化困难仍是竞争解释。是否由反复反馈放大这种损失、是否属于Agent特有难点，尚未确认。

边界：四个转子轨迹来自同一物理划分；两份寿命提交来自同一9/3/3轴承划分，最终15个查询只有3台轴承。六例是选定的旧提交，研究者已见旧分，不估计总体发生率。固定候选复核禁止新训练/权重搜索，不能证明允许重新建模的所有复核均无效；部分组件没有保留的CV证据。没有评价真实维护收益。

核验：95候选资格重放、103项旧预测/指标对照、73份开发指标独立公式检查、六例原生隔离/MCP/预算门通过，全部冻结输入未变。Agent新开发预测经单查询对齐后94/95入口精确匹配；剩余未被选中的参考MLP，批量与单查询差1.5259e-5分钟（0.000916秒），严格1e-5核对的失败记录保留，不影响选择或最终评分。未发现新拟合、外部取数或最终标签访问。资格脚本切片错误、首次额度扫描超时及Agent自身修复的查询构造错误均保留；前两项在0模型请求时恢复，未重跑Agent。

下一步只推进区分机制：在未用于本轮选模的实测单元上，先冻结同候选空间、同训练预算、不同开发反馈时机的对照，再观察开发收益与隔离测试损失是否系统分离。若普通固定流程同样失败、差异不随反馈变化，或普通复核稳定修复，应降低“Agent自适应过拟合”的主张。源资格和盲态未满足前不启动，停止在已开分记录上增加拟合、种子或阈值。持续目标保持active。

证据：[完整数值](../results/review281/NUMERICAL_RESULTS281.json)、[判定](../results/review281/ADJUDICATION281.json)、[实际执行](../results/review281/EXECUTION_SUMMARY281.json)、[运行/归档位置](../results/review281/current_run.json)。
'''
    (root/'REPORT281.md').write_text(report)
    (PROJECT/'docs/frozen_candidate_review281_results.md').write_text(report)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
