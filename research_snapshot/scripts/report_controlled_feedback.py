"""Persist complete feedback-probe outcomes, counterevidence and execution cost."""
import argparse
import json
from pathlib import Path
import shutil

from controlled_feedback import save, utc

PROJECT=Path(__file__).resolve().parents[1]


def report_legacy_uoc(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result=json.loads((root/'NUMERICAL_RESULTS282.json').read_text())
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    execution=[]
    for c in config['cases']:
        run=root/c['id'];job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
        status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        gateway=json.loads((run/'model_gateway'/job['id']/'status.json').read_text())
        receipts=[json.loads(p.read_text()) for p in (run/'model_gateway'/job['id']).glob('*_receipt.json')]
        assert len(receipts)==gateway['requests'] and all(x['upstream_status']==200 for x in receipts)
        assert not gateway['inflight'] and gateway['returned_model_ids']==['deepseek-flash']
        execution.append(dict(case=c['id'],status=status['status'],exit_code=status['exit_code'],seconds=status['elapsed_seconds'],
            requests=gateway['requests'],input_tokens=gateway['input_tokens'],output_tokens=gateway['output_tokens'],
            returned_model_ids=gateway['returned_model_ids'],all_requests_HTTP200=True))
    summary=dict(utc=utc(),cases=execution,normal_completions=sum(x['status']=='finished' and x['exit_code']==0 for x in execution),
        requests=sum(x['requests'] for x in execution),input_tokens=sum(x['input_tokens'] for x in execution),output_tokens=sum(x['output_tokens'] for x in execution),
        formal_estimator_fits=result['real_formal_fits'],cold_processes=result['cold_processes'],independent_metric_checks=len(result['independent_metric_checks']),
        infrastructure_attempts=6,infrastructure_requests=210,total_native_attempts=12,logical_cases=6,
        distinct_final_observations=189,physical_acquisitions_unknown=True)
    save(root/'EXECUTION_SUMMARY282.json',summary)
    candidates=[x for r in result['cases'] for x in r['candidates']]
    decision=dict(utc=utc(),goal_status='active',observed_outcome='Negative mechanism probe with ceiling effects',
        major_available_candidate_losses=sum(r['major_available_candidate_loss'] for r in result['cases']),
        selected_final_accuracy_and_macro_f1=[r['chosen']['final'] for r in result['cases']],
        perfect_final_candidates=sum(c['final']['accuracy']==1 for c in candidates),candidate_attempts=len(candidates),
        unique_recipes=len({json.dumps(c['recipe'],sort_keys=True) for c in candidates}),
        repeated_feedback_causality_confirmed=False,agent_specific_mechanism_confirmed=False,
        all_prior_cross_condition_losses_retracted=False,
        inference='No added final loss or optimism from immediate development feedback on this allocation. Ceiling and unknown original recording dependence limit discrimination and external validity.',
        decision='Stop this opened UoC allocation: no new seeds,features,thresholds or harder subsets to manufacture a failure.',
        next_step='Qualify unopened measured PHMForge acquisitions with authenticated physical grouping; inspect training/development discriminability before freezing another equal-space/equal-budget feedback contrast. Preserve all negative outcomes.')
    save(root/'ADJUDICATION282.json',decision)
    fig,ax=plt.subplots(figsize=(9,4.8))
    correct=np.asarray([[c['final']['correct'] for c in row['candidates']] for row in result['cases']])
    ax.imshow(correct,vmin=187,vmax=189,cmap='Blues',aspect='auto')
    for i,row in enumerate(result['cases']):
        for j,c in enumerate(row['candidates']):
            selected=c['id']==row['selection']['candidate_id']
            ax.text(j,i,str(correct[i,j])+(' *' if selected else ''),ha='center',va='center',color='white' if correct[i,j]==189 else 'black',fontweight='bold' if selected else 'normal')
    ax.set_xticks(range(6),[f'Candidate {i+1}' for i in range(6)])
    ax.set_yticks(range(6),[x['case'].replace('_',' ') for x in result['cases']])
    ax.set_title('UoC real-data feedback probe: all six selected models score 189 / 189')
    ax.set_xlabel('* = sealed Agent selection. Each cell is correct / 189; one shared final allocation.')
    fig.text(.01,.015,'36 actual candidate fits (17 unique recipes); 34 reach 189/189. Unknown acquisition dependence; no cross-device claim.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);(root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/feedback282.png',dpi=180);fig.savefig(root/'figures/feedback282.svg');plt.close(fig)
    rows=[]
    for row in result['cases']:
        c=row['chosen'];recipe=c['recipe']
        rows.append('| '+row['case']+' | '+recipe['model']+' / '+recipe['features']+' | 100% | 100% | 189/189 |')
    text='''# A282：真实训练反馈对照得到反证

**本轮没有复现 Agent 过拟合或错误淘汰更好候选。目标继续 active，不能将“容易过拟合”写成已确认结论。** 在一个 UoC 实测集合划分上，即时反馈和延迟反馈各三次原生会话，六个封存选择的最终 accuracy 与 macro-F1 均为 1.0，两组差值和开发减最终的落差均为 0。

| 会话 | 最终选择 | 训练 CV | 开发 | 最终正确数 |
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

![全部实际候选与封存选择](../results/feedback282/feedback282.png)

这里是 **189 条不同的最终观测、一个划分、六次策略轨迹**，不能算成六套独立设备或 1,134 个独立样本。源包含真实测量派生信号及原作者九类小齿轮状态；完整列划分为 558／189／189，不使用模拟信号、伪造标签或 Stage0/PCA 文件。原始采集／硬件身份缺失，未知相关性或重叠仍限制外推。它是明确适配的 PHMForge 任务与受限配方机制探针，不是原版官方成绩或自由编程全能力评测。

六次恢复会话全部正常完成，**101 次 deepseek-flash 请求全部 HTTP200；输入 4,103,029、输出 120,304 token**。每会话六个不同候选，每候选三折训练 CV 加一次部署拟合，共 **36 个候选、144 次真实估计器拟合**。共享配方会重复，但均实际执行，合计17种不同配方；不把重复配方当17个新设备。资格阶段另有八次不同拟合，其中原四次只复用一次，新端到端检查四次。没有训练基础大模型。

**成绩存在明显上限：34/36 个候选在最终观测上满分，另外两个都只错一条；开发33/36满分，训练全部满分。** 四个选择使用正则化 LDA，另两个为 SVC、ExtraTrees。Agent 在并列时使用较简单模型、已有验证一致性或预先确定的候选顺序，并明确承认采集身份限制。即时反馈没有造成更大的开发乐观偏差；同时，几乎所有候选都满分使这次对照很难检出反馈效应，不能把零差值外推为任何工况都不会过拟合。

选择封存后，对全部候选进行 **72 次冷进程推理**，每个候选各一次开发和最终推理。推理进程没有训练／开发标签或其他案例；输出逐ID对齐。108项训练／CV／开发指标独立重算通过，开发冷预测与实际训练返回逐值相同。65份实际服务响应通过反馈时机审计，延迟条件在六候选确定前未返回开发指标。命令轨迹及工作区扫描未发现额外估计器拟合或另存模型；部署只能使用服务封存的真实权重。所有原输入、模型及选择的哈希在评分前后相同。

执行成本没有隐去：首批另有六次基础设施无效启动，共210次请求、正式候选0，原因是原生命令沙箱拒绝 Unix socket。四次返回了无法执行的报告，两次由主控终止；两条被中断请求的消费未被假定为零。修复为每案例文件请求／只读响应，并通过真实原生工具触发训练之后，才恢复相同数据和相同预算。故总原生启动12次、模型请求311次；有效科学比较使用恢复的六次。旧错误与恢复性工程检查均保留，详情见[协议与恢复记录](feedback282_protocol.md)。

**决策：停止在本次已开分 UoC 划分上增加种子、阈值或人为更难子集。** 这是反证与可辨别性不足的记录，不是新难题。此前电机、HUST、转子和寿命任务的有界选模损失仍存在，但训练／开发反馈复用的因果解释依然未确认。下一轮先核验具备真实采集身份且尚未开分的测量记录，并利用训练／开发信息检查候选之间是否有可比较的差异，再冻结同空间、同预算的反馈对照；不根据最终分数挑新参数。

证据：[数值结果](../results/feedback282/NUMERICAL_RESULTS282.json)、[判定](../results/feedback282/ADJUDICATION282.json)、[完整执行分母](../results/feedback282/EXECUTION_SUMMARY282.json)、[封存与反馈审计](../results/feedback282/CHOICES_SEALED282.json)、[当前状态及归档](../results/feedback282/current_run.json)。
'''
    portable=text.replace('(../results/feedback282/','(').replace('(feedback282.png)','(figures/feedback282.png)').replace('(feedback282_protocol.md)','(workspace_snapshot/docs/feedback282_protocol.md)').replace('(current_run.json)','(ARCHIVE_VERIFIED.json)')
    (root/'REPORT282.md').write_text(portable)
    (PROJECT/'docs/feedback282_results.md').write_text(text)
    for name in ['EXECUTION_SUMMARY282.json','ADJUDICATION282.json']:shutil.copy2(root/name,PROJECT/'results/feedback282'/name)
    shutil.copy2(root/'figures/feedback282.png',PROJECT/'results/feedback282/feedback282.png')
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps({'requests':summary['requests'],'normal_completions':summary['normal_completions'],'new_major_losses':decision['major_available_candidate_losses']}))


def report(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    if config.get('iteration')=='A282':
        return report_legacy_uoc(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result=json.loads((root/'NUMERICAL_RESULTS282.json').read_text())
    audit=json.loads((root/'EXECUTION_AUDIT.json').read_text())
    target=PROJECT/config['result_directory'];tag=config['iteration'].removeprefix('A')
    rows=result['cases'];all_candidates=[c for row in rows for c in row['candidates']]
    effect=result['immediate_minus_delayed']
    signal=bool(effect['mean_final_macro_f1']<=-0.05 and effect['mean_gap']>=0.05) if effect else None
    decision=dict(utc=utc(),goal_status='active',predeclared_descriptive_feedback_signal=signal,
        immediate_minus_delayed=effect,major_available_candidate_losses=sum(x['major_available_candidate_loss'] for x in rows),
        complete_selections=sum(x['delivered'] for x in rows),planned_sessions=len(rows),
        candidate_attempts=sum(x['candidates_attempted'] for x in audit['cases']),complete_candidates=len(all_candidates),
        unique_recipes=len({json.dumps(x['recipe'],sort_keys=True) for x in all_candidates}),
        perfect_final_candidates=sum(x['final']['accuracy']==1 for x in all_candidates),
        perfect_development_imperfect_final_candidates=sum(x['development']['accuracy']==1 and x['final']['accuracy']<1 for x in all_candidates),
        repeated_feedback_causality_confirmed=False,agent_specific_mechanism_confirmed=False,
        population_failure_rate_established=False,scoped_challenge_confirmed=False,
        decision='Preserve all outcomes; do not refit or add seeds on this opened allocation. Interpret trajectory evidence and alternatives before any stronger mechanism claim.',
        limitations=config['limitations'])
    save(root/'ADJUDICATION.json',decision);save(target/'ADJUDICATION.json',decision)
    matrix=np.full((len(rows),config['candidate_fits_per_case']),np.nan)
    for i,row in enumerate(rows):
        for c in row['candidates']:matrix[i,int(c['id'].split('_')[-1])-1]=c['final']['macro_f1']
    fig,ax=plt.subplots(figsize=(13,5))
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i,row in enumerate(rows):
        for c in row['candidates']:
            j=int(c['id'].split('_')[-1])-1;chosen=bool(row['selection'] and c['id']==row['selection']['candidate_id'])
            ax.text(j,i,f'{matrix[i,j]:.2f}'+('*' if chosen else ''),ha='center',va='center',fontsize=8,
                    color='white' if matrix[i,j]>.6 else 'black',fontweight='bold' if chosen else 'normal')
    ax.set_xticks(range(matrix.shape[1]),[str(i+1) for i in range(matrix.shape[1])]);ax.set_yticks(range(len(rows)),[x['case'] for x in rows])
    ax.set_title(config['dataset']+': final macro-F1 of every actual candidate')
    ax.set_xlabel('Candidate fitting order; * = sealed Agent choice. Shared physical allocation, not independent devices.')
    fig.colorbar(im,ax=ax,label='Final macro-F1');fig.tight_layout();(root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures'/('feedback'+tag+'.png'),dpi=180);fig.savefig(root/'figures'/('feedback'+tag+'.svg'));plt.close(fig)
    def fmt(value):return '未交付' if value is None else f'{value:.4f}'
    table=[]
    selector_table=[]
    for row in rows:
        c=row['chosen'];best=next((x for x in row['candidates'] if x['id']==row['posthoc_available_candidate_best']),None)
        if c:
            table.append('| '+row['case']+' | '+c['recipe']['model']+' / '+c['recipe']['features']+' | '+fmt(c['train']['macro_f1'])+' | '+fmt(c['cv']['macro_f1'])+' | '+fmt(c['development']['macro_f1'])+' | '+fmt(c['final']['macro_f1'])+' | '+str(c['final']['correct'])+'/'+str(c['final']['total'])+' | '+fmt(row['macro_f1_regret'])+' |')
        else:table.append('| '+row['case']+' | 未完成选择 | — | — | — | — | — | — |')
        selectors=row['fixed_selectors_on_same_actual_pool']
        selector_table.append('| '+row['case']+' | '+str(c['final']['correct'] if c else '未交付')+' | '+
            str(selectors['cv']['final']['correct'] if selectors['cv'] else '—')+' | '+
            str(selectors['development']['final']['correct'] if selectors['development'] else '—')+' |')
    lead=('至少一组交付不完整，暂不能进行预定两组平均效应比较。' if signal is None else
          '达到预登记的描述性反馈效应信号，仍需检查轨迹和竞争解释。' if signal else
          '未达到预登记的描述性反馈效应信号；不能把这轮结果写成开发反馈导致过拟合的确认。')
    if rows and all(x['delivered'] and x['chosen']['final']['accuracy']==1 for x in rows):
        lead='全部六个封存选择在两个最终工况均满分，本轮得到反证：未复现 Agent 选模过拟合。'
    text=f'''# {config['iteration']}：实测行星齿轮反馈时机对照

**{lead}** 六次会话共享一枚齿轮的同一工况划分，不能当六套独立设备。持续目标 active。

| 会话 | 选择 | 训练 F1 | 整工况 CV F1 | 开发 F1 | 最终 F1 | 最终正确窗 | 实际候选 F1 遗憾 |
|---|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+f'''

即时减延迟：平均最终 macro-F1 **{fmt(effect.get('mean_final_macro_f1'))}**，平均 accuracy **{fmt(effect.get('mean_final_accuracy'))}**，平均开发乐观落差 **{fmt(effect.get('mean_gap'))}**。本轮完成选择 {decision['complete_selections']}/{len(rows)}，重大实际候选选择损失 {decision['major_available_candidate_losses']} 条。上述是此划分上策略轨迹的描述，不是总体发生率、窗口独立显著性或 Agent 特有因果结论。

并非全部候选均容易成功：{decision['perfect_final_candidates']}/{len(all_candidates)} 个实际候选最终满分，{decision['perfect_development_imperfect_final_candidates']} 个候选开发满分而最终未满分。全部候选最终 F1 跨度为 {min(c['final']['macro_f1'] for c in all_candidates):.4f}–{max(c['final']['macro_f1'] for c in all_candidates):.4f}。各会话使用整工况 CV 排除了许多训练／开发分高却跨工况较差的候选。候选之间存在实质差异，不能仅用“所有模型都有成绩天花板”解释这一反证；已选模型满分仍限制小幅差异的辨别能力。

| 会话 | Agent 正确窗／408 | 固定 CV 选择／408 | 固定开发选择／408 |
|---|---:|---:|---:|
'''+ '\n'.join(selector_table)+f'''

固定选择器均按预登记规则最大化该验证 F1、再比较 accuracy、仍并列取最早候选；开发列的失败受这个并列规则影响，不能说任何开发选择策略都会失败。固定 CV 选择器同样全部正确，说明本范围无需 Agent 特有机制也能利用现有验证证据选好模型。各会话候选集合不同，候选数量不当作独立物理设备或总体失败率。

![全部实际候选与封存选择](../{config['result_directory']}/feedback{tag}.png)

实际执行：{audit['normal_completions']}/{len(rows)} 次正常结束，{audit['requests']} 次基础模型请求；{decision['complete_candidates']}/{decision['candidate_attempts']} 个候选完成，{audit['actual_formal_fit_started']} 次真实估计器拟合启动、{audit['actual_formal_fit_completed']} 次完成。另有四次训练内原生工具资格拟合，前轮24次源准入拟合只复用证据，不重复计数。没有训练基础大模型，没有模拟信号、预测或分数。

本轮实际输入{sum(c['input_tokens'] for c in audit['cases']):,}、输出{sum(c['output_tokens'] for c in audit['cases']):,} token；逐会话HTTP状态和实际返回模型保存在执行审计中。共{audit['unique_real_recipes']}种不同配方，{len(audit['identical_recipe_prediction_checks'])}份重复配方的训练／CV／开发预测逐值相同；重复执行保留作成本和流程重复性，不算新设备。

全部选择与候选先封存，再完成 {result['cold_processes']} 次开发／最终独立冷进程推理，逐 ID 对齐；{len(result['independent_metric_checks'])} 组指标独立复算。完整训练／CV／开发证据的 {audit['validation_check_count']} 项检查通过；延迟组的开发行级证据按预定时机释放。固定 CV／开发选择器及每个候选的各工况混淆矩阵保留在数值文件中；这些选择器只比较本会话实际候选，不是完整空间穷举。

数据为 PHMForge 关联原源的研究适配：G1 的 P0/P3/P7 原始裂纹阶段，612／204／408 个非重叠两秒窗。训练为9.8／9.4／9.0 V，开发为9.6 V，最终为9.2／8.8 V；电压是控制设置，不冒充精确转速。载体传感器原始单位mV，名义38400Hz。一个物理齿轮、三份阶段录波跨条件分配，阶段还与采集日期及复装混杂，不能声称跨设备、跨录波或裂纹因果识别。先期可解性筛选只看训练／开发，研究者当时未见最终模型结果。

实验仅改变开发反馈时机，候选空间、真实拟合预算、训练CV和最终选模权一致；两组都能读取与重算已允许的逐样本预测和工况证据。受限配方搜索、三轨迹／组、同一划分及公共数据污染风险限制外推。低最终成绩可能来自工况外推或开发覆盖不足，不能单凭训练高分归因 Agent。

准入工程错误全部保留：MCP注册键规范化，隔离探针路径错误及重启缓存冲突均发生于0正式模型请求、0正式候选拟合；恢复复用已通过的四次真实资格拟合，不把基础设施固定命令响应当模型输出。实际模型请求、命令错误和完整状态见执行审计。轨迹扫描只能提供人工复核线索，不能声称形式化证明不存在额外计算。

被测Agent有五次分析命令错误，均随后恢复完成。两次把CV各折合并后应有的样本复用错误地当作泄漏／重复（immediate_1的item_38、delayed_3的item_48），随后各自修正检查（item_39／item_49），没有据此淘汰有效模型。原报告中个别文字把9.2 V也称为外推；综合解释更正为：9.2 V在训练电压范围内，8.8 V才位于范围之外。原轨迹不改写。

决策：不在本次已开分划分上追加种子、候选或阈值。逐条解释自然选模链、固定选择器表现与反证，研究结论服从本轮真实证据。

证据：[完整数值](../{config['result_directory']}/NUMERICAL_RESULTS282.json)、[真实拟合与执行审计](../{config['result_directory']}/EXECUTION_AUDIT.json)、[判定](../{config['result_directory']}/ADJUDICATION.json)、[冻结协议](feedback{tag}_protocol.md)、[状态及归档](../{config['result_directory']}/current_run.json)。
'''
    (PROJECT/'docs'/('feedback'+tag+'_results.md')).write_text(text)
    portable=text.replace('../'+config['result_directory']+'/','').replace('(feedback'+tag+'.png)','(figures/feedback'+tag+'.png)')
    portable=portable.replace('(feedback'+tag+'_protocol.md)','(workspace_snapshot/docs/feedback'+tag+'_protocol.md)').replace('(current_run.json)','(ARCHIVE_VERIFIED.json)')
    (root/('REPORT'+tag+'.md')).write_text(portable)
    shutil.copy2(root/'figures'/('feedback'+tag+'.png'),target/('feedback'+tag+'.png'))
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(requests=audit['requests'],complete_selections=decision['complete_selections'],major_losses=decision['major_available_candidate_losses'],feedback_signal=signal)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run.resolve())
