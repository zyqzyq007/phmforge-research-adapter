"""Read-only comparison of visible validation evidence and actual PHM choices.

All outcomes predate this audit. Metrics and scope are mechanically extracted;
the program does not score hypotheses or infer a causal feedback effect.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from prepare_verified_benchmarks import extract_bound

PROJECT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Sources:
    def __init__(self,root):
        self.root=root;self.contexts={};self.bindings=[];self.cache={}

    def context(self,name):
        if name not in self.contexts:
            pointer=json.loads((PROJECT/'results'/name/'current_run.json').read_text())
            archive=Path(pointer.get('archive',pointer['run']))
            receipt=json.loads((archive/'ARCHIVE_VERIFIED.json').read_text())
            manifest=archive/receipt['manifest_filename']
            assert sha(manifest)==receipt['manifest_sha256']
            self.contexts[name]=(archive,receipt,{x['path']:x for x in json.loads(manifest.read_text())['files']})
        return self.contexts[name]

    def record(self,name,logical,source):
        archive,receipt,manifest=self.context(name);row=manifest[logical]
        assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
        destination=self.root/'inputs'/name/logical
        if source!=destination:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        self.bindings.append(dict(campaign=name,archive=str(archive),logical_path=logical,bytes=row['bytes'],
            sha256=row['sha256'],archive_manifest_sha256=receipt['manifest_sha256'],audit_copy=str(destination.relative_to(self.root))))
        return destination

    def project(self,name,filename):
        key=(name,filename)
        if key not in self.cache:
            archive,receipt,manifest=self.context(name)
            source=PROJECT/'results'/name/filename;digest=sha(source)
            matches=[k for k,v in manifest.items() if Path(k).name==filename and v['sha256']==digest]
            assert matches,(name,filename,'No matching immutable archive bytes')
            preferred='workspace_snapshot/results/'+name+'/'+filename
            logical=preferred if preferred in matches else min(matches,key=len)
            self.cache[key]=json.loads(self.record(name,logical,source).read_text())
        return self.cache[key]

    def extras(self,name,paths):
        archive,receipt,manifest=self.context(name)
        missing=[path for path in paths if not (archive/path).is_file()]
        if missing:
            extract_bound(archive,self.root/'restored'/name,missing)
        for logical in paths:
            source=archive/logical
            if not source.is_file():source=self.root/'restored'/name/logical
            self.cache[(name,logical)]=json.loads(self.record(name,logical,source).read_text())

    def extra(self,name,path):
        return self.cache[(name,path)]


def one(rows,**fields):
    matches=[row for row in rows if all(row.get(k)==v for k,v in fields.items())]
    assert len(matches)==1,(fields,len(matches))
    return matches[0]


def comparison(ident,task,metric,selected,alternative,cv_scope,cv_comparable=True,notes=None):
    """Use normalized utility only for dominance signs, never pooled effects."""
    sign=1 if metric=='macro_f1' else -1
    complete=cv_comparable and all(side.get('cv') is not None for side in [selected,alternative])
    delta={part:(selected[part]-alternative[part]) for part in ['cv','development','final']
           if selected.get(part) is not None and alternative.get(part) is not None}
    signs={part:sign*v for part,v in delta.items()}
    relation='incomparable_or_missing_cv'
    if complete:
        if signs['cv']>=-1e-12 and signs['development']>=-1e-12:
            relation='selected_no_worse_on_both_visible_aggregates'
        elif signs['cv']<=1e-12 and signs['development']<=1e-12:
            relation='alternative_no_worse_on_both_visible_aggregates'
        else:relation='visible_validation_tradeoff'
    return dict(id=ident,task=task,metric=metric,higher_is_better=sign==1,selected=selected,alternative=alternative,
                raw_selected_minus_alternative=delta,visible_relation=relation,cv_scope=cv_scope,
                cv_comparison_qualified=bool(complete),final_selection_loss=signs.get('final',0)<-1e-12,notes=notes or [])


def main(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());assert config['new_fits']==config['new_LLM_calls']==0
    s=Sources(root)
    motor=s.project('selection270','ADJUDICATION270.json')
    m270=s.project('selection270','NUMERICAL_RESULTS270.json')
    hust=s.project('validation274','NUMERICAL_RESULTS274.json')
    hprobe=s.project('validation274','d_PUBLIC_CHOICE_PROBE274.json')
    s.project('validation274','d_review_PUBLIC_CHOICE_PROBE274.json')
    s.project('validation274','c_PUBLIC_COMPONENT_PROBE274.json')
    s.project('validation274','b_GATE_SUPPORT_DETAIL274.json')
    rotor=s.project('pipeline276','VALIDATION_MECHANISM276.json')
    r276=s.project('pipeline276','NUMERICAL_RESULTS276.json')
    rul=s.project('feedback278','NUMERICAL_RESULTS278.json')
    s.project('feedback278','ADJUDICATION278.json')
    timing={name:s.project(name,'NUMERICAL_RESULTS282.json') for name in ['feedback282','feedback285']}
    reviews=s.project('evidence284','NUMERICAL_RESULTS281.json')
    motor_files=['a/workspaces/s270_motor_a_initial/validation.json',
                 'a_review/workspaces/s270_motor_a_review/validation.json',
                 'a_review/workspaces/s270_motor_a_review/.runtime/repeated_group_cv_weight_audit.json']
    s.extras('selection270',motor_files)
    s.extras('feedback278',['b/workspaces/s278_adaptive_r1/validation.json'])
    save(root/'SOURCE_BINDINGS286.json',dict(utc=utc(),files=s.bindings,new_training=0,new_inference=0))
    # A small inventory keeps all original trajectories and both timing arms.
    inventory=[]
    for campaign,d in [('selection270',m270),('validation274',hust),('pipeline276',r276)]:
        for case in d['cases']:
            stage=list(case['stages'])[-1];v=case['stages'][stage]
            inventory.append(dict(campaign=campaign,case=case['case'],stages=list(case['stages']),
                final=v['final']['metrics'],status=v['status'],unit_count=case.get('test_groups'),
                unit_definition='source recording, not certified independent physical hardware'))
    for case in rul['cohorts']:
        variant=next(v for v in rul['variants'] if v['variant']['id']==case['final_variant'])
        inventory.append(dict(campaign='feedback278',case=case['cohort'],stages=['initial'],
            final=variant['results']['final']['metrics'],status=case['status'],unit_count=3,unit_definition='three shared final bearings'))
    for name,d in timing.items():
        for case in d['cases']:
            inventory.append(dict(campaign=name,case=case['case'],stages=['initial'],
                final=case['chosen']['final'] if case['chosen'] else None,status=case['status'],
                unit_count=None,unit_definition='shared allocation; trajectories are not new physical devices'))
    save(root/'INVENTORY286.json',dict(utc=utc(),original_trajectories=inventory,count=len(inventory),
        followup_reviews=[dict(case=c['case'],selected=c['selected'],outcome=c['outcome']) for c in reviews['cases']],
        scope='Chosen campaigns include all their trajectories. Not a representative frequency denominator.'))
    # Inspect exact schemas before constructing cross-program comparisons.
    schema=dict(motor_initial=s.extra('selection270',motor_files[0]),motor_review=s.extra('selection270',motor_files[1]),
        motor_repeated=s.extra('selection270',motor_files[2]),hust_probe=hprobe,
        rul_validation=s.extra('feedback278','b/workspaces/s278_adaptive_r1/validation.json'))
    save(root/'audit/BOUND_SCHEMA286.json',schema)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(bound_files=len(s.bindings),trajectories=len(inventory),schemas_ready=True,new_fits=0,new_model_calls=0)))


def compare(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    binding=json.loads((root/'SOURCE_BINDINGS286.json').read_text())
    for row in binding['files']:assert sha(root/row['audit_copy'])==row['sha256']
    def load(campaign,filename):
        paths=[p for p in (root/'inputs'/campaign).rglob(filename) if p.name==filename]
        assert len(paths)==1,(campaign,filename,paths)
        return json.loads(paths[0].read_text())
    extra=Sources(root);extra.bindings=binding['files'][:]
    extra.extras('feedback278',['references/candidates/manifest.json','audit/PREFIX_BOUNDARY278.json','CAMPAIGN_CONFIG.json'])
    extra.extras('feedback282',['CAMPAIGN_CONFIG.json'])
    extra.extras('feedback285',['CAMPAIGN_CONFIG.json'])
    review=extra.project('review281','NUMERICAL_RESULTS281.json')
    extra.bindings=list({(x['campaign'],x['logical_path']):x for x in extra.bindings}.values())
    save(root/'SOURCE_BINDINGS286.json',dict(utc=utc(),files=extra.bindings,new_training=0,new_inference=0))
    schema=json.loads((root/'audit/BOUND_SCHEMA286.json').read_text())
    motor=load('selection270','ADJUDICATION270.json');m270=load('selection270','NUMERICAL_RESULTS270.json')
    rotor=load('pipeline276','VALIDATION_MECHANISM276.json');r276=load('pipeline276','NUMERICAL_RESULTS276.json')
    hust=load('validation274','NUMERICAL_RESULTS274.json');rul=load('feedback278','NUMERICAL_RESULTS278.json')
    pairs=[]
    ma=one(motor['cases'],case='motor_a');svc=one(ma['all_prespecified_bank_candidates'],name='svc_c1')
    mc=one(m270['cases'],case='motor_a');selected=mc['stages']['review']['final']['metrics']
    pairs.append(comparison('motor_a','ElectricMotorVibrations','macro_f1',
        dict(name='retained_probability_ensemble',cv=schema['motor_review']['training_group_cv']['reference_folds']['models']['ensemble']['macro_f1'],
             development=schema['motor_review']['development']['metrics']['macro_f1'],final=selected['macro_f1']),
        dict(name='provided_svc_c1',cv=svc['training_group_cv']['macro_f1'],development=svc['development']['macro_f1'],final=svc['test']['macro_f1']),
        'Same supplied three folds of whole source records. Repeated regrouping reuses the same18 records.',notes=['Actual candidate available before selection; test errors concentrate in original label1.']))
    hc=one(hust['cases'],case='hust_geometry_r2');hp=schema['hust_probe']['public_scores']['val']
    pairs.append(comparison('hust_d','HUST','macro_f1',
        dict(name='chosen_loadfamily42',cv=None,development=hp['chosen_loadfamily42']['samples']['macro_f1'],final=hc['stages']['review']['final']['metrics']['macro_f1']),
        dict(name='prior_family57',cv=None,development=hp['prior_family57']['samples']['macro_f1'],final=one(hc['registered_agent_candidates'],candidate='prior_family57')['metrics']['macro_f1']),
        'Selected pipeline filters training specifications and changes normalization; supplied/report CV is not a verified matched score for this complete pipeline.',False,
        ['Development ranking reversal is real. Missing matched CV is retained, not filled with a baseline or another recipe.']))
    rotor_warnings=[]
    for case,ident,component in [('ordinary_r1','rotor_a','subset_2'),('ordinary_r2','rotor_c','component_et_base_sqrt')]:
        v=one(rotor['cases'],case=case);c=one(r276['cases'],case=case)
        cv=v['public_phases']['training_oof'];dev=v['public_phases']['development']
        pairs.append(comparison(ident,'RotorBrokenBar','macro_f1',
            dict(name='retained_ensemble',cv=cv['complete_metrics']['macro_f1'],development=dev['complete_metrics']['macro_f1'],final=c['stages']['initial']['final']['metrics']['macro_f1']),
            dict(name=component,cv=cv['base_metrics']['macro_f1'],development=dev['base_metrics']['macro_f1'],final=one(c['registered_agent_candidates'],candidate=component)['metrics']['macro_f1']),
            'Saved OOF probability aggregation for the actual full ensemble and component on identical complete-load folds.',
            notes=['A284 restored original pre-final evidence; the original Agent had it and the corrected review retained its selection.']))
        groups=[]
        for row in cv['by_condition']:
            groups.append(dict(row,selected_minus_alternative_correct=row['full_correct']-row['base_correct'],
                               selected_minus_alternative_recall=(row['full_correct']-row['base_correct'])/row['n']))
        rotor_warnings.append(dict(case=ident,rows=groups,regressing_condition_class_cells=sum(g['selected_minus_alternative_correct']<0 for g in groups),
            worst_recall_change=min(g['selected_minus_alternative_recall'] for g in groups),
            development_harmed=dev['harmed'],development_rescued=dev['rescued'],
            interpretation='Visible conditional tradeoff, not a prospectively validated predictor of which model wins on unknown loads.'))
    rv=schema['rul_validation'];bank=extra.extra('feedback278','references/candidates/manifest.json');knn=one(bank['models'],name='knn5')
    assert knn['training']['mean_bearing_mae_minutes']<rv['overall_metrics_minutes']['fitting_in_sample']['mean_bearing_mae_minutes']
    variant=lambda name:next(v for v in rul['variants'] if v['variant']['id']==name)
    bfinal=variant('agent_b')['results']['final']['metrics'];kfinal=variant('bank_knn5')['results']['final']['metrics']
    pairs.append(comparison('xjtu_b','XJTU','mean_bearing_mae_minutes',
        dict(name='selected_tree_mlp_50_50',cv=rv['overall_metrics_minutes']['whole_bearing_cv']['mean_bearing_mae_minutes'],
             development=variant('agent_b')['results']['development']['metrics']['mean_bearing_mae_minutes'],final=bfinal['mean_bearing_mae_minutes']),
        dict(name='actually_checked_knn5',cv=knn['training_cv']['mean_bearing_mae_minutes'],development=knn['development']['mean_bearing_mae_minutes'],final=kfinal['mean_bearing_mae_minutes']),
        'Same three held-out whole-bearing folds; primary loss weights each bearing equally. Development has one point/bearing, final has five.',
        notes=['KNN training fitting MAE is lower than the selected combination; literal highest-training-performance preference does not explain this choice.']))
    # Algebraic consistency checks do not execute any estimator.
    metric_checks=[]
    def check_metric(metric,label):
        cm=metric.get('confusion_matrix',metric.get('confusion'))
        if cm:
            n=len(cm);actual=[sum(row) for row in cm];pred=[sum(cm[i][j] for i in range(n)) for j in range(n)]
            f1=sum(2*cm[i][i]/(actual[i]+pred[i]) if actual[i]+pred[i] else 0 for i in range(n))/n
            acc=sum(cm[i][i] for i in range(n))/sum(actual)
            assert abs(f1-metric['macro_f1'])<1e-12 and abs(acc-metric['accuracy'])<1e-12
            metric_checks.append(dict(source=label,passed=True))
    check_metric(schema['motor_review']['training_group_cv']['reference_folds']['models']['ensemble'],'motor_selected_cv')
    for d,name in [(m270,'motor'),(hust,'hust'),(r276,'rotor')]:
        for c in d['cases']:
            for stage,v in c['stages'].items():check_metric(v['final']['metrics'],name+'/'+c['case']+'/'+stage)
    for v in rotor['cases']:
        for part,row in v['public_phases'].items():
            for kind in ['base_metrics','complete_metrics']:check_metric(row[kind],v['case']+'/'+part+'/'+kind)
    for label,value in [('selected_cv',rv['overall_metrics_minutes']['whole_bearing_cv']),('knn_cv',knn['training_cv'])]:
        recomputed=sum(x['mae_minutes'] for x in value['per_bearing'])/len(value['per_bearing'])
        assert abs(recomputed-value['mean_bearing_mae_minutes'])<1e-9
        metric_checks.append(dict(source=label,passed=True))
    boundary=extra.extra('feedback278','audit/PREFIX_BOUNDARY278.json');times={x['sample_id']:x['current_time'] for x in boundary['rows'] if x['split']=='val'}
    devq=variant('bank_knn5')['results']['development']['metrics']['per_query'];finalq=kfinal['per_query']
    support=[dict(unit_id=x['unit_id'],sample_id=x['sample_id'],current_time=times[x['sample_id']],current_rul=x['actual_rul'],
        terminal_nominal_minute=times[x['sample_id']]+x['actual_rul'],conservative_earliest_nominal_time=0,
        upper_bound_rul_any_existing_development_prefix=times[x['sample_id']]+x['actual_rul']) for x in devq]
    cap=max(x['upper_bound_rul_any_existing_development_prefix'] for x in support)
    coverage=dict(development=support,actual_endpoint_rul_range=[min(x['actual_rul'] for x in devq),max(x['actual_rul'] for x in devq)],
        upper_bound_rul_from_any_earlier_existing_prefix=cap,final_rul_range=[min(x['actual_rul'] for x in finalq),max(x['actual_rul'] for x in finalq)],
        final_queries_above_development_trajectory_upper_bound=[dict(sample_id=x['sample_id'],unit_id=x['unit_id'],rul=x['actual_rul']) for x in finalq if x['actual_rul']>cap],
        interpretation='With only these same development bearings, more earlier-prefix queries cannot cover RUL above the derived lifetime upper bound. Training-CV does contain longer trajectories, so this does not establish total absence of usable evidence.')
    knn_units={x['unit_id']:x for x in knn['training_cv']['per_bearing']}
    cv_unit_contrasts=[]
    for row in rv['overall_metrics_minutes']['whole_bearing_cv']['per_bearing']:
        other=knn_units[row['unit_id']];assert row['queries']==other['queries']
        cv_unit_contrasts.append(dict(unit_id=row['unit_id'],queries=row['queries'],selected_mae=row['mae_minutes'],
            knn_mae=other['mae_minutes'],selected_minus_knn=row['mae_minutes']-other['mae_minutes']))
    cv_aggregation=dict(per_bearing=cv_unit_contrasts,
        equal_bearing_selected=rv['overall_metrics_minutes']['whole_bearing_cv']['mean_bearing_mae_minutes'],
        equal_bearing_knn=knn['training_cv']['mean_bearing_mae_minutes'],
        pooled_query_selected=rv['overall_metrics_minutes']['whole_bearing_cv']['pooled_mae_minutes'],
        pooled_query_knn=knn['training_cv']['pooled_mae_minutes'],
        training_fit_selected=rv['overall_metrics_minutes']['fitting_in_sample']['mean_bearing_mae_minutes'],
        training_fit_knn=knn['training']['mean_bearing_mae_minutes'],
        interpretation='Available CV summaries have opposite rankings under different query/bearing weights. The declared objective is equal-bearing MAE; following its lower value is not by itself a reasoning error.')
    ops=[dict(pair=x,original_operator=op,available_in_single_recipe_timing_probes=False,annotation_provenance='Researcher classification of the archived validation reports and reconstructed program definitions.')
         for x,op in [('motor_a','probability ensemble'),('hust_d','training-specification filter plus normalization'),('rotor_a','three-model weighted ensemble'),('rotor_c','six-model ensemble'),('xjtu_b','target transform plus tree/MLP combination')]]
    operator_scope=[]
    for name in ['feedback282','feedback285']:
        c=extra.extra(name,'CAMPAIGN_CONFIG.json');families=list(c['recipe_space']['models']);assert set(families)=={'rf','extra','svc','lda','knn','logistic'}
        operator_scope.append(dict(campaign=name,models=families,task_family='classification',candidate_space='single fixed-feature estimator; no ensemble, filter or target-transform program revision',
            final_results=load(name,'NUMERICAL_RESULTS282.json')['groups'],interpretation='Valid negative evidence for this restricted search; not a matched operator test for the five original selection losses.'))
    result=dict(utc=utc(),exploratory=True,researcher_previously_knew_all_final_results=True,pairs=pairs,
        visible_relations=dict(Counter(x['visible_relation'] for x in pairs)),rotor_conditional_tradeoffs=rotor_warnings,
        rul_development_support=coverage,rul_cv_aggregation=cv_aggregation,operator_coverage=dict(original_loss_pairs=ops,completed_timing_probes=operator_scope),
        metric_consistency_checks=metric_checks,followup_nonrotor_reviews=[dict(case=c['case'],selected=c['selected'],outcome=c['outcome']) for c in review['cases'] if not c['case'].startswith('rotor_')],
        new_fits=0,new_model_inference=0,new_LLM_calls=0,new_independent_cases=0,
        causal_feedback_effect_identified=False,agent_specificity_identified=False,population_frequency_estimated=False)
    save(root/'SELECTION_EVIDENCE286.json',result)
    save(PROJECT/config['result_directory']/'SELECTION_EVIDENCE286.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(pairs=len(pairs),visible_relations=result['visible_relations'],rul_development_upper_bound=cap,
         final_queries_above_bound=len(coverage['final_queries_above_development_trajectory_upper_bound']),new_fits=0,new_model_calls=0)))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());d=json.loads((root/'SELECTION_EVIDENCE286.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));r=one(d['pairs'],id='xjtu_b')
    x=[0,1,2];s=[r['selected'][k] for k in ['cv','development','final']];a=[r['alternative'][k] for k in ['cv','development','final']]
    axes[0].bar([v-.18 for v in x],s,width=.36,label='Agent B combination',color='#c55b50')
    axes[0].bar([v+.18 for v in x],a,width=.36,label='Actually checked KNN',color='#387fb3')
    for i in range(3):
        axes[0].text(i-.18,s[i]+8,f'{s[i]:.1f}',ha='center',fontsize=9);axes[0].text(i+.18,a[i]+8,f'{a[i]:.1f}',ha='center',fontsize=9)
    axes[0].set_xticks(x,['Whole-bearing CV','Development','Final']);axes[0].set_ylabel('Mean per-bearing MAE (minutes; lower is better)');axes[0].legend(fontsize=8)
    axes[0].set_title('Real validation ranking reversal')
    coverage=d['rul_development_support'];values=[coverage['actual_endpoint_rul_range'][1],coverage['upper_bound_rul_from_any_earlier_existing_prefix'],coverage['final_rul_range'][1]]
    axes[1].barh(range(3),values,color=['#438d91','#83b9b2','#c55b50'])
    for i,v in enumerate(values):axes[1].text(v+25,i,f'{v:g}',va='center')
    axes[1].set_yticks(range(3),['Observed dev queries','Any existing dev-prefix\n(conservative upper bound)','Largest final query'])
    axes[1].set_xlim(0,max(values)*1.18);axes[1].set_xlabel('RUL minutes');axes[1].set_title('Four final queries exceed dev-trajectory support')
    fig.suptitle('A286: retrospective evidence audit, no new fits or predictions',fontsize=12);fig.tight_layout()
    (root/'figures').mkdir(exist_ok=True);fig.savefig(root/'figures/selection286.png',dpi=180);fig.savefig(root/'figures/selection286.svg');plt.close(fig)
    rows=[]
    for p in d['pairs']:
        value=lambda side,key:'缺少匹配CV' if p[side].get(key) is None else f"{p[side][key]:.4f}"
        rows.append('| '+p['id']+' | '+('F1 ↑' if p['higher_is_better'] else 'MAE分钟 ↓')+' | '+value('selected','cv')+' / '+value('alternative','cv')+' | '+value('selected','development')+' / '+value('alternative','development')+' | '+value('selected','final')+' / '+value('alternative','final')+' |')
    text='''# A286：真实选择损失的验证依据与实验覆盖审计

**四个重大选择损失，在当时的CV与开发汇总指标上均支持Agent最终选择；第五个缺少可对应完整程序的匹配CV。** 这不支持统一归因为“只看训练分／没验证”。当前仍未识别反复使用开发反馈的因果作用。

本轮仅重算和审查已封存证据，**0新Agent、0拟合、0模型推理、0新物理样本**；所有最终结果此前已被研究者看见，属于后验探索。审计包含四批原实验的全部14条原始轨迹，以及A282/A285的12条控制轨迹，共26条；它们共享若干数据划分，不能作为总体发生率分母。A281/A284复核另作重复证据，不增设备数。

表内每格依次为“所选／当时实际可用的对照”。CV、开发和最终不能当成同一分布的重复测量。

| 真实损失对 | 指标方向 | CV：所选／对照 | 开发：所选／对照 | 最终：所选／对照 |
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Motor和Rotor的对照是当时已有实际拟合模型／组件，XJTU是实际检查过的KNN；HUST d的旧候选同时改变训练规格和归一化，CV范围不完全对应最终程序，保留缺失，不以另一模型的CV补值。这五行是已经核实的重大实际选择损失，不包含HUST c的研究者组件消融，也不把小幅变化追认为重大损失。

## 预警并不相同

Rotor A：所选集成CV总体更高，但在0.5 Nm／三根断条的训练留出记录上，正确数从68/74降至14/74，召回下降72.97个百分点。Rotor C与其实际ET对照也有三个条件×类别单元退步，但最大仅一个窗口，即1.35个百分点。两者开发集上均没有把原本正确的窗口改错，分别修回6／4窗；最终却净多错161／75窗。因此，大幅条件召回退步不是两例共同必要条件，“开发回归测试不退步”也不能认证未见负载的程序改善。A284补全证据后仍保留两方案，说明不是单纯遗漏文件；这仍不能证明原信息足以确定最终赢家。

XJTU B：所选训练拟合MAE18.43分钟，KNN约0.000004分钟，Agent并未挑训练误差最小者。合法CV按轴承等权时所选355.55优于KNN373.85；按全部查询汇总则所选755.74差于KNN730.27。两种汇总对长短寿命轨迹的权重不同；原目标明确采用轴承等权，遵从该主指标本身不是推理错误。逐轴承差值保存在数值文件中。

![寿命排序反转与开发支持范围](../results/selection286/selection286.png)

开发三个查询的RUL为100、82、88分钟，历史当前时刻为23、409、26分钟；由既定名义时间与RUL合同，三个终点为123、491、114分钟。即使把现有开发轴承所有更早的真实前缀都用于评估，其RUL也不可能超过保守上界491分钟。最终RUL范围4–2135分钟，有4/15查询超出该上界。反复查询同三台开发轴承不能补出这些长寿命样本。训练CV仍包含更长轨迹，因此不能称Agent完全没有相关信息；该观察支持代表性／汇总权重的竞争解释，不能单独证实反馈过拟合。

## 前两轮反证的适用边界

A282和A285的12次时机对照均选择了最终满分模型，是有效反证；A285并非全部候选满分。但两轮都是分类任务，只允许固定特征加单个普通估计器，禁止集成、训练数据筛选及空间外目标变换。上述五个自然损失却全部涉及这些操作：Motor概率集成、两个Rotor集成、XJTU目标变换与组合、HUST规格筛选与归一化。分类探针也不能直接回答RUL回归改动的反馈问题。

因此，不能用单模型菜单上的零效应，直接否定原有程序修改操作中的反馈机制；也不能把旧损失反过来当作该机制的因果证明。这是本研究对照覆盖范围的缺口，**不是新发现的Agent失败案例**。A285中“广义假说降级”的判断应限于已测设置；更宽的程序修改机制仍未充分检验。

## 下一项可区分的实验

在尚未开分的真实采集上，先真实训练并封存同一组基础预测器，再让两组Agent在相同、明确覆盖原失败操作的程序空间内搜索。首选可严格计数的概率集成：两组具有相同基础模型、权重／组合空间、候选及时间预算，只改变开发反馈能否影响后续组合生成；完成后两组都拥有相同范围的验证证据作最终选择。保留单模型、预先固定的训练CV选择器及普通复核作为控制。

这是机制对照设计，不是提出改进方法。若只有单模型结果或所有基础模型都不可解，不启动正式会话；若两组都选好、普通固定流程同样失败或仅验证范围解释差异，保留反证。不能重用已开分Motor／HUST／Rotor／XJTU／G1做新权重搜索。原源G2等候选须先核验真实采集身份、阶段语义及数据可得性；新颖性尚未证明。

本次可确认的仍是几份有界真实选模损失。训练过拟合、合法验证的排序反转、验证程序范围不对应、开发支持不足，以及反复反馈的因果作用应分别表述。

证据：[配对数值和范围](../results/selection286/SELECTION_EVIDENCE286.json)、[全部原始轨迹清单](../results/selection286/INVENTORY286.json)、[来源绑定](../results/selection286/SOURCE_BINDINGS286.json)、[协议](../results/selection286/PROTOCOL286.json)、[状态及归档](../results/selection286/current_run.json)。目标active。
'''
    (PROJECT/'docs/selection286_results.md').write_text(text)
    (root/'REPORT286.md').write_text(text.replace('../results/selection286/','').replace('(selection286.png)','(figures/selection286.png)').replace('(current_run.json)','(ARCHIVE_VERIFIED.json)').replace('(PROTOCOL286.json)','(CAMPAIGN_CONFIG.json)'))
    target=PROJECT/config['result_directory'];shutil.copy2(root/'figures/selection286.png',target/'selection286.png')
    for name in ['SOURCE_BINDINGS286.json','INVENTORY286.json']:shutil.copy2(root/name,target/name)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(report='docs/selection286_results.md',new_fits=0,new_model_calls=0,goal_status='active')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--phase',choices=['bind','compare','report'],default='bind')
    a=p.parse_args();{'bind':main,'compare':compare,'report':report}[a.phase](a.run.resolve())
