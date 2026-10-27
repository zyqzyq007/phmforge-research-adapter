#!/usr/bin/env python3
"""Export existing PHM experiment results; never fit, infer, or open unscored tests.

Scope: the real-computation audit and PHMForge research-adaptation line A227,
A238 onward. Every imported source is hashed; numeric cells retain JSON pointers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
import zipfile
from functools import lru_cache
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LIVE305 = Path('/srv/yyf-zyq/PHM_20261010/20261023T052522Z_rotor305')
SOURCES = {}
SUBJECTS, BASELINES, METRICS, COMPARISONS, BATCHES, SPLITS, FEEDBACK = [], [], [], [], [], [], []
TASKS = {
    'ELECTRICMOTORVIBRATIONS': ('电机故障分类', '实测派生', '完整录波组；匿名类别的物理名称尚未核定'),
    'CWRU': ('CWRU轴承分类', '实测／实测派生', '早期特征表身份有限；后期原生9文件已核对。以逐批资格与边界为准'),
    'MFPT': ('MFPT轴承分类', '实测', '发布录波组；文件数不等于独立物理轴承数'),
    'MENDELEY': ('Mendeley轴承分类', '实测', '完整一分钟录波；硬件原型可能重复'),
    'PADERBORN': ('Paderborn轴承分类', '实测', '按完整轴承及其全部工况分组'),
    'HUST': ('HUST轴承诊断', '实测', '故障×规格代理组；不保证各组是独立硬件'),
    'FEMTO': ('FEMTO轴承寿命', '实测', '6条学习寿命、11个测试前缀；10例标签一致子集另列'),
    'ROTORBROKENBAR': ('断条转子诊断', '实测', '完整采集记录和负载；多次采集不等于多台独立电机'),
    'XJTU': ('XJTU轴承寿命', '实测', '9/3/3台轴承；最终15个时点只来自3台轴承'),
    'XJTU_TERMINAL': ('XJTU终态损伤诊断', '实测', '15轴承最后测量、终态拆检标签；9/3/3划分，最终3例均outer；无逐时severity真值'),
    'UOC': ('UoC齿轮诊断', '实测派生', '原始采集/硬件身份缺失；558/189/189列划分'),
    'PLANETARY_G1': ('行星齿轮G1', '实测', '一枚齿轮、三阶段录波；612/204/408个窗口'),
    'PLANETARY_G2': ('行星齿轮G2', '实测', '一枚齿轮、六阶段录波；最终408窗口'),
    'GUIDED_WAVE': ('导波损伤定位', '实测', '同一原任务7个查询；不等于7台独立设备'),
    'CMAPSS_FD001': ('C-MAPSS FD001', '物理仿真', '80台拟合、20台开发、100台官方测试；训练是真实执行'),
    'CMAPSS_FD002': ('C-MAPSS FD002', '物理仿真', '官方仿真发动机轨迹；测试259台'),
    'CMAPSS_FD003': ('C-MAPSS FD003', '物理仿真', '官方仿真发动机轨迹；测试100台'),
    'CMAPSS_FD004': ('C-MAPSS FD004', '物理仿真', '官方仿真发动机轨迹；测试248台'),
    'ASSETOPS_CHILLER': ('AssetOps冷水机预测', '现场来源未核实', '公开telemetry fixture；数据条件选出的28点，仅用于环境核验'),
    'ENGINE_QA_HEALTH': ('Engine单周期组件标注', 'NASA仿真派生QA', '每原题1条600×33源记录，5个组件标注；18源记录不等于18台独立发动机，非物理因果认证'),
    'ENGINE_QA_TREND': ('Engine多周期状态／区间标注', 'NASA仿真派生QA', '每原题1个10源前缀元组；只评作者状态和区间标签，非校准物理失效概率'),
}


def norm_dataset(value):
    s = str(value).upper().replace('-', '_')
    if 'MENDELEY' in s:
        return 'MENDELEY'
    if 'FD00' in s:
        return 'CMAPSS_' + s[s.index('FD00'):s.index('FD00') + 5]
    return s


def source(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    raw = path.read_bytes()
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCES[key] = {
        'id': key, 'absolute_path': str(path), 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'verification': '本地脚本读取与数值一致性检查；非人工论文审稿签署',
    }
    return json.loads(raw) if path.suffix == '.json' else raw.decode('utf-8'), key


def ptr(key, pointer=''):
    return key + '#' + pointer


def scalar_metrics(m):
    names = {'accuracy', 'macro_f1', 'balanced_accuracy', 'mean_group_accuracy',
             'mean_recording_accuracy', 'mae', 'rmse', 'bias', 'challenge_score',
             'mean_bearing_mae_minutes', 'pooled_mae_minutes', 'pooled_rmse_minutes',
             'mean_bearing_rmse_minutes', 'median_error_mm', 'max_error_mm',
             'mean_error_mm', 'cases_within_20mm', 'train_F1', 'CV_F1',
             'development_F1', 'final_F1', 'final_accuracy'}
    names.update({'mae_cycles','rmse_cycles','bias_cycles','overestimation_mass_per_query_minutes','underestimation_mass_per_query_minutes',
        'component_correct','component_n','joint_correct','joint_n','condition_correct','condition_n',
        'probability_bin_correct','probability_bin_n','state_correct','state_n','timing_correct','timing_n'})
    return {k: v for k, v in m.items() if k in names and isinstance(v, (int, float))
            and not isinstance(v, bool)}


def n_samples(m):
    for k in ['n', 'total', 'queries', 'total_cases', 'final_total']:
        if isinstance(m.get(k), int):
            return m[k]
    cm = m.get('confusion_matrix', m.get('confusion'))
    return sum(sum(row) for row in cm) if cm else None


def n_groups(m):
    g = m.get('groups', m.get('bearings'))
    return len(g) if isinstance(g, list) else g if isinstance(g, int) else None


def value_for(m, name):
    if name == 'mae':
        return m.get('mean_bearing_mae_minutes', m.get('mae_cycles',m.get('mae')))
    if name == 'rmse':
        return m.get('pooled_rmse_minutes', m.get('rmse_cycles',m.get('rmse')))
    return m.get(name)


def unit_for(dataset):
    if dataset.startswith('ENGINE_QA_'):return '源QA标注计数'
    return '分钟' if dataset == 'XJTU' else '秒' if dataset == 'FEMTO' else \
        'cycles' if dataset.startswith('CMAPSS') else 'mm' if dataset == 'GUIDED_WAVE' else \
        '源字段单位未核实' if dataset == 'ASSETOPS_CHILLER' else '比例'


def record(batch, dataset, case, stage, status, formal, key, pointer, **kwargs):
    dataset = norm_dataset(dataset)
    title, origin, limit = TASKS.get(dataset, (dataset, '见原记录', '见原记录'))
    row = dict(id=f'{batch}:{case}:{stage}', batch=batch, dataset=dataset, task=title,
               origin=origin, case=case, stage=stage, status=status,
               formal_submission=formal, artifact_kind='正式提交' if formal else '未完成/需看制品状态',
               unit=unit_for(dataset), scope_limit=limit, evidence=ptr(key, pointer),
               model='deepseek-flash', cli='0.164.0' if int(batch[1:4]) >= 305 else '0.163.0',
               official_benchmark_score=False, notes='', metric_sources={})
    row.update(kwargs)
    return row


def attach(row, split, m, key, pointer):
    if not isinstance(m, dict):
        return
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'mae', 'rmse',
                   'median_error_mm', 'max_error_mm', 'cases_within_20mm']:
        v = value_for(m, metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            field = f'{split}_{metric}'
            row[field] = v
            rawkey = ('mean_bearing_mae_minutes' if metric == 'mae' and 'mean_bearing_mae_minutes' in m else
                      'mae_cycles' if metric == 'mae' and 'mae_cycles' in m else
                      'pooled_rmse_minutes' if metric == 'rmse' and 'pooled_rmse_minutes' in m else metric)
            if metric=='rmse' and 'rmse_cycles' in m:rawkey='rmse_cycles'
            row['metric_sources'][field] = ptr(key, pointer + '/' + rawkey)
    if n_samples(m) is not None:
        row[split + '_n'] = n_samples(m)
    if split == 'test' and n_groups(m) is not None:
        row['test_groups'] = n_groups(m)


def collect_metrics(obj, batch, key, pointer='', context=None):
    """Keep metric dictionaries, not confusion cells or individual predictions."""
    context = dict(context or {})
    if isinstance(obj, dict):
        for k in ['dataset', 'case', 'job', 'stage', 'candidate', 'name', 'family', 'split']:
            if isinstance(obj.get(k), (str, int)):
                context[k] = obj[k]
        sm = scalar_metrics(obj)
        if sm:
            for metric, value in sm.items():
                unit=obj.get('unit')
                if 'f1' in metric.lower() or 'accuracy' in metric or metric=='challenge_score':unit='比例'
                elif 'minutes' in metric:unit='分钟'
                elif metric.endswith('_mm'):unit='mm'
                elif metric=='cases_within_20mm':unit='例'
                elif not unit:
                    ds=norm_dataset(context.get('dataset',''))
                    if ds.startswith('CMAPSS') or batch=='A238':unit='cycles'
                    elif ds=='FEMTO' or batch=='A254':unit='秒'
                    elif batch=='A240' and pointer.startswith('/asset'):unit='源字段单位未核实'
                unit={'seconds':'秒','minutes':'分钟'}.get(unit,unit or '见原始数据定义')
                METRICS.append(dict(batch=batch, context=json.dumps(context, ensure_ascii=False),
                                    metric=metric, value=value, n=n_samples(obj),
                                    groups=n_groups(obj), unit=unit,
                                    evidence=ptr(key, pointer + '/' + metric)))
        for k, v in obj.items():
            if k in ['per_class', 'groups', 'per_query', 'per_bearing', 'detail_cases',
                     'confusion_matrix', 'confusion', 'submission_files', 'independent_metric_checks',
                     'evidence', 'compared_to_submissions', 'comparisons', 'difference', 'contrasts']:
                continue
            collect_metrics(v, batch, key, pointer + '/' + k, context)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            collect_metrics(v, batch, key, pointer + '/' + str(i), context)


def import_source(path, batch):
    x, key = source(path)
    collect_metrics(x, batch, key)
    return x, key


NOTES = {
    'A252': '早期探索：测试观测可读，CWRU/电机依赖发布次序及元数据；不视为严格未见工况确认。',
    'A254': '11例原标签与预先指定10例一致子集分别保存；r2预算止但制品可重放。',
    'A256': '整批组恢复假设；一例超时无完整提交，不把参考诊断分当Agent成绩。',
    'A259': '原生MCP配置不合格；两次Paderborn超时无完整提交。',
    'A260': 'MCP修复后重复基线；测试观测可见，研究者此前见过同划分分数。',
    'A262–263': '早期电机/CWRU有测试类别总量先验；FEMTO含标签歧义；不计严格盲测确认。',
    'A264': '旧种子已有类别总量先验，移除元数据不使旧选择成为新盲测；制品溯源混杂保留。',
    'A267': '已知来源探索；测试观测可见；FEMTO首轮部分制品，Paderborn首轮未完成。',
    'A268': '5来源6划分，部分测试组重叠；3阶段预算止、另3次基础设施中断；测试观测可见。',
    'A269': '4分配8阶段；测试观测可见；四次普通复核逐ID预测不变。',
    'A270': '两个相关分配；开发及复核不见测试观测/标签；固定CV和开发选择器也可能失败。',
    'A273': '高负载两阶段完成；低负载两阶段预算止，无完整提交；不把低分当完整决策失败。',
    'A274': '一个跨规格划分；4条初选/复核路径；组件消融与实际弃用候选严格区分。',
    'A276': '一个采集批次/负载划分，4条初选；此批没有配对复核；后续见A281/A284。',
    'A278': '一个9/3/3轴承划分；最终15时点来自3轴承；MAE按轴承平均，RMSE为汇总查询值。',
    'A281': '固定候选复核；0新拟合/最终推理，复用旧冷预测；曾遗漏部分历史验证文件。',
    'A282': '6有效会话；另6次基础设施无效启动单列；采集身份未知，34/36候选最终满分。',
    'A284': '补齐26份验证文件后的固定候选复核；0新拟合/最终推理；不等于开放重建无效。',
    'A285': '一枚G1齿轮；即时/延迟各3会话；固定整工况CV也能全部选对。',
    'A298': '一枚G2齿轮；6初选+6复核；真实基库24次fit，Agent阶段0新fit；固定空间存在能力限制。',
    'A305': '160条新采集、两个记录块；8会话及固定对照已评分。采集块不是独立电机；复核既可改善也可退化。',
    'A306': '四个新增诊断方向；Paderborn预算止、不计正常主体成绩。HUST批次分支变化与弱分类器竞争解释保留。',
    'A307': '五原题六尝试（含一次通信失败及剩余额度恢复）；四FD001共用100台，CWRU仅3最终原记录。',
    'A311': '已知CWRU失败上的一次普通复核；同记录结果选择后的探索，不是新原题或独立确认。',
    'A314': '三道安全题的实测诊断子任务；没有校准风险／合规／维护收益。转子窗口组件损失不导致多数票记录差异。',
    'A315': '四道真实RUL子任务。FEMTO开发非独立；XJTU最终15时点仅3轴承；无完整安全／维护动作合同。',
    'A318': '电机仅振动、XJTU仅终态诊断；缺失声学／逐时severity比较。合法但未出现在测试真值中的类别已按源字典修正评分，预测未改。',
    'A319': '原题001当前资格修复baseline；0新增方向。CWRU原生9录波已见来源，最终与参考持平，不是新物理来源确认。',
    'A312': '18初选＝4正常14超时；只有4正常程序外评，后14题一次补全见A317。',
    'A317': '14个部分程序一次有界补全均正常；0新增原题。参考权重可读但辅助helper遗漏，Root仅补评测导入；强工具完整性命题存在环境限制。',
    'A316': '10方向＝7正常、3预算未交付。正常程序及参考联合均0/7；弱可解性／源语义限制不当Agent特有难题。两程序验证上下文差异单列。',
    'A227': '同一任务7损伤查询、4复核；实测导波+数值算法，无需神经网络训练；局部退化非统一机制。',
    'A238': '真实拟合与评分，但C-MAPSS是物理仿真；4次复核无总体MAE/RMSE退化。',
}


def read_coding_rows(batch, path, pointer='/rows'):
    x, key = import_source(path, batch)
    rows = x['rows'] if pointer == '/rows' else x['femto']
    for i, r in enumerate(rows):
        dataset = norm_dataset(r['dataset']); p = f'{pointer}/{i}'
        formal = bool(r.get('formal_submission', r.get('response_valid', False)))
        out = record(batch, dataset, r['job'], '复核' if 'review' in r['job'] else '初选',
                     r['status'], formal, key, p, elapsed_seconds=r.get('elapsed_seconds'),
                     notes=NOTES[batch])
        final = r.get('final', r.get('results', {}).get('final', {}))
        finalpath = p + ('/final' if 'final' in r else '/results/final')
        reference = r.get('reference', r.get('results', {}).get('reference', {}))
        refpath = p + ('/reference' if 'reference' in r else '/results/reference')
        no_artifact = ((batch in ['A259', 'A260'] and dataset == 'PADERBORN' and not formal)
                       or (batch in ['A256', 'A262–263'] and dataset == 'HUST' and not formal))
        if no_artifact:
            out['artifact_kind'] = '无完整提交；诊断分不计主体成绩'
        elif not formal:
            out['artifact_kind'] = '可重放制品；会话未正常完成'
        if final.get('valid') and not no_artifact:
            if dataset == 'FEMTO':
                attach(out, 'test', final['all11'], key, finalpath + '/all11')
                attach(out, 'test_consistent10', final['consistent10'], key, finalpath + '/consistent10')
                out['test_groups'] = 11
            else:
                attach(out, 'test', final.get('metrics', {}), key, finalpath + '/metrics')
        if reference.get('valid'):
            attach(out, 'reference_test', reference.get('all11', reference.get('metrics', {})),
                   key, refpath + ('/all11' if dataset == 'FEMTO' else '/metrics'))
        SUBJECTS.append(out)


def import_common(batch, path):
    x, key = import_source(path, batch)
    for i, c in enumerate(x['cases']):
        for stage, r in c['stages'].items():
            p = f'/cases/{i}/stages/{stage}'
            out = record(batch, c['dataset'], c['case'], '初选' if stage == 'initial' else '复核',
                         r['status'], bool(r.get('formal_submission')), key, p,
                         job=r['job'], test_n=c.get('test_rows'), test_groups=c.get('test_groups'),
                         elapsed_seconds=r.get('elapsed_seconds'), notes=NOTES[batch])
            if not out['formal_submission']:
                out['artifact_kind'] = '未完成程序的数值诊断；非正式成绩'
            if r.get('final', {}).get('valid'):
                attach(out, 'test', r['final']['metrics'], key, p + '/final/metrics')
            if r.get('reference', {}).get('valid'):
                attach(out, 'reference_test', r['reference']['metrics'], key, p + '/reference/metrics')
            SUBJECTS.append(out)


def attach_public(batch, base):
    for path in sorted(base.glob('*/PUBLIC_SPLIT_SCORES.json')):
        x, key = source(path)
        for i, r in enumerate(x['rows']):
            matches = [s for s in SUBJECTS if s['batch'] == batch and
                       r['job'] in (s.get('job'), s['case'])]
            for s in matches:
                attach(s, 'dev' if r['split'] == 'val' else r['split'], r.get('metrics', {}),
                       key, f'/rows/{i}/metrics')


def import_special():
    x, key = import_source('results/realtrain238/RESULT_INTERPRETATION238.json', 'A238')
    for i, r in enumerate(x['rows']):
        out = record('A238', 'CMAPSS_FD001', r['job'], '复核', 'finished', True, key,
                     f'/rows/{i}', arm=r['condition'], test_groups=100, test_n=100, notes=NOTES['A238'])
        attach(out, 'test', r, key, f'/rows/{i}')
        attach(out, 'reference_test', x['baseline'], key, '/baseline')
        SUBJECTS.append(out)
    x, key = import_source('results/scopereview227/RESULT227.json', 'A227')
    for i, r in enumerate(x['rows']):
        out = record('A227', 'GUIDED_WAVE', r['job'], '复核', 'finished', True, key,
                     f'/rows/{i}', arm=r['arm'], test_n=7, elapsed_seconds=r['elapsed_seconds'], notes=NOTES['A227'])
        attach(out, 'test', r, key, f'/rows/{i}')
        SUBJECTS.append(out)

    x, key = import_source('results/validation267/PAIRED_RESULTS267.json', 'A267')
    for dataset, c in x['classification'].items():
        for stage, r in c['stages'].items():
            p = f'/classification/{dataset}/stages/{stage}'
            out = record('A267', dataset, r['job'], '初选' if stage == 'initial' else '复核',
                         r['status'], bool(r.get('formal_submission')), key, p,
                         elapsed_seconds=r.get('elapsed_seconds'), notes=NOTES['A267'])
            for typ, prefix in [('final', 'test'), ('reference', 'reference_test')]:
                if r.get(typ, {}).get('valid'):
                    attach(out, prefix, r[typ]['metrics'], key, p + f'/{typ}/metrics')
            if not out['formal_submission']:
                out['artifact_kind'] = '无完整提交/冷推理失败'
            SUBJECTS.append(out)
    for stage in ['initial', 'review']:
        r = x['rul'][stage]; p = '/rul/' + stage
        out = record('A267', 'FEMTO', r['job'], '初选' if stage == 'initial' else '复核',
                     r['status'], bool(r.get('formal_submission')), key, p,
                     elapsed_seconds=r.get('elapsed_seconds'), notes=NOTES['A267'])
        if not out['formal_submission']:
            out['artifact_kind'] = '可重放部分制品；缺报告/未正常完成'
        for typ, prefix in [('final', 'test'), ('reference', 'reference_test')]:
            v = r['results'][typ]
            if v.get('valid'):
                attach(out, prefix, v['all11'], key, p + f'/results/{typ}/all11')
                attach(out, prefix + '_consistent10', v['consistent10'], key, p + f'/results/{typ}/consistent10')
        SUBJECTS.append(out)

    x, key = import_source('results/feedback278/NUMERICAL_RESULTS278.json', 'A278')
    variants = {v['variant']['id']: (i, v) for i, v in enumerate(x['variants'])}
    for i, r in enumerate(x['cohorts']):
        j, v = variants[r['final_variant']]
        out = record('A278', 'XJTU', r['case'], '初选', r['status'], bool(r['response_valid']),
                     key, f'/cohorts/{i}', test_groups=3, test_n=15, arm=r['arm'],
                     selected_model=r['selected'], elapsed_seconds=r['elapsed_seconds'], notes=NOTES['A278'])
        for split, name in [('development', 'dev'), ('final', 'test')]:
            attach(out, name, v['results'][split]['metrics'], key, f'/variants/{j}/results/{split}/metrics')
        SUBJECTS.append(out)

    for batch, folder in [('A281', 'review281'), ('A284', 'evidence284')]:
        x, key = import_source(f'results/{folder}/NUMERICAL_RESULTS281.json', batch)
        for i, r in enumerate(x['cases']):
            dataset = 'XJTU' if r['family'] != 'classification' else r['dataset']
            out = record(batch, dataset, r['case'], '固定候选复核', r['status'], bool(r['complete']),
                         key, f'/cases/{i}', selected_model=r['selected'], notes=NOTES[batch])
            attach(out, 'test', r['reviewed'], key, f'/cases/{i}/reviewed')
            attach(out, 'reference_test', r['original'], key, f'/cases/{i}/original')
            SUBJECTS.append(out)
    for batch, folder, dataset in [('A282', 'feedback282', 'UOC'), ('A285', 'feedback285', 'PLANETARY_G1')]:
        x, key = import_source(f'results/{folder}/NUMERICAL_RESULTS282.json', batch)
        for i, r in enumerate(x['cases']):
            out = record(batch, dataset, r['case'], '选模', r['status'], bool(r['delivered']),
                         key, f'/cases/{i}', arm=r['condition'], notes=NOTES[batch],
                         selected_model=json.dumps(r['chosen']['recipe'], ensure_ascii=False))
            for split, name in [('train', 'train'), ('cv', 'cv'), ('development', 'dev'), ('final', 'test')]:
                attach(out, name, r['chosen'][split], key, f'/cases/{i}/chosen/{split}')
            SUBJECTS.append(out)
    x, key = import_source('results/ensemble298/NUMERICAL_SUMMARY298.json', 'A298')
    for i, r in enumerate(x['cases']):
        out = record('A298', 'PLANETARY_G2', r['case'], '初选' if r['stage'] == 'search' else '复核',
                     r['status'], bool(r['delivered']), key, f'/cases/{i}',
                     test_n=r['final_total'], selected_model=str(r['weights']), notes=NOTES['A298'])
        for field, raw in [('train_macro_f1', 'train_F1'), ('cv_macro_f1', 'CV_F1'),
                           ('dev_macro_f1', 'development_F1'), ('test_macro_f1', 'final_F1'),
                           ('test_accuracy', 'final_accuracy')]:
            out[field] = r[raw]; out['metric_sources'][field] = ptr(key, f'/cases/{i}/{raw}')
        SUBJECTS.append(out)


def import_latest():
    x, key = import_source('results/rotor305/NUMERICAL_RESULTS305.json', 'A305')
    assert len(x['rows']) == 8
    for i,r in enumerate(x['rows']):
        p=f'/rows/{i}'
        out = record('A305', 'ROTORBROKENBAR', r['cohort'], '复核' if r['stage']=='review' else '初选',
                     'finished', True, key, p, cohort=r['cohort'], test_n=r['n'],test_groups=r['records'],
                     scoring_status='已评分', notes=NOTES['A305'])
        attach(out,'test',r['metrics'],key,p+'/metrics')
        for field,raw in [('train_accuracy','train_accuracy'),('dev_accuracy','development_accuracy'),
                          ('dev_macro_f1','development_macro_f1'),('reference_test_accuracy','reference_accuracy')]:
            out[field]=r[raw];out['metric_sources'][field]=ptr(key,p+'/'+raw)
        SUBJECTS.append(out)

    for batch,folder in [('A318','diagnostic318'),('A319','cwrudiagnostic319')]:
        x,key=import_source(f'results/{folder}/NUMERICAL_RESULTS{batch[1:]}.json',batch)
        for i,r in enumerate(x['rows']):
            c=r['case'];p=f'/rows/{i}'
            out=record(batch,c['dataset'],c['cohort'],'当前资格修复baseline' if batch=='A319' else '初选',r['status'],r['formal_submission'],key,p,
                original_task_id=c['task_id'],notes=NOTES[batch])
            for part,name in [('development','dev'),('final','test'),('actual_starting_reference','reference_test')]:attach(out,name,r[part],key,p+'/'+part)
            SUBJECTS.append(out)
    state,_=source('results/rotor305/current_run.json')
    return state


def import_completed_campaigns():
    """Import only already opened, persisted result files; never run an evaluator."""
    x,key=import_source('results/coverage306/DETAILED_RESULTS306.json','A306')
    for i,r in enumerate(x['rows']):
        p=f'/rows/{i}';out=record('A306',r['dataset'],r['job'],'初选',r['status'],r['formal_submission'],key,p,
            original_task_id=r['task_id'],notes=NOTES['A306'])
        if r['formal_submission']:
            attach(out,'test',r.get('test',{}),key,p+'/test')
            attach(out,'dev',r.get('development',{}).get('metrics',{}),key,p+'/development/metrics')
            b=r['base_score']
            if b.get('reference',{}).get('valid'):attach(out,'reference_test',b['reference']['metrics'],key,p+'/base_score/reference/metrics')
        else:out['artifact_kind']='未正常交付；起始程序诊断不作Agent成绩'
        SUBJECTS.append(out)
    x,key=import_source('results/expansion307/NUMERICAL_RESULTS307.json','A307')
    for i,r in enumerate(x['rows']):
        p=f'/rows/{i}';dataset='CWRU' if r['task_id']=='pdm_cost_benefit_002' else 'CMAPSS_FD001'
        out=record('A307',dataset,r['job'],'恢复' if 'recovery' in r['job'] else '初选',r['status'],r['formal_submission'],key,p,
            original_task_id=r['task_id'],notes=NOTES['A307'])
        attach(out,'dev',r.get('development',{}),key,p+'/development')
        for part,name in [('final','test'),('reference','reference_test')]:
            v=r.get(part,{})
            if r['formal_submission'] and v.get('valid'):attach(out,name,v.get('metrics',{}),key,p+'/'+part+'/metrics')
        SUBJECTS.append(out)
    x,key=import_source('results/review311/RESULTS311.json','A311');r=x['score']
    out=record('A311','CWRU',r['job'],'复核',r['status'],r['formal_submission'],key,'/score',
        original_task_id=x['original_task_id'],notes=NOTES['A311'])
    for part,name in [('final','test'),('reference','reference_test')]:
        if r[part]['valid']:attach(out,name,r[part]['metrics'],key,'/score/'+part+'/metrics')
    SUBJECTS.append(out)
    x,key=import_source('results/safety314/NUMERICAL_RESULTS314.json','A314')
    for i,r in enumerate(x['rows']):
        p=f'/rows/{i}';out=record('A314',r['dataset'],r['cohort'],'初选','finished',r['normal_submission'],key,p,
            original_task_id=r['task_id'],notes=NOTES['A314'])
        for part,name in [('development','dev'),('final','test'),('actual_starting_reference','reference_test')]:attach(out,name,r[part],key,p+'/'+part)
        SUBJECTS.append(out)
    x,key=import_source('results/rulfollow315/NUMERICAL_RESULTS315.json','A315')
    for i,r in enumerate(x['rows']):
        c=r['case'];p=f'/rows/{i}';out=record('A315',c['dataset'],c['cohort'],'初选',r['status'],r['formal_submission'],key,p,
            original_task_id=c['task_id'],notes=NOTES['A315'],development_scope=r.get('development_scope'))
        for part,name in [('final','test'),('starting','reference_test')]:
            v=r['results'][part]
            if not v['valid']:continue
            m=v['metrics'];mp=p+'/results/'+part+'/metrics'
            if c['dataset']=='FEMTO':
                attach(out,name,m['all'],key,mp+'/all');attach(out,name+'_consistent10',m['consistent10'],key,mp+'/consistent10')
            else:attach(out,name,m,key,mp)
        SUBJECTS.append(out)


def build_splits():
    for batch, name in [('A268','selection268/PREPARATION268.json'),
                        ('A269','overfit269/PREPARATION269.json'),
                        ('A270','selection270/PREPARATION270.json'),
                        ('A274','validation274/PREPARATION274.json'),
                        ('A276','pipeline276/PREPARATION276.json'),
                        ('A305','rotor305/PREPARATION305.json')]:
        x,key=source('results/'+name)
        values=x.get('cases',x.get('blocks',[x] if 'rows' in x else []))
        for i,v in enumerate(values):
            if not isinstance(v,dict) or 'rows' not in v:continue
            case=v.get('case',v.get('block',str(i)))
            SPLITS.append(dict(batch=batch,case=case,dataset=v.get('dataset','ROTORBROKENBAR' if batch in ['A276','A305'] else 'HUST'),
                train_n=v['rows'].get('train'),dev_n=v['rows'].get('val'),test_n=v['rows'].get('test'),
                train_groups=v.get('groups',{}).get('train'),dev_groups=v.get('groups',{}).get('val'),test_groups=v.get('groups',{}).get('test'),
                group_unit='完整采集记录/来源组，非独立设备数',optimizer_steps=v.get('optimizer_steps',v.get('reference_optimizer_steps')),
                evidence=ptr(key,('/cases/'+str(i) if 'cases' in x else '/blocks/'+str(i) if 'blocks' in x else ''))))
            for r in SUBJECTS:
                if r['batch']==batch and (r['case']==case or (batch=='A305' and case in r['case'])):
                    r.update(train_n=v['rows'].get('train'),dev_n=v['rows'].get('val'),test_n=v['rows'].get('test'),
                             test_groups=v.get('groups',{}).get('test'))


def import_engine_campaigns():
    for num,folder in [('312','enginecoding312'),('317','enginecompletion317'),('316','enginetrendcoding316')]:
        batch='A'+num;x,key=import_source(f'results/{folder}/NUMERICAL_RESULTS{num}.json',batch)
        dev,dk=source(f'results/{folder}/DEVELOPMENT_AUDIT{num}.json')
        development={r['case']['cohort']:(i,r) for i,r in enumerate(dev['rows'])}
        for i,r in enumerate(x['rows']):
            c=r['case'];p=f'/rows/{i}';out=record(batch,c['dataset'],c['cohort'],'补全' if num=='317' else '初选',r['status'],r['formal_submission'],key,p,
                original_task_id=c['task_id'],notes=NOTES[batch],test_n=1 if r['formal_submission'] else None,test_groups=1 if r['formal_submission'] else None)
            if not r['formal_submission']:out['artifact_kind']='预算未正常交付；不把起始程序当Agent分数'
            di,dr=development[c['cohort']]
            if dr['normal'] and dr['development'].get('passed'):attach(out,'dev',dr['development']['metrics'],dk,f'/rows/{di}/development/metrics')
            for part,name in [('final','test'),('starting','reference_test')]:
                if not r.get(part,{}).get('valid'):continue
                m=r[part]['metrics']
                for field in ['component_correct','component_n','joint_correct']:
                    out[name+'_'+field]=m[field];out['metric_sources'][name+'_'+field]=ptr(key,p+'/'+part+'/metrics/'+field)
                out[name+'_joint_n']=m['n'];out['metric_sources'][name+'_joint_n']=ptr(key,p+'/'+part+'/metrics/n')
            SUBJECTS.append(out)


def build_feedback():
    for batch,name,ds in [('A282','feedback282/NUMERICAL_RESULTS282.json','UOC'),
                          ('A285','feedback285/NUMERICAL_RESULTS282.json','PLANETARY_G1'),
                          ('A298','ensemble298/NUMERICAL_SUMMARY298.json','PLANETARY_G2')]:
        x,key=source('results/'+name)
        g=x['groups']['search'] if batch=='A298' else x['groups']
        effect=x['effects']['search'] if batch=='A298' else x['immediate_minus_delayed']
        path='/groups/search' if batch=='A298' else '/groups'
        FEEDBACK.append(dict(batch=batch,task=TASKS[ds][0],per_arm_initial_sessions=3,
            immediate_final_f1=g['immediate']['mean_final_macro_f1'],delayed_final_f1=g['delayed']['mean_final_macro_f1'],
            immediate_minus_delayed_f1=effect['mean_final_macro_f1'],
            interpretation='未观察到预期反馈时机效应；不等于证明所有场景无效应。',limitations=NOTES[batch],
            immediate_source=ptr(key,path+'/immediate/mean_final_macro_f1'),
            delayed_source=ptr(key,path+'/delayed/mean_final_macro_f1')))


def build_batches():
    # Requests are copied from checked run ledgers; None means not aggregated here.
    requests={'A252':118,'A254':106,'A259':176,'A260':291,'A262–263':369,'A264':181,
        'A267':442,'A268':790,'A269':627,'A270':178,'A273':184,'A274':746,'A276':343,
        'A278':330,'A281':226,'A282':311,'A284':116,'A285':187,'A298':328,'A305':658,'A318':106,'A319':57,
        'A312':745,'A317':1061,'A316':1030}
    extras={'A268':3,'A282':6}
    docs={'A227':'real_training_and_review238.md','A238':'real_training_and_review238.md',
        'A252':'phmforge_coding_results252.md','A254':'phmforge_femto_results254.md','A256':'phmforge_hust_results256.md',
        'A259':'phmforge_coding_results259.md','A260':'phmforge_coding_results260.md',
        'A262–263':'phmforge_coding_results262_263.md','A264':'phmforge_coding_results262_263.md',
        'A267':'validation_transfer267_results.md','A268':'selection_expansion268_results.md',
        'A269':'phmforge_overfit269_results.md','A270':'choice270_results.md','A273':'recovery273_results.md',
        'A274':'validation274_results.md','A276':'pipeline276_results.md','A278':'xjtu_feedback278_results.md',
        'A281':'frozen_candidate_review281_results.md','A282':'feedback282_results.md','A284':'evidence284_results.md',
        'A285':'feedback285_results.md','A298':'ensemble298_results.md','A305':'rotor305_results.md',
        'A306':'phmforge_coverage306_results.md','A307':'phmforge_expansion307_results.md',
        'A311':'phmforge_ordinary_review311.md','A314':'phmforge_safety314_results.md','A315':'phmforge_rul_followup315_results.md',
        'A318':'phmforge_diagnostic318_319_results.md','A319':'phmforge_diagnostic318_319_results.md',
        'A312':'phmforge_engine_final312_316_317.md','A317':'phmforge_engine_final312_316_317.md','A316':'phmforge_engine_final312_316_317.md'}
    for batch in sorted({r['batch'] for r in SUBJECTS},key=lambda x:int(x[1:4])):
        rows=[r for r in SUBJECTS if r['batch']==batch]
        _,key=source('docs/'+docs[batch])
        BATCHES.append(dict(batch=batch,tasks='、'.join(dict.fromkeys(r['task'] for r in rows)),
            phase_rows=len(rows),normal_finished=sum(r['status']=='finished' for r in rows),
            formal_submissions=sum(r['formal_submission'] for r in rows),
            scored_or_diagnostic_rows=sum(any(k in r for k in ['test_accuracy','test_mae','test_median_error_mm','test_joint_correct']) for r in rows),
            extra_infrastructure_starts=extras.get(batch,0),model_requests=requests.get(batch),
            request_scope='含6次无效启动210请求；有效101请求' if batch=='A282' else '含本批基础设施中断请求' if batch=='A268' else '本批记录',
            status='已结束，见各阶段完成状态',
            notes=NOTES[batch],evidence=key))
    for batch,starts,calls,reason,doc in [
        ('A251',3,30,'有界先导：通道/输入及输出预算，0正式提交','phmforge_coding_results252.md'),
        ('A258',2,53,'实际单次输出上限8192导致补丁截断，0有效最终制品','phmforge_coding_results259.md'),
        ('A272',4,4,'账户402拒绝，0成功模型返回，0Agent任务行为','condition272_transport_failure.md')]:
        _,key=source('docs/'+doc)
        BATCHES.append(dict(batch=batch,tasks='前置失败/恢复记录',phase_rows=0,normal_finished=0,
            formal_submissions=0,scored_or_diagnostic_rows=0,extra_infrastructure_starts=starts,
            model_requests=calls,request_scope='失败尝试，不能当完成的科学会话',status='已关闭/恢复另列',notes=reason,evidence=key))
    BATCHES.sort(key=lambda r:int(r['batch'][1:4]))


def import_qualification():
    x, key = import_source('results/realbench240/QUALIFICATION_RESULTS240.json', 'A240')
    for i, r in enumerate(x['forge']):
        ds = norm_dataset(r['dataset']); out = record('A240', ds, r['job'], '训练资格核验',
            'qualified' if r['qualification_pass'] else 'failed', False, key, f'/forge/{i}',
            model='无新增LLM会话', cli='', artifact_kind='真实拟合基线，非Agent提交',
            optimizer_steps=r['optimizer_steps'], notes='数值环境资格，不计Agent问题验证；数据来源限制见源文件。')
        evaluation = r['evaluation']; m = evaluation['predict']
        attach(out, 'test', m, key, f'/forge/{i}/evaluation/predict')
        BASELINES.append(out)
    asset = x['asset']
    for kind in ['trained', 'pretrained']:
        out = record('A240', 'ASSETOPS_CHILLER', kind, '环境核验', 'qualified', False, key,
                     '/asset/' + kind, model='TTM（真实微调）' if kind == 'trained' else 'TTM（预训练）',
                     cli='', artifact_kind='非Agent实验', test_n=28, notes='按目标非零条件选窗；探索性环境核验，不是盲测泛化成绩。')
        attach(out, 'test', asset[kind], key, '/asset/' + kind)
        BASELINES.append(out)
    x, key = source('results/realbench240/CURRENT_ASSET_QUALIFICATION240.json')
    out = record('A240', 'ASSETOPS_CHILLER', 'current_main', '环境核验', 'qualified', False, key, '',
                 model='TTM（当前main真实微调）', cli='', artifact_kind='非Agent实验', test_n=28,
                 optimizer_steps=x['optimizer_steps_total'], notes='5次拟合共80步；最后一次完整拟合20步；同一探索性28点窗口。')
    out.update(test_mae=x['external_mae'], test_rmse=x['external_rmse'])
    out['metric_sources'] = {'test_mae': ptr(key, '/external_mae'), 'test_rmse': ptr(key, '/external_rmse')}
    BASELINES.append(out)


def build_records():
    for batch, folder in [('A252', 'forgecoding252'), ('A254', 'femtocoding254'),
                          ('A256', 'hustcoding256'), ('A259', 'coding259'), ('A260', 'coding260'),
                          ('A262–263', 'coding262'), ('A264', 'review264')]:
        read_coding_rows(batch, f'results/{folder}/CODING_SCORE.json')
    read_coding_rows('A262–263', 'results/coding262/SUMMARY262_263.json', '/femto')
    for batch, folder, filename in [('A268', 'selection268', 'NUMERICAL_RESULTS268.json'),
        ('A269', 'overfit269', 'NUMERICAL_RESULTS269.json'), ('A270', 'selection270', 'NUMERICAL_RESULTS270.json'),
        ('A273', 'condition273', 'NUMERICAL_RESULTS273.json'), ('A274', 'validation274', 'NUMERICAL_RESULTS274.json'),
        ('A276', 'pipeline276', 'NUMERICAL_RESULTS276.json')]:
        import_common(batch, f'results/{folder}/{filename}')
    import_special()
    x, key = source('results/selection270/ADJUDICATION270.json')
    mapping = {'a': ('motor_a', '初选'), 'a_review': ('motor_a', '复核'),
               'b': ('motor_b', '初选'), 'b_review': ('motor_b', '复核')}
    for i, r in enumerate(x['rows']):
        case, stage = mapping[r['cohort']]
        row = next(s for s in SUBJECTS if s['batch'] == 'A270' and s['case'] == case and s['stage'] == stage)
        for split, m in r['split_metrics'].items():
            attach(row, 'dev' if split == 'val' else split, m, key, f'/rows/{i}/split_metrics/{split}')
    for batch, folder in [('A274', 'validation274'), ('A276', 'pipeline276')]:
        x, _ = source(f'results/{folder}/current_run.json')
        base = Path(x['archive_path'])
        attach_public(batch, base)
    state = import_latest()
    import_completed_campaigns()
    import_engine_campaigns()
    import_qualification()
    SUBJECTS.sort(key=lambda r: (int(r['batch'][1:4]), r['case'], r['stage']))
    assert len(SUBJECTS) == 185, f'Unexpected subject denominator: {len(SUBJECTS)}'
    assert len({s['id'] for s in SUBJECTS}) == len(SUBJECTS)
    return state


def comparison(id_, batch, dataset, name, metric, before, after, unit, key, p_before, p_after,
               n=None, groups=None, kind='实际候选选择', limit='', net_errors=None):
    COMPARISONS.append(dict(id=id_, batch=batch, task=TASKS[dataset][0], dataset=dataset,
        comparison=name, metric=metric, before=before, after=after, delta=after-before,
        unit=unit, n=n, groups=groups, kind=kind, net_extra_errors=net_errors,
        limitation=limit, before_source=ptr(key, p_before), after_source=ptr(key, p_after)))


def build_comparisons():
    x, key = source('results/selection270/NUMERICAL_RESULTS270.json')
    c = x['cases'][0]; j = next(i for i,v in enumerate(c['controls']) if v['name'] == 'svc_c1')
    comparison('C01', 'A270', 'ELECTRICMOTORVIBRATIONS', '可用SVC → Agent最终集成',
        'accuracy', c['controls'][j]['metrics']['accuracy'], c['stages']['review']['final']['metrics']['accuracy'],
        '比例', key, f'/cases/0/controls/{j}/metrics/accuracy', '/cases/0/stages/review/final/metrics/accuracy',
        n=338, groups=6, net_errors=69, limit='一个相关分配；固定CV/开发选择器也可能同选；非Agent特有机制。')
    x, key = source('results/validation274/ADJUDICATION274.json'); d=x['cases']['d']; c=x['cases']['c']
    comparison('C02', 'A274', 'HUST', '实际旧候选 → 最终提交（d）', 'accuracy',
        d['actual_old_candidate_accuracy'], d['selected_accuracy'], '比例', key,
        '/cases/d/actual_old_candidate_accuracy', '/cases/d/selected_accuracy', n=3865, groups=7,
        net_errors=d['actual_old_candidate_net_more_correct'], limit='普通复核保留；CV拟合规则与最终过滤不完全匹配，因果未确认。')
    comparison('C03', 'A274', 'HUST', '关闭复合故障纠错 → 启用纠错（c）', 'accuracy',
        c['without_compound_accuracy'], c['full_accuracy'], '比例', key,
        '/cases/c/without_compound_accuracy', '/cases/c/full_accuracy', n=3865, groups=7,
        kind='固定组件消融，非曾被弃用完整候选',
        net_errors=c['compound_broken_windows']-c['compound_fixed_windows'],
        limit='研究者固定权重对照；不能冒充Agent此前拥有该完整备选。')
    x, key = source('results/pipeline276/ADJUDICATION276.json')
    for i,c in enumerate(x['cases']):
        if 'component' not in c:continue
        comparison('C04' if c['case']=='ordinary_r1' else 'C05', 'A276', 'ROTORBROKENBAR',
            c['robust_component'] + ' → 最终集成（' + c['case'] + '）', 'accuracy',
            c['component']['accuracy'], c['final']['accuracy'], '比例', key,
            f'/cases/{i}/component/accuracy', f'/cases/{i}/final/accuracy', n=737, groups=10,
            net_errors=c['net_extra_errors'], limit='同一负载划分；后续固定候选复核保留，开放重建困难性未证。')
    x, key = source('results/feedback278/ADJUDICATION278.json')
    for i,c in enumerate(x['comparisons']):
        comparison('R'+str(i+1).zfill(2), 'A278', 'XJTU',
            str(c['alternative'])+' → '+str(c['selected']), 'mean_bearing_mae',
            c['final_alternative'], c['final_selected'], '分钟', key,
            f'/comparisons/{i}/final_alternative', f'/comparisons/{i}/final_selected',
            n=15, groups=3, kind='候选/组件对照（见原协议）',
            limit='每轴承5时点；不同对照可能重叠，不计独立失败频率。重大门槛见原判定。')
    x, key = source('results/realtrain238/RESULT_INTERPRETATION238.json')
    for i,r in enumerate(x['rows']):
        comparison('N'+str(i+1), 'A238', 'CMAPSS_FD001', '原MLP → '+r['job'], 'mae',
            x['baseline']['mae'], r['mae'], 'cycles', key, '/baseline/mae', f'/rows/{i}/mae',
            n=100, groups=100, kind='成功/保持反证', limit='数据为物理仿真；无总体退化不代表每台发动机改善。')
    x,key=source('results/rotor305/NUMERICAL_RESULTS305.json')
    for i,r in enumerate(x['rows']):
        if r['stage']!='review':continue
        j=next(j for j,v in enumerate(x['rows']) if v['cohort']==r['cohort'].removesuffix('_review'))
        old=x['rows'][j]
        comparison('A305_'+r['cohort'],'A305','ROTORBROKENBAR','初选程序 → 普通复核程序（'+r['cohort']+'）','accuracy',
            old['final_accuracy'],r['final_accuracy'],'比例',key,f'/rows/{j}/final_accuracy',f'/rows/{i}/final_accuracy',
            n=r['n'],groups=r['records'],kind='配对普通复核；含改善、保持和退化',
            limit='两个采集块不是独立电机；初选与复核总预算不同，不用于估计一般失败率。')
    x,key=source('results/expansion307/NUMERICAL_RESULTS307.json')
    for i,r in enumerate(x['rows']):
        if not r['formal_submission']:continue
        is_cwru=r['task_id']=='pdm_cost_benefit_002';metric='accuracy' if is_cwru else 'mae';rawmetric=metric if is_cwru else 'mae_cycles'
        comparison('A307_'+r['task_id'],'A307','CWRU' if is_cwru else 'CMAPSS_FD001','实际起始程序 → '+r['task_id'],metric,
            r['reference']['metrics'][rawmetric],r['final']['metrics'][rawmetric],'比例' if is_cwru else 'cycles',key,
            f'/rows/{i}/reference/metrics/{rawmetric}',f'/rows/{i}/final/metrics/{rawmetric}',n=177 if is_cwru else 100,groups=3 if is_cwru else 100,
            kind='实际起始程序对照；含成功与失败',limit=NOTES['A307'])
    x,key=source('results/safety314/NUMERICAL_RESULTS314.json')
    for i,r in enumerate(x['rows']):
        comparison('A314_'+r['task_id'],'A314',r['dataset'],'实际起始程序 → '+r['task_id'],'accuracy',
            r['actual_starting_reference']['accuracy'],r['final']['accuracy'],'比例',key,
            f'/rows/{i}/actual_starting_reference/accuracy',f'/rows/{i}/final/accuracy',n=r['final']['n'],
            groups=n_groups(r['final']),kind='实际起始程序对照；安全合同另列',limit=NOTES['A314'])
    x,key=source('/data/yyf-zyq/PHM_20261010/20261023T154500Z_safety314/component_controls314/final/RESULT.json')
    comparison('A314_rotor_component','A314','ROTORBROKENBAR','已有ExtraTrees组件 → 提交集成','accuracy',
        x['metrics']['raw_extra']['accuracy'],x['metrics']['submitted_ensemble']['accuracy'],'比例',key,
        '/metrics/raw_extra/accuracy','/metrics/submitted_ensemble/accuracy',n=1468,groups=20,
        kind='实际已拟合组件对照，非独立会话',limit='所有组件开分前封存；净少对74窗，但整记录多数票均20/20，不能称设备级退化。',net_errors=74)
    x,key=source('results/rulfollow315/NUMERICAL_RESULTS315.json')
    for i,r in enumerate(x['rows']):
        c=r['case'];is_femto=c['dataset']=='FEMTO';m=r['results']['final']['metrics'];base=r['results']['starting']['metrics']
        pointer='all/mae' if is_femto else 'mean_bearing_mae_minutes'
        comparison('A315_'+c['task_id'],'A315',c['dataset'],'实际起始程序 → '+c['task_id'],'mae',
            base['all']['mae'] if is_femto else base[pointer],m['all']['mae'] if is_femto else m[pointer],
            '秒' if is_femto else '分钟',key,f'/rows/{i}/results/starting/metrics/{pointer}',f'/rows/{i}/results/final/metrics/{pointer}',
            n=11 if is_femto else 15,groups=11 if is_femto else 3,kind='实际起始程序对照；含保持与改善',limit=NOTES['A315'])
    i=next(i for i,r in enumerate(x['rows']) if r['case']['task_id']=='pdm_safety_009')
    comparison('A315_xjtu_candidate','A315','XJTU','已检查KNN → 安全题新ET','mean_bearing_mae',
        x['XJTU_control_pool_MAE_minutes']['knn5'],x['rows'][i]['results']['final']['metrics']['mean_bearing_mae_minutes'],'分钟',key,
        '/XJTU_control_pool_MAE_minutes/knn5',f'/rows/{i}/results/final/metrics/mean_bearing_mae_minutes',n=15,groups=3,
        kind='MAE数值选择遗憾',limit='2/3轴承改善，损失集中1个长寿命轴承；ET高估寿命的误差总量更小，但高估次数9对7，不能称安全全面更差；未做本程序普通复核。')
    direction,key=source('results/rulfollow315/ERROR_DIRECTION_COUNTEREVIDENCE315.json')
    collect_metrics(direction,'A315',key,context={'dataset':'XJTU'})
    a=next(i for i,r in enumerate(direction['rows']) if r['case']=='x1');b=next(i for i,r in enumerate(direction['rows']) if r['case']=='x2')
    for field,label in [('overestimation_mass_per_query_minutes','高估寿命误差量／全部15查询'),('underestimation_mass_per_query_minutes','低估寿命误差量／全部15查询')]:
        comparison('A315_'+field,'A315','XJTU','KNN → ET：'+label,field,direction['rows'][a][field],direction['rows'][b][field],'分钟',key,
            f'/rows/{a}/{field}',f'/rows/{b}/{field}',n=15,groups=3,kind='同一预测的误差方向分解，非独立实验',limit='两种误差相互取舍；无已核准的安全／维护效用模型，不宣称更安全或更危险。')
    for batch,folder in [('A318','diagnostic318'),('A319','cwrudiagnostic319')]:
        x,key=source(f'results/{folder}/NUMERICAL_RESULTS{batch[1:]}.json')
        for i,r in enumerate(x['rows']):
            c=r['case'];comparison(batch+'_'+c['task_id'],batch,c['dataset'],'实际起始程序 → '+c['task_id'],'accuracy',
                r['actual_starting_reference']['accuracy'],r['final']['accuracy'],'比例',key,
                f'/rows/{i}/actual_starting_reference/accuracy',f'/rows/{i}/final/accuracy',n=r['final']['n'],groups=n_groups(r['final']),
                kind='持平／改善的实际起始对照',limit=NOTES[batch])
    x,key=source('results/enginecompletion317/COMBINED_SINGLE_CYCLE_RESULTS312_317.json')
    comparison('ENGINE_single_aggregate','A317','ENGINE_QA_HEALTH','18条源查询：实际起点 → 最终程序','joint_annotation_accuracy',
        x['metrics']['actual_starting']['joint_accuracy'],x['metrics']['final']['joint_accuracy'],'比例',key,
        '/metrics/actual_starting/joint_accuracy','/metrics/final/joint_accuracy',n=18,groups=18,
        kind='完整分母改善反证；非独立物理发动机',limit='42阶段的单周期部分只取每原题正常最终提交；3条改善1条退化14条持平，不计算跨来源失败率。')
    i=next(i for i,r in enumerate(x['rows']) if r['task_id']=='engine_health_combined_012')
    comparison('ENGINE_single_012_regret','A317','ENGINE_QA_HEALTH','原题012：实际起点 → 最终程序','component_annotation_count',
        x['rows'][i]['starting_components_correct'],x['rows'][i]['final_components_correct'],'项／5',key,
        f'/rows/{i}/starting_components_correct',f'/rows/{i}/final_components_correct',n=1,groups=1,
        kind='单条源标注退化，非因果确认',limit='真值00011，提交11111；3额外故障标注。参考helper环境缺口单列，不能称此前完整工作程序被改坏。')


@lru_cache(maxsize=None)
def source_json(path):
    return json.loads(Path(path).read_text())


def resolve(ref):
    name, pointer = ref.split('#', 1)
    path = ROOT/name if not name.startswith('/') else Path(name)
    x = source_json(str(path))
    for token in pointer.strip('/').split('/') if pointer else []:
        token = token.replace('~1', '/').replace('~0', '~')
        x = x[int(token)] if isinstance(x, list) else x[token]
    return x


def verify_cells():
    count=0
    for r in SUBJECTS + BASELINES:
        for field, evidence in r['metric_sources'].items():
            assert r[field] == resolve(evidence), (r['id'], field)
            assert math.isfinite(r[field])
            if field.endswith(('accuracy','macro_f1','balanced_accuracy')):
                assert 0 <= r[field] <= 1
            count += 1
    for r in COMPARISONS:
        assert r['before'] == resolve(r['before_source'])
        assert r['after'] == resolve(r['after_source'])
        assert math.isclose(r['delta'], r['after']-r['before'], abs_tol=1e-12)
        count += 2
    for r in FEEDBACK:
        assert r['immediate_final_f1'] == resolve(r['immediate_source'])
        assert r['delayed_final_f1'] == resolve(r['delayed_source'])
        assert math.isclose(r['immediate_minus_delayed_f1'],
                            r['immediate_final_f1']-r['delayed_final_f1'],abs_tol=1e-12)
        count += 2
    for r in METRICS:
        assert r['value'] == resolve(r['evidence']), r['evidence']
        count += 1
    assert all('test_accuracy' in r for r in SUBJECTS if r['batch']=='A305')
    return count


def write_csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore');w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in columns})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    state=build_records();build_comparisons();build_splits();build_batches();build_feedback()
    audit=verify_cells()
    coverage,_=source('results/phmforge75_real_coverage_current.json')
    queue,_=source('results/phmforge_all_real_queue75.json')
    accepted=['measured_task_adaptation','official_simulation_task_adaptation']
    meta=dict(created_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='A227导波复核、A238真实训练复核，以及A240–A319真实计算／研究适配线；包含全部终态、重复与补全，不包含此前333题占位工具分数。',
        experiment_status='execution_and_analysis_complete', new_fits=0,new_model_calls=0,new_inference=0,new_hidden_scores=0,
        subject_rows=len(SUBJECTS),baseline_rows=len(BASELINES),numeric_cells_verified=audit,
        latest_state='全部55个可执行数值方向已有实际终态：51正常外评、4预算未交付；16缺口与4明确合成题另列',latest_subjects=42,latest_final_scores_opened=True,
        phmforge_original_denominator=75,
        phmforge_attempted_directions=sum(coverage['summary'][k]['original_id_count'] for k in accepted),
        phmforge_normally_scored_directions=sum(coverage['summary'][k]['original_ids_with_normal_completed_scored_stage'] for k in accepted),
        potential_numerical_directions=queue['potential_numerical_task_directions'],
        budget_terminal_directions=queue['budget_terminal_directions'],
        qualified_gaps=16,explicit_synthetic_excluded=4,experiment_stage='全部可执行数值方向均有真实终态；已完成结果、反证与下一步计划分析',
        caution='会话、窗口、候选、设备和来源的分母不同；禁止跨批相加为独立设备或总体失败率。')
    bundle=dict(meta=meta,subjects=SUBJECTS,baselines=BASELINES,comparisons=COMPARISONS,
                metrics=METRICS,tasks=TASKS,batches=BATCHES,splits=SPLITS,feedback_controls=FEEDBACK,
                phmforge_queue=queue,sources=list(SOURCES.values()))
    (out/'experiment_data.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('agent_results',SUBJECTS),('baseline_results',BASELINES),
                      ('comparisons',COMPARISONS),('metric_details',METRICS),('batch_inventory',BATCHES),
                      ('data_splits',SPLITS),('feedback_controls',FEEDBACK),('evidence_sources',list(SOURCES.values()))]:
        write_csv(out/(name+'.csv'), rows)
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':
    main()
