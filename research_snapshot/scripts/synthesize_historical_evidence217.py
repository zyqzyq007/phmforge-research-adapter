"""Read-only, zero-model synthesis; never infers scientific failure from completion."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
ARCHIVES = Path('/data/yyf-zyq/PHM_20261010')
INVENTORY = ARCHIVES / '20261018T_benchmark_contract_inventory'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / 'PROTOCOL217.json').is_file():
        raise RuntimeError('Protocol/resource budget must be frozen before synthesis.')
    if (run / 'RESULT217.json').exists():
        raise RuntimeError('No overwrite/reclassification of an existing result.')
    sources = {
        'contracts': INVENTORY / 'compiled/benchmark_contracts.json',
        'recent_state': INVENTORY / 'recent/STATE_EXTRACT.json',
        'historical_verification': INVENTORY / 'ROOT_OUTPUT_VERIFICATION.json',
        'imprl_holdout': ROOT / 'results/imprl89/heldout_comparison89.json',
        'imprl_algebra': ROOT / 'results/imprl89/inspection_value_identity89.json',
        'imprl88': ROOT / 'docs/iteration88.md',
        'imprl90': ROOT / 'docs/iteration90.md',
        'fleet115': ROOT / 'docs/iteration115.md',
        'scoped156': ROOT / 'docs/scoped_problem_review156.md',
        'mera_binding': ARCHIVES / '20261018T040002300334Z_research_check129/TARGET_BINDING_AUDIT160.json',
        'battery185': ARCHIVES / '20261018T142647Z_multiscope185/RESULT185.json',
        'battery185_assessment': ARCHIVES / '20261018T142647Z_multiscope185/ASSESSMENT185.json',
        'scope185': ROOT / 'docs/validation_scope_results185.md',
        'guided_selection': ROOT / 'docs/candidate_validation_selection.md',
        'scope214': ROOT / 'results/fullbench187/final_scope214.json',
        'source_ledger214': ROOT / 'results/fullbench187/final_source_ledger214.jsonl',
        'time215': ROOT / 'results/fullbench187/timeaudit215.json',
        'time_dependency215': ROOT / 'results/fullbench187/prediction_time_dependency_review215.json',
        'time_doc215': ROOT / 'docs/time_and_score_counterevidence215.md',
        'phase216': ROOT / 'results/fullbench187/reciprocity216.json',
        'phase_doc216': ROOT / 'docs/reciprocity_counterevidence216.md',
    }
    provenance = []
    staged = {}
    for name, path in sources.items():
        data = path.read_bytes()
        target = run / 'inputs' / (name + path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        provenance.append(dict(name=name, original_path=str(path), snapshot=str(target.relative_to(run)),
                               bytes=len(data), sha256=sha(data), captured_utc=now()))
        staged[name] = target
    write(run / 'SOURCES217.json', provenance)
    source_map = {p['name']: p for p in provenance}
    assert source_map['contracts']['sha256'] == '298f17504f3b032e9331f88d1569b785213f800cb06ad4a4f70a7928178796dc'
    assert source_map['mera_binding']['sha256'] == '10d6dbc2a05b25098da90b5aa6737cfdc8a6490793ddb4f716279e560d346eaa'
    contracts = json.loads(staged['contracts'].read_text())['rows']
    recent = json.loads(staged['recent_state'].read_text())
    scope = json.loads(staged['scope214'].read_text())
    assert len(contracts) == 141
    assert len({r['audit_row_id'] for r in contracts}) == 141
    assert scope['finished'] == 333 and scope['source_entries'] == 774
    assert scope['unrun_source_entries'] == 441
    assert scope['final_ledger_sha256'] == source_map['source_ledger214']['sha256']
    assert recent['A136']['ordinary_review']['blocked_actions'] == 7
    assert recent['A136']['ordinary_review']['safety_clean_after_guardrail'] == 8
    assert recent['A137']['ordinary_review']['BR00004']['full_episode_signed_f1'] == 1.0
    assert recent['A137']['ordinary_review']['CR00011']['false_positives_in_rule_valid_episodes'] == 23
    assert recent['A137']['ordinary_review']['CT00003']['false_positives_in_rule_valid_episodes'] == 30
    assert recent['A150']['score']['ordinary_affine_inversion_control'] == '8/8'
    holdout = json.loads(staged['imprl_holdout'].read_text())
    policies = {p['policy']: p for p in holdout['rows']}
    assert holdout['native_repair_status'] == 'finished'
    assert abs(policies['r1_holdout']['vs_author']['mean_cost'] - 927.4102852290449) < 1e-9
    assert abs(policies['existing_tool_control_holdout']['vs_author']['mean_cost'] - 779.1616151046493) < 1e-9
    assert '撤回此前依据缺直接通道就断言六例不可辨识的说法' in staged['scoped156'].read_text()

    # Annotations refer to research eligibility, never to an invented common pass/fail scale.
    overlays = {
        'recent_impmarl': ('ordinary_remedy_and_later_success', 'A129 has a real c5 regression (24.4505 vs21.6383), repaired by original inspection timing in8/8 paired seeds; A185 later has two successful configurations. Do not rewrite A129 as two successes.', ['recent_state', 'scope185']),
        'recent_shm_guided': ('retain_local_observation', 'Use A178/A179/A216 updates. Tool rejection and real localization error are observed; information sufficiency, causal mechanism and residual difficulty remain unresolved.', ['guided_selection', 'phase216']),
        'M34': ('same_family_as_retained_observation', 'Same guided-wave source as recent_shm_guided and laterA178/A179; no additional independent source.', ['guided_selection']),
        'recent_shm_baselinefree': ('success_counterevidence', '12/12 on a related localization task; transferred B has different development knowledge, so it is no information-theoretic sufficiency proof.', ['guided_selection']),
        'recent_shm_virtual': ('public_test_already_failed', 'Public synthetic0/6: do not merge into self-test-passes/hidden-fails mechanism.', ['contracts']),
        'M33': ('same_family_public_review_limits', 'Same virtual-baseline source; offline restrictions and restored public docs must remain explicit.', ['contracts']),
        'recent_iot': ('ordinary_visible_guardrail', 'A136 zero-model replay blocks7 unsafe actions; not a newly run Agent repair or independent held-out demonstration. Safe refusal failures include evaluator call-count artifacts.', ['recent_state']),
        'recent_rulelogic': ('semantics_and_incomplete_gold', 'One duration issue is removable; two low-F1 outputs include other rule-valid episodes outside single gold windows. Not physical diagnosis failure.', ['recent_state']),
        'recent_mera': ('ordinary_detection_only', 'A160 detects2/2 target-binding errors,0/3 false alerts; it does not demonstrate repaired predictions or stable generalization.', ['mera_binding', 'scoped156']),
        'recent_measurement': ('ordinary_affine_inverse', 'Four change cases over-abstain; the pre-frozen inversion control is8/8 on these8 synthetic cases. No measured maintenance loss.', ['recent_state']),
        'recent_3w': ('unresolved_not_proven_unidentifiable', 'Do not repeat A152 impossibility inference from absent direct sensors: A156 retracts it. Primary4/18 and post-score11/18 remain; same-information sufficiency and blind-review residual untested.', ['recent_state', 'scoped156']),
        'recent_calb': ('later_local_observation_keep_separate', 'A153 largely ordinary reference/data limits; laterA18525C error299 persists after blind review. Matched program/prefix control changes41.18, leaving257.82; do not attribute all to calibration.', ['recent_state', 'scope185', 'battery185_assessment']),
        'M07': ('real_mechanism_ordinary_control', 'Evaluation truncation mistaken for service horizon is real; simple rolling-index control improves it. Clarification Agent itself timed out, so no completed Agent repair claimed.', ['imprl88']),
        'M08': ('real_mechanism_completed_tool_repair', 'A89 complete information-value omission has residual error before ordinary repair; repair finishes344.23s, same-source new seeds improve927.410->779.162. A90 complete qualifying cost failures0/19.', ['imprl_holdout', 'imprl_algebra', 'imprl90']),
        'M09': ('scope_and_delivery_limits', 'A90 has1complete road policy,2timeouts, complete cost gap+2.03% withCI including0; not three demonstrated decision failures.', ['imprl90']),
        'M35': ('success_counterevidence', 'Three completed boiler programs have strong F1; do not reopen without a new natural failure.', ['contracts']),
        'M36': ('success_counterevidence', 'RLfleet normal4-plane return3930.21875vs3425.03125; lower secondary completion rate cannot reverse original reward success. Other2sessions timed out.', ['fleet115']),
        'M42': ('superseded_source_scope_not_new_replication', 'Earlier6prototype tasks do not equal currentForge75 or paper99; source scores/ambiguity limits remain.', ['scope214']),
        'M43': ('numeric_tool_task_not_full_asset_benchmark', 'Earlier5numeric tasks do not equal AssetOps full benchmark. No current ISO certification inferred.', ['contracts']),
        'recent_asset_hold': ('historical_hold_superseded_execution', 'This old0-subject HOLD is preserved; later selectedAssetOps258 primaries belong to separate frozen contracts. Do not count them twice.', ['scope214']),
        'recent_admit_hold': ('historical_hold_later_adapted_execution', 'Old0subject entry remains0; laterA185 has2 adapted contexts, original gates plus physical function, both successes. Not a claim to complete original source.', ['scope185']),
    }
    invalid_ids = {'recent_mera_invalid', 'recent_ims_invalid', 'recent_pronostia_invalid'}
    rows = []
    for item in contracts:
        row = dict(item)
        recorded_run = not (item['id'].startswith('Q') or item['id'] == 'M44' or item['id'].endswith('_hold'))
        decision = 'historical_contract_only_not_rejudged'
        note = 'Contract metadata retained. Completion is not inferred to be scientific success/failure; no new per-episode correctness or causal review.'
        support = ['contracts']
        if not recorded_run:
            decision = 'qualification_only_no_subject_in_this_record'
            note = 'Original record has no subject run. Later experiments, if any, have separate identity and denominator.'
        if item['id'] in invalid_ids:
            decision = 'invalid_or_leaked_task_contract'
            note = 'Preserve execution and invalidity; do not count it as valid predictive success or Agent failure.'
        if item['id'] in overlays:
            decision, note, support = overlays[item['id']]
        row.update(recorded_Agent_execution_in_this_contract=recorded_run,
                   synthesis217_decision=decision, synthesis217_note=note,
                   synthesis217_sources=support, new_primary_outcome_rejudged=False,
                   confirmed_research_problem=False)
        rows.append(row)
    counts = Counter(r['recorded_Agent_execution_in_this_contract'] for r in rows)
    assert counts == {True: 109, False: 32}
    write(run / 'HISTORICAL_ROWS217.json', {'rows': rows, 'unit': 'contract record, not independent task/device/episode'})
    columns = ['audit_row_id', 'id', 'family', 'period', 'execution_status', 'actual_scope_or_count',
               'recorded_Agent_execution_in_this_contract', 'synthesis217_decision', 'synthesis217_note']
    with (run / 'HISTORICAL_ROWS217.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows)

    cards = [
        dict(id='C01', question='健康参考下，Agent是否以自建测试错误淘汰有效定位工具？', boundary='同一guided-wave-localization任务；开发仅健康参考，冻结函数运行时才见损伤查询。', observation='A178从有效B起点拒绝B，提交在原7例0/7、均误差82.484mm；A179干净普通复核仍0/7、97.920mm。B原7/7，扩展12/12。', consequence='已测定位容差失守；无现场检修/事故损失测量。', control_kind='completed_blind_Agent_review_plus_known_case_numeric_falsification', counterevidence='A179已测试多类损伤；A216只改正反相位后排序仍波形10/10、B0/10，否定相位单因素充分解释。B来自另一任务开发知识，不是同信息可辨识性证明。', decision='保留最强局部观察；未确认难题。', decisive_next_check='先辨别健康数据下的选择信息不足与Agent验证错误；若可见的极少量独立实测锚点或普通保留工具规则稳定解决，则降低挑战性。须预先冻结新未见条件，不能自动重跑A186。', sources=['guided_selection', 'phase216', 'phase_doc216']),
        dict(id='C02', question='RUL组合程序是否沿用了不匹配的子模型校准？', boundary='A185 CALB25℃单目标cycle100；同温参照稀少且删失。', observation='误差299周期、区间不覆盖；盲复核保持。完整程序/前缀匹配对照减少41.18，残余257.82周期。', consequence='实际RUL误差与覆盖遗漏；无维护损失测量。', control_kind='completed_blind_Agent_review_and_prespecified_numeric_variants', counterevidence='35℃误差34且覆盖；材料2低覆盖没有同一程序变更机制。匹配对照同时改变前缀与程序，不能唯一归因。', decision='保留次级观察；不宣称解释大部分误差或已有同信息可靠预测器。', decisive_next_check='先确认同信息强完整流程的可解依据；比较完整/子模型与前缀因素，信息不足不能用Agent难题命名。', sources=['battery185', 'battery185_assessment', 'scope185']),
        dict(id='C03', question='前瞻性维护判断是否依赖当时不可见的事后观测？', boundary='AssetOps compressor一台mp_1 fixture，15题中9题有前瞻措辞；原通用instruction无全局截止。', observation='9/9读未来，8/9最终引用，1/9只访问未引用；4/9私有原rubric要求事后验证。', consequence='已测证据时点越界线索；未做去除未来后的决策反事实或实测预警效果。', control_kind='zero_model_trace_and_data_lineage_audit_not_Agent_repair', counterevidence='10次原forecast调用实际输入时间均截至原窗口；访问未来不等于用未来训练。原合同可能为回顾性分析。', decision='降优先级；不计9个已证错误或高难失败。', decisive_next_check='先澄清原时间合同并区分获取、引用和决策必要依赖；再比较普通时间检查，不先开发方法。', sources=['time215', 'time_dependency215', 'time_doc215']),
        dict(id='C04', question='检查规划是否删除了根据后续观测改选维护行动的信息价值？', boundary='A89原IMPRL一份完整策略；同一k4来源和新随机种子。', observation='完整坏例成本927.410，作者781.898；代数与概率控制支持风险代理遗漏信息价值。', consequence='原仿真折扣成本高18.61%，不是实际货币或现场损失。', control_kind='completed_Agent_existing_tool_repair_evaluated_on_same_source_new_seeds', counterevidence='普通现成工具修复344.23秒正常完成，成本779.162；A90完整策略合格成本坏例0/19。', decision='真实机制已观察，但高挑战未获支持；停止原配置扩量。', decisive_next_check='只有新的自然失败且现成工具流程仍有残余时重开，不重复已见种子。', sources=['imprl_holdout', 'imprl_algebra', 'imprl90']),
        dict(id='C05', question='是否把评价截断误当设备服役终点？', boundary='A88无限期原模型、原20步评分与额外80步机制诊断分开。', observation='完整程序设V20=0，后续停止检修；原20步分数可正常。', consequence='扩展诊断后60步成本增加497.450，重新起算折扣，不能冒充原目标增幅。', control_kind='known_case_numeric_rolling_index_control_and_timeout_Agent_clarification', counterevidence='单行滚动规划控制消除主要损失；澄清Agent虽改善但超时，不能记完整修复成功。', decision='真实局部建模错误；当前不支持高难或新颖性。', decisive_next_check='自然新实例才重开；区别目标终止与外部截断的已有规则先复核。', sources=['imprl88']),
        dict(id='C06', question='维修后未来预测目标是否被换成当前曲线端点？', boundary='A140五个MERA案例，预测下次巡检水平，非物理RUL。', observation='两例目标1245/550被换为1195/500；误差17.069/81.079adu。', consequence='目标时点与区间遗漏属实；不将全部数值误差归于绑定。', control_kind='detection_only_zero_model_timeline_contract', counterevidence='A160检测2/2错误、0/3误报；没有实验证明预测修复，不能写成2/2修好。', decision='保留诊断用例；普通检测可行，残余难度未知。', decisive_next_check='需检测后修复与新实例验证，避免把检测等同完成预测。', sources=['mera_binding', 'scoped156']),
        dict(id='C07', question='维护动作是否违反可见的寿命/阈值约束？', boundary='A136原IoTAgentBench8题，另A1354题独立保留；模板模拟动作。', observation='A136三个任务共7次不安全动作。', consequence='源状态与规则的模拟不安全，不是实际设备损失。', control_kind='zero_model_same_case_visible_data_guardrail_replay', counterevidence='普通RUL>=30不替换、负/非有限阈值不执行，8题均无这些不安全动作；剩余原fail包括安全拒绝调用数计数。', decision='降级，不以重复浅模板扩大难度证据。', decisive_next_check='只有新情景普通规则边界失效且有PHM后果时重开；已有重放不能称新Agent修复。', sources=['recent_state']),
        dict(id='C08', question='关井/重启响应是否被错当故障起因？', boundary='A1513W18个同源回顾性事件，17有效答案；原类别与目标不改。', observation='主类别4/18，top3含真值12/18；事后普通多信号规则11/18。', consequence='原事件类别失准，未测维护动作收益；无定位窗口误差评价。', control_kind='post_score_known_case_numeric_rules_no_blind_Agent_review', counterevidence='部分误报可移除；缺直接DHSV/BSW/choke测点不足以证明不可辨识，A156已撤回旧强断言。', decision='HOLD，不能淘汰成信息论不可能，也不能提升为因果难题。', decisive_next_check='先核全部动态信号与同信息可行基线；再看普通盲复核残余，不能只看通道缺失。', sources=['recent_state', 'scoped156']),
        dict(id='C09', question='校准切换是否使Agent过度弃答真实趋势？', boundary='A1508个合成场景、可见参考量程。', observation='4个真实变化均uncertain；4个传感器变化控制未产生虚假趋势。', consequence='合成趋势分类遗漏；无实测漏维护损失。', control_kind='prespecified_same_information_affine_inverse', counterevidence='普通仿射量程反演8/8。', decision='低挑战反证，不扩大同源合成题。', decisive_next_check='仅在具有新真实数据/非线性条件且普通校准仍不足时重开。', sources=['recent_state']),
        dict(id='C10', question='局部健康排序是否忽略联合检修和长期后果？', boundary='区分A129两配置、A185两配置、A115机队，不能将随机seed算新设备。', observation='A129c5确有成本回退；保持维修规则并用原检查时点在8/8配对等于参考。', consequence='A129原仿真成本差，不是未测现场损失。', control_kind='same_case_timing_control_and_later_complete_policy_counterexamples', counterevidence='A185两配置含关联belief、联合动作与30步成本且优于原参考；A115正常4飞机回报+14.75%。', decision='宽泛假说当前不获支持；不擦除A129原回退。', decisive_next_check='新原任务自然出现动作交互/未来损失时，先给现成规划工具公平机会。', sources=['recent_state', 'scope185', 'fleet115']),
    ]
    for card in cards:
        card.update(confirmed=False, novel_method_designed=False, provenance='AI root retrospective synthesis; not blinded or independent review')
        assert all(k in source_map for k in card['sources'])
    write(run / 'CANDIDATE_CARDS217.json', cards)
    write(run / 'CORRECTIONS217.json', [
        {'old_inference': '缺直接通道证明3W剩余错误不可辨识', 'corrected': 'A156明确撤回；信息限制是竞争解释，不是证明。', 'source': 'scoped156'},
        {'old_inference': 'MERA普通时间检查已修好2个预测', 'corrected': '已检出2/2，尚无预测修复或未见验证。', 'source': 'mera_binding'},
        {'old_inference': 'A129两配置均成功或全部长期维护均失败', 'corrected': 'c5回退真实，普通检查时点控制消除；A185另批成功不能改旧结果。', 'source': 'recent_state'},
        {'old_inference': 'A186从未启动/可等待旧PID或自动重试', 'corrected': 'A186已启动并在PHM工作前execution_error，已归档，无科学评分，按新优先级不自动重试。', 'source': 'current terminal receipt in goal checkpoint'},
    ])
    summary = dict(utc=now(), run_id=run.name, role='retrospective evidence synthesis, not a new PHM experiment',
                   historical_contract_records=141, contracts_recording_Agent_execution=109,
                   qualification_only_contract_records=32, detailed_candidate_cards=len(cards),
                   all_historical_raw_episode_correctness_reassessed=False,
                   original_selected_primary_finished=333, source_contract_entries=774,
                   unrun_source_entries=441, denominators_additive=False,
                   new_Agent_sessions=0, LLM_calls=0, new_solver_runs=0,
                   independently_confirmed_PHM_accuracy=None, new_confirmed_research_problems=0,
                   source_snapshots=len(provenance), source_snapshot_bytes=sum(s['bytes'] for s in provenance),
                   scientific_counts_unchanged=True, full_user_goal_complete=False,
                   conclusion='Retain guided local observation and secondary battery observation; lower compressor time candidate; preserve3W uncertainty. No residual high-difficulty problem confirmed.',
                   next_action='Complete current-source physical-correctness audits before any new primary; test only a pre-frozen distinction that can change candidate status, with valid source/runtime/admission. No automatic A186 retry or original low-score reruns.')
    write(run / 'RESULT217.json', summary)
    lines = [
        '# 全历史证据联合筛选（A217）', '',
        '结论：仍未确认符合当前目标的高难PHM Agent问题。保留导波工具误选为最强局部观察、电池为次级观察；压缩机时点线索降优先级。成功、普通修复、信息限制与评价缺陷不合并为一个机制。', '',
        f'本轮只读覆盖旧清单全部141条合同记录（109条记有Agent执行、32条仅资格/未发模），并接入当前774项来源/333个已执行原题台账；另对{len(cards)}组候选写出分开的判据。这不是141个独立benchmark，不是把所有旧原始输出重新判分，也不能把141与333相加统计失败率。新增Agent、LLM、定位/规划求解均为0。', '',
        '| 具体候选 | 现有观察与普通控制 | 当前决策 |', '| --- | --- | --- |',
    ]
    for c in cards:
        lines.append(f"| {c['id']} {c['question']} | {c['observation']} {c['counterevidence']} | {c['decision']} |")
    lines += ['', '四个必须保留的修正：3W缺测点不能证明不可辨识；MERA2/2为检测而非修复；A129回退不被后续成功抹去；A186实际已因任务前环境错误结束，不能说未启动、等待旧PID或自动重试。', '',
              '材料中的普通控制分为：只检测、同一已见案例数值修正、完成的Agent工具修复、同来源新seed、预声明简单对照与尚未执行。事后规则改善不自动成为稳定解决，复杂错误低分也不自动成为研究难题。', '',
              '下一步只推进能改变裁决的最小证伪，优先核当前原题真实输入、调用和物理结果。不给缺件补造数据，不扩量已降级模板；不依据当前低分补跑/刷分。400配额修订只完成原231题，不自动覆盖新主任务。', '',
              '冻结输入与哈希见SOURCES217.json；全部合同及来源引用见HISTORICAL_ROWS217.json/CSV；逐候选边界、后果、竞争解释和下一检验见CANDIDATE_CARDS217.json。原输出、gold与分数未改。此为root在已见结果后的分析，不是独立或盲审。', '']
    (run / 'REPORT217.md').write_text('\n'.join(lines))
    write(run / 'VALIDATION217.json', dict(utc=now(), immutable_contract_digest_matched=True,
          timeline_evidence_digest_matched=True, current_source_ledger_digest_matched=True,
          contract_ids_unique=True, all141_retained=True, execution_vs_qualification_partition=[109,32],
          selected_numeric_claims_checked_against_retained_json=True,
          raw_correctness_coverage_not_overclaimed=True, original_scores_unchanged=True,
          new_models_or_subagents=0, all_card_source_keys_resolve=True,
          source_bytes=sum(s['bytes'] for s in provenance)))
    write(run / 'RUN_STATUS.json', {'status': 'finished', 'utc': now(), 'all_writers_stopped_after_exit': True})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
