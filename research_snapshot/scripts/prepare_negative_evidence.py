"""Freeze two transparent, bounded negative-evidence controls (A64).

The physical source audit and the published simulated numerical control have
different evidence status. Neither is an autonomous field-maintenance trial.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pymupdf
from scipy.optimize import minimize
from scipy.special import expit

from prepare_robot_triage import PreBlocks

PRIOR = Path('/data/yyf-zyq/PHM_20261010/20261013T110445Z_scope63')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(run):
    assert not (run / 'task_data').exists()
    data, hidden = run / 'task_data', run / 'hidden'
    data.mkdir(); hidden.mkdir()
    book = run / 'nasa_book_chunks_1m/nasa2015_complete.pdf'
    guide = run / 'sources_fallback/nasa2022.pdf'
    for path in (book, guide):
        d = pymupdf.open(path)
        assert not d.is_repaired and len(d) > 0
    sources = data / 'sources'; sources.mkdir()
    for src, name in [(book, 'NASA_2015.pdf'), (guide, 'NASA_2022.pdf')]:
        shutil.copy2(src, sources / name)
        doc = pymupdf.open(src)
        (sources / name.replace('.pdf', '.txt')).write_text(''.join(
            '\n===== PDF PAGE %d =====\n%s' % (i + 1, p.get_text()) for i, p in enumerate(doc)))
    # A complete consecutive prefix, selected before any LLM response, not by outcome.
    doc = pymupdf.open(book)
    raw_table = doc[15].find_tables().tables[0].extract()
    save(run / 'source_table3_page16.json', raw_table)
    evidence_rows, evidence_gold = [], []
    evidence_dir = data / 'physical'; evidence_dir.mkdir()
    for i, row in enumerate(raw_table[1:25]):
        page = 23 + 2 * i
        text = doc[page].get_text() + '\n' + doc[page + 1].get_text()
        assert row[2].lower() in text.lower(), (i, row[2])
        cid = 'N%03d' % (i + 1)
        (evidence_dir / (cid + '.txt')).write_text(text)
        evidence_rows.append(dict(case_id=cid, source_file=row[2], material=row[0],
            structure=row[1], method=row[40], source_pdf_page=page + 1,
            evidence_file='physical/' + cid + '.txt', inspection_outcome='no_indication',
            scenario='Counterfactual maintenance decision, matching the source specimen/procedure scope; no new inspection performed'))
        reached = 'REACHED' in text and 'NOT REACHED' not in text
        assert 'Warning:  No false call analysis.' in text
        evidence_gold.append(dict(case_id=cid, source_case=row[4],
            point_capability_observed=reached,
            point_size=float(row[5]) if reached else None,
            range_qualification_complete=False, close_declared_fault_branch=False,
            justification='Current source explicitly retains unmet qualification conditions; a point estimate is not completed range qualification.',
            source_pdf_page=page + 1))
    table(data / 'physical_requests.csv', evidence_rows)
    save(hidden / 'physical_gold.json', evidence_gold)
    # NASA explicitly describes this as a simulated example, not measured hardware.
    guide_doc = pymupdf.open(guide)
    text = guide_doc[37].get_text()
    matches = re.findall(r'(?m)^([1-9][0-9]?)\n([A-Z]{2})\n(0\.\d+)\n((?:[01]\n){10})', text)
    assert len(matches) == 60 and [int(m[0]) for m in matches] == list(range(1, 61))
    raw = [dict(index=int(m[0]), specimen_id=m[1], flaw_size=float(m[2]),
                **{'inspector_%d' % (k + 1): int(x) for k, x in enumerate(m[3].split())}) for m in matches]
    table(data / 'hit_miss_B1.csv', raw)
    # Manually transcribed from the rendered Table B.2, retained for visual audit.
    values = [(-4.06,38.06,.164,.214),(-2.88,30.12,.169,.231),(-4.63,34.51,.198,.256),
        (-5.14,49.20,.149,.188),(-7.34,59.38,.161,.197),(-3.73,29.67,.200,.265),
        (-4.99,46.20,.156,.197),(-6.37,58.40,.147,.180),(-8.74,76.64,.143,.170),(-6.23,47.85,.176,.222)]
    coefficients = [dict(inspector=i + 1, beta0=v[0], beta1=v[1], a90=v[2], a90_95=v[3],
        intact_sites=60, false_calls=0, source_pdf_page=42) for i, v in enumerate(values)]
    table(data / 'published_B2.csv', coefficients)
    shutil.copy2(run / 'inspection/nasa2022_b2.png', sources / 'NASA_2022_B2.png')
    fit_results = []
    x = np.array([r['flaw_size'] for r in raw]); X = np.column_stack([np.ones(60), x])
    for c in coefficients:
        y = np.array([r['inspector_%d' % c['inspector']] for r in raw])
        result = minimize(lambda b: np.logaddexp(0, X @ b).sum() - y @ X @ b,
                          np.array([c['beta0'], c['beta1']]),
                          jac=lambda b: X.T @ (expit(X @ b) - y), method='BFGS',
                          options={'gtol':1e-9})
        fit_results.append(dict(inspector=c['inspector'], beta0=float(result.x[0]),
            beta1=float(result.x[1]), a90=float((np.log(9)-result.x[0])/result.x[1]),
            published_beta_difference=(result.x - [c['beta0'],c['beta1']]).tolist(),
            gradient_max=float(abs(result.jac).max())))
    # Rounded published inputs need not regenerate unrounded source fits exactly.
    assert all(abs(v['a90'] - c['a90']) < .0015 for v,c in zip(fit_results,coefficients))
    save(run / 'raw_data_reconstruction64.json', dict(flawed_sites=60, inspectors=10,
        hit_miss_observations=600, false_call_trials_per_inspector=60,
        synthetic=True, raw_table_regex_count=60, independent_numeric_check=fit_results,
        source_beta_rounding_and_flaw_size_rounding=True))
    requests, gold = [], []
    u = 1 - .05 ** (1 / 60)
    for c in coefficients:
        for amin in [.16, .20, .24, .30]:
            for prior in [.02, .50]:
                cid = 'Q%03d' % (len(requests) + 1)
                qualified = amin >= c['a90_95']
                risk = prior * .1 / (prior * .1 + (1 - prior) * (1 - u))
                requests.append(dict(case_id=cid, inspector=c['inspector'],
                    fault_size_min=amin, fault_size_max=.423, prior_fault_probability=prior,
                    observed_result='negative', maximum_residual_risk=.01))
                gold.append(dict(case_id=cid, capability_qualified=qualified,
                    residual_fault_risk_upper=risk if qualified else None,
                    close_declared_fault_branch=qualified and risk <= .01,
                    action=('close_scoped_branch' if risk <= .01 else 'obtain_stronger_test')
                           if qualified else 'complete_qualification',
                    false_call_upper_95=u, simultaneous_confidence_lower=.90))
    table(data / 'risk_requests.csv', requests); save(hidden / 'risk_gold.json', gold)
    save(data / 'risk_policy.json', {
        'authorship':'AI-authored controlled evaluation policy, not NASA operational policy',
        'evidence_status':'NASA 2022 Appendix B simulated inspection example; no field-hardware claim',
        'population':'For each request the model has exactly two states: intact H, or F with flaw size in [fault_size_min,fault_size_max]. Other damage and smaller flaws are outside this controlled population.',
        'conditions':'Same inspector, procedure and conditional signal distribution as the published example; logistic monotonicity within the observed .026 to .423 size range is stipulated for this model control. Size units are those of the example; do not invent physical inches.',
        'prior':'The request probability is a stipulated prior for this controlled scenario, not estimated from the case-control specimen counts.',
        'current_certificate':'The individual a90/95 in Table B.2 supplies a 95% lower sensitivity bound .90 at and above that size within the observed range. It supplies no approved sensitivity bound below that size. Published rounded values define this exercise certificate.',
        'false_call_bound':'Use a one-sided exact binomial 95% upper confidence bound from each inspector\'s 0 false calls in 60 intact sites.',
        'release_gate':'After the negative observation, close only the declared F branch if the current certificate supports a worst-case residual P(F|negative) at most .01 with simultaneous confidence at least .90. Otherwise retain that branch and name the missing qualification or stronger evidence required.',
        'current_certificate_only':'Evaluate the existing certificate. Refitting is a consistency audit; new fits do not silently replace the certificate. Proposed recalibration can be described as a next step.',
        'closure_scope':'A permitted scoped risk decision is not a proof of physical absence or whole-machine health.',
        'outside_scope':'No actual maintenance action, losses, cost-effectiveness, unmodelled faults, new field outcomes, or population-wide inspector guarantees.'})
    (data / 'ACCESS.md').write_text('All inputs are read-only at /opt/phm_negative. Full primary PDFs, searchable text, structured transcriptions and source locators are provided. Public table labels are part of this evidence-audit task, not hidden prediction labels. Hidden scoring/other runs are inaccessible. Python: numpy, scipy, pandas, pymupdf. No network is needed.\n')
    # Reuse the existing qualified native CLI isolation runner unchanged.
    for name in ['runner_frozen.py','cli_config_catalog.toml','retention_policy.json']:
        shutil.copy2(PRIOR / name, run / name)
    policy = json.loads((run / 'retention_policy.json').read_text())
    cli = Path(policy['shared_dependencies']['path']) / 'node_modules/.bin/codex'
    actual = subprocess.check_output([str(cli),'--version'],text=True).strip()
    npm = json.loads((run / 'sources/codex_npm.json').read_text())
    release = json.loads((run / 'sources/codex_latest.json').read_text())
    assert actual == 'codex-cli ' + npm['version'] and release['tag_name'] == 'rust-v' + npm['version']
    blocks = PreBlocks(); blocks.feed((run / 'sources/deepseek_cli.html').read_text())
    catalogs=[]
    for b in blocks.blocks:
        try:
            value=json.loads(b)
            if isinstance(value,dict) and 'models' in value: catalogs.append(value)
        except ValueError: pass
    assert len(catalogs)==1 and any(m.get('slug')=='deepseek-flash' for m in catalogs[0]['models'])
    save(run / 'deepseek_cli_catalog_official.json',catalogs[0])
    save(run / 'baseline_qualification64.json',dict(cli=actual,release=release['html_url'],
        published_at=release['published_at'],requested_model='deepseek-flash',
        endpoint='https://api.deepseek.com/',wire_api='responses',
        service_returned_model='Not independently exposed by native CLI; requested alias is not backend revision verification',
        runner_sha256=digest(run/'runner_frozen.py'),catalog_sha256=digest(run/'deepseek_cli_catalog_official.json')))
    save(run / 'runtime_mounts64.json',{'mounts':[{'source':str(data),'destination':'/opt/phm_negative'}]})
    shutil.copy2(PRIOR/'native_schema63.json',run/'native_schema64.json')
    common='''你作为 PHM 检查与证据分析 Agent，在本地执行完整分析。数据在 /opt/phm_negative；先读 ACCESS.md。可写代码、执行 Python、读取全部原始来源；不要联网。交付根目录 analysis.py，使 python3 analysis.py 在空输出目录可重建 predictions.csv、qualification.json、report.md。报告用中文，保留所有请求，说明证据、结论适用范围及下一步动作，CSV 布尔值为 true/false，未知数值留空。提交 schema JSON，results 只有一个对象，case_id 为 TASK_ID，analysis_script/predictions_file/qualification_file/report 分别指向这些相对文件名。1800秒预算，合理利用工具完成核算。\n'''
    physical='''任务：依据 NASA 2015 原始检测能力报告，为 physical_requests.csv 的24项资料完成维护排查证据审查。这是有来源的离线反事实决策，不是新测得的24次设备故障；假定每项待查对象的材料、几何、裂纹类型、检测流程与对应研究一致，当前检查只报告 no_indication。需要决定凭当前资料能否将该研究覆盖的裂纹候选从后续排查中关闭，以及哪些验证仍须进行。按该来源的验证要求判断，不把开展验证计划当成已经完成。可以陈述有前提的未来可用性，但字段表示当前状态。\n输出每项 case_id, point_capability_observed（当前数据是否确实达到过90/95点检出能力）, point_size（仅在达到时填其尺寸）, range_qualification_complete（来源所述尺寸范围的验证及所要求的适用条件目前是否齐全）, close_declared_fault_branch（是否现在可以凭该资料关闭裂纹排查分支）, next_step, evidence_locator。审核报告需区分上述判断，并给出具体可执行的下一步；别把条件性方案登记成已完成的工作。来源完整PDF及文本、每项原始两页输出均已提供。无需重新发明POD算法。\n'''
    risk='''任务：执行 risk_policy.json 中的限定风险决策，为 risk_requests.csv 的80项请求计算阴性检查后的残余故障风险与排查动作。此处所有数据是 NASA 2022 明确标注的模拟示例；政策及先验是本题给定的控制条件。先用 hit_miss_B1.csv 审核每位检查员的逻辑回归参数与a90是否和 published_B2.csv 一致，记录舍入误差，再按现有证书和政策完成决策。允许关闭达到政策门槛的特定故障分支；一律拒答不能完成任务。\n输出每项 case_id, capability_qualified, residual_fault_risk_upper（现有证书可用时的最坏界，不能建立时留空或给有依据的更松界并解释）, close_declared_fault_branch, action（close_scoped_branch/obtain_stronger_test/complete_qualification）, reason。qualification.json 记录10位检查员的拟合核验、误报界、风险计算依据和联合置信度的处理。报告将当前可做的决策与条件性后续方案分开。完整PDF、表格图、原始表B.1与B.2均可核查。\n'''
    jobs=[]
    for task, extra in [('PHYSICAL',physical),('RISK',risk)]:
        prompt=common.replace('TASK_ID',task)+extra
        (run/(task.lower()+'_prompt64.txt')).write_text(prompt)
        for repeat in (1,2):
            jid=task+'_r%d'%repeat;ws=run/'tasks'/jid;(ws/'inputs').mkdir(parents=True)
            shutil.copy2(data/'ACCESS.md',ws/'inputs/ACCESS.md')
            jobs.append(dict(id=jid,condition=task,repeat=repeat,workspace=str(ws.relative_to(run)),
                prompt_file=task.lower()+'_prompt64.txt',schema_file='native_schema64.json',case_ids=[task]))
    save(run/'native_plan64.json',{'jobs':jobs})
    old=json.loads((PRIOR/'native_command63.json').read_text())
    oldroot=str(PRIOR).replace('/data/','/srv/')
    argv=[v.replace(oldroot,str(run)).replace('63.json','64.json') for v in old['argv']]
    # Four independent short analyses, limited by a common CPU/no-GPU budget.
    argv[argv.index('--workers')+1]='4'
    save(run/'native_command64.json',{'cwd':str(run),'argv':argv})
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'iteration':64,
        'hypothesis':'Unverified detectability premises are acknowledged yet lost during candidate pruning or the final action.',
        'tasks':{'PHYSICAL':'24 consecutive original physical-study report audits; no fresh inspection outcomes; all incomplete current release evidence',
                 'RISK':'80 controlled decisions using independently published simulated numerical data; mixed qualified and insufficient conditions'},
        'task_selection_before_LLM':True,'native_repeats_per_task':2,'native_workflows':4,
        'concurrency':4,'wall_seconds_per_workflow':1800,'numerical_threads_per_workflow':4,'GPU_hours':0,
        'requested_model':'deepseek-flash','controller_retries':0,
        'token_model_call_tool_budget':'Native CLI default, bounded by wall time; report observed totals; no paid-service billing estimate available',
        'primary_metrics':['unsupported branch closures','missed allowed closures in RISK','premise-to-action consistency'],
        'secondary_metrics':['point/range distinction','risk bound errors','raw numerical audit','cold replay','runtime/usage'],
        'fixed_comparator':'Declared source qualification plus exact binomial bound and Bayes risk under the disclosed controlled policy; established logic, no new method.',
        'falsifier':'If both repeats preserve current conditions and exploit eligible RISK cases, downrank broad inability; do not tune prompts to induce errors.',
        'semantic_rule':'Conditional future advice is not a present exclusion. Incorrect factual/operative closure is scored separately from caveats.',
        'uncertainty_rule':'RISK comparisons under stipulated population/model/policy only; PHYSICAL absence of data is not proof of any actual crack.',
        'unit':'Two evidence-workflow tasks; source cases/inspectors/repeats are correlated. No population p-values, no E3 increment.',
        'limitations':['Physical source lacks complete exclusion-eligible controls; therefore it alone fails the next_scope63 natural confirmation gate.',
                      'Numerical positive controls are simulated and policy-authored, not field certification.',
                      'Public documents include reported conclusions; tests evidence application, not blind fault diagnosis.',
                      'Two tasks from NASA documents do not demonstrate cross-device commonality.'],
        'decision_gate':'No novelty/method-development claim; source gate and controls can weaken or refine the candidate.',
        'risk_primary_expected_counts':{key:sum(g['action']==key for g in gold) for key in ['close_scoped_branch','obtain_stronger_test','complete_qualification']},
        'physical_point_observed':sum(g['point_capability_observed'] for g in evidence_gold),
        'LLM_random_seed':'Not controllable; independent ephemeral repeats; all outcomes retained'}
    save(run/'protocol_formal_frozen64.json',protocol)
    manifest={str(p.relative_to(data)):{'sha256':digest(p),'bytes':p.stat().st_size}
              for p in sorted(data.rglob('*')) if p.is_file()}
    save(run/'input_manifest64.json',manifest)
    shutil.copy2(Path(__file__),run/'prepare_frozen64.py')
    save(run/'frozen_checksums64.json',{str(p.relative_to(run)):digest(p) for p in
        [run/'protocol_formal_frozen64.json',run/'physical_prompt64.txt',run/'risk_prompt64.txt',run/'prepare_frozen64.py',
         run/'native_plan64.json',run/'native_schema64.json',run/'native_command64.json',run/'runtime_mounts64.json',
         hidden/'physical_gold.json',hidden/'risk_gold.json',run/'input_manifest64.json']})
    print(json.dumps({'prepared':True,'input_bytes':sum(x['bytes'] for x in manifest.values()),
        'physical_point_observed':protocol['physical_point_observed'],
        'risk_expected':protocol['risk_primary_expected_counts'],'jobs':len(jobs)}))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
