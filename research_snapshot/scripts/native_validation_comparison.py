"""Reuse qualified native coding execution for A333's separate legal episodes.

Every RPM holdout gets its own sandbox and fresh sessions. Shared recordings
make the complete study developmental; no episode is a new independent source.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

from controlled_feedback import save, sha, utc
from validation_reliability_policy import best, gate, inspect

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
PRIOR = Path('/data/yyf-zyq/PHM_20261010/20261024T143100Z_m10extension329')

PROMPT = '''Review one measured PHM diagnosis task using the supplied immutable complete candidate programs. This is bounded program selection, not new model generation. Read /data/contract.json, /data/cases.json and the phmforge evidence tool. You may write and execute Python to analyze the supplied real evidence and candidate recipes. Final observations, final labels, final operating conditions, final scores, other holdout tasks and other Agents are unavailable. Do not use a network or other model APIs, create data, fit new models, or spawn Agents.

Each validation result comes from actual fitted whole programs on complete acquisition records. Two legal validation views are supplied: record-group CV and complete-operating-condition CV. Additional nested trials expose how each view behaved on other LEGAL training conditions. In each nested trial, both the trial condition and the hidden final condition were excluded from the inner training and validation. You may inspect, recompute, compare, or disregard these views and trials; no selection rule is imposed. The detailed_rows file includes the same raw evidence. No view is guaranteed to predict the unknown deployment loss.

For EACH of the four supplied possible incumbents, choose one candidate to retain or replace it. These are counterfactual starts on this single task, not four independent tasks. Minimize whole-acquisition classification error on unseen conditions while retaining useful improvements. Each complete program extracts fixed window features, uses its train-fitted scaler and LDA/ExtraTrees model, then votes over windows of each record. Do not assume an incumbent is good or bad. Explain the actual evidence and uncertainty.

Submit exactly four rows (case,incumbent,selected,reason) via phmforge select, and save selection.json and report.md. If needed, tools are also available with python3 /tools/program_update_tools.py client --request followed by a JSON request. Use evidence before selection. Write scratch only under /app/.runtime and never /tmp. Budget:1200seconds,50model requests,2500000input tokens,65536output tokens,8192output per request,2numeric threads. Choices are immutable and externally evaluated only after all sessions seal. A prose claim is not an evaluation result.
'''


def read(path):return json.loads(Path(path).read_text())


def closure(names):
    pending=list(names);found=set()
    while pending:
        name=pending.pop()
        if name in found:continue
        path=PROJECT/'scripts'/(name+'.py')
        assert path.exists(),name
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else
                     [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [])
            pending += [m for m in modules if (PROJECT/'scripts'/(m+'.py')).exists() and m not in found]
    return sorted(found)


def freeze(root, study):
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    release=read(root/'audit/CLI_SOURCE334.json'); prior=read(PRIOR/'CAMPAIGN_CONFIG.json')
    assert release['tag']=='rust-v0.164.0' and sha(Path(prior['cli']))==prior['cli_sha256']
    names=closure(['native_validation_comparison','validation_reliability_policy','screen_program_updates',
                   'run_validation_transfer','program_update_tools','qualify_native_cli_mcp'])
    for name in names:shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in ['program_update_tools.py','controlled_feedback.py','prepare_fault_extensions.py',
                 'acquire_forge_pdmb_sources244.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'shared_tools'/name)
    (root/'scripts').mkdir();shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    config=dict(iteration='A334',frozen_utc=utc(),source_study=str(study),source_protocol_sha256=sha(study/'PROTOCOL333.json'),
        model='deepseek-flash',cli=prior['cli'],cli_sha256=prior['cli_sha256'],
        episode_seconds=1200,episode_requests=50,episode_input_tokens=2500000,episode_output_tokens=65536,
        output_per_request=8192,numerical_threads=2,maximum_parallel_agents=4,parallel_case_workers=2,repeats=2,
        case_count=4,independent_development_studies=1,confirmation_scenarios=0,
        allowed_information='OnecaselegalgroupCVandnestedtrialvalues;allmethodshaveidenticalevidenceandtools.',
        main_method='SameordinaryAgentproposal,thenacceptreplacementonlyifstrictlybetterunderthenested-trial-selectedvalidationview;otherwisekeepincumbent.',
        ablations=['SamegatewithfixedrecordCV','SamegatewithfixedconditionCV'],
        fixed_controls=['keep','record_cv','condition_cv','worst_condition','calibrated_rank'],
        main_anchor='raw_stats__lda_shrink_auto;other3startssecondarycounterfactuals,notnewscenarios.',
        primary='Whole-acquisition classificationerrorandselectionregret;reportharmandgaincapture,windows/F1andcost.',
        model_calls_shared_between_ordinary_and_gate=True,extra_gate_model_calls=0,
        common_evidence_compute_charged_to_all=True,new_PHM_fit_budget=0,
        maximum_new_model_requests=400,maximum_new_disk_GiB=4,
        source_outer_scores_not_read_before_agent_and_gate_choice_seal=True,
        scope='Development-onlyrealdata;fourcorrelatedLOOtasksarenotmajorityconfirmation.',
        fallback='Retain everyfailure;scorefallbackkeepseparately,neveramethodgain.',
        novelty_claimed=False,no_method_or_prompt_tuning_after_source_score_join=True,
        preparation_file='PREPARATION334.json')
    save(root/'CAMPAIGN_CONFIG.json',config);save(root/'PROMPT_FREEZE.json',dict(ordinary=PROMPT))
    save(root/'CODE_FREEZE.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    out=PROJECT/'results/native334';out.mkdir(exist_ok=True)
    save(out/'PROTOCOL334.json',config)
    return config


def source_audit(study, cases, destination):
    """Cold predictions and independent record voting, restricted to legal IDs."""
    fits=read(study/'FITS333.json'); models={(tuple(f['training_records']),f['recipe']):f for f in fits}
    with np.load(study/'FEATURES333.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    checks=[];cache={}
    for case in cases:
        allowed=set(case['allowed_training_records']);jobs=[]
        def add(views):
            for name,view in views.items():
                for j,recipe in enumerate(case['candidate_ids']):
                    for k,fold in enumerate(view['folds']):jobs.append((fold['training'],fold['evaluation'],recipe,view['group_losses'][j][k]))
        add(case['views'])
        for trial in case['calibration']['trials']:
            add(trial['inner_views'])
            for j,recipe in enumerate(case['candidate_ids']):jobs.append((trial['available_records'],trial['evaluation_records'],recipe,trial['observed_losses'][j]))
        for training,evaluation,recipe,expected in jobs:
            assert set(training).isdisjoint(evaluation) and set(training)|set(evaluation) <= allowed
            key=(tuple(sorted(training)),tuple(sorted(evaluation)),recipe)
            if key not in cache:
                fit=models[(key[0],recipe)];path=study/fit['checkpoint'];assert sha(path)==fit['model_sha256']
                model=joblib.load(path);indices=np.flatnonzero(np.isin(d['record'],evaluation));features=recipe.split('__')[0]
                predicted=model.predict(d[features][indices]);records=d['record'][indices];truth=d['labels'][indices]
                wrong=0
                for rec in sorted(evaluation):
                    mask=records==rec;counts=[int(np.sum(predicted[mask]==label)) for label in range(4)]
                    guess=counts.index(max(counts));labels=set(truth[mask].tolist());assert len(labels)==1
                    wrong+=guess!=next(iter(labels))
                cache[key]=wrong/len(evaluation)
            assert abs(cache[key]-expected)<1e-12
            checks.append(dict(case=case['id'],expected=expected,actual=cache[key],legal_ids_only=True))
    save(destination,dict(passed=True,checks=len(checks),actual_cold_predicts=len(cache),
        all_complete_record_votes_match=True,held_final_records_not_used=True,new_fits=0))


def prepare(root):
    cfg=read(root/'CAMPAIGN_CONFIG.json');study=Path(cfg['source_study'])
    assert sha(study/'PROTOCOL333.json')==cfg['source_protocol_sha256']
    assert read(study/'SOURCE_CAPTURE333.json')['passed']
    paths=sorted((study/'public').glob('*/case.json'));assert len(paths)==4
    cases=[read(p) for p in paths];source_audit(study,cases,root/'audit/LEGAL_COLD_AUDIT334.json')
    for k,case in enumerate(cases):
        child=root/'case_runs'/('c'+str(k));child.mkdir()
        for name in ['code','shared_tools','public/rows','scripts','audit','outputs','runtime/tmp','runtime/mpl']:(child/name).mkdir(parents=True)
        for folder in ['code','shared_tools']:
            for p in (root/folder).glob('*.py'):shutil.copy2(p,child/folder/p.name)
        shutil.copy2(root/'scripts/isolated_model_gateway.py',child/'scripts/isolated_model_gateway.py')
        for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json','PROMPT_FREEZE.json','CODE_FREEZE.json']:
            shutil.copy2(root/name,child/name)
        public=copy.deepcopy(case);public['id']='phm_program_choice'
        public['original_incumbent']=public.pop('incumbent')
        public['metric']='Whole-acquisitionclassificationerror,majorityvoteovernonoverlappingwindows.'
        public['calibration'].pop('selected_view',None);public['calibration'].pop('mean_calibration_regret',None)
        for trial in public['calibration']['trials']:
            trial.pop('view_choices',None);trial.pop('view_regrets',None)
        save(child/'public/cases.json',[public]);save(child/'public/rows/phm_program_choice.json',public)
        save(child/'public/contract.json',dict(source='Author-acquiredCOMFAULDArecords,documented6vibrationchannels,actualclock51200Hz.',
            classes=['normal','unbalance','horizontal_misalignment','vertical_misalignment'],
            feature_definitions='48stats:mean,logstd,logRMS,peaktopeak,skewness,kurtosis,crest,meanabs perchannel;raworperwindowchannelRMSnormalization.',
            recipes='Full2x2:raw/normalizedstats;StandardScalertrain-only;LDA_lsqrshrinkauto/ExtraTrees128leaf2seed333.',
            information_boundary='Onlythiscaselegalrecords;finaldata/conditions/labels/scoresandothercasesabsent.',
            candidate_scope='Fourimmutablecompleteprograms;noglobaloptimalityorfreecodingclaim.',
            common_evidence='BothviewsandnestedtrialsareavailabletoordinaryAgentandfixedworkflow,withidenticalcostallocation.'))
        save(child/'PREPARATION334.json',dict(passed=True,source_public_sha256=sha(paths[k]),source_cold_audit=sha(root/'audit/LEGAL_COLD_AUDIT334.json')))
        save(child/'CAMPAIGN_CONFIG.json',dict(cfg,case_source_id=case['id']))
        fixed=[]
        for old in public['candidate_ids']:
            evidence=inspect(public,old);i=public['candidate_ids'].index(old)
            selected={'keep':old,'calibrated_rank':evidence['reranked']}
            for name in ['record_cv','condition_cv']:selected[name]=public['candidate_ids'][best(evidence['risk'][name],i)]
            selected['worst_condition']=public['candidate_ids'][best(np.max(public['views']['condition_cv']['group_losses'],axis=1),i)]
            fixed.append(dict(incumbent=old,selected=selected,evidence=evidence))
        save(child/'FIXED_CHOICES.json',dict(choices=fixed,final_scores_joined=False))
    save(root/'PREPARATION334.json',dict(passed=True,cases=4,ordinary_episodes=8,new_PHM_fits=0,
        case_public_files={str(p.relative_to(root)):sha(p) for p in root.glob('case_runs/*/public/**/*.json')}))


def child_agents(child):
    import screen_program_updates as runner
    runner.OUT=child/'outputs'
    runner.agents(child)


def agents(root):
    assert read(root/'PREPARATION334.json')['passed']
    env=dict(os.environ);env.update(PYTHONPATH=str(DEPS)+':'+str(root/'code'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),MPLCONFIGDIR=str(root/'runtime/mpl'))
    rows=[]
    def execute(child):
        command=[str(PYTHON),'-B',str(root/'code/native_validation_comparison.py'),'child','--run',str(child)]
        with (root/'logs'/(child.name+'.log')).open('w') as log:
            p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        return dict(case=child.name,returncode=p.returncode)
    save(root/'HANDLE334.json',dict(controller_pid=os.getpid(),status='running',phase='native_agents'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,c) for c in sorted((root/'case_runs').iterdir())]):
            rows.append(future.result());save(root/'NATIVE_PROGRESS334.json',rows)
    save(root/'HANDLE334.json',dict(controller_pid=os.getpid(),status='terminal',phase='native_agents',rows=rows))


def seal(root):
    assert read(root/'HANDLE334.json')['status']=='terminal'
    all_choices=[]
    for child in sorted((root/'case_runs').iterdir()):
        case=read(child/'public/cases.json')[0]
        result=read(child/'AGENT_CHOICES.json')
        for episode in result['episodes']:
            if not episode['completed'] or episode['fallback']:
                all_choices.append(dict(case=child.name,episode=episode['id'],completed=False,fallback=True,choices=None));continue
            decisions=[]
            for row in episode['choice']['choices']:
                variants={};evidence={}
                for method,view in [('calibrated_gate','calibrated'),('record_gate','record_cv'),('condition_gate','condition_cv')]:
                    variants[method],evidence[method]=gate(case,row['incumbent'],row['selected'],view)
                decisions.append(dict(incumbent=row['incumbent'],ordinary=row['selected'],variants=variants,evidence=evidence))
            all_choices.append(dict(case=child.name,episode=episode['id'],completed=True,fallback=False,choices=decisions))
    save(root/'ALL_METHOD_CHOICES_SEALED334.json',dict(utc=utc(),episodes=all_choices,
        code_sha256=sha(root/'code/validation_reliability_policy.py'),source_final_score_join_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','agents','child','seal']);p.add_argument('--run',type=Path,required=True);p.add_argument('--study',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze(a.run,a.study)
    elif a.action=='child':child_agents(a.run)
    else:globals()[a.action](a.run)
