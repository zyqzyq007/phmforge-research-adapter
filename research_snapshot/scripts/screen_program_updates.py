"""A328: bounded screening of ten program-selection mechanisms on real banks.

Researcher-opened archival tasks are development evidence, never fresh
confirmation. Every candidate is also enumerated as an incumbent in a separate
counterfactual update panel; this creates no new physical observations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

import numpy as np

from controlled_feedback import save, sha, utc
from program_selection_policies import METHODS, run_all, choose, validate
from run_full_benchmark_campaign import clean_env
from run_frozen_candidate_review import source_context

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
SOURCES = {
    'rotor': Path('/data/yyf-zyq/PHM_20261010/20261024T103209Z_hypotheses325'),
    'gear': Path('/data/yyf-zyq/PHM_20261010/20261022T174700Z_feedback285'),
    'rul': Path('/data/yyf-zyq/PHM_20261010/20261022T081740Z_feedback278'),
}
OUT = PROJECT/'results/policy_screen328'
CODE = ['screen_program_updates.py', 'program_selection_policies.py', 'program_update_tools.py',
        'controlled_feedback.py', 'controlled_program_checks.py', 'prepare_fault_extensions.py',
        'acquire_forge_pdmb_sources244.py', 'run_full_benchmark_campaign.py', 'run_validation_transfer.py',
        'replay_diagnostic_cli.py', 'isolated_model_gateway.py', 'qualify_native_cli_mcp.py',
        'evaluate_controlled_feedback.py', 'run_frozen_candidate_review.py',
        'real_xjtu_rul_model.py', 'real_bearing_rul_model.py', 'real_rul_model.py']

PROMPT = '''You are reviewing genuinely fitted PHM programs using an existing coding Agent environment. This is a bounded candidate-selection task, not free model development. Read /data/contract.json, /data/cases.json, and use the phmforge evidence tool for every case. Each candidate is a complete immutable trained predictor. Whole-program CV and development predictions are real and hash-bound; cold development replays have been verified. No new fitting is needed or permitted. You may write Python to recompute metrics, analyze groups, and choose programs. Raw validation rows are in /data/rows/. Final observations, labels, conditions, scores, previous final conclusions, and other Agents are not available. Do not access the network or other model APIs, guess final outcomes, or spawn other agents.

For each case, choose a candidate for EACH supplied possible incumbent. This exhaustive panel evaluates sensitivity to starting-program quality; these are repeated update decisions on the same measured data, not independent datasets. Keep or replace the supplied incumbent using the available evidence. Do not assume that an incumbent is good or bad. Minimize future loss under the stated case metric, retaining useful improvements while avoiding harmful replacements. No rule about which validation group predicts the future is supplied. You may examine recipes and full group evidence, quantify uncertainty and use reasonable analysis. No invented safety or distribution-shift guarantee is allowed. All candidates have equal legal information and opportunity.

Submit all (case,incumbent,selected,reason) rows once using phmforge select, save selection.json and report.md. Use evidence for each case first; retain uncertainty and negative evidence. Tools are also accessible as:
python3 /tools/program_update_tools.py client --request '{"action":"evidence","case_id":"CASE"}'
python3 /tools/program_update_tools.py client --request '{"action":"select","choices":[...]}'
The actual immutable chosen program is scored externally after all runs seal their choices. A prose score is not an outcome. Store all scratch under /app/.runtime, never /tmp. Budget:1200seconds,50model requests,2500000input tokens,65536output tokens,8192output/request,2numeric threads. Do not retry this task with another model or changed candidates. Your report should describe actual checks and selection limitations.
'''


def update(root, **changes):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text()) if (root/'CAMPAIGN_CONFIG.json').exists() else {}
    path = root/config.get('status_file','STATUS328.json')
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updated_utc=utc(), **changes)
    destination=PROJECT/'results'/config['result_key'] if config.get('result_key') else OUT
    save(path, state); save(destination/'current_run.json', dict(state, run=str(root)))


def env_for(root):
    env = clean_env(root)
    env.update(PYTHONPATH=str(DEPS)+':'+str(root/'code')+':'+str(PROJECT/'scripts'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def freeze(root):
    root.mkdir(exist_ok=False)
    for name in ['runtime/tmp', 'runtime/mpl', 'logs', 'audit', 'code', 'shared_tools', 'public/rows', 'evaluation_inputs']:
        (root/name).mkdir(parents=True)
    contexts = {k: source_context(v) for k, v in SOURCES.items()}
    original = json.loads((SOURCES['rotor']/'CAMPAIGN_CONFIG.json').read_text())
    url = 'https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'PHM-research-provenance'}), timeout=30) as response:
        release = json.load(response)
    assert release['tag_name'] == 'rust-v0.164.0', 'Resolve updated stable CLI before formal freeze'
    assert sha(Path(original['cli'])) == original['cli_sha256']
    save(root/'audit/CLI_SOURCE.json', dict(url=url, checked_utc=utc(), tag=release['tag_name'], published_at=release['published_at']))
    config = dict(iteration='A328', user_authorization='尝试10个方法 来真正形成有效的提升', frozen_utc=utc(),
        methods=METHODS, method_kind='Ten executable fixed selection layers; not ten novel Agent systems or official paper reproductions.',
        source_archives={k: dict(path=str(v), manifest_sha256=contexts[k][1]['manifest_sha256']) for k, v in SOURCES.items()},
        original_anchors=dict(rotor='P00', gear='candidate_1 (first chronological proposal)', rul='rf200'),
        panels=['9 original-anchor decisions', '88 exhaustive candidate-as-incumbent counterfactual decisions'],
        cases='2 rotor acquisition blocks; all6 G1 search pools from the same physical split;1 XJTU frozen bank. No outcome-based case filtering.',
        repeats=2, baseline='Fresh ordinary native coding Agent on exactly the same legal evidence and candidates',
        cli=original['cli'], cli_sha256=original['cli_sha256'], model='deepseek-flash',
        maximum_parallel_agents=2, episode_seconds=1200, episode_requests=50, episode_input_tokens=2500000,
        episode_output_tokens=65536, output_per_request=8192, numerical_threads=2,
        new_fit_budget=0, real_training_reused=True, new_final_forward_budget=0,
        validation_replay='All72 G1 deployed classifiers;all6 XJTU bank models;all60 rotor CV/full checkpoint identities (72 load operations because blend reuses components).',
        primary='Source-balanced normalized selection regret on original anchors; also native task losses, harmful updates and useful improvement capture.',
        normalization='(selected final loss - candidate minimum)/(candidate maximum-candidate minimum); zero when range=0. Final values used only by external evaluator, never selection.',
        weighting='Within pool average repetitions/anchors, then within source average related pools, then equal-weight3 sources. Also print unaggregated native-unit results.',
        opportunity='Loss of supplied incumbent minus best candidate loss; gain capture=min(1,max(0,old-selected)/opportunity),undefined when no opportunity.',
        improvement_rule='Exploratory promise requires lower source-balanced regret than both ordinary repeats, positive gain capture where opportunity exists, and no higher source-balanced harm. Fresh confirmation still required.',
        statistical_policy='No significance claim from9 related pools or88 overlapping anchors. Ten-way search is exploratory; all10 shown, no best-seed selection.',
        resampling='4000 paired group bootstrap resamples,seed328,upper Bonferroni .05/(K-1) quantile; descriptive sensitivity,not OOD coverage guarantee.',
        sign_test='Exact one-sided group win sign test,discard ties,Bonferroni .05/(K-1);group dependence limits inferential interpretation.',
        unknown_deployment_information_not_added=True, researcher_previously_knows_final_scores=True,
        confirmation_experiment=False, no_adaptive_method_threshold_changes=True,
        source_limitations=['One laboratory rotor;two related blocks','Six G1 candidate pools share one gear/split','XJTU has3 final bearings;fixed bank excludes adaptive programs',
            'All final source outcomes were opened previously;this is development screening','Exhaustive incumbents are counterfactual decisions,not naturally observed Agent failures'],
        maximum_new_disk_GiB=8, maximum_new_model_requests=100,
        failure_policy='Preserve both ordinary attempts including timeout/invalid/missing choices. Fallback keep is scored separately and never advertised as method improvement.')
    save(root/'CAMPAIGN_CONFIG.json', config); save(OUT/'PROTOCOL328.json', config)
    save(root/'PROMPT_FREEZE.json', dict(ordinary=PROMPT))
    for name in CODE:
        shutil.copy2(PROJECT/'scripts'/name, root/'code'/name)
    for name in ['program_update_tools.py', 'controlled_feedback.py', 'prepare_fault_extensions.py', 'acquire_forge_pdmb_sources244.py', 'isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name, root/'shared_tools'/name)
    for name in ['base_cli_config.toml', 'model_catalog187.json']:
        assert sha(SOURCES['rotor']/name) == contexts['rotor'][0][name]['sha256']
        shutil.copy2(SOURCES['rotor']/name, root/name)
    save(root/'CODE_FREEZE.json', {name: sha(root/'code'/name) for name in CODE})
    policy = json.loads((SOURCES['rotor']/'retention_policy.json').read_text())
    policy.update(final_results_not_yet_opened_at_policy_creation=False,
                  retain=['Frozen methods/protocol,legal evidence,all decisions,real cold replay logs,model/source SHA references,full model traffic,independent evaluation'])
    save(root/'retention_policy.json', policy)
    save(root/'OWNERSHIP.json', dict(user_request=config['user_authorization'], created_utc=utc(), run=str(root), archive='/data/yyf-zyq/PHM_20261010/'+root.name))
    update(root, status='protocol_frozen', new_agent_sessions=0, new_fits=0, new_final_scores_joined=False)
    print(json.dumps(dict(run=str(root), methods=10, baseline_sessions=2)))


def prepare(root):
    from evaluate_controlled_feedback import independent, cold
    assert not (root/'PREPARATION328.json').exists()
    bindings = []
    contexts = {k: source_context(v) for k, v in SOURCES.items()}
    def bound(source, relative):
        path = SOURCES[source]/relative
        row = contexts[source][0][relative]
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        bindings.append(dict(source=source, path=relative, sha256=row['sha256'], bytes=row['bytes']))
        return path
    def read(source, relative):
        return json.loads(bound(source, relative).read_text())
    cases = []; cold_jobs = []; checks = []
    # Genuine rotor record-majority evidence, including every legal condition.
    for block in ['block78', 'block910']:
        reports = read('rotor', 'bank/'+block+'/VALIDATION_REPORTS.json')
        ids = ['P00', 'P10', 'P01', 'P11', 'P11B']
        groups = []
        for part in ['cv', 'development']:
            for load in sorted({r['load'] for r in reports['P00'][part]['records']}):
                groups.append(dict(id=part+':'+str(load), split=part, condition_load_Nm=load, unit='whole acquisition record'))
        risk = []; rows = {}
        for cid in ids:
            records = reports[cid]; per = []
            for group in groups:
                rr = [r for r in records[group['split']]['records'] if r['load'] == group['condition_load_Nm']]
                per.append(sum(r['actual'] != r['predicted'] for r in rr)/len(rr))
            risk.append(per); rows[cid] = {part: records[part]['records'] for part in ['cv', 'development']}
            bound('rotor', f'bank/{block}/{cid}_predictions.npz')
        for file in sorted((SOURCES['rotor']/'bank'/block).rglob('*.joblib')):
            bound('rotor', str(file.relative_to(SOURCES['rotor'])))
        for filename in ['training_designs.npz', 'development_designs.npz', 'folds.json']:
            bound('rotor', f'bank/{block}/'+filename)
        for part in ['train','val']:
            bound('rotor', f'hidden/{block}/{part}.npz')
        cid = 'rotor_'+block
        case = dict(id=cid, source='rotor', metric='whole-record classification error', candidate_ids=ids,
            candidate_descriptions=['raw+spatial+spectral,old ET','spectral,old ET','raw+spatial+spectral,new ET','spectral,new ET','.9 spectral new ET+.1 raw standardized logistic'],
            groups=groups, group_losses=risk, development_columns=[5], original_incumbent='P00',
            limitations='One laboratory motor;whole-load CV;unknown final conditions. Condition groups are not independent motors.')
        cases.append(case); save(root/'public/rows'/(cid+'.json'), rows)
        cold_jobs.append(('rotor', block))
    # All six original G1 pools, including successful and unsuccessful recipes.
    config = read('gear', 'CAMPAIGN_CONFIG.json')
    with np.load(bound('gear', 'hidden/train.npz'), allow_pickle=False) as z:
        train = {k:z[k] for k in ['sample_ids','labels','groups','condition_id']}
    with np.load(bound('gear', 'hidden/val.npz'), allow_pickle=False) as z:
        dev = {k:z[k] for k in ['sample_ids','labels','groups','condition_id']}
        input_dir = root/'evaluation_inputs/gear_dev'; input_dir.mkdir(exist_ok=True)
        if (input_dir/'observations.npz').exists():
            with np.load(input_dir/'observations.npz',allow_pickle=False) as prior:
                assert np.array_equal(prior['waveforms'],z['waveforms']) and np.array_equal(prior['sample_ids'],z['sample_ids'])
        else:
            np.savez_compressed(input_dir/'observations.npz', waveforms=z['waveforms'], sample_ids=z['sample_ids'])
    for original in config['cases']:
        old = original['id']; state = read('gear', old+'/broker/STATE.json')
        trials = state['trials']; assert len(trials) == 12 and all(t['status']=='complete' for t in trials)
        groups = [dict(id=part+':'+str(cond), split=part, operating_condition=str(cond), unit='recorded operating-condition segment')
                  for part, data in [('cv',train),('development',dev)] for cond in sorted(set(data['condition_id'].tolist()))]
        risk=[]; rows={}; recipes=[]; ids=[]
        for trial in trials:
            name=trial['id']; prefix=old+'/broker/'+name+'/'; ids.append(name); recipes.append(trial['recipe'])
            report=read('gear',prefix+'fit_report.json')
            assert report['real_fits']==4 and report['cold_reload_equal']
            checkpoint=bound('gear',prefix+'checkpoint.joblib')
            with np.load(bound('gear',prefix+'predictions.npz'),allow_pickle=False) as z:
                pred={p:z[p] for p in ['cv','development']}
            values=[]; detailed={}
            for part,data in [('cv',train),('development',dev)]:
                score=independent(data['labels'],pred[part],config['classes'])
                assert score['confusion']==report[part]['confusion']
                detailed[part]=[dict(sample_id=str(i),actual=int(y),predicted=int(p),group=str(g),condition=str(c))
                    for i,y,p,g,c in zip(data['sample_ids'],data['labels'],pred[part],data['groups'],data['condition_id'])]
                for cond in sorted(set(data['condition_id'].tolist())):
                    mask=data['condition_id']==cond
                    values.append(1-independent(data['labels'][mask],pred[part][mask],config['classes'])['macro_f1'])
                checks.append(dict(case='gear_'+old,candidate=name,split=part,score_recomputed=True))
            risk.append(values);rows[name]=detailed
            cold_jobs.append(('gear','gear_'+old,name,checkpoint,pred['development']))
        cid='gear_'+old
        cases.append(dict(id=cid,source='gear',metric='1 - macro-F1 (all3 original stage labels)',candidate_ids=ids,
            candidate_descriptions=recipes,groups=groups,group_losses=risk,development_columns=[3],original_incumbent=ids[0],
            limitations='All6 candidate pools share one measured gear and identical train/development/final observations. Do not count independent experiments.'))
        save(root/'public/rows'/(cid+'.json'),rows)
    # Existing XJTU six-model bank. Preserve the complete old prediction scope.
    manifest=read('rul','references/candidates/manifest.json'); models=manifest['models']
    assert len(models)==6
    groups=[dict(id=part+':'+str(row['unit_id']),split=part,bearing_id=row['unit_id'],unit='whole bearing')
        for part,key in [('cv','training_cv'),('development','development')] for row in models[0][key]['per_bearing']]
    risk=[]
    for row in models:
        risk.append([record['mae_minutes'] for key in ['training_cv','development'] for record in row[key]['per_bearing']])
        for key in ['training_cv','development']:
            assert abs(np.mean([r['mae_minutes'] for r in row[key]['per_bearing']])-row[key]['mean_bearing_mae_minutes'])<1e-9
        bound('rul','references/candidates/'+row['checkpoint'])
        bound('rul','references/candidates/'+row['name']+'_OOF.npz')
        for fold in range(3):bound('rul','references/candidates/cv_models/'+row['name']+'_fold'+str(fold)+'.joblib')
    for file in ['train.npz','val.npz']:bound('rul','public_data/datasets/XJTU/'+file)
    bound('rul','references/training_cv_folds.json')
    cases.append(dict(id='xjtu_bank',source='rul',metric='mean bearing MAE in minutes',candidate_ids=[r['name'] for r in models],
        candidate_descriptions=[r['name'] for r in models],groups=groups,group_losses=risk,
        development_columns=[9,10,11],original_incumbent='rf200',
        limitations='CV uses3 whole-bearing folds;9 correlated bearing error summaries. Development3 bearings,one query each;CV has denser queries. Candidate scope is unchanged6-model bank,not adaptive original code.'))
    save(root/'public/rows/xjtu_bank.json',{'models':models,'note':'Detailed legal rows and exact checkpoints remain hash-bound in source; per-bearing loss summaries are provided.'})
    cold_jobs.append(('rul',))
    assert len(cases)==9 and sum(len(c['candidate_ids']) for c in cases)==88
    for case in cases:validate(case)
    save(root/'public/cases.json',cases)
    save(root/'public/contract.json',dict(
        task='PHMForge-linked real program-selection research adaptation',final_information_unavailable=True,
        data='Measured rotor,measured G1 planetary gear,measured XJTU bearing RUL',
        scope='Choose from actual immutable complete predictors; no new neural training or code modification.',
        group_losses='Rows follow candidate_ids,columns follow groups. Smaller is better. All complete held-out CV/development predictions were generated with training-only fits.',
        group_mean_is_not_a_future_guarantee=True,
        anchors='Every candidate is considered once as a supplied incumbent; these are counterfactual update choices on existing records.',
        evidence='cases/evidence tools and /data/rows; candidate labels are local opaque recipe IDs,not an indication of final quality.'))
    save(root/'audit/SOURCE_BINDINGS.json',bindings)
    # Cold execution uses immutable source checkpoints. No source final data is
    # passed to any worker; G1 uses the established bwrap label-free executor.
    outcomes=[]
    def replay_one(job):
        if job[0]=='gear':
            _,case,candidate,checkpoint,expected=job
            saved=root/'evaluation282'/case/candidate/'gear_dev/predictions.json'
            reused=saved.exists()
            actual=json.loads(saved.read_text())['predictions'] if reused else cold(root,case,candidate,'gear_dev',checkpoint=checkpoint)
            keyed={r['sample_id']:r['predicted_label'] for r in actual}
            got=np.asarray([keyed[str(i)] for i in dev['sample_ids']])
            assert np.array_equal(got,expected)
            return dict(kind='gear',case=case,candidate=candidate,passed=True,checkpoint_reloads=1,new_fits=0,reused_completed_cold_prediction=reused)
        folder=root/'audit/cold_checks'/('_'.join(job));folder.mkdir(parents=True)
        result=subprocess.run([str(PYTHON/'bin/python3.12'),'-B',str(root/'code/screen_program_updates.py'),
            'cold','--run',str(root),'--kind',job[0],*(['--block',job[1]] if len(job)>1 else [])],
            env=env_for(root),capture_output=True,text=True,timeout=300)
        (folder/'stdout').write_text(result.stdout);(folder/'stderr').write_text(result.stderr)
        assert result.returncode==0,result.stderr[-1200:]
        return json.loads(result.stdout.splitlines()[-1])
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(replay_one,job) for job in cold_jobs]
        for f in as_completed(futures):
            outcomes.append(f.result())
            update(root,status='real_validation_replay',completed_replay_jobs=len(outcomes),total_replay_jobs=len(cold_jobs))
    save(root/'audit/REAL_REPLAY_CHECKS.json',outcomes)
    save(root/'audit/METRIC_CHECKS.json',checks)
    result=dict(passed=True,utc=utc(),cases=9,source_families=3,candidate_programs=88,counterfactual_anchor_decisions=88,
        original_anchor_decisions=9,new_fits=0,reused_component_fits=60+288+24,
        cold_replay_jobs=len(outcomes),checkpoint_reloads=sum(r['checkpoint_reloads'] for r in outcomes),
        protocol_sha256=sha(root/'CAMPAIGN_CONFIG.json'),legal_cases_sha256=sha(root/'public/cases.json'),
        final_scores_joined=False,researcher_previously_knows_final_scores=True)
    save(root/'PREPARATION328.json',result);save(OUT/'PREPARATION328.json',result)
    update(root,status='real_evidence_qualified',new_fits=0,planned_agent_sessions=2)
    print(json.dumps(result))


def cold_private(root,kind,block):
    # Only allowlist the source files needed for legal validation. The bwrap
    # worker cannot read original final inputs even though they share an archive.
    folder=root/'audit/cold_checks'/('_'.join([kind,block]) if block else kind)
    folder.mkdir(parents=True,exist_ok=True)
    mounts=[]
    if kind=='rotor':
        source=SOURCES['rotor']
        mounts=[(source/'bank'/block,'/snapshot/bank/'+block),
                (source/'hidden'/block/'train.npz','/snapshot/hidden/'+block+'/train.npz'),
                (source/'hidden'/block/'val.npz','/snapshot/hidden/'+block+'/val.npz')]
        code="""from pathlib import Path
import json
from controlled_program_checks import replay_check,CANDIDATES
reports=[replay_check(Path('/snapshot'),BLOCK,c,Path('/output')/c) for c in CANDIDATES]
print(json.dumps(dict(kind='rotor',block=BLOCK,passed=True,checkpoint_reloads=sum(r['actual_cold_models_replayed'] for r in reports),new_fits=0)))
""".replace('BLOCK',repr(block))
    else:
        source=SOURCES['rul']
        mounts=[(source/'references/candidates','/snapshot/candidates'),
                (source/'references/training_cv_folds.json','/snapshot/folds.json'),
                (source/'public_data/datasets/XJTU','/snapshot/data')]
        code="""import json,joblib,numpy as np
from pathlib import Path
from real_xjtu_rul_model import predict_bank,metrics
with np.load('/snapshot/data/train.npz',allow_pickle=False) as z:train=dict(z)
with np.load('/snapshot/data/val.npz',allow_pickle=False) as z:dev=dict(z)
catalog=json.loads(Path('/snapshot/candidates/manifest.json').read_text())
folds=json.loads(Path('/snapshot/folds.json').read_text())['folds']
count=0
for row in catalog['models']:
 name=row['name'];cv=np.zeros(len(train['labels']))
 for fold in folds:
  mask=np.isin(train['unit_ids'],fold['heldout_unit_ids']);sub={k:(v if v.ndim==0 else v[mask]) for k,v in train.items()}
  model=joblib.load('/snapshot/candidates/cv_models/'+name+'_fold'+str(fold['fold'])+'.joblib')
  cv[mask]=predict_bank(model,sub);count+=1
 with np.load('/snapshot/candidates/'+name+'_OOF.npz',allow_pickle=False) as z:assert np.allclose(cv,z['predictions'],rtol=0,atol=1e-10)
 model=joblib.load('/snapshot/candidates/'+row['checkpoint']);p=predict_bank(model,dev);count+=1
 actual=metrics(p,dev['labels'],dev['unit_ids'])
 assert abs(actual['mean_bearing_mae_minutes']-row['development']['mean_bearing_mae_minutes'])<1e-9
 Path('/output/'+name+'.json').write_text(json.dumps(dict(predictions=p.tolist(),cv_match=True,development=actual)))
print(json.dumps(dict(kind='rul',passed=True,checkpoint_reloads=count,new_fits=0)))
"""
    command=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():command+=['--ro-bind',p,p]
    for src,dst in [(PYTHON,'/opt/python312'),(DEPS,'/opt/deps'),(root/'code','/tools'),*mounts]:
        command+=['--ro-bind',str(src),dst]
    for d in ['tmp','mpl']:(folder/d).mkdir(exist_ok=True)
    command+=['--bind',str(folder),'/output','--proc','/proc','--dev','/dev','--chdir','/output','--remount-ro','/',
              '/opt/python312/bin/python3.12','-B','-c',code]
    save(folder/'command.json',command)
    env=clean_env(root);env.update(PYTHONPATH='/opt/deps:/tools',TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=280)
    (folder/'worker_stderr').write_text(p.stderr)
    assert p.returncode==0,p.stderr[-2000:]
    print(p.stdout,end='')


def fixed(root):
    assert json.loads((root/'PREPARATION328.json').read_text())['passed']
    assert not (root/'FIXED_CHOICES.json').exists()
    cases=json.loads((root/'public/cases.json').read_text())
    start=time.monotonic();decisions=run_all(cases)
    save(root/'FIXED_CHOICES.json',dict(utc=utc(),choices=decisions,seconds=time.monotonic()-start,
        policy_code_sha256=sha(root/'code/program_selection_policies.py'),cases_sha256=sha(root/'public/cases.json'),final_scores_joined=False))
    save(OUT/'FIXED_CHOICES.json',json.loads((root/'FIXED_CHOICES.json').read_text()))
    update(root,status='ten_methods_and_controls_sealed',fixed_methods=10,fixed_control_methods=3,new_fits=0)
    print(json.dumps(dict(methods=10,controls=3,decisions_per_method=88)))


def args_for(root,row):
    cfg=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());run=row['run']
    return SimpleNamespace(cli=Path(cfg['cli']),model='deepseek-flash',output=run/'primary',plan=run/'plans/unused.json',
        timeout=cfg['episode_seconds'],analysis_tools=True,permission_profile='iot-offline',isolation_config=run/'cli_config187.toml',
        model_gateway_socket=run/'ipc/s.sock',model_catalog=run/'model_catalog187.json',runtime_mounts_file=run/'mounts.json',
        workspace_alias='/app',credential_env_file=None,credential_env_name='DEEPSEEK_API_KEY',snapshot_generated_source=True,cleanup_runtime_cache=True)


def agents(root):
    from replay_diagnostic_cli import replay,isolate_filesystem
    from run_validation_transfer import start_gateway
    from qualify_native_cli_mcp import probe
    assert not (root/'AGENT_CHOICES.json').exists()
    cfg=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert json.loads((root/cfg.get('preparation_file','PREPARATION328.json')).read_text())['passed']
    first_case=json.loads((root/'public/cases.json').read_text())[0]['id']
    prompts=json.loads((root/'PROMPT_FREEZE.json').read_text())
    for name,digest in json.loads((root/'CODE_FREEZE.json').read_text()).items():assert sha(root/'code'/name)==digest
    episodes=[]
    for condition,rep in [(c,r) for c in cfg.get('selector_conditions',['ordinary']) for r in range(1,cfg.get('repeats',2)+1)]:
        run=root/'episodes'/(condition+'_'+str(rep))
        for d in ['app/.runtime/tmp','app/.runtime/mpl','app/artifacts','runtime/tmp','runtime/mpl','logs','plans','audit','broker','primary','feedback']:(run/d).mkdir(parents=True)
        (run/'app/AGENTS.md').write_text('Use only the supplied real candidate evidence;no new fitting. Final information unavailable. Write only under /app;temporary files under /app/.runtime.\n')
        text=(root/'base_cli_config.toml').read_text()+'''[mcp_servers.phmforge]
command = "/opt/python312/bin/python3.12"
args = ["-B", "/tools/program_update_tools.py", "mcp"]
required = true
env = { PYTHONPATH = "/opt/phm_python/extra:/tools", PYTHONDONTWRITEBYTECODE = "1", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2" }
startup_timeout_sec = 60
tool_timeout_sec = 180
'''
        (run/'cli_config187.toml').write_text(text);shutil.copy2(root/'model_catalog187.json',run/'model_catalog187.json')
        save(run/'mounts.json',dict(mounts=[dict(source=str(p),destination=q) for p,q in [(PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'shared_tools','/tools'),(root/'public','/data'),(run/'feedback','/feedback')]]))
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=run.name,socket=str(run/'ipc/s.sock'))],
            base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=cfg.get('episode_requests',50),max_input_tokens=cfg.get('episode_input_tokens',2500000),
            max_output_tokens=cfg.get('episode_output_tokens',65536),max_output_per_request=cfg.get('output_per_request',8192),max_inflight=1,
            denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent','followup_task','interrupt_agent','list_agents','send_message','wait_agent'],buffer_verified_responses=True))
        save(run/'BROKER_CONFIG.json',dict(public=str(root/'public'),request_directory='app/.runtime/feedback_requests',**cfg.get('broker_config_extra',{})))
        save(run/'PROTOCOL187.json',dict(cli=cfg['cli'],expected_mcp_tools=['mcp__phmforge__'+k for k in cfg.get('expected_mcp_actions',['cases','evidence','select'])]))
        row=dict(id=run.name,run=run,repeat=rep,condition=condition);episodes.append(row)
        # A local socket is sufficient for a network-free filesystem probe;
        # no credential or actual model request is needed for qualification.
        import socket
        socket_dir=run/'ipc';socket_dir.mkdir()
        fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        fixture.bind(str(socket_dir/'s.sock'));fixture.listen(1)
        (run/'audit/isolation').mkdir()
        ns=args_for(root,row)
        check="""import os,socket,json
from pathlib import Path
c=dict(private_paths_absent=not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/data/yyf-zyq/PHM_20261010','/srv/yyf-zyq/PHM_20261010/_shared','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010']),keys_absent=not any('API_KEY' in k or 'SECRET' in k for k in os.environ),temporary_path=os.environ['TMPDIR'].startswith('/app/.runtime/'),public_allowlist={p.name for p in Path('/data').iterdir()}=={'cases.json','contract.json','rows'})
try:s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();c['network_blocked']=False
except OSError:c['network_blocked']=True
print(json.dumps(c));assert all(c.values())
"""
        command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',check],ns,run/'app',run/'audit/isolation',None)
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp')
        try:p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        finally:fixture.close();(socket_dir/'s.sock').unlink()
        save(run/'audit/ISOLATION.json',dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr[-1000:]
        native=probe(run,run,dict(id=run.name,mounts=str(run/'mounts.json')),'fixed_discovery',text)
        assert native['passed']
        # Exercise actual scoped file transport through the native CLI sandbox
        # before any provider gateway or scientific model request starts.
        with (run/'logs/broker_qualification.log').open('w') as log:
            worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/program_update_tools.py'),'serve','--run',str(run)],env=env_for(root),stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if (run/'BROKER_READY.json').exists():break
                assert worker.poll() is None;time.sleep(.1)
            assert (run/'BROKER_READY.json').exists()
            fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(socket_dir/'s.sock'));fixture.listen(1)
            folder=run/'audit/transport';folder.mkdir()
            program="import json;from program_update_tools import client;r=client({'action':'evidence','case_id':"+repr(first_case)+"});assert r['ok'] and r['result']['id']=="+repr(first_case)+";print(json.dumps({'passed':True,'real_evidence':True,'model_calls':0}))"
            command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',program],ns,run/'app',folder,None)
            env['PYTHONPATH']='/opt/phm_python/extra:/tools'
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=210)
            save(folder/'RESULT.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,model_calls=0))
            assert result.returncode==0,result.stderr[-1000:]
            if cfg.get('qualify_submission_chain',False):
                # Exercise the real delivery path too; qualification artifacts
                # are moved aside before the fresh scientific session starts.
                program="""import json
from program_update_tools import client
cases=client({'action':'cases'})['result']['cases']
rows=[]
for case in cases:
 r=client({'action':'evidence','case_id':case['id']});assert r['ok']
 for old in case.get('incumbent_ids',case['candidate_ids']):
  rows.append(dict(case=case['id'],incumbent=old,selected=old,reason='Transport qualification only;not a scientific choice.'))
r=client({'action':'select','choices':rows});assert r['ok']
print(json.dumps(dict(passed=True,choices=len(r['result']['choices']),model_calls=0)))
"""
                if cfg.get('qualification_query_slot'):
                    program=program.replace("rows=[]", "q=client({'action':'query','slot':"+repr(cfg['qualification_query_slot'])+"});assert q['ok'] and q['result']['labels_consumed']==4\nrows=[]")
                if cfg.get('qualification_selected'):
                    program=program.replace('selected=old,', 'selected='+repr(cfg['qualification_selected'])+',')
                submission_folder=folder/'submission';submission_folder.mkdir()
                command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',program],ns,run/'app',submission_folder,None)
                result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=210)
                save(folder/'SUBMISSION_CHAIN.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,model_calls=0))
                assert result.returncode==0,result.stderr[-1000:]
                shutil.move(str(run/'broker/FINAL_CHOICE.json'),str(folder/'QUALIFICATION_CHOICE.json'))
        finally:
            worker.terminate();worker.wait(timeout=10)
            fixture.close();(socket_dir/'s.sock').unlink(missing_ok=True)
            (run/'BROKER_READY.json').unlink(missing_ok=True)
            if cfg.get('reset_qualification_transport',False):
                # No qualification labels or requests can enter the scientific
                # actor's mounted response directory or cumulative query state.
                keep=run/'audit/transport/qualification_state';keep.mkdir()
                for source,label in [(run/'broker','broker'),(run/'feedback/responses','responses'),
                                     (run/'app/.runtime/feedback_requests','requests')]:
                    if source.exists():shutil.move(str(source),str(keep/label))
                (run/'broker').mkdir()
    save(root/'audit/ALL_NATIVE_GATES.json',dict(passed=True,episodes=len(episodes),actual_model_calls=0,public_cases_sha256=sha(root/'public/cases.json')))
    update(root,status='ordinary_agents_running',planned_agent_sessions=len(episodes),controller_pid=os.getpid(),completed_agent_sessions=0)
    def execute(row):
        run=row['run'];broker=None;gateway=None;started=time.monotonic()
        try:
            with (run/'logs/broker.log').open('w') as log:
                broker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/program_update_tools.py'),'serve','--run',str(run)],env=env_for(root),stdout=log,stderr=subprocess.STDOUT)
            for _ in range(600):
                if (run/'BROKER_READY.json').exists():break
                assert broker.poll() is None;time.sleep(.1)
            assert (run/'BROKER_READY.json').exists()
            save(run/'EPISODE_BUDGET.json',dict(start_epoch=time.time(),deadline_epoch=time.time()+cfg['episode_seconds'],stage=cfg.get('stage','selection')))
            gateway=start_gateway(run)
            prompt=run/'plans/ordinary.txt';prompt.write_text(prompts[row['condition']])
            job=dict(id=row['id'],condition=row['condition'],repeat=row['repeat'],workspace=str(run/'app'),prompt_file=str(prompt),answer_mode='text')
            result=replay(job,args_for(root,row))
            path=run/'broker/FINAL_CHOICE.json';choice=json.loads(path.read_text()) if path.exists() else None
            result.update(id=row['id'],condition=row['condition'],repeat=row['repeat'],choice=choice,completed=result['status']=='finished' and choice is not None,
                          fallback=choice is None,seconds=time.monotonic()-started)
        except Exception as exc:
            result=dict(id=row['id'],condition=row['condition'],repeat=row['repeat'],completed=False,fallback=True,error_type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-started)
        finally:
            if broker is not None:
                broker.terminate()
                try:broker.wait(timeout=10)
                except subprocess.TimeoutExpired:broker.kill();broker.wait()
            if gateway is not None:
                (run/'MODEL_GATEWAY_STOP').touch();gateway.wait(timeout=650)
        save(run/'EPISODE_RESULT.json',result);return result
    results=[]
    with ThreadPoolExecutor(max_workers=cfg.get('maximum_parallel_agents',2)) as pool:
        for f in as_completed([pool.submit(execute,row) for row in episodes]):
            results.append(f.result());update(root,status='ordinary_agents_running',completed_agent_sessions=len(results))
    save(root/'AGENT_CHOICES.json',dict(utc=utc(),episodes=results,final_scores_joined=False))
    files=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in [root/'CAMPAIGN_CONFIG.json',root/'FIXED_CHOICES.json',root/'AGENT_CHOICES.json',root/'public/cases.json',*sorted((root/'code').glob('*.py'))]]
    save(root/'ALL_CHOICES_SEALED.json',dict(utc=utc(),files=files,ordinary_episodes=results,final_score_join_not_yet_run=True,
        researcher_previously_knows_final_scores=cfg.get('researcher_previously_knows_final_scores',True)))
    destination=PROJECT/'results'/cfg['result_key'] if cfg.get('result_key') else OUT
    save(destination/'AGENT_CHOICES.json',json.loads((root/'AGENT_CHOICES.json').read_text()))
    update(root,status='all_choices_sealed',completed_agent_sessions=len(episodes),normal_agent_sessions=sum(r['completed'] for r in results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','cold','fixed','agents'])
    p.add_argument('--run',required=True,type=Path);p.add_argument('--kind');p.add_argument('--block');a=p.parse_args()
    root=a.run.resolve()
    if a.action=='cold':cold_private(root,a.kind,a.block)
    else:globals()[a.action](root)
