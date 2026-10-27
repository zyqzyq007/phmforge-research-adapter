"""Qualify a budgeted native decision stage on already-scored real proposals.

This is an ordinary-Agent development control, not a new method, a rerun of
training, or confirmation. No final performance is copied into Actor mounts.
"""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from run_frozen_candidate_review import source_context
import screen_program_updates as runner

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261024T181000Z_freeprogram337r1')
OUT = PROJECT/'results/decision340'

PROMPT = '''Review four pools of actual coding-Agent-authored, genuinely trained PHM programs. This is a bounded decision stage, not new model development. Read /data/contract.json and /data/cases.json; use phmforge evidence for EACH case. Each pool has one existing program p00 and several immutable alternatives. Choose exactly one registered candidate per pool, keeping p00 or replacing it. No other starting programs are evaluated. The four pools share the SAME measured physical training, validation and final-record split; they are not independent datasets.

You may inspect the actual program source, recompute metrics and analyze acquisition-level predictions under /data/rows/. All supplied CV, development and independent-check results are legal validation evidence, with the fitting and evaluation conditions explicitly identified. CV uses models trained on one other condition; development and independent-check results use the unchanged full-training checkpoint that would be deployed. The independent check is a permitted validation set, not the hidden final evaluation. No validation condition is guaranteed to predict future performance. Whole-acquisition error, using a majority vote of nine nonoverlapping windows, is the primary task loss. Window counts do not create independent machines. Minimize future loss while retaining useful improvements and avoiding harmful replacements. No candidate, view or choice rule is prescribed as correct.

Final measurements, labels, operating conditions and scores, source filenames, other Agent sessions, historical experiment conclusions and model APIs are unavailable. No fitting, new mixtures, edited predictors, invented labels or network use. You may write analysis code, retain uncertainty and compare evidence. Do not spawn Agents.

Submit exactly four (case,incumbent,selected,reason) rows with incumbent p00 via phmforge select, then save selection.json and report.md. Read evidence for every case first. A successful tool submission is the immutable deliverable. Tool file fallback: python3 /tools/program_update_tools.py client --request followed by a JSON action. Scratch stays under /app/.runtime;never /tmp. This stage has its OWN reserved budget:600seconds,20model requests,600000input tokens,65536output tokens total,16384output tokens/request,2numeric threads. Numeric remaining budget is available in cases/evidence responses. Deliver your choice before exhausting the budget. All sessions are externally evaluated only after every choice is sealed;prose claims are not scores.
'''


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    assert not (root/'CAMPAIGN_CONFIG.json').exists(), 'No frozen experiment overwrite'
    if root.exists():
        assert (root/'attempts/PRE_ADMISSION_FAILURE340.json').exists(), 'Only recorded zero-model preparation repair is resumable'
    root.mkdir(exist_ok=True)
    for directory in ['runtime/tmp','runtime/mpl','logs','audit','code','shared_tools','scripts','public/rows','evaluation_inputs']:
        (root/directory).mkdir(parents=True,exist_ok=True)
    index, receipt = source_context(SOURCE)

    def bound(relative):
        path = SOURCE/relative
        assert sha(path) == index[relative]['sha256'], relative
        return path

    original = read(bound('PROTOCOL337.json'))
    url = 'https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research-provenance'}),timeout=30) as response:
        release = json.load(response)
    assert release['tag_name'] == 'rust-v0.164.0', 'Qualify new stable CLI before freezing'
    assert sha(Path(original['cli'])) == original['cli_sha256']
    save(root/'audit/CLI_SOURCE340.json',dict(url=url,checked_utc=utc(),tag=release['tag_name'],published_at=release['published_at']))
    config = dict(iteration='A340',stage='reserved_selection_development_control',frozen_utc=utc(),
        model='deepseek-flash',cli=original['cli'],cli_sha256=original['cli_sha256'],
        episode_seconds=600,episode_requests=20,episode_input_tokens=600000,
        episode_output_tokens=65536,output_per_request=16384,numerical_threads=2,
        repeats=2,maximum_parallel_agents=2,preparation_file='PREPARATION340.json',qualify_submission_chain=True,
        source_archive=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],
        source_final_results_already_known_to_researcher=True,confirmation_experiment=False,
        physical_development_cohorts=1,related_actual_program_pools=4,independent_confirmation_scenarios=0,
        new_fits=0,new_PHM_predictions=0,new_final_forward_budget=0,
        hypothesis='A fresh explicit decision budget can qualify ordinary selection without treating build-budget exhaustion as quality. Ordinary full-checkpoint validation may already solve these pools.',
        main_comparison='Ordinary native choices vs keep,mean condition CV,mean full-checkpoint legal validation,and retired absolute minimax. No new method claimed.',
        failure_policy='Retain all attempts;no quality conclusion from missing submission;no retries selected by outcomes.',
        fixed_ties='retain p00,then registration order',
        metrics='Whole-record final error;common-pool selection regret;harm;gain capture where p00 has an improvement opportunity. Average repetitions and pools within one shared physical cohort.',
        maximum_new_disk_GiB=2,maximum_new_model_requests=40,
        next_end_to_end_budget_contract=dict(total_seconds=2400,total_requests=50,total_input_tokens=2500000,total_output_tokens=131072,
            build=dict(seconds=1800,requests=30,input_tokens=1900000,output_tokens=65536),
            choice=dict(seconds=600,requests=20,input_tokens=600000,output_tokens=65536),
            note='Proposed common ceilings only;this pilot executes choice alone. A live two-stage controller is not claimed qualified. Input usage may overshoot admission cap by one actual reply.'))
    save(root/'CAMPAIGN_CONFIG.json',config);save(OUT/'PROTOCOL340.json',config)
    save(root/'PROMPT_FREEZE.json',dict(ordinary=PROMPT))
    names = closure(['qualify_program_decision','screen_program_updates','program_update_tools','run_validation_transfer','qualify_native_cli_mcp'])
    for name in names:
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in ['program_update_tools.py','controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'shared_tools'/name)
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    for name in ['base_cli_config.toml','model_catalog187.json']:
        shutil.copy2(bound(name),root/name)
    save(root/'CODE_FREEZE.json',{name+'.py':sha(root/'code'/(name+'.py')) for name in names})

    cases=[];mapping={};audits=[];fixed={name:[] for name in ['keep','mean_cv','full_checkpoint_validation','retired_absolute_minimax']}
    for number, subject in enumerate(['ordinary_coding_1','ordinary_coding_2','witness_transaction_1','witness_transaction_2'],1):
        case_id='pool_'+str(number)
        state=read(bound('subjects/'+subject+'/broker/STATE.json'))
        packet='subjects/'+subject+('/independent_packet/REPORT.json' if state['probe'] else '/controller_packet/REPORT.json')
        check=read(bound(packet))['programs']
        ids=sorted(state['programs']);scores=[];rows={};descriptions=[]
        for ident in ids:
            prefix='reference/p00' if ident=='p00' else 'subjects/'+subject+'/programs/'+ident
            report=read(bound(prefix+'/REPORT.json'))
            validation=[*report['cv'],report['development'],check[ident]]
            assert report['status']=='complete' and check[ident]['status']=='complete'
            for result in validation:
                records=result['records']
                assert len(records)==4
                error=sum(r['actual']!=r['predicted'] for r in records)/len(records)
                assert error==result['record_error']
                audits.append(dict(case=case_id,candidate=ident,record_error=error,actual_record_count=4))
            scores.append([result['record_error'] for result in validation])
            rows[ident]=dict(cv=report['cv'],development=report['development'],independent_check=check[ident],
                source_directory='/data/rows/'+case_id+'/source/'+ident)
            source=SOURCE/prefix/'source'
            for path in sorted(source.rglob('*')):
                if path.is_file():
                    relative=str(path.relative_to(SOURCE));bound(relative)
                    target=root/'public/rows'/case_id/'source'/ident/path.relative_to(source)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            claim=read(bound(prefix+'/CLAIM.json')) if (SOURCE/prefix/'CLAIM.json').exists() else dict(claim='Starting raw-statistics StandardScaler/LDA program',rival='No claim of optimality')
            descriptions.append(dict(program_id=ident,claim=claim['claim'],rival=claim['rival'],source_directory=rows[ident]['source_directory']))
        case=dict(id=case_id,source='one_measured_rotating_machine_rig',metric='whole-acquisition classification error',
            candidate_ids=ids,incumbent_ids=['p00'],original_incumbent='p00',candidate_descriptions=descriptions,
            groups=['CV/train32/evaluate16','CV/train16/evaluate32','full/train16+32/development24','full/train16+32/independent_check48'],
            group_losses=scores,development_columns=[2],full_checkpoint_validation_columns=[2,3],
            evaluation_contexts=[dict(column=0,train_hz=[32],evaluate_hz=16,checkpoint='fold'),
                dict(column=1,train_hz=[16],evaluate_hz=32,checkpoint='fold'),
                dict(column=2,train_hz=[16,32],evaluate_hz=24,checkpoint='full'),
                dict(column=3,train_hz=[16,32],evaluate_hz=48,checkpoint='full')],
            counts=dict(train_acquisitions=8,development_acquisitions=4,independent_check_acquisitions=4,windows_per_acquisition=9),
            relationship='All four pools use identical physical records. Not independent scenes.',
            training='Real earlier authored-program fits,with fixed weights. No new fitting during selection.')
        cases.append(case);mapping[case_id]=subject;save(root/'public/rows'/(case_id+'.json'),rows)
        key=lambda values,ident:(values,ident!='p00',ids.index(ident))
        selections=dict(keep='p00',
            mean_cv=min(zip(ids,scores),key=lambda r:key(sum(r[1][:2])/2,r[0]))[0],
            full_checkpoint_validation=min(zip(ids,scores),key=lambda r:key(sum(r[1][2:])/2,r[0]))[0],
            retired_absolute_minimax=min(zip(ids,scores),key=lambda r:(max(r[1]),sum(r[1])/4,r[0]!='p00',ids.index(r[0])))[0])
        for method,selected in selections.items():
            fixed[method].append(dict(case=case_id,incumbent='p00',selected=selected))
    save(root/'public/cases.json',cases)
    save(root/'public/contract.json',dict(task='Select a complete immutable trained PHM program;four related actual proposal pools,one original incumbent each.',
        labels=dict(normal=0,unbalance=1,horizontal_misalignment=2,vertical_misalignment=3),
        evidence='Whole-acquisition row labels/predictions from actual code and weights,source hashes verified. Independent check is legal validation,not final evaluation.',
        sampling='Physical laboratory measured six vibration channels,51200Hz inferred from time grid,9nonoverlapping25600point windows/record.',
        metric='Majority vote per acquisition;classification error. Ties in votes use smallest integer class. Do not infer population prevalence from four class-balanced validation acquisitions.',
        permission='Public cases,contract,legal validation rows and registered sources only. No fitting or additional model APIs.',
        deployment='Future conditions unknown;no guarantee that CV or full-checkpoint checks transfer.',
        budget=dict(seconds=600,requests=20,input_tokens=600000,output_tokens=65536,per_reply=16384),
        stage='Fresh choice-only pilot;not continuation or recovery of old coding sessions.'))
    save(root/'evaluation_inputs/SOURCE_MAPPING.json',mapping)
    save(root/'FIXED_CHOICES.json',dict(utc=utc(),choices=fixed,final_scores_joined=False))
    save(root/'PREPARATION340.json',dict(passed=True,utc=utc(),candidate_versions=sum(len(c['candidate_ids']) for c in cases),
        metric_row_checks=len(audits),audit_rows=audits,source_manifest_sha256=receipt['manifest_sha256'],new_fits=0,new_predictions=0,
        final_source_files_read_during_preparation=False,public_files=[dict(path=str(p.relative_to(root/'public')),sha256=sha(p)) for p in sorted((root/'public').rglob('*')) if p.is_file()]))
    shutil.copy2(bound('retention_policy.json'),root/'retention_policy.json')
    save(root/'OWNERSHIP.json',dict(user_request='围绕程序错误替换持续迭代；资格化双方共同选模预算并保留强完整模型验证控制',run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name,created_utc=utc()))
    runner.OUT=OUT;runner.update(root,status='prepared',new_fits=0,planned_agent_sessions=2)
    print(json.dumps(dict(run=str(root),passed=True,programs=sum(len(c['candidate_ids']) for c in cases),metric_checks=len(audits),new_fits=0)))


def analyze(root):
    assert (root/'ALL_CHOICES_SEALED.json').exists()
    mapping=read(root/'evaluation_inputs/SOURCE_MAPPING.json')
    index,_=source_context(SOURCE)
    source_result=SOURCE/'RESULTS337.json'
    assert sha(source_result)==index['RESULTS337.json']['sha256']
    final=read(source_result)['all_results']
    cases=read(root/'public/cases.json');bycase={c['id']:c for c in cases}

    def scored(rows):
        result=[]
        for row in rows:
            case=bycase[row['case']];subject=mapping[case['id']]
            get=lambda ident:final['common_p00' if ident=='p00' else subject+'__'+ident]['record_error']
            old=get('p00');selected=get(row['selected']);oracle=min(get(i) for i in case['candidate_ids'])
            result.append(dict(row,record_error=selected,selection_loss=selected-oracle,harm=max(0,selected-old),
                gain_capture=max(0,old-selected)/(old-oracle) if old>oracle else None))
        return dict(rows=result,mean_record_error=sum(r['record_error'] for r in result)/len(result),
            mean_selection_loss=sum(r['selection_loss'] for r in result)/len(result),mean_harm=sum(r['harm'] for r in result)/len(result),
            mean_gain_capture=sum(r['gain_capture'] for r in result if r['gain_capture'] is not None)/sum(r['gain_capture'] is not None for r in result))

    fixed={name:scored(rows) for name,rows in read(root/'FIXED_CHOICES.json')['choices'].items()}
    agents=[]
    for episode in read(root/'AGENT_CHOICES.json')['episodes']:
        agents.append(dict(id=episode['id'],completed=episode['completed'],seconds=episode['seconds'],
            scores=scored(episode['choice']['choices']) if episode['completed'] else None))
    traffic=[]
    for path in sorted((root/'episodes').glob('*/model_gateway/*/status.json')):
        traffic.append(dict(id=path.parents[2].name,usage=read(path)))
    result=dict(iteration='A340',utc=utc(),ordinary_episodes=agents,fixed_controls=fixed,gateway_usage=traffic,
        physical_development_cohorts=1,related_pools=4,independent_confirmation_scenarios=0,new_fits=0,new_predictions=0,
        budget_stage='Choice stage qualified independently;full build-plus-choice controller remains unqualified.',
        novelty_claim=False,majority_goal_complete=False,source_final_scores_previously_opened=True)
    save(root/'RESULT340.json',result);save(OUT/'RESULT340.json',result)
    runner.OUT=OUT;runner.update(root,status='analysis_complete',valid_agent_submissions=sum(a['completed'] for a in agents),scientific_experiment_terminal=True)
    print(json.dumps(dict(valid=sum(a['completed'] for a in agents),ordinary_errors=[a['scores']['mean_record_error'] if a['scores'] else None for a in agents],
        full_checkpoint_error=fixed['full_checkpoint_validation']['mean_record_error'],new_fits=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','agents','analyze']);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='agents':runner.OUT=OUT;runner.agents(root)
    else:analyze(root)
