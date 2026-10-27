"""R6 prospective full-coding pilot, using the shared actual-training runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_free_program_comparison import independent_score
from native_validation_comparison import closure
from program_revision_runtime import Runtime, load, score
from run_frozen_candidate_review import source_context
import free_program_comparison as coding

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261025T012000Z_source346r1')
LIVE_SOURCE = Path('/srv/yyf-zyq/PHM_20261010/20261025T012000Z_source346r1')
TEMPLATE = Path('/data/yyf-zyq/PHM_20261010/20261024T202500Z_decision340')
OUT = PROJECT / 'results/local347'

COMMON = '''Improve or retain the genuinely trained PHM four-class diagnosis program. You are a full coding Agent: arbitrary genuine preprocessing, features, supported classifiers, ensembles and input-observable local rules are allowed. You are not restricted to a small model bank or rotation-nearest routing. Read /data/metadata.json, /reference/source/program.py and /reference/REPORT.json, then phmforge info.

The data are real laboratory acquisitions from one rotating-machine rig, not simulation. Training has 8 complete acquisitions at two conditions; development has 8 acquisitions at two other conditions. Each acquisition contributes9nonoverlapping windows,not9independent machines. Six documented vibration channels and48statistics are supplied. The ninth raw source column is excluded. The optional /tools/phm_program_features.py can help; you may write other genuine features. Final conditions, observations, identities, class counts, labels, scores and other runs are unavailable. No network, other model service or agent spawning.

All supervised candidate fitting MUST go through register_program(files,claim,rival). Write program.py plus up to9.py/.json helpers under/app, using the reference fit/predict CLI. Registering freezes the complete source, actually fits two whole-condition leave-out models and one full-training checkpoint, then cold-evaluates development and ID invariance. Use fixed seeds and at most2numeric threads. At most4new complete versions besides p00; errors consume a version. No fabricated waveforms/labels, constant/mock training, batch quotas, identifier rules or unregistered fitting. Learned gates count as learning and must be fitted with training data through the same tool. Development/check labels may guide program selection but may not enter supervised fitting. Prediction receives one acquisition at a time without labels, with artifacts read-only.

Two shared tools support verification. paired_evidence(incumbent,challenger) joins actual record repairs/harms, source/checkpoint hashes and observed condition/log-RMS summaries. Its fold models differ from the deployed full checkpoint; do not mix their contexts. probe evaluates all not-yet-checked exact versions on the same4legal development-check acquisitions. It may be called after later registrations; cached exact-version results are reused. This is adaptively reusable DEVELOPMENT evidence, not independent certification. A later edit must register and check anew. No validation view guarantees transfer. Both tools, all candidates and arbitrary code are equally available to both workflows.

Minimize whole-acquisition error on unseen conditions while preserving useful improvements and avoiding harmful replacement. Majority vote over9windows is the primary metric. Investigate conflicting evidence rather than presume the initial program or a new version must be wrong. Retain or choose any complete registered program, including p00, via submit(program_id,reason). This immutable tool submission is the scientific choice; submit it before spending time polishing report.md. Explain actual checks, discarded alternatives and uncertainty. No controller replaces your choice with a preferred rule. If MCP discovery fails, python3 /tools/program_revision_tools.py client --request accepts the same JSON action.

Budget1800seconds/60modelrequests/4000000cumulative-input admission threshold/98304totaloutput/16384output per reply/4newversions. Each complete version has450seconds, each fit120seconds, each per-record prediction45seconds. Tool work and checks consume the episode wall budget. Usage is counted after replies and may overshoot an admission threshold by the last admitted reply; both workflows use the same rule. Numeric remaining budget appears in info/register. Use/app/.runtime for scratch, never/tmp. No claim of a final score without external evaluation.
'''

LOCAL = '''
Additional R6 paired-evidence local-revision workflow under test:
1. Produce a genuine first candidate without assuming a gate is needed. Keep the starting program and all earlier complete versions.
2. Before a further revision, call paired_evidence on p00 and the candidate. Write repair_plan.json specifying actual repaired/changed-to-wrong acquisitions, exact source/checkpoint and view, and two competing explanations for the change. Identify an observable input property whose behavior would distinguish them. Actual class labels describe validation outcomes; they are NOT deployable routing variables.
3. If a change has both benefits and harms, investigate whether the useful computation can be confined to a region recognized from legal input observations, while retaining the older working computation elsewhere. You may rewrite a feature/decision path or genuinely train a combined predictor. A local rule must be part of one complete registered program, not an evaluator-side oracle. Do not write rules based on acquisition IDs or hidden target conditions. Do not force a gate when no defensible observable separator exists; a supported global version or p00 remains allowed.
4. Use actual full-program paired outcomes and, when useful, probe the exact new version to test the proposed repair. Track what it newly breaks as well as what it repairs. Compare full-program evidence before final submission. Do not inherit a previous version's result or demand strict improvement on every group. At most two such evidence/revision cycles fit within the same4registrations and total budget.

There is no fixed minimax, coverage threshold or automatic rejection rule. This hypothesis may fail: sparse physical evidence may not identify a transferable local repair. More explanations or a smaller acceptance rate alone are not a benefit. Retain contradictory results and make the best actual program choice.
'''


def read(path):
    return json.loads(Path(path).read_text())


def freeze(root):
    assert not root.exists()
    root.mkdir()
    for d in ['code', 'common_tools', 'scripts', 'logs', 'audit', 'runtime/tmp', 'runtime/mpl',
              'private_inputs', 'public_data', 'reference/p00/source']:
        (root / d).mkdir(parents=True)
    plan = read(OUT / 'LOCAL_REVISION_PROTOCOL347.json')
    cli = read(OUT / 'CLI_SOURCE347.json')
    assert sha(Path(cli['cli'])) == cli['cli_sha256'] and cli['latest_tag'] == 'rust-v0.164.0'
    cfg = dict(plan, cli=cli['cli'], cli_sha256=cli['cli_sha256'], source_archive=str(SOURCE),
               source_protocol_sha256=sha((SOURCE if SOURCE.exists() else LIVE_SOURCE) / 'PROTOCOL346.json'),
               max_registered_programs=4, enable_paired_evidence=True, probe_reusable=True,
               prompt_file='PROMPTS347.json', build_stage_only=True, defer_independent_check=False, stage='full_coding')
    save(root / 'LOCAL_REVISION_PROTOCOL347.json', cfg)
    save(root / 'PROTOCOL337.json', cfg)
    save(root / 'PROMPTS347.json', dict(ordinary_coding=COMMON, local_revision=COMMON + LOCAL))
    shutil.copy2(OUT / 'CLI_SOURCE347.json', root / 'audit/CLI_SOURCE347.json')
    for name in closure(['local_program_revision', 'free_program_comparison', 'program_revision_tools', 'program_revision_runtime',
                         'program_training_entry', 'qualify_native_cli_mcp']):
        shutil.copy2(PROJECT / 'scripts' / (name + '.py'), root / 'code' / (name + '.py'))
    for name in ['program_revision_tools', 'program_training_entry', 'phm_program_features', 'controlled_feedback',
                 'prepare_fault_extensions', 'acquire_forge_pdmb_sources244', 'isolated_model_gateway']:
        shutil.copy2(PROJECT / 'scripts' / (name + '.py'), root / 'common_tools' / (name + '.py'))
    inventory, receipt = source_context(TEMPLATE)
    for name in ['base_cli_config.toml', 'model_catalog187.json', 'retention_policy.json']:
        assert sha(TEMPLATE / name) == inventory[name]['sha256']
        shutil.copy2(TEMPLATE / name, root / name)
    shutil.copy2(PROJECT / 'scripts/phm_starting_program.py', root / 'reference/p00/source/program.py')
    save(root / 'IMPLEMENTATION_FREEZE347.json', {str(p.relative_to(root)): sha(p) for folder in ['code', 'common_tools'] for p in (root / folder).glob('*.py')})
    save(root / 'OWNERSHIP.json', dict(run=str(root), archive='/data/yyf-zyq/PHM_20261010/' + root.name, created_utc=utc(), goal='A330',
         scope='R6 same-budget full native coding pilot;not method confirmation;all failures retained.'))
    save(OUT / 'EFFECTIVE_PROTOCOL347.json', cfg)
    save(OUT / 'current_run.json', dict(run=str(root), status='implementation_frozen_source_pending', new_LLM_calls=0, new_fits=0))
    print(json.dumps(dict(status='implementation_frozen', native_sessions=4, source_pending=not SOURCE.exists(), code_files=len(read(root / 'IMPLEMENTATION_FREEZE347.json')))))


def prepare(root):
    assert not (root / 'PREPARED347.json').exists()
    cfg = read(root / 'LOCAL_REVISION_PROTOCOL347.json')
    for relative, digest in read(root / 'IMPLEMENTATION_FREEZE347.json').items():
        assert sha(root / relative) == digest
    inventory, receipt = source_context(SOURCE)
    assert sha(SOURCE / 'PROTOCOL346.json') == cfg['source_protocol_sha256']
    qualification = read(SOURCE / 'QUALIFICATION346.json'); assert qualification['passed']
    packaging = []
    for role in ['train', 'development', 'independent_check']:
        file = SOURCE / 'qualification' / (role + '.npz')
        assert sha(file) == qualification['files'][file.name]['sha256'] == inventory[str(file.relative_to(SOURCE))]['sha256']
        values = load(file); groups = np.unique(values['groups']); np.random.default_rng(347).shuffle(groups)
        order = np.concatenate([np.flatnonzero(values['groups'] == g) for g in groups])
        values = {k: v[order] for k, v in values.items()}
        target = root / 'private_inputs' / file.name; np.savez_compressed(target, **values)
        if role != 'independent_check':
            shutil.copy2(target, root / 'public_data' / file.name)
        else:
            np.savez_compressed(root / 'public_data/independent_check_observations.npz', **{k: v for k, v in values.items() if k != 'labels'})
        packaging.append(dict(role=role, records=len(groups), windows=len(values['labels']), source_sha256=sha(file), private_sha256=sha(target)))
    save(root / 'public_data/metadata.json', dict(task='Whole-acquisition four-class diagnosis on future unknown conditions',
        classes=dict(normal=0, unbalance=1, horizontal_misalignment=2, vertical_misalignment=3),
        source='Author-measured COMFAULDA laboratory recordings;one physical rig,not simulated.',
        training_acquisitions=8, development_acquisitions=8, train_conditions_hz=[15,35], development_conditions_hz=[25,51],
        legal_reusable_check='4acquisitions at42Hz;actual results only through probe for exact registered versions;adaptive development evidence,not independent certification.',
        windows_per_acquisition=9,window_points=25600,sampling_rate_hz=51200,sensor_columns='Six documented vibration channels;unknown9thcolumnexcluded',
        available_arrays=['waveforms','features','rotation_hz','sampling_rate_hz','condition_id','sample_ids','groups'],
        features='48statistics from6documentedvibrationchannels;RMS and standard-deviation features are logged;no source amplitude calibration asserted.',
        fit_policy='Training labels only;complete-condition CV;learned gates and all preprocessing part of genuine registered fit. Development/check not supervised fitting data.',
        action_space='Arbitrary genuine complete programs,including learned input-observable local updates;no ID/classquota routing.',
        possible_future_rotation_range_hz=[10,60],actual_final_conditions_unknown=True,final_arrays_absent=True,
        primary='Whole-acquisition error by9windowmajorityvote;windowsnotindependentphysicaldevices',budgets={k:cfg[k] for k in ['episode_seconds','episode_requests','max_registered_programs']}))
    save(root / 'ENGINE_CONFIG.json', dict(input_root=str(root / 'private_inputs'), common_tools=str(root / 'common_tools'),
        fit_seconds=cfg['fit_seconds'], predict_seconds=cfg['predict_seconds'], program_seconds=cfg['program_seconds']))
    report = Runtime(root).fit_program(root / 'reference/p00')
    assert report['fit_entry_executions'] == 3 and report['identity_invariance_passed']
    coding.OUT = OUT; coding.prepare(root)
    save(root / 'SOURCE_BINDING347.json', dict(archive_receipt=receipt, qualification_sha256=sha(SOURCE / 'QUALIFICATION346.json'),
        legal_packaging=packaging, final_numeric_arrays_read=False))
    save(root / 'PREPARED347.json', dict(passed=True, baseline_fit_entries=3, baseline_prediction_processes=24, new_LLM_calls=0, utc=utc()))
    print(json.dumps(dict(status='prepared', baseline_real_fit_entries=3, baseline_cold_prediction_processes=24, final_data_read=False)))


def preflight(root):
    assert not (root / 'PREFLIGHT347.json').exists()
    run = root / 'qualification_subject'; run.mkdir()
    for d in ['app/.runtime/feedback_requests', 'feedback', 'broker', 'logs']:
        (run / d).mkdir(parents=True)
    shutil.copy2(root / 'ENGINE_CONFIG.json', run / 'ENGINE_CONFIG.json')
    files = []
    for p in (root / 'reference/p00/source').rglob('*'):
        if p.is_file():
            rel = p.relative_to(root / 'reference/p00/source'); destination = run / 'app' / rel
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, destination); files.append(str(rel))
    save(run / 'BROKER_CONFIG.json', dict(baseline=str(root / 'reference/p00'), contract=str(root / 'public_data/metadata.json'),
        max_programs=2, request_directory='app/.runtime/feedback_requests', enable_paired_evidence=True, probe_reusable=True))
    with (run / 'logs/service.log').open('w') as log:
        worker = subprocess.Popen([str(coding.PYTHON / 'bin/python3.12'), '-B', '-u', str(root / 'code/program_revision_tools.py'),
            'serve', '--run', str(run)], env=coding.environment(root), stdout=log, stderr=subprocess.STDOUT)
    def call(value):
        key = uuid.uuid4().hex; save(run / 'app/.runtime/feedback_requests' / (key + '.json'), value)
        reply = run / 'feedback/responses' / (key + '.json'); end = time.monotonic() + 690
        while time.monotonic() < end:
            if reply.exists():
                result = read(reply); assert result['ok'], result; return result['result']
            assert worker.poll() is None; time.sleep(.1)
        raise TimeoutError('Preflight real service timed out')
    try:
        for _ in range(600):
            if (run / 'BROKER_READY.json').exists(): break
            assert worker.poll() is None; time.sleep(.1)
        assert (run / 'BROKER_READY.json').exists()
        initial = call(dict(action='paired_evidence', incumbent='p00', challenger='p00'))
        assert not initial['reusable_check_available'] and all(v['repairs'] == v['harms'] == 0 for v in initial['views'])
        one = call(dict(action='register_program', files=files, claim='Exact actual baseline copy;transport qualification only', rival='Success response can be serialized incorrectly'))
        assert one['program_id'] == 'p01' and one['confirmed_estimator_fits'] == 3
        before = call(dict(action='paired_evidence', incumbent='p00', challenger='p01'))
        assert all(v['repairs'] == v['harms'] == 0 for v in before['views'])
        first = call(dict(action='probe')); cached = call(dict(action='probe'))
        assert set(first['results']) == {'p00','p01'} and cached['cached'] and cached['new_prediction_processes'] == 0
        # Interface fixture only:both observable branches use the same genuinely
        # fitted model,so outcomes should be unchanged. This is not a PHM method.
        file=run/'app/program.py';text=file.read_text();old='labels=model.predict(features(data));rows='
        branch='''inputs=features(data);high=data['rotation_hz']>30
            labels=np.empty(len(inputs),dtype=int)
            for mask in [high,~high]:
                if mask.any():labels[mask]=model.predict(inputs[mask])
            print(json.dumps(dict(branch_low_windows=int((~high).sum()),branch_high_windows=int(high.sum()),qualification_fixture=True)))
            rows='''
        assert old in text;file.write_text(text.replace(old,branch))
        two = call(dict(action='register_program', files=files, claim='Post-check observable-branch interface fixture;same genuinely fitted classifier,not a scientific candidate', rival='A changed program might inherit a previous version check or fail branch execution'))
        assert two['program_id'] == 'p02'
        missing = call(dict(action='paired_evidence', incumbent='p00', challenger='p02')); assert not missing['reusable_check_available']
        refreshed = call(dict(action='probe')); assert refreshed['new_program_ids'] == ['p02']
        after = call(dict(action='paired_evidence', incumbent='p00', challenger='p02'))
        assert after['reusable_check_available'] and all(v['repairs'] == v['harms'] == 0 for v in after['views'])
        chosen = call(dict(action='submit', program_id='p02', reason='Qualification only;not a scientific Agent choice')); assert chosen['program_id'] == 'p02'
        audits = [read(p) for p in run.rglob('training_audit.json')]
        assert sum(x.get('mode') == 'fit' for x in audits) == 6 and sum(x.get('mode') == 'predict' for x in audits) == 60
        executed_branches=set()
        for p in (run/'programs/p02').rglob('PROCESS.json'):
            process=read(p)
            if process['mode']=='predict':
                for line in process['stdout'].splitlines():
                    value=json.loads(line)
                    if value['branch_low_windows']:executed_branches.add('low')
                    if value['branch_high_windows']:executed_branches.add('high')
        assert executed_branches=={'low','high'}
        save(root / 'PREFLIGHT347.json', dict(passed=True, actual_fit_entries=6, actual_prediction_processes=60, true_pair_checks=True,
            check_cache_zero_forward=True, new_version_not_inherited=True, new_version_actually_rechecked=True,
            arbitrary_observable_branches_both_actually_executed=True,branch_fixture_not_method_evaluation=True,
            register_info_pair_probe_refresh_submit=True, new_LLM_calls=0, final_data_read=False, utc=utc()))
    finally:
        worker.terminate(); worker.wait(timeout=10)
    print(json.dumps(read(root / 'PREFLIGHT347.json')))


def qualify(root):
    assert read(root / 'PREFLIGHT347.json')['passed']
    coding.OUT = OUT; coding.qualify(root)
    # Qualification is outside scientific usage:do not leave its broker state or
    # responses in an Actor-visible mount at episode admission.
    for row in read(root / 'SESSIONS337.json'):
        run = Path(row['path']); saved = run / 'audit/pre_scientific_transport'; saved.mkdir()
        for relative in ['broker/STATE.json', 'feedback/responses', 'app/.runtime/feedback_requests']:
            p = run / relative
            if p.exists():
                shutil.move(str(p), str(saved / relative.replace('/', '_')))
                if p.suffix != '.json': p.mkdir(parents=True)
    save(root / 'QUALIFIED347.json', dict(passed=True, native_isolation=True, native_MCP_5_tools=True,
        true_training_pair_refresh_chain=True, qualification_state_outside_Actor=True, new_LLM_calls=0, utc=utc()))
    print(json.dumps(read(root / 'QUALIFIED347.json')))


def agents(root):
    assert read(root / 'QUALIFIED347.json')['passed']
    coding.OUT = OUT; coding.agents(root)


def evaluate(root):
    assert not (root / 'RESULT347.json').exists()
    cfg = read(root / 'LOCAL_REVISION_PROTOCOL347.json'); sealed = read(root / 'ALL_CHOICES_SEALED337.json')
    assert len(sealed['episodes']) == 4 and read(root / 'HANDLE337.json')['status'] == 'terminal'
    for rel,digest in read(root / 'EFFECTIVE_CODE_FREEZE337.json').items(): assert sha(root / rel) == digest
    qualification = read(SOURCE / 'QUALIFICATION346.json'); path = SOURCE / 'qualification/final.npz'
    assert sha(path) == qualification['files']['final.npz']['sha256']
    save(root / 'FINAL_STARTED347.json', dict(utc=utc(),all4choices_sealed=True,seal_sha256=sha(root / 'ALL_CHOICES_SEALED337.json'),
        final_input_sha256=sha(path),prespecified_primary='whole-acquisitionerror',new_data_confirmations=0))
    data = load(path); catalog = {'common_p00':root / 'reference/p00'}
    for e in sealed['episodes']:
        for k,v in read(root / 'subjects' / e['id'] / 'broker/STATE.json')['programs'].items():
            if k != 'p00': catalog[e['id'] + '__' + k] = Path(v['path'])
    def one(item):
        key,program=item; report=read(program / 'REPORT.json')
        for rel,digest in report['artifacts'].items(): assert sha(program / rel) == digest
        runtime=Runtime(root); runtime.deadline=time.monotonic()+600; out=root / 'evaluation' / key
        try:
            a,aa=runtime.predict_registered(out / 'forward',program,data)
            b,ab=runtime.predict_registered(out / 'cold_repeat',program,data)
            assert np.array_equal(a,b); value=score(data,a); independent=independent_score(data,a)
            assert value['confusion']==independent['confusion'] and value['record_error']==independent['record_error']
            value.update(status='complete',actual_prediction_processes=len(aa+ab),cold_repeat_and_independent_vote_passed=True)
        except Exception as error: value=dict(status='evaluation_error',error_type=type(error).__name__,error=str(error))
        save(out / 'RESULT.json',value); return key,value
    values={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,item) for item in catalog.items()]):
            key,value=f.result();values[key]=value;save(root / 'EVALUATION_PROGRESS347.json',values)
    assert values['common_p00']['status']=='complete'; baseline=values['common_p00']['record_error']; rows=[]
    for e in sealed['episodes']:
        state=read(root / 'subjects' / e['id'] / 'broker/STATE.json'); ids=list(state['programs']); view=lambda k:values['common_p00' if k=='p00' else e['id']+'__'+k]
        fixed={'keep':'p00','development':min(ids,key=lambda k:(state['programs'][k]['report']['development']['record_error'],k!='p00',k)),
            'mean_cv':min(ids,key=lambda k:(np.mean([v['record_error'] for v in state['programs'][k]['report']['cv']]),k!='p00',k))}
        packet=state.get('probe'); checked=packet['reports'] if packet and packet['status']=='complete' else {}
        eligible=[k for k in ids if k in checked and checked[k]['status']=='complete']
        if eligible:
            fixed['full_checkpoint_validation']=min(eligible,key=lambda k:((state['programs'][k]['report']['development']['record_error']+checked[k]['record_error'])/2,k!='p00',k))
            risk=lambda k:[z['record_error'] for z in state['programs'][k]['report']['cv']]+[state['programs'][k]['report']['development']['record_error'],checked[k]['record_error']]
            fixed['minimax_all_legal']=min(eligible,key=lambda k:(max(risk(k)),np.mean(risk(k)),k!='p00',k))
        choices=dict(fixed)
        if e.get('submission'):choices['actor']=e['submission']['program_id']
        scores={}
        for method,k in choices.items():
            value=view(k);r=dict(program_id=k,evaluation_status=value['status'])
            if value['status']=='complete':r.update(record_error=value['record_error'],harm=max(0.,value['record_error']-baseline),gain=baseline-value['record_error'])
            scores[method]=r
        rows.append(dict(id=e['id'],arm=e['arm'],repeat=e['repeat'],completed=e['completed'],native_status=e.get('native',{}).get('status'),
            paired_tool_calls=len(state.get('paired_evidence_calls',[])),programs=ids,check_covered_programs=eligible,methods=scores,seconds=e['seconds']))
    summary={}
    for arm in cfg['arms']:
        rr=[r for r in rows if r['arm']==arm];valid=[r for r in rr if r['completed'] and r['methods'].get('actor',{}).get('evaluation_status')=='complete']
        summary[arm]=dict(planned=len(rr),completed=len(valid),mean_record_error=float(np.mean([r['methods']['actor']['record_error'] for r in valid])) if len(valid)==len(rr) else None,
            mean_harm=float(np.mean([r['methods']['actor']['harm'] for r in valid])) if len(valid)==len(rr) else None)
    receipts=[read(p) for p in root.glob('subjects/*/model_gateway/*/*_receipt.json')]
    fit=[read(p) for p in root.rglob('training_audit.json') if read(p).get('mode')=='fit']
    result=dict(iteration='A347',utc=utc(),status='complete',primary=summary,baseline_error=baseline,episodes=rows,all_program_results=values,
        physical_development_cohorts=1,independent_method_confirmation=0,ownbankoracle_not_crossarm_improvement=True,
        cost=dict(request_receipts=len(receipts),HTTP200=sum(r.get('upstream_status')==200 for r in receipts),all_usage_present=all(r.get('usage') for r in receipts),
            returned_model_ids=sorted({m for r in receipts for m in r.get('returned_model_ids',[])}),input_tokens=sum((r.get('usage')or{}).get('input_tokens',0) for r in receipts),
            output_tokens=sum((r.get('usage')or{}).get('output_tokens',0) for r in receipts),real_fit_entries=len(fit),
            completed_estimator_fits=sum(r['completed_estimator_fits'] for r in fit),actual_optimizer_steps=sum(r['optimizer_steps'] for r in fit),
            final_prediction_processes=sum(r.get('actual_prediction_processes',0) for r in values.values())),
        majority_goal_complete=False,novelty_established=False,confirmation338_performance_unopened=True)
    save(root / 'RESULT347.json',result);save(OUT / 'RESULT347.json',result)
    save(OUT / 'current_run.json',dict(run=str(root),status='evaluated',primary=summary,goal_complete=False))
    print(json.dumps(dict(primary=summary,cost=result['cost'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','prepare','preflight','qualify','agents','evaluate']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
