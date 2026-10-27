"""Run previously unexecuted PHMForge detailed tasks on measured observations.

Reuse the native isolated coding runner. Original questions remain verbatim;
source corrections and added evaluation outputs are explicit task addenda.
This is coverage expansion on known sources, not unseen-source confirmation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_fault_extensions import features
from real_fault_model import train
from run_validation_transfer import prepare, freeze, execute_wave, state

PROJECT = Path(__file__).resolve().parents[1]
ARTIFICIAL = set('KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08'.split())
NATURAL = set('KA04 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI04 KI14 KI16 KI17 KI18 KI21'.split())
HEALTHY = {f'K{i:03d}' for i in range(1, 7)}
SPECS = [('h', 'HUST', 'pdm_fault_007'), ('p', 'PADERBORN', 'pdm_fault_006'),
         ('m', 'MENDELEY', 'pdm_fault_009'), ('f', 'MFPT', 'pdm_fault_008')]
ADDENDA = {
    'h': '''\nAdditional original-task requirements: retain all seven published classes (healthy plus six fault categories). HUST has five bearing types 6204..6208. The arrays bearing_type, load_w and shaft_frequency_hz come from original recording metadata. MAT fs is shaft frequency, not acquisition rate; acquisition is 51200 Hz. Compute actual development accuracy by fault class, bearing type, load and speed/load combination, with counts. Different load recordings of the same bearing proxy stay together. Explain unsupported or small strata. Save these development breakdowns in stratified_validation.json. Final metrics will be computed by the external evaluator.\n''',
    'p': '''\nAdditional original-task requirements: predict BOTH fault category and damage origin for every supplied sample. Each prediction row must also contain predicted_origin, one of "healthy", "artificial", "accelerated_life". Train/val origin_labels are actual publisher annotations. "Natural" in the question means damage produced by accelerated laboratory lifetime testing, not observed field degradation. Healthy bearings have no damage origin and use "healthy". Compute development fault accuracy separately by TRUE origin, plus origin accuracy and joint fault-and-origin accuracy, with counts; save stratified_validation.json. Never infer origin from hidden bearing identities. A separately fitted train-only origin reference is available as /reference/origin_reference.joblib; the starting program combines it with the genuine fault reference. You may retain or replace either. Original fault labels remain 0=healthy, 1=inner, 2=outer, 3=inner+outer; there is no ball-fault class in this source. All windows of each physical bearing stay in one split.\n''',
    'm': '''\nSource correction and additional original-task requirements: the actual publisher data has varying 0..900 RPM profiles, two measured bearing states (healthy and outer-ring defect), 96000 Hz, and 60-second recordings. The question's 600..1800 RPM premise is incorrect for this source. Use the actual observed profiles; do not claim any evidence at absent speeds. This correction preserves the variable-speed diagnosis objective and is a documented research adaptation. The speed_range_rpm and speed_trend fields describe each recording's published profile; exact tachometer samples and vibration are in record_files, relative to the dataset directory. Compute actual development fault accuracy separately by speed_range_rpm and speed_trend with denominators; do not call a whole-record profile metric instantaneous speed accuracy. Save stratified_validation.json. Raw tachometer outliers are retained and any filtering must be explicit. The same physical prototypes are reused across recordings; this is not a cross-bearing test.\n''',
    'f': '''\nAdditional original-task requirements: classify healthy/inner/outer states, retaining the measured healthy control records rather than forcing them into fault classes. All signals are native MFPT MAT recordings from the MathWorks official example data repository. Actual native load (lb), shaft frequency (Hz), and sampling rate are supplied as load_lb, shaft_frequency_hz and sampling_hz. Native load fields can be strings; their values have been parsed, not inferred. Compute actual development fault accuracy by load with counts and per-fault precision/recall/F1; save stratified_validation.json. Use a whole recording as the split unit. The official example's six test files stay withheld, including 250/300 lb fault recordings and 270 lb healthy/outer recordings; this deployment load range is known before selection. Development comprises one complete lowest-lexicographic training file per class; the other eleven files are for fitting. Shared hardware across recordings prevents a cross-bearing generalization claim.\n''',
}


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as stream:
        return dict(stream)


def configure(root):
    assert not (root / 'CAMPAIGN_CONFIG.json').exists()
    config = dict(iteration='A306', result_directory='results/coverage306', planned_sessions=4,
        authorization='用户2026-10-23：9／75，其余的继续推进真实实验啊。',
        purpose='Four previously unexecuted detailed original-task directions on measured data; explicit source corrections; first-pass coverage, no automatic review or retry.',
        test_inputs_visible=False, candidate_bank_from_training_cv=False,
        cohorts={c:dict(family='classification', specs=[dict(id='detail306_'+c, dataset=d, task_id=t)]) for c,d,t in SPECS},
        cases=[dict(cohort=c, dataset=d, task_id=t, arm=c) for c,d,t in SPECS],
        arm_prompt_addenda=ADDENDA,
        session_budget=dict(subject_timeout_seconds=3600, max_model_requests_per_subject=180,
            max_total_input_tokens_per_subject=32000000, max_total_output_tokens_per_subject=524288,
            max_output_tokens_per_request=32768, mcp_tool_timeout_seconds=180, max_concurrency=4,
            local_max_concurrency=1, artifact_budget_bytes=40*1024**3),
        prior_source_outcomes_known_to_researcher=True, prior_solutions_or_scores_given_to_agent=False,
        ordinary_review='Deferred until first-pass new-task coverage; no unregistered extra model sessions.',
        primary_metrics=['fault_accuracy','macro_f1','balanced_accuracy','complete_ID_coverage'],
        additional_metrics={'h':['bearing_type','load_w','shaft_frequency_hz','condition'],
            'p':['fault_accuracy_by_true_origin','origin_accuracy','joint_accuracy'],
            'm':['speed_profile','speed_trend'], 'f':['load_lb']},
        failure_policy='Retain timeout, invalid output, omitted strata, execution error and good outcomes; no upper accuracy rejection.',
        estimate=dict(max_wall_seconds_after_preparation=4200, numerical_threads_per_subject=2,
            gpu_hours=0, max_model_requests=720, cost_cap='Token/request/session caps; provider prices not assumed.',
            expected_data_bytes=5*1024**3, artifact_cap_bytes=40*1024**3),
        data_plan=dict(hust='Reuse A255 whole-bearing split, add exact source conditions.',
            paderborn='Reuse A257 whole-bearing split, add publisher Tables4/5/7 origin annotations.',
            mendeley='Reuse A257 whole-record split and native recordings; correct impossible speed premise.',
            mfpt='All20 original MathWorks native records. Official6 test unchanged. One lexicographically first train file per label becomes development; remaining11 train. Nonoverlapping2560 samples, incomplete tails retained in source but not windowed. Real MLP50epochs, seed2026102306.'))
    save(root/'CAMPAIGN_CONFIG.json', config)
    save(root/'audit/PREPARATION_PROTOCOL306.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        configuration_sha256=sha256(root/'CAMPAIGN_CONFIG.json'), no_final_scores_opened=True,
        source_measurement_metadata_may_be_inspected=True, source_qualification_only_before_agent=True))


def derived(root, case, name):
    origin = root/'source/recovered257'
    if case == 'h':
        old = root/'h/restored/HUST'
        origin = root/'source/recovered255'
        assert old.exists() and not origin.exists()
        shutil.move(str(old), origin)
    source = origin/'public/datasets'/name
    target = root/case/'restored'/name/'data'
    target.mkdir(parents=True)
    info = read(source/'dataset.json')
    info['coverage306_scope'] = 'New detailed original task; reused source and pre-existing split; not unseen-source confirmation.'
    private = read(origin/'hidden'/name/'labels.json')
    if case == 'h':
        source_rows = read(origin/'audit/HUST_SPLIT.json')['observations']
        speeds = {v['filename']:v['shaft_frequency_hz'] for v in read(origin/'audit/NATIVE_VALUE_CORRESPONDENCE.json')['native_records']}
        by_id = {v['sample_id']:v for v in source_rows}
        def condition(row):
            m = re.fullmatch(r'(N|I|O|B|IO|IB|OB)([4-8])0([024])\.mat', row['filename']); assert m
            return dict(bearing_type='620'+m[2], load_w=int(m[3])*100, shaft_frequency_hz=speeds[row['filename']])
        for row in private:row.update(condition(row))
        info.update(condition_fields=['bearing_type','load_w','shaft_frequency_hz'],
            fs_semantics='Native fs = shaft frequency Hz; acquisition rate51200Hz.',
            waveform_scale='Full-record centered/scaled before windowing, exactly verified to publisher MAT; offline classification only.')
    elif case == 'p':
        source_rows = read(origin/'audit/PU_SPLIT.json')['source_rows']
        by_id = {v['sample_id']:v for v in source_rows}
        assert {v['bearing'] for v in source_rows} == ARTIFICIAL | NATURAL | HEALTHY
        def damage(row):
            b=row['bearing']
            return 'artificial' if b in ARTIFICIAL else 'accelerated_life' if b in NATURAL else 'healthy'
        for row in private:row['origin_label']=damage(row)
        save(root/'audit/PU_ORIGIN_MAP.json', dict(source='Lessmeier et al.2016, publisher PDF Tables4,5,7',
            pdf_sha256=sha256(root/'source/PU_PRIMARY2016.pdf'), artificial=sorted(ARTIFICIAL),
            accelerated_life=sorted(NATURAL),healthy=sorted(HEALTHY)))
        info.update(origin_classes=['healthy','artificial','accelerated_life'],
            origin_annotations='Original publisher Tables4/5/7; accelerated laboratory lifetime damage is not field degradation.',
            origin_field_train_validation='origin_labels; target unavailable at final inference')
    else:
        source_rows = read(origin/'audit/MENDELEY_SPLIT.json')['source_records']
        by_id = {v['sample_id']:v for v in source_rows}
        shutil.copytree(source/'records', target/'records')
        info['question_erratum']='Actual source variable0..900RPM, not stated600..1800; use actualprofiles only.'
    group_alias = {}
    counts = {}
    for part in ['train','val','test']:
        data = arrays(source/(part+'.npz'))
        rows=[by_id[str(s)] for s in data['sample_ids']]
        if case=='h':
            for key in ['bearing_type','load_w','shaft_frequency_hz']:
                data[key]=np.asarray([condition(v)[key] for v in rows])
        if case=='p' and part!='test':data['origin_labels']=np.asarray([damage(v) for v in rows])
        if part!='test':
            data['groups']=np.asarray([group_alias.setdefault(str(g),'group_'+secrets.token_hex(12)) for g in data['groups']])
        assert case!='p' or part!='test' or 'origin_labels' not in data
        np.savez_compressed(target/(part+'.npz'),**data)
        counts[part]=len(data['sample_ids'])
        assert np.array_equal(data['features'], arrays(source/(part+'.npz'))['features'])
    # Opaque aliases prevent any original fault prefix becoming a public ID.
    info.pop('train_validation_groups', None)
    save(target/'dataset.json', info)
    save(target.parent/'hidden/labels.json', private)
    old_model = next((origin/'outputs'/name).glob('*/model.pt'))
    dest=target.parent/'reference';dest.mkdir()
    for filename in ['model.pt','training_audit.json']:shutil.copy2(old_model.parent/filename,dest/filename)
    save(root/'audit'/(name+'_DERIVATION.json'), dict(source=str(source), source_metadata_sha256=sha256(source/'dataset.json'),
        rows=counts, features_and_fault_labels_unchanged=True, split_unchanged=True,
        raw_fault_labels_not_exposed_in_test=True, group_alias=group_alias,
        source_binding=str(root/'audit'/('HUST_BINDINGS.json' if case=='h' else 'SOURCE257_BINDINGS.json'))))
    if case=='p':fit_origin(target)
    return dict(case=case,dataset=name,rows=counts,passed=True)


def fit_origin(target):
    dest=target.parent/'reference';path=dest/'origin_reference.joblib'
    assert not path.exists(), 'No repeat of a completed origin fit'
    fit=arrays(target/'train.npz');dev=arrays(target/'val.npz')
    model=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=2026102306,n_jobs=2)
    model.fit(fit['features'],fit['origin_labels'])
    joblib.dump(model,path)
    cold=joblib.load(path);pred=cold.predict(dev['features'])
    assert np.array_equal(pred,model.predict(dev['features']))
    save(dest/'origin_training_audit.json',dict(actual_fit=True,train_rows=len(fit['labels']),
        fit_input_sha256=sha256(target/'train.npz'), n_trees=len(model.estimators_),
        actual_tree_nodes=sum(t.tree_.node_count for t in model.estimators_),
        development_accuracy=float(np.mean(pred==dev['origin_labels'])),
        frozen_recipe='RF300,min_samples_leaf2,balanced,seed2026102306',
        checkpoint_sha256=sha256(path),cold_reload_equal=True,final_scores_read=False))


def mfpt(root):
    source=root/'source/mfpt_native';acq=read(source/'ACQUISITION_RESULT.json');assert acq['passed']
    manifest={x['path']:x for x in acq['files']}
    paths=sorted(source.rglob('*.mat'));assert len(paths)==20
    def label(p):
        return 1 if p.name.startswith('InnerRace') else 2 if p.name.startswith('OuterRace') else 0
    development={min(p for p in paths if p.parent.name=='train_data' and label(p)==c) for c in [0,1,2]}
    save(root/'audit/MFPT_SPLIT_BEFORE_FIT.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        test_rule='All6MathWorks test_datafiles unchanged',
        development_files=[str(p.relative_to(source)) for p in sorted(development)],
        train_rule='Remaining11MathWorks train_datafiles',window_samples=2560,seed=2026102306,
        no_model_fit_or_final_score_yet=True))
    data=root/'f/restored/MFPT/data';data.mkdir(parents=True)
    splits={k:[] for k in ['train','val','test']};native=[]
    for path in paths:
        relative=str(path.relative_to(source));assert sha256(path)==manifest[relative]['sha256']
        b=loadmat(path,simplify_cells=True)['bearing'];signal=np.asarray(b['gs'],float).reshape(-1)
        assert np.isfinite(signal).all() and len(signal)>2560
        load=float(b['load']);sr=float(b['sr']);rate=float(b['rate'])
        assert load>=0 and sr in [48828,97656] and rate==25
        part='test' if path.parent.name=='test_data' else 'val' if path in development else 'train'
        alias='record_'+secrets.token_hex(12)
        native.append(dict(file=relative,sha256=sha256(path),samples=len(signal),load_lb=load,
            sampling_hz=sr,shaft_frequency_hz=rate,part=part,label=label(path),group=alias))
        for start in range(0,len(signal)-2560+1,2560):
            wave=signal[start:start+2560]
            splits[part].append(dict(sample_id=secrets.token_hex(16),features=features(wave[:,None]),
                waveforms=wave.astype(np.float32),group=alias,label=label(path),source_file=relative,
                start=start,load_lb=load,sampling_hz=sr,shaft_frequency_hz=rate))
    for part,rows in splits.items():
        np.random.default_rng(secrets.randbits(64)).shuffle(rows)
        payload={k:np.asarray([v[k] for v in rows]) for k in ['sample_id','features','waveforms','load_lb','sampling_hz','shaft_frequency_hz']}
        payload['sample_ids']=payload.pop('sample_id');payload['dataset']=np.asarray('MFPT')
        if part!='test':payload.update(labels=np.asarray([v['label'] for v in rows]),groups=np.asarray([v['group'] for v in rows]))
        np.savez_compressed(data/(part+'.npz'),**payload)
    save(data.parent/'hidden/labels.json',[{k:v for k,v in row.items() if k not in ['features','waveforms']} for row in splits['test']])
    save(root/'audit/MFPT_NATIVE_RECORDS.json',native)
    info=dict(dataset='MFPT',seed=2026102306,class_names=[0,1,2],label_mapping={'0':'healthy','1':'inner race','2':'outer race'},
        architectures=['mlp'],source_sha256=sha256(source/'GIT_TREE.json'),source_pin=read(source/'ACQUISITION_PLAN.json')['tree_sha'],
        rows_by_split={k:len(v) for k,v in splits.items()},groups_by_split={k:len({v['group'] for v in x}) for k,x in splits.items()},
        source_rows=20,group_definition='Whole native recording; no recording crosses splits; multiple recordings may share a physical bearing.',
        preprocessing='Nonoverlapping2560native samples, seven time and eight FFT features. No simulated signals or whole-record normalization.',
        feature_names=['mean','std','rms','peak_abs','peak_to_peak','skewness','kurtosis']+[f'log1p_fft_band_{i}' for i in range(8)],
        units='Native MFPT gs acceleration; native sr in Hz, rate in shaftHz, load in pounds.',
        source_url='https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data',
        deployed_loads_lb=[250,270,300],known_deployment_condition_not_label_prior=True,
        labels_in_test_input=False,source_native_values_verified=True,original_benchmark_score_claimed=False)
    save(data/'dataset.json',info)
    audit=train(data,data.parent/'reference','mlp',50)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    return dict(case='f',dataset='MFPT',rows=info['rows_by_split'],actual_fit=audit['optimizer_steps'],passed=True)


def install_details(root,case,name):
    cohort=root/case;job,=read(cohort/'PLAN187.json')['jobs']
    reference=cohort/'references'/job['id'];workspace=cohort/job['workspace']
    source=cohort/'restored'/name/'data'
    if case=='p':
        for fn in ['origin_reference.joblib','origin_training_audit.json']:shutil.copy2(source.parent/'reference'/fn,reference/fn)
        (workspace/'solution.py').write_text('''"""Actual fitted fault and origin reference candidates."""
import argparse,json,sys
from pathlib import Path
import numpy as np,joblib
sys.path.insert(0,'/tools')
from real_fault_model import infer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=infer(Path('/reference/model.pt'),a.input,a.output)
    with np.load(a.input,allow_pickle=False) as f:
        origins=joblib.load('/reference/origin_reference.joblib').predict(f['features'])
    for row,origin in zip(value['predictions'],origins):row['predicted_origin']=str(origin)
    a.output.write_text(json.dumps(value)+'\\n')
''')
    if case=='m':
        for part in ['train','val','test']:
            dest=(cohort/'evaluation_data' if part=='test' else cohort/'public_data')/job['id']/'datasets'/name
            for rel in arrays(source/(part+'.npz'))['record_files']:
                target=dest/str(rel);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/str(rel),target)
        # Check that each public record belongs to train/dev only, and every test record is external.
        public={str(p.relative_to(cohort/'public_data'/job['id']/'datasets'/name)) for p in (cohort/'public_data'/job['id']/'datasets'/name/'records').glob('*.npz')}
        test=set(arrays(source/'test.npz')['record_files'].tolist())
        assert public.isdisjoint(test) and len(public)==64 and len(test)==15
    freeze(cohort)


def build(root):
    assert not (root/'PREPARATION306.json').exists()
    configure(root)
    rows=[derived(root,c,n) for c,n,_ in SPECS if c!='f']
    rows.append(mfpt(root))
    for case,name,_ in SPECS:
        prepare(root,root/case,'classification','initial')
        install_details(root,case,name)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'PREPARATION306.json',dict(passed=True,rows=rows,planned_original_tasks=[x[2] for x in SPECS],
        no_model_sessions_yet=True,no_final_scores_read=True,configuration_sha256=sha256(root/'CAMPAIGN_CONFIG.json')))
    state(root,status='prepared_four_new_original_task_directions',planned_model_sessions=4,
        new_model_sessions=0,task_count=4,original_coverage_before=9)


def run(root):
    assert read(root/'PREPARATION306.json')['passed']
    execute_wave(root,'new_original_tasks',[root/c for c,_,_ in SPECS])
    state(root,status='four_initial_subjects_terminal_waiting_closed_choice_audit',completed_model_sessions=4,
        no_final_scores_opened=True,next_action='Register actually used candidates, score final and all task-specific stratifications externally.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['build','run'],required=True)
    a=p.parse_args();(build if a.phase=='build' else run)(a.run.resolve())
