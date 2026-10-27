"""Frozen M10 transfer and paired ordinary-Agent gate on archival real banks.

No refitting, outcome-guided tuning, or new physical observations. A329 is an
extension to sources excluded from A328 screening, not prospective confirmation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import ceil
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

import numpy as np

from controlled_feedback import save, sha, utc
import screen_program_updates as runner
from program_selection_policies import choose, validate, EPS

PROJECT = runner.PROJECT
OUT = PROJECT/'results/m10extension329'
PRIOR = Path('/data/yyf-zyq/PHM_20261010/20261024T132000Z_policies328')
SOURCES = dict(motor=Path('/data/yyf-zyq/PHM_20261010/20261021T174126Z_choice270'),
               gear=Path('/data/yyf-zyq/PHM_20261010/20261023T002126Z_ensemble298'))
POLICIES = ['keep','development','cv','M01_group_mean','M10_broad_support',
            'M10_without_coverage','M10_without_dev_guard']


def read(path):
    return json.loads(Path(path).read_text())


def update(root, **changes):
    state=read(root/'STATUS329.json') if (root/'STATUS329.json').exists() else {}
    state.update(updated_utc=utc(), **changes)
    save(root/'STATUS329.json',state);save(OUT/'current_run.json',dict(state,run=str(root)))


def source_index(name):
    source=SOURCES[name];receipt=read(source/'ARCHIVE_VERIFIED.json')
    p=source/receipt['manifest_filename'];assert sha(p)==receipt['manifest_sha256']
    return {r['path']:r for r in read(p)['files']},receipt


def bind_many(root, name, paths):
    """Allowlisted extraction only, without expanding the whole historical run."""
    index,receipt=source_index(name); paths=set(paths);assert paths<=set(index)
    folder=root/'source_views'/name
    missing=[p for p in paths if not (folder/p).exists()]
    if receipt.get('archive_format'):
        bundle=SOURCES[name]/'RUN_CONTENTS.tar.gz'
        marker=root/'audit'/('BUNDLE_'+name+'.json')
        if not marker.exists():
            assert sha(bundle)==receipt['bundle_sha256']
            save(marker,dict(path=str(bundle),sha256=receipt['bundle_sha256'],utc=utc()))
        if missing:
            pending=set(missing)
            with tarfile.open(bundle,'r|gz') as tf:
                for member in tf:
                    rel=member.name.removeprefix('./')
                    if rel not in pending:continue
                    assert member.isfile() and not Path(rel).is_absolute() and '..' not in Path(rel).parts
                    p=folder/rel;p.parent.mkdir(parents=True,exist_ok=True)
                    with tf.extractfile(member) as inp,p.open('xb') as output:shutil.copyfileobj(inp,output)
                    pending.remove(rel)
            assert not pending,pending
    else:
        for rel in missing:
            p=folder/rel;p.parent.mkdir(parents=True,exist_ok=True)
            # Source archive is immutable; retain small exact files needed here.
            shutil.copy2(SOURCES[name]/rel,p)
    bindings=read(root/'SOURCE_BINDINGS329.json') if (root/'SOURCE_BINDINGS329.json').exists() else []
    known={(r['source'],r['path']) for r in bindings}
    for rel in sorted(paths):
        p=folder/rel;r=index[rel];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],rel
        if (name,rel) not in known:bindings.append(dict(source=name,path=rel,**{k:r[k] for k in ['bytes','sha256']}))
    save(root/'SOURCE_BINDINGS329.json',bindings)
    return folder


def freeze(root):
    root.mkdir(exist_ok=False)
    for d in ['runtime/tmp','runtime/mpl','audit','logs','code','shared_tools','public/rows','validation_inputs']:(root/d).mkdir(parents=True)
    assert sha(PROJECT/'scripts/program_selection_policies.py')==sha(PRIOR/'code/program_selection_policies.py')
    original=read(PRIOR/'CAMPAIGN_CONFIG.json')
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research-provenance'}),timeout=30) as r:release=json.load(r)
    assert release['tag_name']=='rust-v0.164.0','Resolve newer stable CLI before freeze'
    assert sha(Path(original['cli']))==original['cli_sha256']
    save(root/'audit/CLI_SOURCE.json',dict(url=url,checked_utc=utc(),tag=release['tag_name'],published_at=release['published_at']))
    config=dict(iteration='A329',user_authorization='继续尝试M10的方法',frozen_utc=utc(),
        cli=original['cli'],cli_sha256=original['cli_sha256'],model='deepseek-flash',
        episode_seconds=1200,episode_requests=50,episode_input_tokens=2500000,episode_output_tokens=65536,
        output_per_request=8192,numerical_threads=2,maximum_parallel_agents=2,repeats=2,
        preparation_file='PREPARATION329.json',methods=POLICIES,new_fit_budget=0,new_final_forward_budget=0,
        candidate_policy='All6 originally supplied fixed Motor bank models for each allocation;all6 G2 basis models plus every distinct actually submitted initial/review mixture. No final-score-based candidate selection.',
        source_archives={k:dict(path=str(p),manifest_sha256=source_index(k)[1]['manifest_sha256']) for k,p in SOURCES.items()},
        group_policy='Motor:each complete source record,18 CV records+6 development records,record-equal error;G2:each original held-condition CV/development group,1-macroF1 over all6 classes.',
        primary_panel='All eligible candidates as counterfactual incumbents,first pool-equal then source-equal;not independent tasks or natural failure frequency.',
        secondary_anchors='Motor original bank starting_model;G2 first original registered basis. No outcome-based anchors.',
        gate='For each actual ordinary-Agent proposal,accept only if exactly the frozen M10 eligibility holds vs incumbent;otherwise keep. No new Agent call or alternate selection. Separately report full-bank M10 reranking.',
        ablations='Remove coverage alone,or development non-regression alone;keep mean strict improvement,tie policy and other conditions fixed. No threshold search.',
        primary='Normalized selection regret;jointly report harmful updates,gain capture,native metrics,counts,cost. No significance claim from2 source families.',
        promise_rule='All-incumbent source-equal regret lower than both ordinary repeats,source-equal harm no greater,and positive useful gain. Report misses separately even if this rule passes.',
        M10_code_sha256=sha(PROJECT/'scripts/program_selection_policies.py'),M10_rule_unchanged=True,
        researcher_previously_knows_final_scores=True,confirmation_experiment=False,
        disclosure='Cross-source extension after A328,not new data. The motor is distinct from RotorBrokenBar;G2 is a second gear from the G1 laboratory source,not a new institution.',
        maximum_new_disk_GiB=8,maximum_model_requests=100,
        missing_evidence_policy='Stop before Agent admission if exact per-group CV or development cannot be bound;do not invent group values or refit opened data.',
        failed_episode_policy='Retain every attempt;fallback keep is distinct from successful method output.')
    save(root/'CAMPAIGN_CONFIG.json',config);save(OUT/'PROTOCOL329.json',config)
    save(root/'PROMPT_FREEZE.json',dict(ordinary=runner.PROMPT))
    for p in (PROJECT/'scripts').glob('*.py'):shutil.copy2(p,root/'code'/p.name)
    for p in (PRIOR/'shared_tools').glob('*.py'):shutil.copy2(p,root/'shared_tools'/p.name)
    for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:shutil.copy2(PRIOR/name,root/name)
    save(root/'CODE_FREEZE.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    save(root/'OWNERSHIP.json',dict(run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name,user_request=config['user_authorization']))
    update(root,status='frozen_before_extension_decisions',new_fits=0,new_agent_sessions=0,final_scores_joined=False)
    print(json.dumps(dict(run=str(root),status='frozen')))


def cold(root, caseid, kind, jobs):
    """Cold checkpoints see legal observation-only inputs and no old final files."""
    folder=root/'audit/cold'/caseid;folder.mkdir(parents=True)
    for d in ['tmp','mpl']:(folder/d).mkdir()
    tasks=[];mounts=[]
    for i,j in enumerate(jobs):
        tasks.append(dict(name=j['name'],kind=kind,model='/models/'+str(i)+'.joblib',input='/inputs/'+str(i)+'.npz'))
        mounts.extend([(j['model'],'/models/'+str(i)+'.joblib'),(j['input'],'/inputs/'+str(i)+'.npz')])
    save(folder/'jobs.json',tasks)
    code="""import json,joblib,numpy as np
from pathlib import Path
from controlled_feedback import transform
for j in json.loads(Path('/output/jobs.json').read_text()):
 d=np.load(j['input'],allow_pickle=False);assert 'labels' not in d.files and 'groups' not in d.files
 m=joblib.load(j['model'])
 p=m.predict(d['features'].astype(float)) if j['kind']=='motor' else m['model'].predict_proba(transform(d['waveforms'],m['recipe']['features']))
 np.savez_compressed('/output/'+j['name']+'.npz',prediction=p,sample_ids=d['sample_ids'])
print(json.dumps(dict(passed=True,loads=len(json.loads(Path('/output/jobs.json').read_text())),new_fits=0)))
"""
    cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    for p,q in [(runner.PYTHON,'/opt/python312'),(runner.DEPS,'/opt/deps'),(root/'code','/tools'),*mounts]:cmd+=['--ro-bind',str(p),q]
    cmd+=['--bind',str(folder),'/output','--proc','/proc','--dev','/dev','--chdir','/output','--remount-ro','/',
          '/opt/python312/bin/python3.12','-B','-c',code]
    env=runner.env_for(root);env.update(PYTHONPATH='/opt/deps:/tools',TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/mpl')
    started=time.monotonic();result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=240)
    save(folder/'STATUS.json',dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,seconds=time.monotonic()-started))
    assert result.returncode==0,result.stderr[-2000:]
    return folder


def prepare(root):
    from evaluate_program_update_screen import classification
    assert not (root/'PREPARATION329.json').exists()
    cases=[];checks=[];jobs_by_case={};motor_index,_=source_index('motor')
    paths=[]
    for cohort in ['a','b']:
        tag='s270_motor_'+cohort+'_initial'
        paths += [cohort+'/public_data/'+tag+'/datasets/ELECTRICMOTORVIBRATIONS/'+p for p in ['train.npz','val.npz']]
        paths += [p for p in motor_index if p.startswith('bank_selection/motor_'+cohort+'/')]
        paths += [p for p in motor_index if p.startswith(cohort+'/references/'+tag+'/candidates/')]
    source=bind_many(root,'motor',paths)
    for cohort in ['a','b']:
        caseid='motor_'+cohort;tag='s270_motor_'+cohort+'_initial'
        data=source/cohort/'public_data'/tag/'datasets/ELECTRICMOTORVIBRATIONS'
        train=dict(np.load(data/'train.npz',allow_pickle=False));dev=dict(np.load(data/'val.npz',allow_pickle=False))
        bank=source/cohort/'references'/tag/'candidates';catalog=read(bank/'manifest.json')
        cache=source/'bank_selection'/caseid;cv=read(cache/'TRAINING_CV.json')
        assert sha(data/'train.npz')==cv['train_sha256'] and sha(data/'val.npz')==cv['dev_sha256']
        assert not set(train['groups'])&set(dev['groups'])
        folder=root/'validation_inputs'/caseid;folder.mkdir()
        np.savez_compressed(folder/'development.npz',features=dev['features'],sample_ids=dev['sample_ids'])
        masks={};seen=[]
        for fold in cv['folds']:
            mask=np.isin(train['sample_ids'],fold['heldout_ids']);masks[fold['fold']]=mask
            assert not set(fold['fit_ids'])&set(fold['heldout_ids'])
            assert set(fold['fit_ids'])==set(train['sample_ids'])-set(fold['heldout_ids'])
            assert not set(train['groups'][mask])&set(train['groups'][~mask])
            seen+=fold['heldout_ids'];np.savez_compressed(folder/('fold'+str(fold['fold'])+'.npz'),features=train['features'][mask],sample_ids=train['sample_ids'][mask])
        assert len(seen)==len(set(seen))==len(train['labels'])
        models=[m for m in catalog['models'] if 'checkpoint' in m];assert len(models)==6
        jobs=[]
        for m in models:
            jobs.append(dict(name=m['name']+'_dev',model=bank/m['checkpoint'],input=folder/'development.npz'))
            for fold in cv['folds']:jobs.append(dict(name=m['name']+'_cv'+str(fold['fold']),model=cache/'cv_models'/(m['name']+'_fold'+str(fold['fold'])+'.joblib'),input=folder/('fold'+str(fold['fold'])+'.npz')))
        jobs_by_case[caseid]=('motor',jobs)
        groups=[dict(id=part+':'+str(g),split=part,source_record=str(g),unit='complete source record')
                for part,d in [('cv',train),('development',dev)] for g in sorted(set(d['groups']))]
        cases.append(dict(id=caseid,source='electric_motor',metric='record-equal window classification error',
            candidate_ids=[m['name'] for m in models],candidate_descriptions=[m['name']+';original train-only fitted checkpoint' for m in models],
            groups=groups,development_columns=[i for i,g in enumerate(groups) if g['split']=='development'],
            original_incumbent=catalog['starting_model'],limitations='Two related allocations of the same30source records;unknown physical device identity;integer labels have no verified physical fault names. No cross-device claim.'))
        # Bind exact existing rows; after cold execution compare all CV/full outputs.
        save(folder/'LEGAL_METADATA.json',dict(train={k:train[k].tolist() for k in ['sample_ids','labels','groups']},development={k:dev[k].tolist() for k in ['sample_ids','labels','groups']},models=models))
    gear_index,_=source_index('gear')
    paths=['CAMPAIGN_CONFIG.json','BASIS_QUALIFICATION.json','bank/probabilities.npz','hidden/val.npz']
    paths += [p for p in gear_index if p.startswith('basis/') and p.endswith(('.json','.npz','.joblib'))]
    cfg=read(SOURCES['gear']/'CAMPAIGN_CONFIG.json')
    paths += [prefix+c['id']+'/broker/STATE.json' for prefix in ['', 'review/'] for c in cfg['cases']]
    gear=bind_many(root,'gear',paths);basis=read(gear/'BASIS_QUALIFICATION.json');bank=dict(np.load(gear/'bank/probabilities.npz',allow_pickle=False))
    assert basis['passed'] and basis['wrapper_fit_calls_completed']==24 and sha(gear/'bank/probabilities.npz')==basis['bank_sha256']
    names=bank['basis_names'].tolist();assert len(names)==6
    weights=[np.eye(6,dtype=int)[i].tolist() for i in range(6)];ids=list(names);submissions=[]
    for prefix in ['', 'review/']:
        for c in cfg['cases']:
            state=read(gear/(prefix+c['id']+'/broker/STATE.json'));w=state['final_choice']['weights']
            assert state['final_choice'] is not None
            submissions.append(dict(phase=prefix or 'initial',case=c['id'],weights=w))
            # Proportional mixtures denote the same unchanged probability program.
            if not any(np.allclose(np.asarray(w)/sum(w),np.asarray(q)/sum(q),rtol=0,atol=1e-15) for q in weights):
                weights.append(w);ids.append('submitted_mixture_'+str(len(ids)-5))
    assert len(ids)==7 and len(submissions)==12
    save(root/'G2_CANDIDATE_BINDING.json',dict(ids=ids,weights=weights,original_submissions=submissions))
    groups=[dict(id=part+':'+str(g),split=part,condition=str(g),unit='complete operating-condition segment')
            for part in ['cv','development'] for g in sorted(set(bank[part+'_conditions']))]
    rows={};risk=[]
    for cid,w in zip(ids,weights):
        rows[cid]={};rr=[]
        for part in ['cv','development']:
            p=np.asarray(bank['classes'])[np.einsum('nmc,m->nc',bank[part],np.asarray(w)/sum(w)).argmax(1)]
            rows[cid][part]=[dict(sample_id=str(i),actual=int(y),predicted=int(v),group=str(g)) for i,y,v,g in zip(bank[part+'_ids'],bank[part+'_labels'],p,bank[part+'_conditions'])]
            for g in sorted(set(bank[part+'_conditions'])):
                mask=bank[part+'_conditions']==g;rr.append(1-classification(bank[part+'_labels'][mask],p[mask],bank['classes'].tolist())['macro_f1'])
        risk.append(rr)
    cases.append(dict(id='gear_g2',source='gear_g2',metric='1-macroF1 over six published crack stages',candidate_ids=ids,
        candidate_descriptions=[dict(weights=w,basis_names=names) for w in weights],groups=groups,group_losses=risk,
        development_columns=[i for i,g in enumerate(groups) if g['split']=='development'],original_incumbent=ids[0],
        limitations='One actual gear;four related validation conditions. Bank excludes other searched mixtures;no claim of globally optimal coding or weight search.'))
    save(root/'public/rows/gear_g2.json',rows)
    folder=root/'validation_inputs/gear_g2';folder.mkdir()
    dev=dict(np.load(gear/'hidden/val.npz',allow_pickle=False));assert np.array_equal(dev['sample_ids'],bank['development_ids'])
    np.savez_compressed(folder/'development.npz',waveforms=dev['waveforms'],sample_ids=dev['sample_ids'])
    jobs_by_case['gear_g2']=('gear',[dict(name=n,model=gear/'basis'/n/'checkpoint.joblib',input=folder/'development.npz') for n in names])
    with ThreadPoolExecutor(max_workers=3) as pool:
        future={cid:pool.submit(cold,root,cid,kind,jobs) for cid,(kind,jobs) in jobs_by_case.items()}
        outputs={k:f.result() for k,f in future.items()}
    for case in cases[:2]:
        cid=case['id'];metadata=read(root/'validation_inputs'/cid/'LEGAL_METADATA.json');rows={};risk=[]
        cache=source/'bank_selection'/cid;folds=read(cache/'TRAINING_CV.json')['folds']
        for name in case['candidate_ids']:
            rr=[];rows[name]={}
            for part,key in [('cv','train'),('development','development')]:
                d=metadata[key];pred={}
                files=[outputs[cid]/(name+'_cv'+str(f['fold'])+'.npz') for f in folds] if part=='cv' else [outputs[cid]/(name+'_dev.npz')]
                for f in files:
                    with np.load(f,allow_pickle=False) as z:
                        for i,p in zip(z['sample_ids'],z['prediction']):assert str(i) not in pred;pred[str(i)]=int(p)
                assert set(pred)==set(d['sample_ids'])
                if part=='cv':assert pred=={r['sample_id']:r['predicted_label'] for r in read(cache/(name+'_OOF.json'))['predictions']}
                p=np.asarray([pred[i] for i in d['sample_ids']]);y=np.asarray(d['labels']);g=np.asarray(d['groups'])
                rows[name][part]=[dict(sample_id=i,actual=int(a),predicted=int(b),group=s) for i,a,b,s in zip(d['sample_ids'],y,p,d['groups'])]
                for s in sorted(set(g)):rr.append(float(np.mean(p[g==s]!=y[g==s])))
                m=classification(y,p,list(range(4)))
                original=next(v for v in metadata['models'] if v['name']==name)
                expected=original['training_group_cv']['macro_f1'] if part=='cv' else original['development_macro_f1']
                assert abs(m['macro_f1']-expected)<1e-12
                checks.append(dict(case=cid,candidate=name,part=part,matched_original_predictions_or_metrics=True))
            risk.append(rr)
        case['group_losses']=risk;save(root/'public/rows'/(cid+'.json'),rows)
    for i,n in enumerate(names):
        z=np.load(outputs['gear_g2']/(n+'.npz'),allow_pickle=False)
        assert np.array_equal(z['sample_ids'],bank['development_ids']) and np.allclose(z['prediction'],bank['development'][:,i,:],rtol=0,atol=1e-12)
        report=read(gear/'basis'/n/'fit_report.json');assert report['real_fits']==4
        for fold in report['cv_folds']:assert not set(fold['fitting_ids'])&set(fold['checking_ids']) and not set(fold['fitting_conditions'])&set(fold['checking_conditions'])
        checks.append(dict(case='gear_g2',candidate=n,development_cold_equal=True,whole_condition_CV_disjoint=True,CV_weights_not_saved_originally=True))
    for c in cases:validate(c)
    save(root/'public/cases.json',cases)
    save(root/'public/contract.json',dict(scope='Three real archival candidate pools,Motor two related splits and G2 one gear. No final data or outcomes.',
        tasks='Select complete immutable predictors using CV/development evidence. All possible incumbents are counterfactual choices;not independent PHM experiments.',
        loss=read(root/'CAMPAIGN_CONFIG.json')['group_policy'],training='All previously fitted on real measured data. No fitting allowed.',
        candidates='6+6+7 complete programs;G2 includes every distinct actual historical final mixture plus its6 bases. No other mixture search in this experiment.'))
    result=dict(passed=True,utc=utc(),cases=3,source_families=2,candidate_incumbents=sum(len(c['candidate_ids']) for c in cases),
        checkpoint_cold_reloads=54,reused_component_fits=72,new_fits=0,new_final_forwards=0,
        verified_CV_cold_models=36,G2_CV_evidence='18 genuine original fit calls and SHA-bound OOF probabilities;fold checkpoints not retained,not reloaded or refitted.',
        checks=checks,cases_sha256=sha(root/'public/cases.json'),confirmation_experiment=False)
    save(root/'PREPARATION329.json',result);save(OUT/'PREPARATION329.json',result)
    update(root,status='real_evidence_qualified',planned_agent_sessions=2,validation_checkpoints_reloaded=54)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


def select(case, incumbent, method):
    if method in ['M10_without_coverage','M10_without_dev_guard']:
        ids,r=validate(case);old=ids.index(incumbent);mu=r.mean(1);dev=r[:,case['development_columns']].mean(1)
        eligible=[old]+[j for j in range(len(ids)) if mu[j]<mu[old]-EPS
            and (method=='M10_without_coverage' or int((r[j]<r[old]-EPS).sum())>=max(2,ceil(r.shape[1]/2)))
            and (method=='M10_without_dev_guard' or dev[j]<=dev[old]+EPS)]
        best=min(mu[j] for j in eligible);tied=[j for j in eligible if mu[j]<=best+EPS]
        chosen=old if old in tied else min(tied)
        return ids[chosen],dict(eligible=[ids[j] for j in eligible],selected=ids[chosen])
    return choose(case,incumbent,method)


def fixed(root):
    assert read(root/'PREPARATION329.json')['passed'] and not (root/'FIXED_CHOICES.json').exists()
    cases=read(root/'public/cases.json');start=time.monotonic();choices={}
    for method in POLICIES:
        choices[method]=[dict(case=c['id'],incumbent=i,selected=select(c,i,method)[0],evidence=select(c,i,method)[1]) for c in cases for i in c['candidate_ids']]
    save(root/'FIXED_CHOICES.json',dict(utc=utc(),choices=choices,seconds=time.monotonic()-start,final_scores_joined=False,
        policy_code_sha256=sha(root/'code/program_selection_policies.py'),cases_sha256=sha(root/'public/cases.json')))
    save(OUT/'FIXED_CHOICES.json',read(root/'FIXED_CHOICES.json'))
    update(root,status='frozen_M10_controls_and_ablations_sealed')


def agents(root):
    # The inherited gateway launcher resolves PROJECT/scripts when imported
    # from a run snapshot. Install that exact frozen file before admission.
    gateway=root/'scripts/isolated_model_gateway.py'
    if not gateway.exists():
        gateway.parent.mkdir(exist_ok=True)
        shutil.copy2(root/'code/isolated_model_gateway.py',gateway)
    assert sha(gateway)==read(root/'CODE_FREEZE.json')['isolated_model_gateway.py']
    runner.OUT=OUT;runner.update=update
    runner.agents(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','fixed','agents']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
