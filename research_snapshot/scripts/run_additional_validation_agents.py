"""A418: ordinary native Agent qualification on shared real calibration evidence.

Reuses the established CLI/MCP/broker/gateway implementation. This is a
development-only, fixed complete-program selection study, not a new method.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import numpy as np

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from screen_program_updates import env_for, agents


PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = '/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    cfg = read(root/'PROTOCOL418.json')
    source, programs = Path(cfg['prediction_archive']), Path(cfg['program_archive'])
    manifest = read(source/'SOURCE_MANIFEST.json')
    check = read(source/'ARCHIVE_VERIFIED.json')
    assert sha(source/'SOURCE_MANIFEST.json') == cfg['prediction_manifest_sha256'] == check['manifest_sha256']
    names = {'PREPARED417.json','AUDIT417.json'}
    for acq in [7,8]:
        tag = f'a{acq}_l30'
        names.update([f'private/{tag}/gold.json',f'observations/{tag}/datasets/ROTORBROKENBAR/test.npz'])
        names.update(f'replays/{tag}/{c}/0/output/complete.json' for c in cfg['cohorts'])
    entries = {r['path']:r for r in manifest['files'] if r['path'] in names}
    assert len(entries)==len(names)
    with tarfile.open(source/check['bundle_filename'],'r|gz') as archive:
        for member in archive:
            if member.name not in entries:
                continue
            raw = archive.extractfile(member).read()
            entry = entries[member.name]
            assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
            dest = root/'source'/member.name
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            names.remove(member.name)
            if not names:
                break
    assert not names
    save(root/'SOURCE_RESTORED418.json',dict(utc=utc(),files=entries,
        archive=str(source),manifest_sha256=cfg['prediction_manifest_sha256']))
    code_names = closure(['screen_program_updates','program_update_tools','native_validation_comparison'])
    for name in code_names:
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    template = Path(cfg['runner_template'])
    candidate_map = cfg['candidate_map']
    model_manifest = {r['path']:r for r in read(programs/'SOURCE_MANIFEST.json')['files']}
    all_fixed = []
    for acq in [7,8]:
        child = root/f'cal{acq}'
        for folder in ['code','shared_tools','scripts','public/rows/programs','runtime/tmp','runtime/mpl','audit']:
            (child/folder).mkdir(parents=True)
        for file in (root/'code').glob('*.py'):
            shutil.copy2(file,child/'code'/file.name)
        for name in ['program_update_tools.py','controlled_feedback.py','isolated_model_gateway.py']:
            shutil.copy2(PROJECT/'scripts'/name,child/'shared_tools'/name)
        shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',child/'scripts/isolated_model_gateway.py')
        for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:
            shutil.copy2(template/name,child/name)
        tag = f'a{acq}_l30'
        gold = read(root/'source/private'/tag/'gold.json')
        expected_ids = {g['sample_id'] for g in gold}
        observation = root/'source/observations'/tag/'datasets/ROTORBROKENBAR/test.npz'
        shutil.copy2(observation,child/'public/rows/calibration_observations.npz')
        grouped = {}
        for row in gold:
            grouped.setdefault(row['group'],[]).append(row)
        catalog, candidate_losses, old_losses, detailed, source_files = [],[],[],{},[]
        for cohort in cfg['cohorts']:
            identity = candidate_map[cohort]
            predictions = read(root/'source/replays'/tag/cohort/'0/output/complete.json')['predictions']
            mapping = {p['sample_id']:int(p['predicted_label']) for p in predictions}
            assert set(mapping)==expected_ids and len(mapping)==len(predictions)
            group_losses = [sum(mapping[x['sample_id']]!=x['label'] for x in group)/len(group) for _,group in sorted(grouped.items())]
            complete, = [row for row in read(programs/f'registered_replays/val/{cohort}/RESULT.json')['rows'] if row['name']=='complete']
            prior_loss = 1-complete['metrics']['mean_group_accuracy']
            candidate_losses.append(group_losses);old_losses.append(prior_loss)
            detailed[identity] = dict(predictions=predictions,calibration_group_losses=group_losses,
                historical_full_program_validation=complete['metrics'])
            app = programs/f'registered_replays/test/{cohort}/app'
            selected_files = ['solution.py']+[n for n in ['modeling.py','rotor_model.py','rotor_pipeline.py'] if (app/n).exists()]
            for name in selected_files:
                file = app/name;rel=str(file.relative_to(programs));entry=model_manifest[rel]
                assert sha(file)==entry['sha256']
                dest = child/'public/rows/programs'/identity/name
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
                source_files.append(dict(path=str(dest.relative_to(child/'public')),sha256=entry['sha256']))
            catalog.append(dict(id=identity,role='complete original frozen Agent submission',
                source_code_directory='/data/rows/programs/'+identity,
                supervised_fit_scope='Measured acquisitions3/4' if cohort[0] in 'ab' else 'Measured acquisitions5/6',
                fitting_loads_Nm=[.5,1.,1.5,2.,2.5],historical_validation_load_Nm=3.,
                no_group_ids_at_inference=True,original_unsupervised_batch_transforms_retained=True))
        public = dict(id='rotor_program_portfolio',candidate_ids=[candidate_map[c] for c in cfg['cohorts']],
            incumbent_ids=[candidate_map[c] for c in 'abcd'],catalog=catalog,
            primary_metric=cfg['primary_metric'],calibration_records=len(grouped),calibration_windows=len(gold),
            calibration_load_Nm=3.,declared_deployment_loads_Nm=[3.5,4.],
            calibration_group_ids=sorted(grouped),calibration_group_losses=candidate_losses,
            historical_validation_mean_record_errors=old_losses,
            measured_prediction_cache=True,new_supervised_fits=0,
            warnings=['All eight complete programs are selectable;not their individual components.',
                'A counterfactual incumbent is not assumed to be best or wrong.',
                'Source programs used different fitting acquisitions. Compare their measured predictions on the common calibration records.',
                'Calibration evidence does not guarantee future deployment accuracy.'])
        save(child/'public/cases.json',[public])
        save(child/'public/rows/rotor_program_portfolio.json',dict(gold=gold,candidates=detailed,
            source_files=source_files,observation_path='/data/rows/calibration_observations.npz',
            sampling_hz=7600,window_points=2048,feature_order='statistic-major then channel;7time and8FFTband stats/channel',
            raw_measurements='5published vibrationchannels,nonoverlapping complete windows, no synthesis'))
        save(child/'public/contract.json',dict(task='PHMForge pdm_fault_004 research adaptation:measured rotor broken-bar severity',
            classes=[0,1,2,3,4],record_definition='One complete measured acquisition at one fault state and load',
            source_scope='One experimental motor;no independent motor claim',mode='Fixed original complete program selection',
            declared_deployment_loads_Nm=[3.5,4.],primary_metric=cfg['primary_metric'],
            information='Historical complete-program validation plus one additional measured3Nm acquisition perclass;all candidates share the same extra calibration data.',
            evaluation_boundary='Other-acquisition3.5/4Nmobservations,labels,predictions and prior finalconclusions are absent.',
            cached_evidence='Actual original frozen program predictions from independent cold execution;cached explicitly,not newly trained or fabricated scores.',
            operations='Analyze provided rows and source;select exactly one existing complete program per initial incumbent;no new fitting,weights,ensembles or new predictors.'))
        fixed = []
        current = np.mean(candidate_losses,axis=1)
        old = np.asarray(old_losses)
        ids=public['candidate_ids']
        for incumbent in public['incumbent_ids']:
            def best(values):
                lowest=min(values);ties=[ids[i] for i,x in enumerate(values) if abs(x-lowest)<1e-12]
                return incumbent if incumbent in ties else min(ties)
            fixed.append(dict(case=public['id'],incumbent=incumbent,selected=dict(keep=incumbent,
                additional_calibration=best(current),historical_development=best(old),
                mean_old_and_new=best((current+old)/2),worst_old_and_new=best(np.maximum(current,old)))))
        save(child/'FIXED_CHOICES.json',dict(utc=utc(),choices=fixed,final_scores_joined=False))
        all_fixed.append(dict(calibration_acquisition=acq,held_acquisition=15-acq,choices=fixed))
        cc = dict(iteration='A418',cli=cfg['cli'],cli_sha256=cfg['cli_sha256'],model='deepseek-flash',
            episode_seconds=1200,episode_requests=50,episode_input_tokens=2500000,
            episode_output_tokens=65536,output_per_request=8192,maximum_parallel_agents=2,
            repeats=2,selector_conditions=['ordinary'],preparation_file='PREPARATION418.json',
            status_file='STATUS418.json',result_key=f'calibration418/cal{acq}',
            qualify_submission_chain=True,reset_qualification_transport=True,
            researcher_previously_knows_final_scores=True)
        save(child/'CAMPAIGN_CONFIG.json',cc)
        save(child/'PROMPT_FREEZE.json',dict(ordinary=cfg['prompt']))
        save(child/'CODE_FREEZE.json',{p.name:sha(p) for p in (child/'code').glob('*.py')})
        save(child/'PREPARATION418.json',dict(passed=True,new_fits=0,new_cold_forwards=0,
            public_files={str(p.relative_to(child/'public')):sha(p) for p in (child/'public').rglob('*') if p.is_file()},
            original_prediction_manifest=cfg['prediction_manifest_sha256']))
    save(root/'FIXED_CHOICES418.json',dict(utc=utc(),rows=all_fixed,final_scores_joined=False))
    save(root/'PREPARED418.json',dict(utc=utc(),passed=True,children=['cal7','cal8'],episodes=4,
        raw_records=10,complete_programs=8,incumbents=4,source_clusters=1,
        new_fits=0,new_forwards=0,prediction_cache_bound=True))


def run(root):
    assert read(root/'PREPARED418.json')['passed']
    start=time.monotonic()
    def child(acq):
        path=root/f'cal{acq}'
        command=[PYTHON,'-B',str(path/'code/screen_program_updates.py'),'agents','--run',str(path)]
        save(path/'COMMAND418.json',command)
        with (path/'controller.stdout').open('wb') as out,(path/'controller.stderr').open('wb') as err:
            process=subprocess.Popen(command,env=env_for(path),stdout=out,stderr=err)
            save(path/'CONTROLLER418.json',dict(pid=process.pid,status='running',started_utc=utc()))
            code=process.wait()
            save(path/'CONTROLLER418.json',dict(pid=process.pid,status='terminal',exit_code=code,ended_utc=utc()))
        return dict(child=path.name,exit_code=code)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(child,[7,8]))
    save(root/'NATIVE_EXECUTION418.json',dict(utc=utc(),rows=rows,seconds=time.monotonic()-start,terminal=True))
    assert all(row['exit_code']==0 for row in rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();{'prepare':prepare,'run':run}[args.mode](args.run)
