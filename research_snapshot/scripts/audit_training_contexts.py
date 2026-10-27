"""A339: compare existing fitted contexts on identical legal acquisitions."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save,sha,utc
from program_revision_runtime import Runtime,load,score

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T181000Z_freeprogram337r1')


def read(path):return json.loads(Path(path).read_text())


def main(root):
    assert not root.exists();root.mkdir(parents=True)
    for name in ['code','outputs','runtime/tmp','runtime/mpl','audit']:(root/name).mkdir(parents=True)
    receipt=read(SOURCE/'ARCHIVE_VERIFIED.json');manifest=SOURCE/receipt['manifest_filename'];assert sha(manifest)==receipt['manifest_sha256']
    inventory={x['path']:x for x in read(manifest)['files']}
    def bound(path):
        relative=str(path.relative_to(SOURCE));assert sha(path)==inventory[relative]['sha256'];return path
    subject=SOURCE/'subjects/witness_transaction_1';programs={i:subject/'programs'/i for i in ['p02','p03']}
    files={role:bound(SOURCE/'private_inputs'/(role+'.npz')) for role in ['development','independent_check']}
    source_bindings={}
    for ident,path in programs.items():
        report=read(bound(path/'REPORT.json'))
        for relative,digest in report['artifacts'].items():assert sha(bound(path/relative))==digest
        source_bindings[ident]=dict(report_sha256=sha(path/'REPORT.json'),artifacts=report['artifacts'])
    protocol=dict(iteration='A339',frozen_utc=utc(),goal='program_update_iteration_goal330',source_archive=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],
        scope='Posthoc developmental mechanism audit,not new independent task or confirmation.',programs=['p02','p03'],
        training_contexts={'cv_16':[32],'cv_32':[16],'full_fit':[16,32]},evaluation_roles=['development','independent_check'],
        hypothesis='On identical legal evaluation acquisitions,the ranking can depend on the checkpoint training context;single-condition CV scores need not rank full-trained checkpoints.',
        rival='Ranking differences mainly reflect different evaluation conditions,without a training-context ranking reversal.',
        identification_limit='Training sample count and condition coverage change together;cannot identify either factor alone.',
        max_new_prediction_processes=32,max_parallel_groups=2,group_seconds=180,total_seconds=600,new_fit_budget=0,new_LLM_budget=0,
        full_trained_predictions_reused=True,source_final_data_used=False,future_confirmation_data_used=False,novelty_claimed=False)
    save(root/'PROTOCOL339.json',protocol);save(root/'audit/SOURCE_BINDINGS339.json',source_bindings)
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(SOURCE/'private_inputs'),common_tools=str(SOURCE/'common_tools'),fit_seconds=0,predict_seconds=90,program_seconds=180))
    for name in ['audit_training_contexts.py','program_revision_runtime.py','controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    save(root/'CODE_FREEZE339.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    shutil.copy2(SOURCE/'retention_policy.json',root/'retention_policy.json')
    out=PROJECT/'results/contextaudit339';out.mkdir(exist_ok=True);save(out/'PROTOCOL339.json',protocol)
    data={role:load(path) for role,path in files.items()};rows=[];deadline=time.monotonic()+protocol['total_seconds']
    save(root/'HANDLE339.json',dict(pid=os.getpid(),status='running',phase='existing_checkpoint_replay',started_utc=utc()))
    for ident,path in programs.items():
        for role,d in data.items():
            pred=bound(path/'development/PREDICTIONS.npz') if role=='development' else bound(subject/'independent_packet'/ident/'PREDICTIONS.npz')
            with np.load(pred,allow_pickle=False) as z:
                assert np.array_equal(z['sample_ids'],d['sample_ids']);result=score(d,z['prediction'])
            rows.append(dict(program=ident,training_context='full_fit',training_conditions=[16,32],evaluation_role=role,checkpoint_reused=True,**result))
    jobs=[(ident,context,role) for ident in programs for context in ['cv_16','cv_32'] for role in data]
    def replay(job):
        ident,context,role=job;path=programs[ident];training=bound(path/context/'train.npz')
        with np.load(training,allow_pickle=False) as z:conditions=sorted(set(z['condition_id'].tolist()))
        assert conditions==protocol['training_contexts'][context]
        engine=Runtime(root);engine.deadline=min(deadline,time.monotonic()+protocol['group_seconds'])
        prediction,audits=engine.predict(root/'outputs'/ident/context/role,path/'source',path/context/'fit/artifacts',data[role])
        assert all(a['completed_estimator_fits']==0 and a['optimizer_steps']==0 for a in audits)
        return dict(program=ident,training_context=context,training_conditions=conditions,evaluation_role=role,new_prediction_processes=len(audits),**score(data[role],prediction))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(replay,j) for j in jobs]):rows.append(f.result());save(root/'PROGRESS339.json',rows)
    reversals=[]
    for role in data:
        get=lambda ident,ctx:next(r['record_error'] for r in rows if r['program']==ident and r['training_context']==ctx and r['evaluation_role']==role)
        full=get('p02','full_fit')-get('p03','full_fit')
        for context in ['cv_16','cv_32']:
            delta=get('p02',context)-get('p03',context)
            reversals.append(dict(evaluation_role=role,single_training_context=context,single_error_difference=delta,full_error_difference=full,strict_rank_reversal=delta*full<0))
    result=dict(iteration='A339',utc=utc(),status='complete',rows=rows,rank_tests=reversals,
        strict_training_context_rank_reversals=sum(x['strict_rank_reversal'] for x in reversals),new_prediction_processes=sum(x.get('new_prediction_processes',0) for x in rows),
        new_fits=0,new_LLM_calls=0,independent_confirmation_scenarios=0,source_final_data_read=False)
    assert result['new_prediction_processes']==32
    save(root/'RESULTS339.json',result);save(out/'RESULTS339.json',result);save(root/'HANDLE339.json',dict(status='terminal',phase='existing_checkpoint_replay',utc=utc()))
    print(json.dumps(dict(rank_tests=reversals,new_prediction_processes=32,new_fits=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
