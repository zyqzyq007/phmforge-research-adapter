"""Bind omitted historical CV evidence to its original pre-test seal.

Read-only numeric audit: no fitting, no model inference, no final predictions.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from acquire_forge_pdmb_sources244 import save,sha256
from run_frozen_candidate_review import source_context

FILES = {
 'a': ['base3_forward.npz','ensemble_probabilities.npz','ensemble_results.json',
       'extrapolation_probabilities.npz','extrapolation_results.json','knn_probabilities.npz',
       'knn_results.json','research_results.json','spectral_results.json'],
 'b': ['cv_experiments.py','final_train_stdout.json'],
 'c': ['candidate_probabilities.npz','candidate_probability_summary.json',
       'et_sweep_probabilities.npz','et_sweep_results.json','spectral_results.json',
       'build_candidate_probabilities.py'],
 'd': ['ensemble_probabilities.npz','ensemble_summary.json','feature_summary.json',
       'model_summary.json','refine_ensemble_summary.json','spectral_refine_summary.json',
       'spectral_summary.json','validation.pretty.json','model_metadata.pretty.json'],
}


def main(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    prior=Path(config['prior_review_archive']);previous_manifest,previous_receipt=source_context(prior)
    rows=[]
    for case in config['cases']:
        source=Path(case['source_archive']);manifest,receipt=source_context(source)
        cohort=case['cohort'];job=json.loads((source/cohort/'PLAN187.json').read_text())['jobs'][0]
        ws=source/cohort/job['workspace'];prefix=str(ws.relative_to(source))
        sealing=json.loads((source/'audit/ALL_CHOICES_SEALED276.json').read_text())
        audit_path=source/cohort/'CODING_CHOICE_AUDIT.json'
        assert sha256(audit_path)==sealing['original_choice_audit_hashes'][cohort]
        audit=json.loads(audit_path.read_text());assert audit['new_hidden_test_scores_not_read']
        assert audit['utc']<sealing['utc']
        sealed_files={x['path']:x for x in audit['rows'][0]['files']}
        old_run=prior/case['id'];frozen=json.loads((old_run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())['files']
        frozen_hashes={x['sha256'] for x in frozen}
        dataset=old_run/'public_data/datasets/ROTORBROKENBAR'
        train=np.load(dataset/'train.npz',allow_pickle=False)
        dev=np.load(dataset/'val.npz',allow_pickle=False)
        # Bind public data to the verified prior archive; never open final files.
        for name in ['train.npz','val.npz']:
            rel=str((dataset/name).relative_to(prior))
            assert sha256(dataset/name)==previous_manifest[rel]['sha256']
        qualified=[]
        for name in FILES[cohort]:
            relative='.runtime/'+name;path=ws/relative;full=prefix+'/'+relative
            proof=sealed_files[relative];assert proof['sha256']==manifest[full]['sha256']==sha256(path)
            assert proof['bytes']==manifest[full]['bytes']==path.stat().st_size
            row=dict(source_path=full,relative_path=relative,basename=name,sha256=proof['sha256'],bytes=proof['bytes'],
                     pre_final_sealed=True,absent_from_previous_frozen_files=proof['sha256'] not in frozen_hashes)
            if path.suffix=='.npz':
                arrays=np.load(path,allow_pickle=False);summary=[]
                for key in arrays.files:
                    value=arrays[key]
                    assert np.issubdtype(value.dtype,np.number) and np.isfinite(value).all(),(case['id'],name,key)
                    item=dict(key=key,shape=list(value.shape),dtype=str(value.dtype),finite=True)
                    if value.ndim==2 and value.shape[1]==5:
                        assert value.min()>=-1e-8 and value.max()<=1+1e-8
                        assert np.allclose(value.sum(axis=1),1,rtol=0,atol=1e-6)
                        if 'oof' in key:
                            part='train';ids=train['sample_ids'];gold=train['labels']
                        elif key.endswith('_dev') or key.startswith(('dev_','dev__')):
                            part='development';ids=dev['sample_ids'];gold=dev['labels']
                        else:
                            load=float(key.rsplit('_',1)[1]);sel=train['operating_load_Nm']==load
                            assert sel.any();part='forward_training_load_'+str(load);ids=train['sample_ids'][sel];gold=train['labels'][sel]
                        assert len(value)==len(ids)
                        pred=value.argmax(axis=1);cm=np.zeros((5,5),dtype=int)
                        for y,p in zip(gold,pred):cm[int(y),int(p)]+=1
                        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(5),where=den!=0)
                        item.update(split=part,probability_simplex=True,correct=int(np.trace(cm)),
                                    accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(f.mean()))
                    elif key=='y':assert np.array_equal(value,train['labels']);item['matches']='training labels'
                    elif key=='y_dev':assert np.array_equal(value,dev['labels']);item['matches']='development labels'
                    elif key=='loads':assert np.array_equal(value,train['operating_load_Nm']);item['matches']='training load'
                    else:raise ValueError('Unidentified historical array: '+key)
                    summary.append(item)
                row['arrays']=summary
            elif path.suffix=='.json':
                # Parsing all values rejects malformed evidence without executing it.
                text=path.read_text()
                strict=lambda x: (_ for _ in ()).throw(ValueError(x))
                try:
                    value=json.loads(text,parse_constant=strict);row['format']='json'
                except json.JSONDecodeError:
                    decoder=json.JSONDecoder(parse_constant=strict);value=[];position=0
                    while position<len(text):
                        while position<len(text) and text[position].isspace():position+=1
                        if position==len(text):break
                        record,position=decoder.raw_decode(text,position);value.append(record)
                    assert value;row['format']='concatenated_json';row['record_count']=len(value)
                row['json_parsed']=True;row['top_level_type']=type(value).__name__
            elif path.suffix=='.py':
                import ast
                ast.parse(path.read_text());row['code_inspected_not_executed']=True
            else:raise ValueError('Unapproved historical evidence type')
            qualified.append(row)
        case['historical_evidence']=[x['relative_path'] for x in qualified]
        rows.append(dict(case=case['id'],source_workspace=prefix,files=qualified,
                         original_pre_final_seal_utc=audit['utc'],original_final_opening_freeze_utc=sealing['utc'],
                         source_manifest_sha256=receipt['manifest_sha256'],
                         train_sha256=sha256(dataset/'train.npz'),development_sha256=sha256(dataset/'val.npz'),
                         probability_array_count=sum(len(x.get('arrays',[])) for x in qualified),
                         absent_files=sum(x['absent_from_previous_frozen_files'] for x in qualified)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),cases=rows,new_fits=0,new_model_inference=0,
                new_LLM_calls=0,final_predictions_opened=False,
                problem='Original .runtime exclusion omitted pre-final CV evidence from A281 review.',
                interpretation='Correct review information scope; original choice losses and initial Agent evidence remain valid.')
    save(root/'audit/HISTORICAL_EVIDENCE284.json',result)
    save(root/'CAMPAIGN_CONFIG.json',config)
    print(json.dumps(dict(cases=len(rows),files=sum(len(x['files']) for x in rows),
                         arrays=sum(x['probability_array_count'] for x in rows),
                         absent_files=sum(x['absent_files'] for x in rows),new_fits=0,final_predictions_opened=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
