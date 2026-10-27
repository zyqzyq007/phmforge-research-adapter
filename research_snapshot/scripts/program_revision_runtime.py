"""Actual arbitrary-code training and label-isolated prediction for PHM."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from controlled_feedback import save,sha,utc


PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def subset(data,mask):return {k:v[mask] for k,v in data.items()}


def observations(data):return {k:v for k,v in data.items() if k!='labels'}


def score(data,prediction):
    assert len(prediction)==len(data['labels'])
    actual=[];pred=[];records=[]
    for group in sorted(set(data['groups'].tolist())):
        mask=data['groups']==group;truth=set(data['labels'][mask].tolist());assert len(truth)==1
        y=int(next(iter(truth)));counts=np.bincount(prediction[mask],minlength=4)
        p=int(np.flatnonzero(counts==counts.max())[0]);actual.append(y);pred.append(p)
        records.append(dict(group=group,actual=y,predicted=p,condition_hz=float(data['condition_id'][mask][0]),
                            window_correct=int(np.sum(prediction[mask]==data['labels'][mask])),windows=int(mask.sum())))
    confusion=[[sum(y==i and p==j for y,p in zip(actual,pred)) for j in range(4)] for i in range(4)]
    f1=[]
    for i in range(4):
        denominator=sum(confusion[i])+sum(row[i] for row in confusion)
        f1.append(2*confusion[i][i]/denominator if denominator else 0.)
    return dict(record_error=sum(y!=p for y,p in zip(actual,pred))/len(actual),
        record_accuracy=sum(y==p for y,p in zip(actual,pred))/len(actual),macro_F1=sum(f1)/4,
        window_error=float(np.mean(prediction!=data['labels'])),confusion=confusion,records=records)


class Runtime:
    def __init__(self,root):
        self.root=Path(root);self.config=json.loads((self.root/'ENGINE_CONFIG.json').read_text())
        self.inputs=Path(self.config['input_root']);self.common=Path(self.config['common_tools'])

    def command(self,source,artifacts,input_file,output,mode,extra_readonly_mounts=()):
        output.mkdir(parents=True,exist_ok=False);(output/'tmp/mpl').mkdir(parents=True)
        command=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
            if Path(p).exists():command+=['--ro-bind',p,p]
        for p,destination in [(PYTHON,'/opt/python312'),(DEPS,'/opt/deps'),(self.common,'/tools'),
            (source,'/program'),(input_file,'/input.npz'),*extra_readonly_mounts]:command+=['--ro-bind',str(p),destination]
        if mode=='predict':command+=['--ro-bind',str(artifacts),'/artifacts']
        command+=['--bind',str(output),'/work','--proc','/proc','--dev','/dev','--chdir','/program','--remount-ro','/']
        call=['/opt/python312/bin/python3.12','-B','/tools/program_training_entry.py','--audit','/work/training_audit.json',
              '--mode',mode,'/program/program.py',mode,'--model-dir','/work/artifacts' if mode=='fit' else '/artifacts']
        call+=['--train','/input.npz'] if mode=='fit' else ['--input','/input.npz','--output','/work/predictions.json']
        command+=call
        env=dict(PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps:/tools:/program',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR='/work/tmp',TMP='/work/tmp',TEMP='/work/tmp',MPLCONFIGDIR='/work/tmp/mpl',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        started=time.monotonic()
        try:
            limit=self.config['fit_seconds'] if mode=='fit' else self.config['predict_seconds']
            if hasattr(self,'deadline'):
                limit=min(limit,self.deadline-time.monotonic());assert limit>0,'Wholeprogramtimebudgetexhausted'
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=limit)
            result=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-started,mode=mode,command=command)
        except subprocess.TimeoutExpired:
            result=dict(returncode=None,error='program_timeout',seconds=time.monotonic()-started,mode=mode,command=command)
        save(output/'PROCESS.json',result)
        assert result['returncode']==0,('Authored program failed: '+str(output.relative_to(self.root))+' '+result.get('stderr',result.get('error',''))[-2400:])
        audit=json.loads((output/'training_audit.json').read_text())
        if mode=='fit':
            assert audit['completed_estimator_fits']>0 or audit['nonzero_sampled_optimizer_steps']>0,'Noauditedclassifierfitoroptimizerupdate'
            assert (output/'artifacts').is_dir() and any(p.is_file() for p in (output/'artifacts').rglob('*'))
        elif self.config.get('allow_past_capacity_online_fit',False):
            assert audit['optimizer_steps']<=self.config['online_optimizer_steps_max'],'Onlineupdatebudgetexhausted'
        else:assert audit['completed_estimator_fits']==0 and audit['optimizer_steps']==0,'Fittingduringpredictionisforbidden'
        return result,audit

    def predict(self,folder,source,artifacts,data,rename=False):
        folder.mkdir(parents=True,exist_ok=False);predicted={};audits=[]
        for index,group in enumerate(sorted(set(data['groups'].tolist()))):
            mask=data['groups']==group;obs=observations(subset(data,mask));original=obs['sample_ids'].copy()
            if rename:
                obs['sample_ids']=np.asarray(['newid_'+str(index)+'_'+str(i) for i in range(len(original))])
                obs['groups']=np.repeat('newgroup_'+str(index),len(original))
            path=folder/('input'+str(index)+'.npz');np.savez_compressed(path,**obs)
            out=folder/('prediction'+str(index));_,audit=self.command(source,artifacts,path,out,'predict')
            rows=json.loads((out/'predictions.json').read_text())['predictions']
            mapping={r['sample_id']:r['predicted_label'] for r in rows}
            assert len(rows)==len(mapping)==len(original) and set(mapping)==set(obs['sample_ids'])
            assert all(isinstance(v,int) and v in range(4) for v in mapping.values())
            predicted.update({str(a):int(mapping[str(b)]) for a,b in zip(original,obs['sample_ids'])});audits.append(audit)
        assert set(predicted)==set(data['sample_ids'])
        p=np.asarray([predicted[str(k)] for k in data['sample_ids']],dtype=int)
        np.savez_compressed(folder/'PREDICTIONS.npz',sample_ids=data['sample_ids'],prediction=p)
        return p,audits

    def fit_program(self,program_dir):
        self.deadline=time.monotonic()+self.config['program_seconds']
        source=program_dir/'source';train=load(self.inputs/'train.npz');dev=load(self.inputs/'development.npz')
        full=[];fold_scores=[];fit_audits=[]
        for condition in sorted(set(train['condition_id'].tolist())):
            mask=train['condition_id']==condition;fold=program_dir/('cv_'+str(int(condition)));fold.mkdir()
            f=fold/'train.npz';np.savez_compressed(f,**subset(train,~mask))
            fitout=fold/'fit';_,audit=self.command(source,None,f,fitout,'fit');fit_audits.append(audit)
            assert not set(train['groups'][mask])&set(train['groups'][~mask])
            pred,_=self.predict(fold/'check',source,fitout/'artifacts',subset(train,mask))
            fold_scores.append(dict(condition_hz=float(condition),**score(subset(train,mask),pred)))
        fitout=program_dir/'full_fit';_,audit=self.command(source,None,self.inputs/'train.npz',fitout,'fit');fit_audits.append(audit)
        prediction,_=self.predict(program_dir/'development',source,fitout/'artifacts',dev)
        renamed,_=self.predict(program_dir/'identity_check',source,fitout/'artifacts',dev,rename=True)
        assert np.array_equal(prediction,renamed),'Predictiondependsongroup/sampleidentifiers'
        assert not any(p.is_symlink() for p in program_dir.rglob('*')),'Symlinkartifactsnotallowed'
        artifacts={str(p.relative_to(program_dir)):sha(p) for p in program_dir.rglob('*') if p.is_file() and ('artifacts' in p.parts or 'source' in p.parts)}
        result=dict(status='complete',program_id=program_dir.name,cv=fold_scores,development=score(dev,prediction),
            fit_entry_executions=len(fit_audits),confirmed_estimator_fits=sum(a['completed_estimator_fits'] for a in fit_audits),
            actual_optimizer_steps=sum(a['optimizer_steps'] for a in fit_audits),
            nonzero_sampled_optimizer_steps=sum(a['nonzero_sampled_optimizer_steps'] for a in fit_audits),
            training_only=True,whole_condition_split=True,identity_invariance_passed=True,
            artifacts=artifacts,independent_check_used=False,final_data_read=False,utc=utc())
        save(program_dir/'REPORT.json',result);return result

    def predict_ensemble(self,folder,members,data,incumbent='p00',rename=False):
        """Cold-run complete trained programs,then compose their record decisions.

        This reuses actual fitted artifacts;it is not a new training operation.
        Each record shares the existing prediction time limit across all members.
        Callers must use separate Runtime instances for concurrent ensembles.
        """
        assert incumbent in members and members
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        previous_deadline=getattr(self,'deadline',None)
        overall=previous_deadline if previous_deadline is not None else time.monotonic()+self.config['program_seconds']
        predicted={};audits=[];records=[]
        try:
            for index,group in enumerate(sorted(set(data['groups'].tolist()))):
                mask=data['groups']==group;obs=observations(subset(data,mask));original=obs['sample_ids'].copy()
                assert 'labels' not in obs
                if rename:
                    obs['sample_ids']=np.asarray(['newid_'+str(index)+'_'+str(i) for i in range(len(original))])
                    obs['groups']=np.repeat('newgroup_'+str(index),len(original))
                record=folder/('record_'+str(index));record.mkdir()
                input_file=record/'observations.npz';np.savez_compressed(input_file,**obs)
                self.deadline=min(overall,time.monotonic()+self.config['predict_seconds'])
                started=time.monotonic();votes={};member_rows={}
                for name,member in sorted(members.items()):
                    process,audit=self.command(Path(member['source']),Path(member['artifacts']),input_file,record/name,'predict')
                    rows=json.loads((record/name/'predictions.json').read_text())['predictions']
                    mapping={r['sample_id']:r['predicted_label'] for r in rows}
                    assert len(rows)==len(mapping)==len(original) and set(mapping)==set(obs['sample_ids'])
                    assert all(type(v) is int and v in range(4) for v in mapping.values())
                    values=np.array([mapping[str(s)] for s in obs['sample_ids']],dtype=int)
                    counts=np.bincount(values,minlength=4);votes[name]=int(np.flatnonzero(counts==counts.max())[0])
                    member_rows[name]=dict(zip(original.tolist(),values.tolist()))
                    audits.append(dict(group=str(group),member=name,seconds=process['seconds'],audit=audit))
                counts=np.bincount(list(votes.values()),minlength=4);tied=np.flatnonzero(counts==counts.max()).tolist()
                selected=votes[incumbent] if votes[incumbent] in tied else min(tied)
                predicted.update({str(s):selected for s in original})
                row=dict(group=str(group),members=votes,member_window_predictions=member_rows,predicted=selected,
                         tied_labels=tied,incumbent_tiebreak=len(tied)>1 and votes[incumbent] in tied,
                         seconds=time.monotonic()-started,source_input_sha256=sha(input_file),labels_in_input=False)
                records.append(row);save(record/'ENSEMBLE.json',row)
            assert set(predicted)==set(data['sample_ids'])
            values=np.array([predicted[str(s)] for s in data['sample_ids']],dtype=int)
            np.savez_compressed(folder/'PREDICTIONS.npz',sample_ids=data['sample_ids'],prediction=values)
            save(folder/'ENSEMBLE_REPORT.json',dict(records=records,member_count=len(members),actual_prediction_processes=len(audits),
                aggregation='Original window-majority per member,then equal member vote;ties prefer incumbent if tied,otherwise smallest class.',
                inherited_fitted_models=True,new_fits=0,record_time_limit=self.config['predict_seconds'],
                window_metric_scope='Output is one acquisition decision repeated for its windows;do not compare its window error to original per-window classifiers.',utc=utc()))
            return values,audits
        finally:
            if previous_deadline is None:
                if hasattr(self,'deadline'):del self.deadline
            else:self.deadline=previous_deadline

    def verified_members(self,program_dir):
        """Verify host-sealed artifacts; return base members of a composition.

        Relative member paths survive relocation of the complete campaign.
        This marker is outside actor-written source/ and only the broker writes
        it. Nested compositions are deliberately excluded from this protocol.
        """
        program_dir=Path(program_dir);report=json.loads((program_dir/'REPORT.json').read_text())
        for relative,digest in report['artifacts'].items():
            assert sha(program_dir/relative)==digest,'Registered artifact changed'
        if report.get('program_kind')!='equal_record_vote':return None
        manifest=json.loads((program_dir/'COMPOSITION.json').read_text())
        assert manifest['rule']=='equal_record_vote' and manifest['tie_anchor'] in manifest['members']
        members={}
        for name,binding in manifest['members'].items():
            parent=(program_dir/binding['relative_path']).resolve()
            assert sha(parent/'REPORT.json')==binding['report_sha256'],'Parent report changed'
            parent_report=json.loads((parent/'REPORT.json').read_text())
            assert parent_report.get('program_kind')!='equal_record_vote','Nested composition forbidden'
            assert self.verified_members(parent) is None
            members[name]=dict(source=parent/'source',artifacts=parent/'full_fit/artifacts')
        return manifest,members

    def predict_registered(self,folder,program_dir,data,rename=False):
        """Execute either a registered trained program or its sealed composition."""
        program_dir=Path(program_dir);composition=self.verified_members(program_dir)
        if composition is None:
            return self.predict(folder,program_dir/'source',program_dir/'full_fit/artifacts',data,rename=rename)
        manifest,members=composition
        prediction,wrapped=self.predict_ensemble(folder,members,data,incumbent=manifest['tie_anchor'],rename=rename)
        return prediction,[dict(row['audit'],composition_member=row['member'],acquisition_group=row['group']) for row in wrapped]

    def independent(self,programs,packet):
        self.deadline=time.monotonic()+self.config['program_seconds']
        data=load(self.inputs/'independent_check.npz');reports={}
        for name,folder in programs.items():
            manifest=json.loads((folder/'REPORT.json').read_text())
            for relative,digest in manifest['artifacts'].items():assert sha(folder/relative)==digest
            try:
                p,_=self.predict_registered(packet/name,folder,data)
                reports[name]=dict(status='complete',**score(data,p))
            except Exception as error:reports[name]=dict(status='prediction_error',error_type=type(error).__name__,error=str(error))
        save(packet/'REPORT.json',dict(programs=reports,real_record_labels=True,labels_from_author_fixture=True,
            physical_record_count=len(set(data['groups'])),new_fits=0,final_data_read=False,utc=utc()))
        return reports
