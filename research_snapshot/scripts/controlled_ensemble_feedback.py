"""Counted probability-ensemble search with privately timed development feedback.

Basis probabilities must come from actual saved fitting/CV. This service never
fits estimators and never accepts code, paths, new models or final observations.
The existing file-queue transport and native coding runner remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from controlled_feedback import client,metrics,save,serve,sha,utc


def save_arrays(path,**arrays):
    stage=path.with_name(path.name+'.new')
    with stage.open('wb') as stream:np.savez_compressed(stream,**arrays)
    stage.replace(path)


class EnsembleBroker:
    def __init__(self,run):
        self.run=Path(run);self.config=json.loads((self.run/'BROKER_CONFIG.json').read_text())
        assert self.config['condition'] in ['immediate','delayed']
        self.stage=self.config.get('stage','search');assert self.stage in ['search','review']
        assert (self.stage=='review')==('prior_pool' in self.config)
        self.budget=int(self.config['candidate_budget']);self.denominator=int(self.config['weight_denominator'])
        assert 1<=self.budget<=1000 and self.denominator>=1
        self.bank=np.load(self.config['bank'],allow_pickle=False)
        allowed={'classes','basis_names','cv','development'}|{p+'_'+k for p in ['cv','development'] for k in ['labels','ids','conditions','groups']}
        assert set(self.bank.files)<=allowed,'Undeclared bank arrays must not enter feedback'
        self.classes=self.bank['classes'].tolist();self.basis=self.bank['basis_names'].tolist()
        assert self.classes and len(set(self.classes))==len(self.classes)
        assert len(set(self.basis))==len(self.basis) and len(self.basis)>=2
        for part in ['cv','development']:
            values=self.bank[part];labels=self.bank[part+'_labels'];ids=self.bank[part+'_ids']
            assert values.shape==(len(labels),len(self.basis),len(self.classes))
            assert len(ids)==len(labels)==len(set(ids.tolist()))
            assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
            assert np.allclose(values.sum(axis=2),1,rtol=0,atol=1e-8)
            assert set(labels.tolist())<=set(self.classes)
            for key in ['conditions','groups']:
                if part+'_'+key in self.bank.files:assert len(self.bank[part+'_'+key])==len(labels)
        assert not set(self.bank['cv_ids'].tolist())&set(self.bank['development_ids'].tolist())
        assert not (self.run/'broker/STATE.json').exists(),'No implicit restart'
        self.state=dict(started_utc=utc(),condition=self.config['condition'],stage=self.stage,trials=[],final_choice=None,prior_choice=None)
        # These immutable single-model fallbacks are genuine actual candidates.
        for i,name in enumerate(self.basis):
            vector=[0]*len(self.basis);vector[i]=self.denominator
            self.state['trials'].append(self.score('base_'+str(i+1),vector,'basis_anchor',name))
        if self.stage=='review':
            prior_path=self.run/self.config['prior_pool']
            assert prior_path.resolve().is_relative_to(self.run.resolve())
            prior=json.loads(prior_path.read_text())
            assert prior['source_phase_terminal'] is True
            assert prior['candidate_pool_complete'] is True and prior['source_kind'] in ['Agent_session','numeric_fixture']
            assert prior['bank_sha256']==sha(self.config['bank'])
            assert prior['classes']==self.classes and prior['basis_names']==self.basis
            assert prior['weight_denominator']==self.denominator
            mapping={};known={tuple(x['weights']):x for x in self.state['trials']}
            assert len({x['id'] for x in prior['trials']})==len(prior['trials'])
            assert len({self.validate(x['weights']) for x in prior['trials']})==len(prior['trials'])
            for old in prior['trials']:
                weights=self.validate(old['weights']);current=known.get(weights)
                if current is None:
                    current=self.score('prior_'+old['id'],old['weights'],'prior_ensemble',old.get('name'))
                    current['original_candidate_id']=old['id'];self.state['trials'].append(current);known[weights]=current
                assert current['cv']==old['cv'] and current['development']==old['development']
                mapping[old['id']]=current['id']
            choice=prior['final_choice']
            # An undelivered initial session remains in the denominator. Review
            # can use its real candidate pool without inventing a prior choice.
            if choice is not None:
                assert choice['bank_sha256']==prior['bank_sha256']
                assert choice['weight_denominator']==self.denominator
                original=next(x for x in prior['trials'] if x['id']==choice['candidate_id'])
                assert original['weights']==choice['weights']
                self.state['prior_choice']=dict(candidate_id=mapping[choice['candidate_id']],original_candidate_id=choice['candidate_id'],
                    weights=choice['weights'],source_pool_sha256=sha(prior_path),source_kind=prior['source_kind'])
            self.state['prior_delivery']=prior.get('delivered')
        save(self.run/'broker/STATE.json',self.state)
        self.publish()

    def validate(self,vector):
        assert isinstance(vector,list) and len(vector)==len(self.basis)
        assert all(type(v) is int and v>=0 for v in vector) and sum(vector)==self.denominator
        return tuple(vector)

    def predictions(self,vector,part):
        probability=np.einsum('nbc,b->nc',self.bank[part],np.asarray(vector,dtype=np.float64))/self.denominator
        assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1),1,rtol=0,atol=1e-8)
        return np.asarray(self.classes)[probability.argmax(axis=1)]

    def score(self,ident,vector,kind,name=None):
        self.validate(vector)
        return dict(id=ident,weights=vector,kind=kind,name=name,committed_utc=utc(),
            cv=metrics(self.bank['cv_labels'],self.predictions(vector,'cv'),self.classes),
            development=metrics(self.bank['development_labels'],self.predictions(vector,'development'),self.classes))

    def count(self):return sum(t['kind']=='ensemble' for t in self.state['trials'])

    def released(self):return self.stage=='review' or self.config['condition']=='immediate' or self.count()==self.budget

    def raw_development_released(self):return self.stage=='review' or self.count()==self.budget

    def public(self,row):
        return {k:row[k] for k in ['id','weights','kind','name','cv']} | dict(
            development=row['development'] if self.released() else 'withheld_until_all_vectors_committed')

    def publish(self):
        folder=self.run/'feedback';folder.mkdir(parents=True,exist_ok=True)
        public_cv=folder/'training_cv_probabilities.npz'
        if not public_cv.exists():
            fields={k:self.bank[k] for k in self.bank.files if k.startswith('cv_')}
            save_arrays(public_cv,probabilities=self.bank['cv'],classes=self.bank['classes'],basis_names=self.bank['basis_names'],**fields)
        # Raw development basis vectors are withheld even in the immediate arm
        # until commitment ends. Earlier per-candidate evidence has labels and
        # hard predictions only, enabling arithmetic/condition checks without
        # exposing unlimited uncounted soft-ensemble evaluation.
        if self.raw_development_released():
            public_dev=folder/'development_probabilities.npz'
            if not public_dev.exists():
                fields={k:self.bank[k] for k in self.bank.files if k.startswith('development_')}
                save_arrays(public_dev,probabilities=self.bank['development'],classes=self.bank['classes'],basis_names=self.bank['basis_names'],**fields)
        value=dict(condition=self.config['condition'],stage=self.stage,basis_names=self.basis,classes=self.classes,
            weight_denominator=self.denominator,candidate_budget=self.budget,ensemble_candidates=self.count(),
            remaining=self.budget-self.count(),development_released=self.released(),
            candidates=[self.public(row) for row in self.state['trials']],prior_choice=self.state['prior_choice'],final_choice=self.state['final_choice'])
        save(folder/'catalog.json',value)
        return dict(condition=self.config['condition'],stage=self.stage,ensemble_candidates=self.count(),remaining=self.budget-self.count(),
            basis_anchors=len(self.basis),development_released=self.released(),catalog='/feedback/catalog.json',
            training_cv_probabilities='/feedback/training_cv_probabilities.npz',
            all_development_probabilities='/feedback/development_probabilities.npz' if self.raw_development_released() else None,
            prior_choice=self.state['prior_choice'],final_choice=self.state['final_choice'])

    def evidence(self,row,part):
        assert part=='cv' or part=='development' and self.released(),'Development evidence withheld'
        pred=self.predictions(row['weights'],part);labels=self.bank[part+'_labels'];ids=self.bank[part+'_ids'];records=[]
        for i in range(len(labels)):
            record=dict(sample_id=str(ids[i]),actual=int(labels[i]),predicted=int(pred[i]))
            for key in ['conditions','groups']:
                if part+'_'+key in self.bank.files:record[key]=str(self.bank[part+'_'+key][i])
            records.append(record)
        output=self.run/'feedback/evidence'/row['id']/(part+'.json')
        save(output,dict(candidate_id=row['id'],weights=row['weights'],classes=self.classes,split=part,
                        metrics=row[part],rows=records))
        return '/feedback/evidence/'+row['id']+'/'+part+'.json'

    def dispatch(self,request):
        action=request.get('action')
        if action=='status':assert set(request)=={'action'}
        elif action=='evaluate':
            assert set(request)=={'action','vectors'} and self.state['final_choice'] is None
            vectors=request['vectors'];assert isinstance(vectors,list) and 1<=len(vectors)<=50
            canonical=[self.validate(v) for v in vectors]
            assert len(set(canonical))==len(canonical),'Repeated vector inside batch'
            old={tuple(t['weights']) for t in self.state['trials']}
            assert not old&set(canonical),'Vector already available; no duplicate evaluations'
            assert self.count()+len(vectors)<=self.budget,'Candidate budget exhausted'
            start=self.count()
            for i,vector in enumerate(vectors):self.state['trials'].append(self.score('ensemble_'+str(start+i+1),vector,'ensemble'))
        elif action=='evidence':
            assert set(request)=={'action','candidate_id','split'} and request['split'] in ['cv','development']
            row=next(t for t in self.state['trials'] if t['id']==request['candidate_id'])
            path=self.evidence(row,request['split'])
            return dict(evidence=path,**self.publish())
        elif action=='select':
            assert set(request)=={'action','candidate_id','reason'}
            assert (self.stage=='review' or self.count()==self.budget) and self.state['final_choice'] is None
            row=next(t for t in self.state['trials'] if t['id']==request['candidate_id'])
            reason=request['reason'];assert isinstance(reason,str) and reason.strip() and len(reason)<=4096
            self.state['final_choice']=dict(candidate_id=row['id'],weights=row['weights'],weight_denominator=self.denominator,
                reason=reason,sealed_utc=utc(),bank_sha256=sha(self.config['bank']),stage=self.stage)
        else:raise ValueError('Allowed actions:status,evaluate,evidence,select')
        save(self.run/'broker/STATE.json',self.state)
        return self.publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['serve','client']);parser.add_argument('--run',type=Path);parser.add_argument('--request')
    a=parser.parse_args()
    if a.action=='serve':serve(a.run,broker_class=EnsembleBroker)
    else:print(json.dumps(client(json.loads(a.request)),indent=2))
