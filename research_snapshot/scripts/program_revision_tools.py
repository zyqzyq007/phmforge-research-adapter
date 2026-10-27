"""MCP broker for actual authored-code fitting and one independent check."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import time
import uuid

from controlled_feedback import save,sha,serve,utc,remaining_budget


def client(request,timeout_seconds=690):
    directory=Path('/app/.runtime/feedback_requests');directory.mkdir(parents=True,exist_ok=True)
    ident=uuid.uuid4().hex;save(directory/(ident+'.json'),request)
    response=Path('/feedback/responses')/(ident+'.json');end=time.monotonic()+timeout_seconds
    while time.monotonic()<end:
        if response.exists():return json.loads(response.read_text())
        time.sleep(.05)
    raise TimeoutError('Realprogramserviceexceeded'+str(timeout_seconds)+'seconds')


def public(report):
    return {k:v for k,v in report.items() if k not in ['artifacts']}


class Broker:
    def __init__(self,run):
        from program_revision_runtime import Runtime
        self.root=Path(run);self.config=json.loads((self.root/'BROKER_CONFIG.json').read_text())
        self.engine=Runtime(self.root);self.workspace=self.root/'app'
        self.path=self.root/'broker/STATE.json'
        if self.path.exists():self.state=json.loads(self.path.read_text())
        else:
            baseline=Path(self.config['baseline']);report=json.loads((baseline/'REPORT.json').read_text())
            self.state=dict(programs={'p00':dict(path=str(baseline),report=report)},attempts=0,probe=None,submission=None)
            save(self.path,self.state)

    def persist(self):save(self.path,self.state)

    def budget(self):
        result=remaining_budget(Path(self.config.get('budget_root',self.root)))
        clock=self.root/'EPISODE_BUDGET.json'
        if result is not None and clock.exists():
            result['wall_seconds_remaining']=max(0,json.loads(clock.read_text())['deadline_epoch']-time.time())
        return result

    def dispatch(self,request):
        action=request['action']
        if action=='info':
            result=dict(contract=json.loads(Path(self.config['contract']).read_text()),
                training_versions_remaining=self.config['max_programs']-self.state['attempts'],independent_checks_remaining=int(self.state['probe'] is None),
                programs={i:public(v['report']) for i,v in self.state['programs'].items()},submission=self.state['submission'],episode_budget=self.budget())
            if self.config.get('probe_reusable',False):
                result.update(independent_checks_remaining=None,validation_policy='Same legal development-check acquisitions may be reused on exact registered versions;cached results reused. Adaptive evidence,not independent certification.')
            if self.config.get('enable_submit_review',False):
                result.update(submission_review_required=self.config.get('submission_review_required',False),
                    submission_review_policy='review_update exposes actual complete-program harms AND missed gains before committing. Both arms have this tool. Required-policy submits must acknowledge a current receipt for the exact choice and bank; no performance veto is applied.')
            if self.config.get('enable_composition',False):
                result.update(compositions_remaining=self.config['max_compositions']-self.state.get('composition_attempts',0),
                    composition_policy='Reuse 2-5 distinct actually trained complete programs. Equal acquisition vote; ties prefer your declared member, then smallest class. All members share the same per-acquisition prediction time limit. No new fitting; complete-program CV comes from bound out-of-fold predictions, development is cold-executed twice. Nested compositions unavailable.')
            return result
        assert self.state['submission'] is None,'Submissionalreadysealed'
        if action=='compose_programs':
            assert self.config.get('enable_composition',False),'Composition not enabled'
            return self.compose_programs(request['program_ids'],request['tie_anchor'],request['reason'])
        if action=='review_update':
            assert self.config.get('enable_submit_review',False),'Submitreviewnotenabledinthisprotocol'
            return self.review_update(request['program_id'])
        if action=='paired_evidence':
            assert self.config.get('enable_paired_evidence',False),'Pairedtoolnotenabledinthisprotocol'
            return self.paired_evidence(request['incumbent'],request['challenger'])
        if action=='register_program':
            assert self.state['attempts']<self.config['max_programs'],'Actualtrainingversionbudgetexhausted'
            files=request['files'];assert isinstance(files,list) and 'program.py' in files and 1<=len(files)<=10 and len(files)==len(set(files))
            self.state['attempts']+=1;ident='p'+str(self.state['attempts']).zfill(2)
            destination=self.root/'programs'/ident/'source';destination.mkdir(parents=True,exist_ok=False)
            for relative in files:
                pure=PurePosixPath(relative)
                assert not pure.is_absolute() and '..' not in pure.parts and pure.suffix in ['.py','.json']
                path=self.workspace/relative;assert path.resolve().is_relative_to(self.workspace.resolve()) and not path.is_symlink()
                before=path.stat();assert before.st_size<=4*1024**2
                raw=path.read_bytes();after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            save(destination.parent/'CLAIM.json',dict(claim=request.get('claim','unspecified'),rival=request.get('rival','unspecified'),
                 code_hashes={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},utc=utc()))
            self.persist()
            try:
                report=self.engine.fit_program(destination.parent)
                self.state['programs'][ident]=dict(path=str(destination.parent),report=report);self.persist()
                return dict(public(report),episode_budget=self.budget())
            except Exception as error:
                failure=dict(program_id=ident,status='authored_program_error',error_type=type(error).__name__,error=str(error),attempt_consumed=True)
                save(destination.parent/'ERROR.json',failure);self.persist();return failure
        if action=='probe':
            assert not self.config.get('defer_independent_check',False),'The common controller checks the entire frozen bank after the build stage;no builder-stage label packet is available.'
            previous=self.state['probe']
            assert previous is None or self.config.get('probe_reusable',False),'Independentpacketcanberevealedonlyonce'
            if previous is not None:
                assert previous['status']=='complete','Priorvalidationcheckdidnotcomplete'
                missing={i:Path(p['path']) for i,p in self.state['programs'].items() if i not in previous['program_ids']}
                if not missing:
                    return dict(real_independent_condition=True,results=previous['reports'],cached=True,new_prediction_processes=0,
                        remaining_checks=None,adaptive_validation_reuse=True,final_data_read=False)
                packet=self.root/('independent_packet_'+uuid.uuid4().hex[:12]);packet.mkdir()
                save(packet/'PRECOMMIT.json',dict(utc=utc(),new_program_ids=sorted(missing),
                    code_and_weights={i:self.state['programs'][i]['report']['artifacts'] for i in missing},
                    legal_validation_labels_previously_revealed=True,not_independent_certification=True))
                reports=dict(previous['reports']);reports.update(self.engine.independent(missing,packet))
                self.state['probe']=dict(status='complete',program_ids=sorted(self.state['programs']),reports=reports,
                    adaptive_validation_reuse=True,utc=utc());self.persist()
                return dict(real_independent_condition=True,results=reports,cached=False,new_program_ids=sorted(missing),
                    remaining_checks=None,adaptive_validation_reuse=True,final_data_read=False)
            # The complete currently registered bank is precommitted before labels.
            programs={i:Path(p['path']) for i,p in self.state['programs'].items()}
            packet=self.root/'independent_packet';packet.mkdir()
            receipt=dict(utc=utc(),precommitted_program_ids=sorted(programs),
                code_and_weights={i:self.state['programs'][i]['report']['artifacts'] for i in programs},label_reveal_started=False)
            save(packet/'PRECOMMIT.json',receipt);self.state['probe']=dict(status='started',program_ids=sorted(programs));self.persist()
            reports=self.engine.independent(programs,packet)
            self.state['probe']=dict(status='complete',program_ids=sorted(programs),reports=reports,utc=utc());self.persist()
            return dict(real_independent_condition=True,results=reports,all_codes_frozen_before_label_reveal=True,
                        remaining_checks=None if self.config.get('probe_reusable',False) else 0,final_data_read=False)
        if action=='submit':
            ident=request['program_id'];assert ident in self.state['programs'] and isinstance(request.get('reason'),str) and request['reason'].strip()
            if self.config.get('submission_review_required',False):
                review=self.state.get('submission_review')
                current=(review is not None and request.get('review_id')==review['review_id']
                    and review['program_id']==ident and review['bank_signature']==self.bank_signature())
                if not current:
                    evidence=self.review_update(ident)
                    return dict(status='review_required',submission_sealed=False,review=evidence,
                        next_action='Read the actual harms, benefits and alternative. Revise/register if useful, or submit the same choice with this review_id and a reason addressing the evidence. Retention is reviewed for missed gains too. No extra time or model budget is added.')
            result=dict(program_id=ident,reason=request['reason'],utc=utc(),program_report_sha256=sha(Path(self.state['programs'][ident]['path'])/'REPORT.json'))
            if self.config.get('enable_submit_review',False):
                result['review_id']=request.get('review_id')
            self.state['submission']=result;self.persist();save(self.root/'broker/FINAL_CHOICE.json',result);return result
        raise ValueError('Unknownaction')

    def compose_programs(self,identifiers,tie_anchor,reason):
        """Seal a relocatable composition of real models, never pretend to fit."""
        import numpy as np
        from program_revision_runtime import load,subset,score
        assert isinstance(identifiers,list) and 2<=len(identifiers)<=5 and len(set(identifiers))==len(identifiers)
        assert tie_anchor in identifiers and isinstance(reason,str) and reason.strip()
        assert self.state.get('composition_attempts',0)<self.config['max_compositions'],'Composition budget exhausted'
        parents={i:self.state['programs'][i] for i in identifiers}
        for value in parents.values():
            assert json.loads((Path(value['path'])/'REPORT.json').read_text())==value['report'],'Report changed'
            assert value['report'].get('program_kind')!='equal_record_vote','Nested composition forbidden'
            assert self.engine.verified_members(Path(value['path'])) is None
        self.state['composition_attempts']=self.state.get('composition_attempts',0)+1
        ident='e'+str(self.state['composition_attempts']).zfill(2);folder=self.root/'programs'/ident
        folder.mkdir(parents=True,exist_ok=False);self.persist()
        manifest=dict(rule='equal_record_vote',tie_anchor=tie_anchor,reason=reason,utc=utc(),
            members={i:dict(relative_path=os.path.relpath(v['path'],folder),report_sha256=sha(Path(v['path'])/'REPORT.json')) for i,v in parents.items()})
        save(folder/'COMPOSITION.json',manifest)
        try:
            train=load(self.engine.inputs/'train.npz');dev=load(self.engine.inputs/'development.npz')
            def cached_prediction(data,reports):
                mapped={i:{row['group']:row for row in report['records']} for i,report in reports.items()}
                groups=set(data['groups'].tolist());assert all(set(rows)==groups for rows in mapped.values())
                result=np.empty(len(data['groups']),dtype=int)
                for group in sorted(groups):
                    mask=data['groups']==group;truth=set(data['labels'][mask].tolist());assert len(truth)==1
                    for rows in mapped.values():
                        assert rows[group]['actual']==next(iter(truth)) and rows[group]['windows']==int(mask.sum())
                        assert rows[group]['condition_hz']==float(data['condition_id'][mask][0])
                    votes={i:rows[group]['predicted'] for i,rows in mapped.items()};counts=np.bincount(list(votes.values()),minlength=4)
                    tied=np.flatnonzero(counts==counts.max()).tolist();result[mask]=votes[tie_anchor] if votes[tie_anchor] in tied else min(tied)
                return result
            folds=[]
            for hz in sorted(set(train['condition_id'].tolist())):
                data=subset(train,train['condition_id']==hz)
                reports={i:next(r for r in v['report']['cv'] if r['condition_hz']==hz) for i,v in parents.items()}
                folds.append(dict(condition_hz=float(hz),**score(data,cached_prediction(data,reports))))
            provisional=dict(status='complete',program_id=ident,program_kind='equal_record_vote',artifacts={'COMPOSITION.json':sha(folder/'COMPOSITION.json')})
            save(folder/'REPORT.json',provisional)
            allowance=self.engine.config['program_seconds'];budget=self.budget()
            if budget and 'wall_seconds_remaining' in budget:allowance=min(allowance,budget['wall_seconds_remaining'])
            self.engine.deadline=time.monotonic()+allowance
            prediction,audits=self.engine.predict_registered(folder/'development',folder,dev)
            renamed,more=self.engine.predict_registered(folder/'identity_check',folder,dev,rename=True)
            expected=cached_prediction(dev,{i:v['report']['development'] for i,v in parents.items()})
            assert np.array_equal(prediction,renamed) and np.array_equal(prediction,expected),'Cold composition differs from bound records'
            result=dict(provisional,cv=folds,development=score(dev,prediction),fit_entry_executions=0,
                confirmed_estimator_fits=0,actual_optimizer_steps=0,nonzero_sampled_optimizer_steps=0,
                reused_trained_members=identifiers,tie_anchor=tie_anchor,training_only=True,whole_condition_split=True,
                cv_scope='Exact aggregation of actual bound parent out-of-fold acquisition predictions;no new fit.',
                window_metric_scope='Acquisition decision repeated across windows.',actual_cold_prediction_processes=len(audits)+len(more),
                identity_invariance_passed=True,independent_check_used=False,final_data_read=False,utc=utc())
            save(folder/'REPORT.json',result);self.state['programs'][ident]=dict(path=str(folder),report=result);self.persist()
            return dict(public(result),episode_budget=self.budget())
        except Exception as error:
            failure=dict(program_id=ident,status='composition_error',error_type=type(error).__name__,error=str(error),attempt_consumed=True)
            save(folder/'ERROR.json',failure);self.persist();return failure

    def bank_signature(self):
        payload=dict(program_reports={i:sha(Path(v['path'])/'REPORT.json') for i,v in self.state['programs'].items()},
                     probe=self.state.get('probe'),attempts=self.state['attempts'])
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def review_update(self,selected):
        """A real-evidence challenge at commit time, without an oracle decision."""
        assert selected in self.state['programs'] and self.config.get('enable_paired_evidence',False)
        ranking=[]
        packet=self.state.get('probe') or {}
        checks=packet.get('reports',{}) if packet.get('status')=='complete' else {}
        # Choose the alternative using the same CV/development views for ALL
        # programs. Unequal optional-check coverage must not change its rank.
        for ident,value in self.state['programs'].items():
            report=value['report'];cv=[float(v['record_error']) for v in report['cv']]
            dev=float(report['development']['record_error'])
            ranking.append(dict(program_id=ident,mean_CV_error=sum(cv)/len(cv),development_error=dev,
                                common_view_mean=(sum(cv)/len(cv)+dev)/2,
                                extra_check_available=ident in checks and checks[ident]['status']=='complete'))
        alternatives=[r for r in ranking if r['program_id']!=selected]
        alternative=min(alternatives,key=lambda r:(r['common_view_mean'],r['program_id']!='p00',r['program_id']))['program_id'] if alternatives else None
        pairs=[]
        for old,new,purpose in [('p00',selected,'selected_change_vs_start'),(selected,alternative,'opportunity_if_an_alternative_is_chosen')]:
            if new is None or (old==new and pairs):continue
            actual=self.paired_evidence(old,new)
            compact=[]
            for view in actual['views']:
                records=[{k:r[k] for k in ['group','actual','old_prediction','new_prediction','condition_hz','transition']} for r in view['records']]
                n=len(records)
                compact.append(dict(view=view['view'],records=n,repairs=view['repairs'],harms=view['harms'],
                    old_error=sum(r['actual']!=r['old_prediction'] for r in records)/n,
                    new_error=sum(r['actual']!=r['new_prediction'] for r in records)/n,
                    changed_records=[r for r in records if r['old_prediction']!=r['new_prediction']]))
            pairs.append(dict(purpose=purpose,incumbent=old,challenger=new,views=compact,
                              source_and_report_binding=actual['source_and_report_binding']))
        review=dict(review_id=uuid.uuid4().hex,utc=utc(),program_id=selected,bank_signature=self.bank_signature(),
                    ranking_for_counterevidence_only=ranking,alternative=alternative,comparisons=pairs,
                    scope='Recorded outcomes of exact COMPLETE registered programs. CV checkpoints differ from full checkpoints. Reused development checks are adaptive. These data do not certify unseen conditions; no automatic keep/accept rule or hidden result is used.',
                    requires_decision_not_automatic_veto=True,new_fits=0,new_predictions=0,final_data_read=False,episode_budget=self.budget())
        self.state['submission_review']=review;self.persist()
        save(self.root/'broker/submission_reviews'/(review['review_id']+'.json'),review)
        return review

    def paired_evidence(self,incumbent,challenger):
        """Join genuine complete-program records;never synthesize a test label."""
        import numpy as np
        assert incumbent in self.state['programs'] and challenger in self.state['programs']
        programs={k:self.state['programs'][k] for k in [incumbent,challenger]}
        for value in programs.values():
            path=Path(value['path']);assert json.loads((path/'REPORT.json').read_text())==value['report'],'Reportchangedafterregistration'
            self.engine.verified_members(path)
        left=programs[incumbent]['report'];right=programs[challenger]['report']
        summary={}
        for role in ['train','development','independent_check']:
            with np.load(self.engine.inputs/(role+'.npz'),allow_pickle=False) as data:
                for group in np.unique(data['groups']):
                    mask=data['groups']==group
                    summary[str(group)]=dict(rotation_hz=float(np.median(data['rotation_hz'][mask])),
                        log_RMS_six_documented_channels=np.median(data['features'][mask,12:18],axis=0).tolist())
        views=[('development/full_checkpoint',left['development'],right['development'],'full_fit/artifacts/')]
        assert len(left['cv'])==len(right['cv'])
        for a,b in zip(left['cv'],right['cv']):
            assert a['condition_hz']==b['condition_hz']
            hz=int(a['condition_hz']);views.append(('CV/fold_checkpoint/'+str(hz),a,b,'cv_'+str(hz)+'/fit/artifacts/'))
        packet=self.state.get('probe')
        if packet and packet['status']=='complete' and all(k in packet['reports'] and packet['reports'][k]['status']=='complete' for k in programs):
            views.append(('reusable_check/full_checkpoint',packet['reports'][incumbent],packet['reports'][challenger],'full_fit/artifacts/'))
        rows=[]
        for view,a,b,prefix in views:
            aa={r['group']:r for r in a['records']};bb={r['group']:r for r in b['records']};assert set(aa)==set(bb)
            paired=[]
            for group,x in aa.items():
                y=bb[group];assert x['actual']==y['actual'] and x['condition_hz']==y['condition_hz']
                before=x['actual']==x['predicted'];after=y['actual']==y['predicted']
                paired.append(dict(group=group,actual=x['actual'],old_prediction=x['predicted'],new_prediction=y['predicted'],
                    condition_hz=x['condition_hz'],transition='repair' if after and not before else 'harm' if before and not after else 'both_correct' if after else 'both_wrong',
                    observable_summary=summary[group]))
            rows.append(dict(view=view,records=paired,repairs=sum(r['transition']=='repair' for r in paired),harms=sum(r['transition']=='harm' for r in paired),
                checkpoint_hashes={k:{f:d for f,d in v['report']['artifacts'].items() if f.startswith(prefix)} for k,v in programs.items()}))
        result=dict(incumbent=incumbent,challenger=challenger,source_and_report_binding={k:dict(report_sha256=sha(Path(v['path'])/'REPORT.json'),
            source_sha256={f:d for f,d in v['report']['artifacts'].items() if f.startswith('source/') or f=='COMPOSITION.json'}) for k,v in programs.items()},views=rows,
            scope='Actual labels and full-program predictions on legal acquisitions;fold checkpoints differ from full checkpoint. Physical summaries are observations,not causal proof or deployment truth.',
            reusable_check_available=any(x[0].startswith('reusable_check') for x in views),new_fits=0,new_predictions=0,final_data_read=False,episode_budget=self.budget())
        ident=uuid.uuid4().hex;save(self.root/'broker/pairs'/(ident+'.json'),result)
        self.state.setdefault('paired_evidence_calls',[]).append(dict(id=ident,incumbent=incumbent,challenger=challenger,utc=utc()));self.persist()
        return result


async def mcp():
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server=Server('phm-code-revision')
    fields={'info':{},'register_program':{'files':{'type':'array','items':{'type':'string'}},'claim':{'type':'string'},'rival':{'type':'string'}},
            'probe':{},'submit':{'program_id':{'type':'string'},'reason':{'type':'string'}}}
    descriptions={'info':'Read actual baseline/CV/development evidence,interface,budgets and registered programs.',
        'register_program':'Freeze your declared program.py and helper source files,then genuinely fit2whole-condition CVmodels and1fullmodel;independently cold-evaluate development and ID invariance.Consumesoneversion.',
        'probe':'Precommit all current fitted complete programs,then evaluate them on one independent real acquired condition with genuine labels.Onlyonce;notfinaltest.',
        'submit':'Seal one actually fitted registered complete program;no final score feedback.'}
    if os.environ.get('PHM_PAIRED_EVIDENCE')=='1':
        fields['paired_evidence']={'incumbent':{'type':'string'},'challenger':{'type':'string'}}
        descriptions['paired_evidence']='Join exact registered programs actual record repairs/harms,source/checkpoint bindings and observable physical summaries. No fitting or invented outcomes.'
        descriptions['probe']='Actually evaluate all not-yet-checked registered versions on the same legal development-check acquisitions;reuse cached exact-version results. Adaptive validation,not independent certification;never final test.'
    if os.environ.get('PHM_SUBMIT_REVIEW')=='1':
        fields['review_update']={'program_id':{'type':'string'}}
        fields['submit']['review_id']={'type':'string'}
        descriptions['review_update']='Review exact complete-program validation harms, benefits and best common-view alternative before a decision;retaining p00 is also checked for missed gains.No fits,synthetic outcomes or final labels.'
        descriptions['submit']='Request sealing of a registered program. If info says review is mandatory, first receive review_required,then acknowledge the current review_id after considering both harm and missed gains.No performance veto;stale-bank receipts cannot seal.'
    if os.environ.get('PHM_COMPOSITION')=='1':
        fields['compose_programs']={'program_ids':{'type':'array','items':{'type':'string'}},'tie_anchor':{'type':'string'},'reason':{'type':'string'}}
        descriptions['compose_programs']='Seal an equal acquisition vote of 2-5 distinct registered trained complete programs. Ties prefer tie_anchor if tied,otherwise smallest class. Reuses real weights,0newfits;independent cold execution verifies development and ID invariance. All members share prediction time limit. No nesting. Counts against a separate composition budget;submit returned eXX like a single program.'
    @server.list_tools()
    async def list_tools():return [types.Tool(name=n,description=descriptions[n],inputSchema=dict(type='object',properties=v,required=[k for k in v if k!='review_id'],additionalProperties=False)) for n,v in fields.items()]
    @server.call_tool()
    async def call_tool(name,arguments):
        assert name in fields
        result=await asyncio.to_thread(client,dict(action=name,**arguments))
        return [types.TextContent(type='text',text=json.dumps(result,ensure_ascii=False))]
    async with stdio_server() as (read,write):await server.run(read,write,server.create_initialization_options())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['mcp','serve','client']);p.add_argument('--run',type=Path);p.add_argument('--request');a=p.parse_args()
    if a.action=='mcp':asyncio.run(mcp())
    elif a.action=='serve':serve(a.run,broker_class=Broker)
    else:print(json.dumps(client(json.loads(a.request)),ensure_ascii=False))
