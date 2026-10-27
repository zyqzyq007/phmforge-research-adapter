"""Real PHM program recipes, paired checks and an outcome-blind task broker.

No final input is used by preparation fitting or the Agent broker. The five
programs share genuinely fitted components; checking reloads those parameters.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import secrets
import time

from controlled_feedback import save, sha, utc, client, serve


CANDIDATES = ['P00', 'P10', 'P01', 'P11', 'P11B']
COMPONENTS = ['P00', 'P10', 'P01', 'P11', 'RAW_LR']


def features_module(root):
    spec = importlib.util.spec_from_file_location('rotor_feature_recipe', root/'shared_tools/rotor_feature_recipe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def designs(root, data):
    import numpy as np
    module = features_module(root)
    raw = data['features'].astype(float)
    spectral = module.spectral_features(data['waveforms'])
    return dict(raw=raw, spectral=spectral,
                all=np.concatenate([raw, module.spatial_features(raw), spectral], axis=1))


def estimator(config, component):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if component == 'RAW_LR':
        return make_pipeline(StandardScaler(), LogisticRegression(
            **config['logistic'], random_state=config['numerical_seed']))
    parameter = 'old' if component in ['P00', 'P10'] else 'new'
    return ExtraTreesClassifier(**config['tree_parameters'][parameter], class_weight='balanced',
        random_state=config['numerical_seed'], n_jobs=config['numerical_threads_per_worker'])


def feature_kind(component):
    return 'raw' if component == 'RAW_LR' else ('all' if component in ['P00', 'P01'] else 'spectral')


def metrics(data, prediction):
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    y = np.asarray(data['labels']); pred = np.asarray(prediction)
    assert y.shape == pred.shape and set(pred.tolist()) <= set(range(5))
    records = []
    for group in np.unique(data['groups']):
        mask = data['groups'] == group
        labels = np.unique(y[mask]); loads = np.unique(data['operating_load_Nm'][mask])
        assert len(labels) == len(loads) == 1
        voted = int(np.bincount(pred[mask].astype(int), minlength=5).argmax())
        records.append(dict(group=str(group), actual=int(labels[0]), predicted=voted,
            load=float(loads[0]), windows=int(mask.sum()), correct_windows=int((y[mask] == pred[mask]).sum())))
    correct = sum(r['actual'] == r['predicted'] for r in records)
    return dict(window_accuracy=float(np.mean(y == pred)),
        window_macro_f1=float(f1_score(y, pred, labels=range(5), average='macro', zero_division=0)),
        window_correct=int(np.sum(y == pred)), window_total=len(y),
        window_confusion=confusion_matrix(y, pred, labels=range(5)).tolist(),
        record_correct=correct, record_total=len(records), record_error=1-correct/len(records), records=records,
        by_load={str(load):dict(record_correct=sum(r['actual']==r['predicted'] for r in records if r['load']==load),
            record_total=sum(r['load']==load for r in records)) for load in sorted({r['load'] for r in records})})


def paired(old, new):
    a = {r['group']: r for r in old['records']}; b = {r['group']: r for r in new['records']}
    assert a.keys() == b.keys()
    fixed = [g for g in a if a[g]['actual'] != a[g]['predicted'] and b[g]['actual'] == b[g]['predicted']]
    broken = [g for g in a if a[g]['actual'] == a[g]['predicted'] and b[g]['actual'] != b[g]['predicted']]
    return dict(fixed_records=fixed, broken_records=broken, fixed=len(fixed), broken=len(broken),
                error_change=new['record_error']-old['record_error'])


def prepare_block(root, config, block, source_records):
    import joblib
    import numpy as np
    from prepare_fault_extensions import features
    bank = root/'bank'/block['id']; bank.mkdir(parents=True)
    data_dir = root/'hidden'/block['id']; data_dir.mkdir(parents=True)
    selected = [r for r in source_records if r['acquisition'] in block['acquisitions']]
    assert len(selected) == 80 and not any(r['previously_opened'] for r in selected)
    rows = {p: [] for p in ['train','val','test']}; provenance = []
    for record in selected:
        path = Path(config['source_archive'])/record['export_path']
        assert sha(path) == record['export_sha256']
        with np.load(path, allow_pickle=False) as z:
            signal = z['signals']; label = int(z['broken_bars']); load = float(z['load_nm'])
            assert label == record['broken_bars'] and load == record['load_nm']
            assert int(z['acquisition']) == record['acquisition'] and int(z['sampling_hz']) == 7600
            part, = [p for p, loads in config['load_allocation'].items() if load in loads]
            group = secrets.token_hex(12)
            for start in range(0, len(signal)-2048+1, 2048):
                wave = signal[start:start+2048]
                assert np.isfinite(wave).all()
                rows[part].append(dict(features=features(wave), waveforms=wave.T.copy(), labels=label,
                    groups=group, sample_ids=secrets.token_hex(16), sampling_hz=7600, operating_load_Nm=load))
            provenance.append(dict(group=group, split=part, source_path=record['export_path'],
                source_sha256=record['export_sha256'], acquisition=record['acquisition']))
    rng = np.random.default_rng(config['numerical_seed'])
    for part in rows:
        rng.shuffle(rows[part]); payload = {k:np.asarray([x[k] for x in rows[part]]) for k in rows[part][0]}
        np.savez_compressed(data_dir/(part+'.npz'), **payload)
    save(root/'audit'/(block['id']+'_source_binding.json'), provenance)
    del rows, payload, signal, wave
    with np.load(data_dir/'train.npz', allow_pickle=False) as z: train = dict(z)
    with np.load(data_dir/'val.npz', allow_pickle=False) as z: dev = dict(z)
    assert len(set(train['groups'])) == 50 and len(set(dev['groups'])) == 10
    assert not set(train['groups']) & set(dev['groups'])
    x, dx = designs(root, train), designs(root, dev)
    np.savez_compressed(bank/'training_designs.npz', **x)
    np.savez_compressed(bank/'development_designs.npz', **dx)
    y = train['labels']; folds = []
    for index, load in enumerate(sorted(set(train['operating_load_Nm']))):
        check = np.flatnonzero(train['operating_load_Nm'] == load)
        fit = np.flatnonzero(train['operating_load_Nm'] != load)
        assert not set(train['groups'][fit]) & set(train['groups'][check])
        folds.append(dict(fold=index, load=float(load), fit=fit.tolist(), check=check.tolist()))
    save(bank/'folds.json', folds)
    events = []; probabilities = {}
    for component in COMPONENTS:
        kind = feature_kind(component); folder = bank/component; folder.mkdir()
        cv = np.zeros((len(y), 5)); devprob = None
        for fold in folds + [dict(fold='full', fit=list(range(len(y))), check=[])]:
            fit = np.asarray(fold['fit']); held = np.asarray(fold['check'], dtype=int)
            model = estimator(config, component)
            started = time.monotonic(); role = str(fold['fold'])
            event = dict(component=component, fold=role, fitting_rows=len(fit), started_utc=utc(), status='started')
            with (bank/'fit_events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
            model.fit(x[kind][fit], y[fit])
            assert model.classes_.tolist() == list(range(5))
            if component == 'RAW_LR':
                assert np.allclose(model.named_steps['standardscaler'].mean_,x[kind][fit].mean(0),rtol=1e-10,atol=1e-10)
            target = dx[kind] if role == 'full' else x[kind][held]
            pred = model.predict_proba(target)
            filename = folder/('fold_'+role+'.joblib'); joblib.dump(model, filename, compress=3)
            cold = joblib.load(filename).predict_proba(target)
            assert np.allclose(cold,pred,rtol=0,atol=1e-12)
            if role == 'full': devprob = pred
            else: cv[held] = pred
            event.update(status='completed',elapsed_seconds=time.monotonic()-started,
                checkpoint_sha256=sha(filename), cold_reload_equal=True,
                model_class=type(model).__name__, output_sha256=__import__('hashlib').sha256(pred.tobytes()).hexdigest())
            with (bank/'fit_events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
            events.append(event)
            save(bank/'FIT_PROGRESS.json',dict(completed=len(events),planned=30,last=event,final_scores_opened=False))
        assert np.allclose(cv.sum(1),1) and np.allclose(devprob.sum(1),1)
        probabilities[component] = (cv,devprob)
        np.savez_compressed(folder/'probabilities.npz',cv=cv,development=devprob)
    reports = {}
    for cid in CANDIDATES:
        if cid == 'P11B': cv, dp = [0.9*a+0.1*b for a,b in zip(probabilities['P11'],probabilities['RAW_LR'])]
        else: cv, dp = probabilities[cid]
        reports[cid] = dict(candidate_id=cid, cv=metrics(train,cv.argmax(1)),development=metrics(dev,dp.argmax(1)))
        np.savez_compressed(bank/(cid+'_predictions.npz'),cv=cv.argmax(1),development=dp.argmax(1),
                            cv_ids=train['sample_ids'],development_ids=dev['sample_ids'])
    save(bank/'VALIDATION_REPORTS.json',reports)
    proof = dict(block=block['id'],actual_component_fits=len(events),cold_checkpoint_replays=len(events),
        train_records=50,development_records=10,train_windows=len(y),development_windows=len(dev['labels']),
        data_hashes={p:sha(data_dir/(p+'.npz')) for p in ['train','val','test']},
        source_records=80,source_sha256s=[r['export_sha256'] for r in selected],
        actual_final_model_predictions=0,final_scores_opened=False)
    save(bank/'QUALIFICATION.json',proof)
    print(json.dumps(dict(block=block['id'],fits=len(events),status='real_training_complete')),flush=True)
    return proof


def prepare(root):
    config = json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert not (root/'PREPARATION325.json').exists()
    source = Path(config['source_archive']); manifest = source/'SOURCE_MANIFEST.json'
    assert sha(manifest) == config['source_manifest_sha256']
    filemap = {x['path']:x for x in json.loads(manifest.read_text())['files']}
    index = source/'audit/FULL_NATIVE_RECORD_INDEX304.json'
    assert sha(index) == filemap['audit/FULL_NATIVE_RECORD_INDEX304.json']['sha256']
    assert sha(root/'shared_tools/rotor_feature_recipe.py') == config['feature_recipe_sha256']
    records = json.loads(index.read_text())['records']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare_block,root,config,b,records) for b in config['blocks']]
        results = [f.result() for f in futures]
    assert not set(results[0]['source_sha256s']) & set(results[1]['source_sha256s'])
    proof = dict(passed=True,blocks=results,actual_component_fits=sum(r['actual_component_fits'] for r in results),
        actual_cold_checkpoint_replays=sum(r['cold_checkpoint_replays'] for r in results),
        protocol_sha256=sha(root/'CAMPAIGN_CONFIG.json'), final_scores_opened=False, new_agent_sessions=0)
    save(root/'PREPARATION325.json',proof)
    save(Path(config['project'])/config['result_directory']/'PREPARATION325.json',proof)


def replay_check(root, block, cid, output):
    """Cold replay the complete fixed program, including every fitted fold."""
    import joblib
    import numpy as np
    bank=root/'bank'/block; data_dir=root/'hidden'/block
    with np.load(data_dir/'train.npz',allow_pickle=False) as z:
        train={k:z[k] for k in ['labels','groups','sample_ids','operating_load_Nm']}
    with np.load(data_dir/'val.npz',allow_pickle=False) as z:
        dev={k:z[k] for k in ['labels','groups','sample_ids','operating_load_Nm']}
    with np.load(bank/'training_designs.npz',allow_pickle=False) as z: x=dict(z)
    with np.load(bank/'development_designs.npz',allow_pickle=False) as z: dx=dict(z)
    foldspec=json.loads((bank/'folds.json').read_text()); components=['P11','RAW_LR'] if cid=='P11B' else [cid]
    combined=[]; weights=[.9,.1] if cid=='P11B' else [1.0]; checked=[]
    for component,weight in zip(components,weights):
        kind=feature_kind(component); cv=np.zeros((len(train['labels']),5))
        for fold in foldspec:
            held=np.asarray(fold['check']); path=bank/component/('fold_'+str(fold['fold'])+'.joblib')
            cv[held]=joblib.load(path).predict_proba(x[kind][held]); checked.append(sha(path))
        path=bank/component/'fold_full.joblib'; dp=joblib.load(path).predict_proba(dx[kind]); checked.append(sha(path))
        combined.append((weight*cv,weight*dp))
    cv=sum(p[0] for p in combined).argmax(1); dp=sum(p[1] for p in combined).argmax(1)
    with np.load(bank/(cid+'_predictions.npz'),allow_pickle=False) as z:
        assert np.array_equal(cv,z['cv']) and np.array_equal(dp,z['development'])
    report=dict(candidate_id=cid,cv=metrics(train,cv),development=metrics(dev,dp),
        actual_cold_models_replayed=len(checked),checkpoint_sha256s=checked,actual_new_fits=0,
        final_input_read=False,complete_program_matched=True)
    for part,data,p in [('cv',train,cv),('development',dev,dp)]:
        save(output/(part+'_rows.json'),[dict(sample_id=str(i),group=str(g),load=float(l),actual=int(y),predicted=int(v))
            for i,g,l,y,v in zip(data['sample_ids'],data['groups'],data['operating_load_Nm'],data['labels'],p)])
    reference=json.loads((bank/'VALIDATION_REPORTS.json').read_text())['P00']
    report['versus_P00']={part:paired(reference[part],report[part]) for part in ['cv','development']}
    save(output/'report.json',report)
    return report


def compact(report):
    return {part:{k:v for k,v in report[part].items() if k!='records'} for part in ['cv','development']}


def accepted_choice(proposed, reports, checked):
    if proposed not in CANDIDATES or proposed not in checked:
        return 'P00','invalid_or_unchecked_choice'
    if proposed=='P00':return 'P00','retained_start'
    score=lambda c:(reports[c]['cv']['record_error']+reports[c]['development']['record_error'])/2
    if score(proposed)<score('P00')-1e-12:return proposed,'accepted_validation_improvement'
    return 'P00','no_strict_joint_validation_improvement'


class Broker:
    def __init__(self, run):
        self.run=Path(run);self.config=json.loads((self.run/'BROKER_CONFIG.json').read_text())
        self.root=Path(self.config['campaign']);self.block=self.config['block']
        self.reports=json.loads((self.root/'bank'/self.block/'VALIDATION_REPORTS.json').read_text())
        self.state=dict(checks={},hypotheses=[],plans=[],final_choice=None,actual_new_fits=0)
        assert not (self.run/'broker/STATE.json').exists()
        save(self.run/'broker/STATE.json',self.state)
    def dispatch(self,request):
        assert isinstance(request,dict) and len(json.dumps(request))<=60000
        action=request['action']; phase=json.loads((self.run/'PHASE.json').read_text())['role']
        ordinary=self.config['arm']=='ordinary'
        if action=='status':
            return dict(candidates=CANDIDATES,phase=phase,checked=self.state['checks'],final_choice=self.state['final_choice'],
                remaining_distinct_checks=5-len(self.state['checks']))
        if self.state['final_choice'] is not None:raise ValueError('Choice already sealed')
        if action=='check':
            assert ordinary or phase in ['builder_proposal','builder_checks']
            cid=request['candidate_id'];assert cid in CANDIDATES
            if not ordinary and phase=='builder_proposal':assert cid in ['P00','P11B']
            if not ordinary and phase=='builder_checks':
                planned={c for p in self.state['plans'] for c in p['candidate_ids']}
                assert cid in planned or cid in ['P00','P11B'], 'Check must be registered before execution'
            if cid not in self.state['checks']:
                output=self.run/'feedback'/cid;started=time.monotonic()
                report=replay_check(self.root,self.block,cid,output)
                self.state['checks'][cid]=dict(candidate_id=cid,metrics=compact(report),evidence='/feedback/'+cid+'/report.json',
                    versus_P00=report['versus_P00'],elapsed_seconds=time.monotonic()-started,
                    actual_cold_models_replayed=report['actual_cold_models_replayed'])
            result=self.state['checks'][cid]
        elif action=='hypotheses':
            assert ordinary or phase in ['hypothesis','reviewer_plan','coordinator_plan']
            rows=request['hypotheses'];assert isinstance(rows,list) and 1<=len(rows)<=6
            for row in rows:
                assert all(isinstance(row.get(k),str) and row[k].strip() for k in ['id','explanation','prediction','contradiction'])
            self.state['hypotheses'].append(dict(utc=utc(),rows=rows));result={'registered':len(rows)}
        elif action=='plan':
            assert ordinary or phase in ['experiment','reviewer_plan','coordinator_plan']
            assert ordinary or self.state['hypotheses']
            plan={k:request[k] for k in ['candidate_ids','contrasts','fixed_factors','interpretation']}
            assert isinstance(plan['candidate_ids'],list) and 1<=len(plan['candidate_ids'])<=5
            assert set(plan['candidate_ids'])<=set(CANDIDATES)
            assert all(isinstance(plan[k],str) and plan[k].strip() for k in ['contrasts','fixed_factors','interpretation'])
            plan['utc']=utc();self.state['plans'].append(plan);result={'registered':plan}
        elif action=='select':
            assert ordinary or phase in ['hypothesis_decision','reviewer_decision','coordinator_decision']
            assert isinstance(request.get('reason'),str) and request['reason'].strip()
            cid=request['candidate_id'];selected,reason=accepted_choice(cid,self.reports,self.state['checks'])
            self.state['final_choice']=dict(proposed=cid,selected=selected,gate_reason=reason,
                agent_reason=request['reason'],utc=utc(),phase=phase)
            result=self.state['final_choice']
        else:raise ValueError('Allowed actions:status,check,hypotheses,plan,select')
        save(self.run/'broker/STATE.json',self.state)
        return result


async def mcp_server():
    import asyncio
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server=Server('phm-controlled-checks')
    fields={
      'status':{},
      'check':{'candidate_id':{'type':'string','enum':CANDIDATES}},
      'hypotheses':{'hypotheses':{'type':'array','items':{'type':'object','properties':{k:{'type':'string'} for k in ['id','explanation','prediction','contradiction']},'required':['id','explanation','prediction','contradiction'],'additionalProperties':False}}},
      'plan':{'candidate_ids':{'type':'array','items':{'type':'string','enum':CANDIDATES}},**{k:{'type':'string'} for k in ['contrasts','fixed_factors','interpretation']}},
      'select':{'candidate_id':{'type':'string','enum':CANDIDATES},'reason':{'type':'string'}}}
    descriptions={'status':'List the common program catalog, real checks and remaining budget.',
      'check':'Cold-replay a complete candidate on legal whole-load CV and development records; expose real paired evidence. No new fitting or final data.',
      'hypotheses':'Register competing explanations with predictions and contradiction criteria before further checks.',
      'plan':'Register controlled comparisons, fixed factors and outcome interpretation before execution.',
      'select':'Seal a candidate choice subject to the common validation acceptance rule.'}
    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=k,description=descriptions[k],inputSchema={'type':'object','properties':v,'required':list(v),'additionalProperties':False}) for k,v in fields.items()]
    @server.call_tool()
    async def call_tool(name,arguments):
        assert name in fields
        result=await asyncio.to_thread(client,dict(action=name,**arguments))
        return [types.TextContent(type='text',text=json.dumps(result,ensure_ascii=False))]
    async with stdio_server() as (read,write):await server.run(read,write,server.create_initialization_options())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','serve','client','mcp'])
    p.add_argument('--run',type=Path);p.add_argument('--request');a=p.parse_args()
    if a.action=='prepare':prepare(a.run.resolve())
    elif a.action=='serve':serve(a.run,broker_class=Broker)
    elif a.action=='client':print(json.dumps(client(json.loads(a.request)),ensure_ascii=False))
    else:__import__('asyncio').run(mcp_server())
