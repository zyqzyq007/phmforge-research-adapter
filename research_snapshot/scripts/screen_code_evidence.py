"""Source-disjoint code/evidence ranker probe on real archived PHM programs.

Code facts are extracted before historical outcomes are restored. Meta-model
workers receive other-source labels and held-source features only.
"""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from isolated_model_gateway import gateway
from mlestar_phm_adapter import post_socket
from screen_score_blinding import restore, output_text, json_output
from screen_cross_source_ranking import ranks, normalized, choice


FACTS = {
 'query_span_aggregation':'Uses a statistic over an entire supplied sequence/record in features, including values for earlier rows. This says dependence, not future leakage.',
 'scale_preserving_inputs':'At least one predictive channel preserves the original between-record physical amplitude/level. Global training-set standardization preserves that information; separate per-record normalization removes it.',
 'within_query_normalization':'Divides or centers measurements by the same record/sequence own magnitude, mean, initial measurement or standard deviation before prediction.',
 'derivative_or_spectrum':'Explicit finite differences, slopes, derivatives, FFT or spectral features contribute to predictions.',
 'multiple_top_level_predictors':'Uses predictions of two or more explicitly fitted top-level predictors. Internal trees of one forest do not alone count.',
 'learned_output_combination':'Learns combination weights, stacking or gating from labeled examples; simple averaging or voting alone does not count.',
 'handcoded_prediction_override':'A manually specified condition replaces/overrides a learned prediction. Unconditional voting, generic output clipping and nonnegativity alone do not count.',
 'finite_upper_output_cap':'Applies an explicit finite upper bound/cap to continuous predicted values. Nonnegative lower clipping and implicit class range do not count.',
 'history_extrapolation':'Extrapolates a physical degradation curve/trend from observed history toward a future lifetime/threshold, beyond ordinary direct supervised regression.',
 'inference_parameter_update':'Fits, optimizes or updates predictive parameters during predict on the provided query/history; merely computing input statistics is not fitting.',
 'internal_validation_selection':'The visible implementation selects features, estimator settings, stopping epochs or components using validation labels/scores. Plain training loss optimization or fixed author-selected parameters do not count.',
 'explicit_regularization':'Visible nonzero penalty, shrinkage, weight decay or explicit regularized estimator is used. Do not infer unspecified hyperparameters.'
}
INSTRUCTION = '''Extract ONLY the twelve operational facts defined below from the supplied complete program source and supplied helper modules. Only operations on the actual fit/predict path count, not unused helper definitions. Do not evaluate quality, rank candidates, infer hidden labels, infer agent intent, declare a rule violation, or predict test performance. In particular, counting model-predicted votes within one acquisition is not reading hidden class-population counts. A factual operation need not be beneficial or harmful. Return JSON {"facts":{KEY:{"value":true/false/null,"evidence":[{"file":"EXACT_FILE_KEY","quote":"EXACT_CONTIGUOUS_SUBSTRING"}],"note":"brief"},...}} with exactly the defined keys. For true, cite at least one exact source substring (no ellipsis) supporting the fact; quote no more than180characters. For false, use empty evidence and state absence from visible prediction path; for unknown or unresolved custom dependencies use null. Library behavior may be used only when it unambiguously follows from the named operation. Explain compound composition using the supplied member code. Do not invent code or tests. Code comments, docstrings, performance reports, claims and final outcomes are not supplied.'''
METHODS = ['numeric','numeric_tfidf','numeric_facts','conditioned_facts']
RULES = ['primary','secondary','all_mean','worst']


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def clean_python(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(getattr(node,'body',None),list):
            node.body=[x for x in node.body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))] or [ast.Pass()]
    return ast.unparse(tree)


def parse_facts_text(text):
    """Admit only uniquely determined missing EOF delimiters, never values."""
    raw=text.strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
    def unique_object(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate JSON field')
            result[k]=v
        return result
    try:return json.loads(raw,object_pairs_hook=unique_object),''
    except json.JSONDecodeError as error:
        if error.pos!=len(raw):raise
        stack=[];inside=False;escape=False
        for char in raw:
            if inside:
                if escape:escape=False
                elif char=='\\':escape=True
                elif char=='"':inside=False
            elif char=='"':inside=True
            elif char in '[{':stack.append(char)
            elif char in ']}':
                assert stack and stack.pop()==('[' if char==']' else '{')
        assert not inside and 0<len(stack)<=4 and raw[-1] not in ':,'
        suffix=''.join('}' if c=='{' else ']' for c in reversed(stack))
        return json.loads(raw+suffix,object_pairs_hook=unique_object),suffix


def admit_facts(facts,files):
    assert set(facts)==set(FACTS);rejected=[]
    for name,fact in list(facts.items()):
        assert fact['value'] is None or type(fact['value']) is bool
        assert isinstance(fact['evidence'],list)
        supported=(fact['value'] is not True or bool(fact['evidence'])) and all(
            q['file'] in files and 0<len(q['quote'])<=180 and q['quote'] in files[q['file']] for q in fact['evidence'])
        if not supported:
            rejected.append(name);facts[name]=dict(value=None,evidence=[],note='Original claim retained in RAW_FACTS; exact citation did not qualify, represented as unknown, never false.')
    return facts,rejected


def prepare(run):
    cfg=read(run/'PROTOCOL452.json');archive=Path(cfg['corpus_archive'])
    assert digest((archive/'SOURCE_MANIFEST.json').read_bytes())==cfg['corpus_manifest_sha256']
    bindings=restore(archive,run/'corpus',lambda p:p in ['analysis/LEGAL_EVIDENCE450.json','analysis/EPISODES450.json'] or p.startswith('inputs/') and ('/source/' in p or p.endswith('/COMPOSITION.json')))
    bindings+=restore(Path(cfg['task_archive']),run/'tasks',lambda p:p.startswith('original_tasks/') and p.endswith('.json'))
    for helper in cfg['helpers']:
        bindings+=restore(Path(helper['archive']),run/'helpers'/str(helper['batch']),lambda p:p==helper['member'])
    legal=read(run/'corpus/analysis/LEGAL_EVIDENCE450.json');episodes=read(run/'corpus/analysis/EPISODES450.json')
    tasks={family:read(run/'tasks/original_tasks'/str(batch)/f'PROMPTS{batch}.json')['ordinary'] for family,batch in [('COMFAULDA',427),('CMAPSS_FD001',435),('HUST_BATTERY',442)]}
    lookup={r['id']:r for r in legal};payloads={};mapping={};rendered={}
    def code_for(row,seen=None):
        seen=set() if seen is None else seen
        assert row['id'] not in seen;seen=seen|{row['id']};files={}
        for relative in row['code_files']:
            p=run/'corpus'/relative
            name=relative.split('/source/',1)[1] if '/source/' in relative else p.name
            assert name not in files
            if name=='COMPOSITION.json':
                c=read(p);files[name]=json.dumps(dict(rule=c['rule'],tie_anchor=c['tie_anchor'],members=list(c['members'])))
                for member in c['members']:
                    dep=lookup[row['episode']+'/'+member]
                    for fn,body in code_for(dep,seen).items():files['member_'+member+'/'+fn]=body
            else:
                files[name]=clean_python(p.read_text()) if p.suffix=='.py' else p.read_text()
        for helper in cfg['helpers']:
            if helper['batch']==row['batch'] and any(helper['module'] in text for text in files.values()):
                p=run/'helpers'/str(helper['batch'])/helper['member'];files['shared/'+p.name]=clean_python(p.read_text())
        return files
    for row in legal:
        files=code_for(row);task=tasks[row['source_family']]
        # Task semantics, not scores; source/program identifiers absent from key metadata.
        payload=dict(task=task,definitions=FACTS,entrypoint='program.py' if 'program.py' in files else 'COMPOSITION.json',
                     source_scope='Only the root entrypoint and its actual imports/calls execute. Nested historical draft files are supplied under distinct relative paths; do not confuse them with the root entrypoint.',source_files=files)
        key=digest(json.dumps(payload,sort_keys=True).encode())[:24]
        payloads[key]=payload;mapping[row['id']]=key;rendered[row['id']]='\n'.join(files.values())
    for key,payload in payloads.items():save(run/'visible'/(key+'.json'),payload)
    save(run/'FACT_DEFINITIONS452.json',FACTS);save(run/'SOURCE_TO_FACT452.json',mapping)
    save(run/'RENDERED_SOURCE452.json',rendered)
    jobs=[dict(id=key,bytes=(run/'visible'/(key+'.json')).stat().st_size) for key in payloads]
    jobs.sort(key=lambda x:(-x['bytes'],x['id']))
    save(run/'JOBS452.json',jobs)
    for row in legal:validation_maps(row)
    save(run/'PREPARATION452.json',dict(bindings=bindings,episodes=16,unique_artifacts=len({r['content_identity'] for r in legal}),memberships=len(legal),unique_extraction_jobs=len(jobs),no_historical_outcomes_restored=True,source_render_sha={k:digest(v.encode()) for k,v in rendered.items()}))
    assert len(episodes)==16 and len(legal)==134 and len(jobs)<=59
    print(json.dumps(dict(ready=True,jobs=len(jobs),largest_bytes=jobs[0]['bytes'])))


def extract(run):
    cfg=read(run/'PROTOCOL452.json');jobs=read(run/'JOBS452.json')
    assert not (run/'private_outcomes').exists() and not (run/'FACTS_SEALED452.json').exists()
    save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=j['id'],socket=str(run/'ipc'/(j['id']+'.sock'))) for j in jobs],
        base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=1,max_input_tokens=cfg['input_tokens_per_extraction'],
        max_output_tokens=cfg['output_tokens_per_extraction'],max_output_per_request=cfg['output_tokens_per_extraction'],max_inflight=1,buffer_verified_responses=True))
    started=time.monotonic();thread=threading.Thread(target=gateway,args=(run,));thread.start()
    while not (run/'MODEL_GATEWAY_READY.json').exists():
        assert thread.is_alive() and time.monotonic()-started<30;time.sleep(.1)
    def one(job):
        folder=run/'extractions'/job['id'];folder.mkdir(parents=True,exist_ok=False);payload=read(run/'visible'/(job['id']+'.json'))
        try:
            assert time.monotonic()-started<cfg['extraction_wall_seconds']
            cache=run/'reused_extractions'/job['id']
            if cache.exists():
                bound=read(run/'REUSE_BINDINGS452.json')[job['id']]
                assert digest((cache/'RESPONSE.json').read_bytes())==bound['response_sha256']
                request=read(cache/'REQUEST.json')
                assert request['input']==json.dumps(payload) and request['instructions']==INSTRUCTION
                response=read(cache/'RESPONSE.json')
            else:
                response=post_socket(run/'ipc'/(job['id']+'.sock'),dict(model='deepseek-flash',instructions=INSTRUCTION,input=json.dumps(payload),tools=[],stream=False,max_output_tokens=cfg['output_tokens_per_extraction']))
            save(folder/'RESPONSE.json',response);value,suffix=parse_facts_text(output_text(response));save(folder/'RAW_FACTS.json',value)
            facts,rejected=admit_facts(value['facts'],payload['source_files'])
            terminal=dict(id=job['id'],status='complete',facts=facts,unsupported_citations_as_unknown=rejected,EOF_delimiter_suffix=suffix,reused_prior_response=cache.exists())
        except Exception as error:
            terminal=dict(id=job['id'],status='incomplete',error_type=type(error).__name__,error=str(error)[:700])
        save(folder/'TERMINAL.json',terminal);print(json.dumps(dict(id=job['id'],status=terminal['status'])),flush=True);return terminal
    results=[]
    try:
        # Largest three sources qualify structured extraction, without reading outcomes.
        with ThreadPoolExecutor(max_workers=3) as pool:results.extend(pool.map(one,jobs[:3]))
        if all(r['status']=='complete' for r in results):
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                futures=[pool.submit(one,j) for j in jobs[3:]]
                for f in as_completed(futures):results.append(f.result())
        else:
            for j in jobs[3:]:
                t=dict(id=j['id'],status='not_started_qualification_failed');save(run/'extractions'/j['id']/'TERMINAL.json',t);results.append(t)
    finally:
        (run/'MODEL_GATEWAY_STOP').touch();thread.join(timeout=640);assert not thread.is_alive()
    save(run/'FACTS_SEALED452.json',dict(utc=datetime.now(timezone.utc).isoformat(),jobs=results,complete=all(r['status']=='complete' for r in results),
         elapsed_seconds=time.monotonic()-started,no_final_outcomes_read=True,hashes={r['id']:digest((run/'extractions'/r['id']/'TERMINAL.json').read_bytes()) for r in results}))


def validation_maps(row):
    report=row['legal_report'];family=row['source_family'];v={};t={}
    if family=='HUST_BATTERY':
        v={str(c['cell']):c['MAE'] for fold in report['cv'] for c in fold['cells']}
    elif family=='CMAPSS_FD001':
        v={str(c['engine']):c['MAE'] for c in report['development']['engine_errors']}
        if row['additional_legal_check']:
            t={str(c['engine']):c['MAE'] for c in row['additional_legal_check']['engine_errors']}
    else:
        v={str(c['condition_hz']):c['record_error'] for c in report['cv']}
        groups=defaultdict(list)
        for c in report['development']['records']:
            groups[str(c['condition_hz'])].append(float(c['actual']!=c['predicted']))
        t={k:sum(a)/len(a) for k,a in groups.items()}
        check=row['additional_legal_check']
        if check:t['additional_check']=check['record_error']
    assert v and all(np.isfinite(x) and x>=0 for x in [*v.values(),*t.values()])
    return v,t


def make_features(run):
    legal=read(run/'corpus/analysis/LEGAL_EVIDENCE450.json');episodes=read(run/'corpus/analysis/EPISODES450.json');mapping=read(run/'SOURCE_TO_FACT452.json');text=read(run/'RENDERED_SOURCE452.json')
    seal=read(run/'FACTS_SEALED452.json');assert seal['complete']
    facts={r['id']:r['facts'] for r in seal['jobs']};cases=[]
    for ep in episodes:
        rows=[r for r in legal if r['episode']==ep['id']];maps=[validation_maps(r) for r in rows]
        vk=sorted(set.intersection(*(set(v) for v,t in maps)));tk=sorted(set.intersection(*(set(t) for v,t in maps)))
        assert vk
        a=np.array([[v[k] for k in vk] for v,t in maps]);b=np.array([[t[k] for k in tk] for v,t in maps]) if tk else a.copy()
        both=np.column_stack([a,b]);ar=ranks(a.mean(1));br=ranks(b.mean(1));gap=(b.mean()-a.mean())/(b.mean()+a.mean()+1e-12)
        n=np.column_stack([ar,br,ranks(a.max(1)),ranks(b.max(1)),ranks(both.std(1)),-(a<=a.min(0)+1e-12).mean(1),-(b<=b.min(0)+1e-12).mean(1),(ar-br)*gap])
        n-=n.mean(0)
        z=np.array([[v for name in FACTS for v in (float(facts[mapping[r['id']]][name]['value'] is True),float(facts[mapping[r['id']]][name]['value'] is not None))] for r in rows])
        z=z/np.sqrt(z.shape[1]);z-=z.mean(0)
        conditioned=np.column_stack([z,z*(ar-br)[:,None],z*n[:,4,None]])
        conditioned-=conditioned.mean(0)
        cases.append(dict(id=ep['id'],source=ep['source_family'],ids=[r['program_id'] for r in rows],contents=[r['content_identity'] for r in rows],
             native_completed=ep['completed'],native_arm=ep['arm'],historical_selected=ep['selected'] if ep['completed'] else None,
             numeric=n.tolist(),facts=z.tolist(),conditioned=conditioned.tolist(),texts=[text[r['id']] for r in rows],
             controls=dict(primary=a.mean(1).tolist(),secondary=b.mean(1).tolist(),all_mean=both.mean(1).tolist(),worst=both.max(1).tolist()),
             primary_group_ids=vk,secondary_group_ids=tk,secondary_fallback_to_primary=not bool(tk)))
    save(run/'LEGAL_FEATURES452.json',cases);return cases


def fit_fold(path):
    fold=read(path);train=fold['train'];test=fold['test'];sources={c['source'] for c in train}
    assert len(sources)==2 and fold['held_source'] not in sources
    vocab=TfidfVectorizer(max_features=512,token_pattern=r'(?u)\b[A-Za-z_0-9]+\b',lowercase=False,sublinear_tf=True)
    # Vocabulary/idf fit on deduplicated other-source code only.
    documents={h:t for c in train for h,t in zip(c['contents'],c['texts'])}
    vocab.fit(list(documents.values()))
    def x(c,method):
        n=np.asarray(c['numeric'])
        if method=='numeric':return n
        if method=='numeric_tfidf':z=vocab.transform(c['texts']).toarray();z-=z.mean(0)
        else:z=np.asarray(c['facts' if method=='numeric_facts' else 'conditioned'])
        return np.column_stack([n,z])
    counts=Counter((c['source'],h) for c in train for h in c['contents'])
    unique={s:len({h for c in train if c['source']==s for h in c['contents']}) for s in sources}
    w=np.array([1/(len(sources)*unique[c['source']]*counts[(c['source'],h)]) for c in train for h in c['contents']]);assert np.isclose(w.sum(),1)
    y=np.concatenate([np.asarray(c['target']) for c in train]);outputs={}
    for method in METHODS:
        matrix=np.vstack([x(c,method) for c in train]);prior=np.zeros(matrix.shape[1]);prior[:2]=.5
        lhs=matrix.T@(w[:,None]*matrix)+fold['lambda']*np.eye(matrix.shape[1]);rhs=matrix.T@(w*y)+fold['lambda']*prior
        beta=np.linalg.solve(lhs,rhs)
        outputs[method]=dict(beta=beta.tolist(),prior=prior.tolist(),train_matrix=matrix.tolist(),y=y.tolist(),weights=w.tolist(),
            training_sources=sorted(sources),scores={c['id']:(x(c,method)@beta).tolist() for c in test},test_matrices={c['id']:x(c,method).tolist() for c in test},
            residual=float(np.max(np.abs(lhs@beta-rhs))))
    return dict(held_source=fold['held_source'],models=outputs,actual_meta_fits=4,tfidf_vocabulary={k:int(v) for k,v in vocab.vocabulary_.items()},tfidf_idf=vocab.idf_.tolist())


def train(run):
    assert read(run/'SEMANTIC_REVIEW452.json')['review_complete'], 'Pre-label source-fact review is required'
    cfg=read(run/'PROTOCOL452.json');cases=make_features(run)
    save(run/'FEATURES_FROZEN452.json',dict(sha256=digest((run/'LEGAL_FEATURES452.json').read_bytes()),historical_outcomes_absent=not (run/'private_outcomes').exists()))
    restore(Path(cfg['corpus_archive']),run/'private_outcomes',lambda p:p=='analysis/HISTORICAL_OUTCOMES450.json')
    historical=read(run/'private_outcomes/analysis/HISTORICAL_OUTCOMES450.json');gold={r['id']:r['final_loss'] for r in historical};fold_paths=[]
    for held in sorted({c['source'] for c in cases}):
        training=[];test=[];history=[]
        for case in cases:
            c={k:v for k,v in case.items() if k not in ['historical_selected','native_arm','native_completed']}
            if case['source']==held:test.append(c);continue
            target=normalized([gold[case['id']+'/'+pid] for pid in case['ids']]);c['target']=(target-target.mean()).tolist();training.append(c)
            old=case['ids'].index('p00');history.append(dict(source=case['source'],**{rule:float(target[choice(case['controls'][rule],old)]) for rule in RULES}))
        means={rule:float(np.mean([np.mean([r[rule] for r in history if r['source']==s]) for s in sorted({r['source'] for r in history})])) for rule in RULES}
        selected_rule=min(RULES,key=lambda name:(means[name],RULES.index(name)))
        p=run/'fold_inputs'/(held+'.json');save(p,dict(held_source=held,train=training,test=test,lambda_=cfg['ridge_lambda'],**{'lambda':cfg['ridge_lambda']},history_rule=selected_rule,history_training_scores=means));fold_paths.append(p)
    fit_and_seal(run,cases)


def fit_and_seal(run,cases):
    assert not (run/'ALL_CHOICES_SEALED452.json').exists()
    assert read(run/'FEATURES_FROZEN452.json')['sha256']==digest((run/'LEGAL_FEATURES452.json').read_bytes())
    fold_paths=sorted((run/'fold_inputs').glob('*.json'));assert len(fold_paths)==3
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:outputs=list(pool.map(fit_fold,fold_paths))
    outputs={v['held_source']:v for v in outputs};decisions=[]
    for held,value in outputs.items():save(run/'fold_outputs'/(held+'.json'),value)
    for c in cases:
        o=outputs[c['source']];fold=read(run/'fold_inputs'/(c['source']+'.json'));scores=dict(c['controls']);scores['history_rule']=scores[fold['history_rule']]
        scores.update({m:o['models'][m]['scores'][c['id']] for m in METHODS});old=c['ids'].index('p00')
        for method,values in scores.items():decisions.append(dict(case=c['id'],source=c['source'],method=method,selected=c['ids'][choice(values,old)]))
        decisions.append(dict(case=c['id'],source=c['source'],method='keep',selected='p00'))
    save(run/'DECISIONS452.json',decisions)
    save(run/'ALL_CHOICES_SEALED452.json',dict(utc=datetime.now(timezone.utc).isoformat(),held_source_outcomes_absent_from_each_fit_input=True,actual_meta_fits=12,
       fit_seconds=time.monotonic()-started,sha256=digest((run/'DECISIONS452.json').read_bytes()),feature_sha256=digest((run/'LEGAL_FEATURES452.json').read_bytes())))
    print(json.dumps(dict(meta_fits=12,choices=len(decisions),seconds=time.monotonic()-started)))


def score(run):
    seal=read(run/'ALL_CHOICES_SEALED452.json');assert seal['sha256']==digest((run/'DECISIONS452.json').read_bytes())
    cases=read(run/'LEGAL_FEATURES452.json');gold={r['id']:r['final_loss'] for r in read(run/'private_outcomes/analysis/HISTORICAL_OUTCOMES450.json')};lookup={c['id']:c for c in cases};rows=[]
    for d in read(run/'DECISIONS452.json'):
        c=lookup[d['case']];losses=[gold[c['id']+'/'+pid] for pid in c['ids']];y=normalized(losses);new=c['ids'].index(d['selected']);old=c['ids'].index('p00')
        rows.append(dict(**d,loss=losses[new],anchor=losses[old],regret=float(y[new]),harm=float(max(0,y[new]-y[old])),gain=float(max(0,y[old]-y[new]))))
    summaries={}
    for method in sorted({r['method'] for r in rows}):
        by={s:{k:float(np.mean([r[k] for r in rows if r['source']==s and r['method']==method])) for k in ['loss','regret','harm','gain']} for s in sorted({c['source'] for c in cases})}
        summaries[method]=dict(by_source=by,**{k:float(np.mean([v[k] for v in by.values()])) for k in ['regret','harm','gain']})
    main=summaries['conditioned_facts'];comparisons={}
    for name,baseline in summaries.items():
        if name=='conditioned_facts':continue
        deltas=[main['by_source'][s]['regret']-baseline['by_source'][s]['regret'] for s in main['by_source']]
        comparisons[name]=dict(wins=sum(x < -1e-12 for x in deltas),ties=sum(abs(x)<=1e-12 for x in deltas),losses=sum(x>1e-12 for x in deltas))
    gate=(all(main['regret']<v['regret']-1e-12 for k,v in summaries.items() if k not in ['conditioned_facts','keep']) and
          all(main['harm']<=summaries[k]['harm']+1e-12 and main['gain']>=summaries[k]['gain']-1e-12 for k in ['numeric','numeric_tfidf','history_rule']) and
          all(comparisons[k]['wins']>=2 for k in ['numeric','numeric_tfidf','history_rule']))
    recovery=read(run/'SERIALIZATION_RECOVERY452.json') if (run/'SERIALIZATION_RECOVERY452.json').exists() else {}
    result=dict(iteration=read(run/'PROTOCOL452.json')['iteration'],rows=rows,summaries=summaries,comparisons=comparisons,primary_gate_passed=gate,
                distinct_meta_rankers=12,actual_meta_fits=12+recovery.get('completed_but_unpersisted_meta_fits',0),
                new_PHM_fits=0,new_native_Agent_stages=0,independent_confirmation=0,exploratory=True,
                inference_scope='Source-disjoint capability probe; old native Agents did not have other-source outcome histories, so not an equal-information Agent superiority claim.')
    save(run/'RESULT452.json',result);print(json.dumps(dict(gate=gate,summaries=summaries,comparisons=comparisons)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','extract','train','score'],required=True);a=ap.parse_args();run=a.run.resolve();assert Path('/tmp') not in run.parents
    dict(prepare=prepare,extract=extract,train=train,score=score)[a.stage](run)
