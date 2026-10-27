"""Frozen, ordinary input-conditional risk controls on all real native pools.

Training consumes only original training-unit CV losses. Query measurements
and frozen base predictions are restored after all selector fits are sealed.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from screen_score_blinding import restore


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def get(run,key,role,predicate):
    cfg=read(run/'PROTOCOL454.json')['archives'][key];archive=Path(cfg['path'])
    assert sha(archive/'SOURCE_MANIFEST.json')==cfg['manifest_sha256']
    selected=[r for r in read(archive/'SOURCE_MANIFEST.json')['files'] if predicate(r['path'])]
    assert selected,'No archive member matched '+key
    dest=run/role/key;bindings=[];missing=[]
    for item in selected:
        src=archive/item['path'];out=dest/item['path']
        if out.exists():assert sha(out)==item['sha256']
        elif src.is_file():
            assert sha(src)==item['sha256'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
        else:missing.append(item['path']);continue
        bindings.append(dict(archive=str(archive),member=item['path'],sha256=item['sha256'],copy=str(out.relative_to(run))))
    if missing:
        b=restore(archive,dest,lambda p:p in set(missing))
        bindings.extend(dict(x,copy=str((dest/x['member']).relative_to(run))) for x in b)
    path=run/(role+'_BINDINGS454.json');save(path,(read(path) if path.exists() else [])+bindings)
    return dest


def features(data,family):
    if family=='HUST_BATTERY':
        ids=data['groups'].astype(str).tolist();assert len(set(ids))==len(ids)
        x=np.column_stack([data['x'],data['past_cap'],data['rates'],data['age'],data['cycle']])
        return ids,x
    ids=sorted(set(data['groups'].astype(str)));rows=[]
    for group in ids:
        mask=data['groups']==group;x=data['features'][mask]
        rows.append(np.r_[x.mean(0),x.std(0),data['rotation_hz'][mask].mean()])
    return ids,np.asarray(rows)


def prepare(run):
    get(run,'corpus','train',lambda p:p=='analysis/LEGAL_EVIDENCE450.json')
    legal=[r for r in read(run/'train/corpus/analysis/LEGAL_EVIDENCE450.json') if r['source_family'] in ['COMFAULDA','HUST_BATTERY']]
    assert len({r['episode'] for r in legal})==12 and len({r['content_identity'] for r in legal})==45
    get(run,'COMFAULDA','train',lambda p:p=='private_inputs/train.npz')
    get(run,'HUST_BATTERY','train',lambda p:p in ['private_inputs/training_queries_private.npz','JOBS366.json','public_data/metadata.json'])
    datasets={s:load(run/'train'/s/'private_inputs'/('train.npz' if s=='COMFAULDA' else 'training_queries_private.npz')) for s in ['COMFAULDA','HUST_BATTERY']}
    sources={}
    for family,data in datasets.items():
        ids,x=features(data,family);assert np.isfinite(x).all()
        assert len(ids)==(8 if family=='COMFAULDA' else 47)
        sources[family]=dict(unit_ids=ids,x=x.tolist())
    cases=[]
    for episode in sorted({r['episode'] for r in legal}):
        rows=[r for r in legal if r['episode']==episode];family=rows[0]['source_family'];source=sources[family];errors=[]
        for row in rows:
            if family=='COMFAULDA':
                units=[u for fold in row['legal_report']['cv'] for u in fold['records']]
                values={u['group']:float(u['actual']!=u['predicted']) for u in units}
                for u in units:
                    data=datasets[family];truth=set(data['labels'][data['groups']==u['group']].tolist());assert truth=={u['actual']}
            else:
                values={u['cell']:u['MAE'] for f in row['legal_report']['cv'] for u in f['cells']}
            assert set(values)==set(source['unit_ids'])
            errors.append([values[i] for i in source['unit_ids']])
        cases.append(dict(case=episode,source=family,ids=[r['program_id'] for r in rows],errors=np.asarray(errors).T.tolist(),
                          contents=[r['content_identity'] for r in rows],native_completed=rows[0]['native_completed']))
    save(run/'LEGAL_MEMBERS454.json',legal);save(run/'TRAIN_INPUT454.json',dict(sources=sources,cases=cases))
    save(run/'PREPARATION454.json',dict(pools=len(cases),memberships=len(legal),unique_artifacts=45,
        original_normal=sum(c['native_completed'] for c in cases),original_incomplete=sum(not c['native_completed'] for c in cases),
        sources={s:dict(units=len(d['unit_ids']),raw_features=len(d['x'][0])) for s,d in sources.items()},
        query_data_loaded=False,outcomes_loaded=False,train_input_sha256=sha(run/'TRAIN_INPUT454.json')))
    print(json.dumps(read(run/'PREPARATION454.json')))


def fit(run):
    assert not (run/'query').exists() and not (run/'FITS_SEALED454.json').exists()
    data=read(run/'TRAIN_INPUT454.json');start=time.monotonic();fits=[]
    def one_source(item):
        family,source=item;x=np.asarray(source['x']);scaler=StandardScaler().fit(x)
        pca=PCA(n_components=min(8,len(x)-1,x.shape[1]),whiten=True,svd_solver='full').fit(scaler.transform(x))
        z=pca.transform(scaler.transform(x));assert np.isfinite(z).all()
        folder=run/'fitted'/family;folder.mkdir(parents=True)
        joblib.dump(dict(scaler=scaler,pca=pca),folder/'transform.joblib')
        save(folder/'TRANSFORM.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),pca_mean=pca.mean_.tolist(),
            components=pca.components_.tolist(),explained_variance=pca.explained_variance_.tolist(),z=z.tolist(),unit_ids=source['unit_ids']))
        records=[]
        for case in [c for c in data['cases'] if c['source']==family]:
            y=np.asarray(case['errors']);out=run/'fitted'/case['case'];out.mkdir(parents=True)
            models=dict(local_knn=KNeighborsRegressor(n_neighbors=math.ceil(math.sqrt(len(x))),algorithm='brute',weights='uniform').fit(z,y),
                        local_ridge=Ridge(alpha=1.,fit_intercept=True,solver='svd').fit(z,y))
            for method,model in models.items():
                joblib.dump(model,out/(method+'.joblib'));records.append(dict(case=case['case'],method=method,sha256=sha(out/(method+'.joblib'))))
            ridge=models['local_ridge']
            save(out/'FIT.json',dict(y=y.tolist(),knn_k=models['local_knn'].n_neighbors,ridge_coef=ridge.coef_.tolist(),ridge_intercept=ridge.intercept_.tolist()))
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        for records in pool.map(one_source,data['sources'].items()):fits.extend(records)
    assert len(fits)==24
    save(run/'FITS_SEALED454.json',dict(utc=datetime.now(timezone.utc).isoformat(),risk_estimator_fit_calls=24,scaler_fits=2,PCA_fits=2,
        seconds=time.monotonic()-start,query_data_absent=True,outcomes_absent=True,
        train_sha256=sha(run/'TRAIN_INPUT454.json'),files={str(p.relative_to(run)):sha(p) for p in (run/'fitted').rglob('*') if p.is_file()}))
    print(json.dumps(dict(risk_fits=24,seconds=time.monotonic()-start)))


def predict(run):
    seal=read(run/'FITS_SEALED454.json');assert not (run/'ALL_PREDICTIONS_SEALED454.json').exists()
    for p,digest in seal['files'].items():assert sha(run/p)==digest
    cfg=read(run/'PROTOCOL454.json');legal=read(run/'LEGAL_MEMBERS454.json');train=read(run/'TRAIN_INPUT454.json')
    requests=defaultdict(set);locations={}
    for row in legal:
        if row['source_family']=='COMFAULDA':
            name='common_p00' if row['program_id']=='p00' else row['episode'].split('/')[1]+'__'+row['program_id']
            key='COMFAULDA';member='evaluation/'+name+'/cold_repeat/PREDICTIONS.npz'
        elif row['initial']:
            key='HUST442' if row['batch']==442 else 'HUST445';member='qualification/parents/'+row['program_id']+'/PREDICTIONS.npz'
        else:
            loc=row['artifact_locations']['source/program.py'];key=next(k for k,v in cfg['archives'].items() if v['path']==loc['archive'])
            member=loc['member'].rsplit('/source/program.py',1)[0]+'/development/PREDICTIONS.npz'
        requests[key].add(member);locations[row['id']]=dict(key=key,member=member)
    for i in range(8):requests['COMFAULDA'].add(f'evaluation/common_p00/cold_repeat/input{i}.npz')
    requests['HUST_BATTERY'].add('private_inputs/development_observations.npz')
    for key,names in requests.items():
        get(run,key,'query',lambda p,names=names:p in names)
    com=[load(run/f'query/COMFAULDA/evaluation/common_p00/cold_repeat/input{i}.npz') for i in range(8)]
    assert all('labels' not in d for d in com)
    com={k:np.concatenate([d[k] for d in com]) for k in ['features','groups','sample_ids','rotation_hz']}
    hust=load(run/'query/HUST_BATTERY/private_inputs/development_observations.npz');assert 'y' not in hust and 'cap' not in hust
    query_data=dict(COMFAULDA=com,HUST_BATTERY=hust);sources={}
    for family,data in query_data.items():
        ids,x=features(data,family);assert len(ids)==8
        model=joblib.load(run/'fitted'/family/'transform.joblib');z=model['pca'].transform(model['scaler'].transform(x))
        sources[family]=dict(unit_ids=ids,x=x.tolist(),z=z.tolist())
    rows=[];bank=[]
    for case in train['cases']:
        family=case['source'];q=sources[family];base=[]
        for pid in case['ids']:
            loc=locations[case['case']+'/'+pid];d=load(run/'query'/loc['key']/loc['member']);mapping=dict(zip(d['sample_ids'].astype(str),d['prediction']))
            if family=='COMFAULDA':
                data=com;values=[]
                for unit in q['unit_ids']:
                    samples=data['sample_ids'][data['groups']==unit];votes=[int(mapping[s]) for s in samples]
                    counts=np.bincount(votes,minlength=4);values.append(int(np.flatnonzero(counts==max(counts))[0]))
            else:values=[float(mapping[str(s)]) for s in hust['sample_ids']]
            base.append(values)
        base=np.asarray(base).T;assert base.shape==(8,len(case['ids'])) and np.isfinite(base).all()
        scores={m:joblib.load(run/'fitted'/case['case']/(m+'.joblib')).predict(q['z']) for m in ['local_knn','local_ridge']}
        scores['global_cv']=np.tile(np.mean(case['errors'],axis=0),(8,1))
        old=case['ids'].index('p00');selections={}
        for method,s in scores.items():
            selections[method]=[old if s[i,old]<=s[i].min()+1e-12 else int(np.flatnonzero(s[i]<=s[i].min()+1e-12)[0]) for i in range(8)]
        selections['keep']=[old]*8
        for method,ix in selections.items():
            rows.append(dict(case=case['case'],source=family,method=method,selected=[case['ids'][i] for i in ix],
                predictions=base[np.arange(8),ix].tolist(),unit_ids=q['unit_ids']))
        if family=='COMFAULDA':
            ensemble=[]
            for i in range(8):
                counts=np.bincount(base[i].astype(int),minlength=4);tied=np.flatnonzero(counts==max(counts));anchor=int(base[i,old])
                ensemble.append(anchor if anchor in tied else int(tied[0]))
        else:ensemble=base.mean(1).tolist()
        rows.append(dict(case=case['case'],source=family,method='equal_ensemble',selected=None,predictions=ensemble,unit_ids=q['unit_ids']))
        bank.append(dict(case=case['case'],source=family,ids=case['ids'],predictions=base.tolist(),unit_ids=q['unit_ids'],risk_scores={m:s.tolist() for m,s in scores.items()}))
    for p,digest in seal['files'].items():assert sha(run/p)==digest
    save(run/'QUERY_FEATURES454.json',sources);save(run/'BASE_PREDICTIONS454.json',bank);save(run/'PREDICTIONS454.json',rows)
    save(run/'PREDICTION_BINDINGS454.json',locations)
    save(run/'ALL_PREDICTIONS_SEALED454.json',dict(utc=datetime.now(timezone.utc).isoformat(),rows=60,
        prediction_sha256=sha(run/'PREDICTIONS454.json'),base_sha256=sha(run/'BASE_PREDICTIONS454.json'),outcomes_absent=not (run/'outcomes').exists(),
        new_PHM_forward=0,new_selector_prediction_vectors=24))
    print(json.dumps(dict(prediction_rows=len(rows),new_PHM_forward=0)))


def score(run):
    assert sha(run/'PREDICTIONS454.json')==read(run/'ALL_PREDICTIONS_SEALED454.json')['prediction_sha256']
    get(run,'COMFAULDA','outcomes',lambda p:p=='evaluation/common_p00/RESULT.json')
    get(run,'HUST_BATTERY','outcomes',lambda p:p=='private_inputs/discovery_truth.npz')
    get(run,'corpus','outcomes',lambda p:p in ['analysis/HISTORICAL_OUTCOMES450.json','analysis/EPISODES450.json'])
    com=read(run/'outcomes/COMFAULDA/evaluation/common_p00/RESULT.json');h=load(run/'outcomes/HUST_BATTERY/private_inputs/discovery_truth.npz')
    obs=load(run/'query/HUST_BATTERY/private_inputs/development_observations.npz');assert np.array_equal(h['sample_ids'],obs['sample_ids'])
    truths={'COMFAULDA':{x['group']:x['actual'] for x in com['records']},'HUST_BATTERY':dict(zip(obs['groups'].astype(str),h['y']))}
    bank={x['case']:x for x in read(run/'BASE_PREDICTIONS454.json')};gold={r['id']:r['final_loss'] for r in read(run/'outcomes/corpus/analysis/HISTORICAL_OUTCOMES450.json')}
    rows=[]
    for d in read(run/'PREDICTIONS454.json'):
        y=np.asarray([truths[d['source']][i] for i in d['unit_ids']]);base=bank[d['case']];p=np.asarray(base['predictions'])
        errors=(p!=y[:,None]).astype(float) if d['source']=='COMFAULDA' else abs(p-y[:,None])
        for j,pid in enumerate(base['ids']):assert abs(errors[:,j].mean()-gold[d['case']+'/'+pid])<1e-7
        actual=np.asarray(d['predictions']);loss=(actual!=y).astype(float) if d['source']=='COMFAULDA' else abs(actual-y)
        old=errors[:,base['ids'].index('p00')]
        rows.append(dict(**d,loss=float(loss.mean()),harm=float(np.maximum(loss-old,0).mean()),gain=float(np.maximum(old-loss,0).mean()),
            errors=loss.tolist(),best_global_program=float(errors.mean(0).min()),per_query_oracle=float(errors.min(1).mean()),
            regret_to_query_oracle=float((loss-errors.min(1)).mean())))
    summaries={m:{s:{k:float(np.mean([r[k] for r in rows if r['method']==m and r['source']==s])) for k in ['loss','harm','gain','regret_to_query_oracle']} for s in truths} for m in read(run/'PROTOCOL454.json')['methods']}
    qualified={m:all(summaries[m][s]['loss']<summaries[b][s]['loss']-1e-12 for s in truths for b in ['global_cv','equal_ensemble']) and
               all(summaries[m][s]['harm']<=summaries['global_cv'][s]['harm']+1e-12 and summaries[m][s]['gain']>=summaries['global_cv'][s]['gain']-1e-12 for s in truths)
               for m in ['local_knn','local_ridge']}
    result=dict(iteration='A454',rows=rows,summaries=summaries,capability_qualified=qualified,novelty_established=False,
                new_native_Agent_stages=0,new_PHM_fits=0,new_PHM_forward=0,new_LLM_calls=0,meta_risk_fit_calls=24,
                independent_confirmation=0,exploratory=True,scope='2sources/12correlatedpools/45programs;not a native Agent comparison or independent confirmation.')
    save(run/'RESULT454.json',result);print(json.dumps(dict(summaries=summaries,qualified=qualified)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','fit','predict','score'],required=True)
    a=p.parse_args();globals()[a.stage](a.run)
