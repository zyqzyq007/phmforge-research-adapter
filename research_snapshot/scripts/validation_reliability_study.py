"""Strictly nested, acquired-record development study of selection evidence.

Four speed holdouts share records and are one correlated development study.
This script runs actual sklearn fits; it never calls or simulates an LLM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_comfaulda_study import save


RECIPES = ['raw_stats__lda_shrink_auto', 'raw_stats__extra128_leaf2',
           'rms_normalized_stats__lda_shrink_auto', 'rms_normalized_stats__extra128_leaf2']
VIEWS = ['condition_cv', 'record_cv']  # A view tie favors whole-condition validation.
EPS = 1e-12


def feature(block):
    centered = block - block.mean(axis=0)
    std = np.sqrt(np.mean(centered**2, axis=0))
    rms = np.sqrt(np.mean(block**2, axis=0))
    z = centered / np.maximum(std, 1e-12)
    return np.concatenate([block.mean(axis=0), np.log(np.maximum(std, 1e-12)),
        np.log(np.maximum(rms, 1e-12)), np.ptp(block, axis=0),
        np.mean(z**3, axis=0), np.mean(z**4, axis=0),
        np.max(np.abs(block), axis=0)/np.maximum(rms, 1e-12), np.mean(np.abs(block), axis=0)])


def load_features(root, protocol):
    data = []; raw_features = []; normalized_features = []; identities = []
    for row in protocol['selected_records']:
        receipt = json.loads((root/'records'/row['unit_id']/'RECEIPT.json').read_text())
        assert receipt['passed'] and receipt['CSV_CRC_verified']
        path = Path(receipt['raw_path']); raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
        delimiter = ';' if b';' in raw.splitlines()[0] else ','
        a = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        assert a.shape[1] >= 8 and a.shape[0] > protocol['window_points'] and np.isfinite(a).all()
        dt = np.diff(a[:,0]); assert np.all(dt > 0)
        fs = (len(a)-1)/(a[-1,0]-a[0,0])
        expected = a[0,0]+np.arange(len(a))/fs
        assert np.max(np.abs(a[:,0]-expected)) < 2e-8
        n = protocol['window_points']; window_count = len(a)//n
        identities.append(dict(unit_id=row['unit_id'], source_sha256=receipt['sha256'],
                               shape=list(a.shape), fs=fs, windows=window_count,
                               unused_tail_rows=len(a)-window_count*n,
                               selected_columns=protocol['sensor_columns_zero_based']))
        for w in range(window_count):
            block = a[w*n:(w+1)*n, protocol['sensor_columns_zero_based']]
            raw_features.append(feature(block))
            scale = np.sqrt(np.mean(block**2, axis=0))
            normalized_features.append(feature(block/np.maximum(scale,1e-12)))
            data.append((row['unit_id'], row['label'], row['target_condition_Hz'], w))
    X = dict(raw_stats=np.asarray(raw_features), rms_normalized_stats=np.asarray(normalized_features))
    assert all(np.isfinite(x).all() for x in X.values())
    record = np.asarray([d[0] for d in data]); labels = np.asarray([d[1] for d in data])
    condition = np.asarray([d[2] for d in data])
    np.savez_compressed(root/'FEATURES333.npz', **X, record=record, labels=labels, condition=condition)
    save(root/'FEATURE_IDENTITIES333.json', identities)
    return X, record, labels, condition


def best(loss, incumbent=0):
    tied = np.flatnonzero(np.asarray(loss) <= np.min(loss)+EPS).tolist()
    return incumbent if incumbent in tied else tied[0]


class Study:
    def __init__(self, root, protocol):
        self.root, self.protocol = root, protocol
        self.X,self.record,self.y,self.condition = load_features(root,protocol)
        self.records = sorted(set(self.record.tolist()))
        self.record_conditions = {r:int(self.condition[np.flatnonzero(self.record == r)[0]]) for r in self.records}
        self.record_labels = {r:int(self.y[np.flatnonzero(self.record == r)[0]]) for r in self.records}
        self.fits = []; self.models = {}; self.eval_cache = {}; self.forwards = 0
        self.started = time.monotonic()

    def predict(self, training, evaluation, recipe_index):
        train, test = tuple(sorted(training)), tuple(sorted(evaluation))
        assert set(train).isdisjoint(test) and set(train) <= set(self.records) and set(test) <= set(self.records)
        assert set(self.record_labels[r] for r in train) == {0,1,2,3}
        recipe = RECIPES[recipe_index]
        cache_key = (train, recipe_index)
        key = hashlib.sha256(json.dumps([train,recipe]).encode()).hexdigest()[:24]
        feature_name, model_name = recipe.split('__')
        features = self.X[feature_name]
        train_idx = np.flatnonzero(np.isin(self.record, train))
        test_idx = np.flatnonzero(np.isin(self.record, test))
        if cache_key not in self.models:
            assert len(self.fits) < self.protocol['CPU_model_fit_cap']
            assert time.monotonic()-self.started < self.protocol['fit_wall_seconds']
            classifier = (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                if model_name == 'lda_shrink_auto' else
                ExtraTreesClassifier(n_estimators=128,min_samples_leaf=2,random_state=333,n_jobs=1))
            model = make_pipeline(StandardScaler(), classifier)
            start = time.monotonic(); model.fit(features[train_idx],self.y[train_idx])
            path = self.root/'fit_cache'/(key+'.joblib');joblib.dump(model,path,compress=3)
            cold = joblib.load(path)
            live_prediction = model.predict(features[test_idx]); cold_prediction = cold.predict(features[test_idx])
            self.forwards += 2
            assert np.array_equal(live_prediction,cold_prediction)
            # All preprocessing must have been fitted on these training rows.
            assert np.allclose(model[0].mean_,features[train_idx].mean(axis=0),rtol=1e-11,atol=1e-11)
            self.models[cache_key] = cold
            item = dict(id=key,recipe=recipe,training_records=list(train),training_rows=len(train_idx),
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint=str(path.relative_to(self.root)),
                cold_reload_matches=True,seconds=time.monotonic()-start,nonzero_standardizer_variance=int((model[0].var_>0).sum()),
                classifier_classes=model[-1].classes_.tolist(),genuine_fit=True)
            self.fits.append(item);save(self.root/'FITS333.json',self.fits)
        prediction = self.models[cache_key].predict(features[test_idx]);self.forwards += 1
        return test_idx, prediction, key

    def evaluate(self, training, evaluation, recipe_index):
        key = (tuple(sorted(training)),tuple(sorted(evaluation)),recipe_index)
        if key in self.eval_cache:return self.eval_cache[key]
        indices, prediction, model_id = self.predict(*key)
        recs = self.record[indices];record_truth=[];record_prediction=[]
        for r in sorted(evaluation):
            counts = np.bincount(prediction[recs==r],minlength=4)
            record_prediction.append(int(np.flatnonzero(counts==counts.max())[0]))
            record_truth.append(self.record_labels[r])
        value = dict(loss=float(np.mean(np.asarray(record_prediction)!=record_truth)),
            window_error=float(np.mean(prediction!=self.y[indices])),
            macro_F1=float(f1_score(record_truth,record_prediction,labels=[0,1,2,3],average='macro',zero_division=0)),
            model_id=model_id,records=sorted(evaluation),record_truth=record_truth,record_prediction=record_prediction,
            confusion=confusion_matrix(record_truth,record_prediction,labels=[0,1,2,3]).tolist())
        self.eval_cache[key] = value
        return value

    def views(self, available):
        conditions = sorted({self.record_conditions[r] for r in available})
        assert len(conditions)>=2
        folds = {'condition_cv':[], 'record_cv':[]}
        for g in conditions:
            test=[r for r in available if self.record_conditions[r]==g]
            folds['condition_cv'].append((sorted(set(available)-set(test)),test))
        ranks = {c:j for j,c in enumerate(conditions)}
        for fold in range(2):
            test=[r for r in available if (ranks[self.record_conditions[r]]+self.record_labels[r])%2==fold]
            folds['record_cv'].append((sorted(set(available)-set(test)),test))
        return {name:dict(folds=[dict(training=t,evaluation=e) for t,e in pairs],
                    group_losses=[[self.evaluate(t,e,j)['loss'] for t,e in pairs] for j in range(len(RECIPES))])
                for name,pairs in folds.items()}

    def calibration(self, available):
        conditions = sorted({self.record_conditions[r] for r in available}); trials=[]
        for held in conditions:
            test=[r for r in available if self.record_conditions[r]==held]
            training=sorted(set(available)-set(test));v=self.views(training)
            observed=[self.evaluate(training,test,j)['loss'] for j in range(len(RECIPES))]
            policies={name:best(np.mean(value['group_losses'],axis=1)) for name,value in v.items()}
            trials.append(dict(held_condition=held,available_records=training,evaluation_records=test,
                inner_views=v,observed_losses=observed,view_choices=policies,
                view_regrets={name:observed[j]-min(observed) for name,j in policies.items()}))
        mean_regret={name:float(np.mean([t['view_regrets'][name] for t in trials])) for name in VIEWS}
        selected=min(VIEWS,key=lambda name:(mean_regret[name],VIEWS.index(name)))
        return dict(selected_view=selected,mean_calibration_regret=mean_regret,trials=trials)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run.resolve();protocol=json.loads((root/'PROTOCOL333.json').read_text())
    s=Study(root,protocol);cases=[];choices=[];evaluations=[]
    # Every outer decision excludes its held recordings from every inner fit.
    # No outer model predicts or scores the outer recordings until all choices seal.
    for k,condition in enumerate(sorted(set(s.condition.tolist()))):
        test=[r for r in s.records if s.record_conditions[r]==condition]
        training=sorted(set(s.records)-set(test));views=s.views(training);cal=s.calibration(training)
        losses={name:np.mean(v['group_losses'],axis=1).tolist() for name,v in views.items()}
        decision={'keep':0,'record_cv':best(losses['record_cv']),
                  'condition_cv':best(losses['condition_cv']),
                  'worst_condition':best(np.max(views['condition_cv']['group_losses'],axis=1)),
                  'validation_reliability':best(losses[cal['selected_view']])}
        case=dict(id='development_case_'+str(k),candidate_ids=RECIPES,incumbent=RECIPES[0],
            views=views,calibration=cal,allowed_training_records=training,
            scope='Onecorrelated acquired-record development study;not independentconfirmation.',
            hidden_final_data_included=False)
        save(root/'public'/case['id']/'case.json',case)
        cases.append(case);choices.append(dict(case=case['id'],selected={m:RECIPES[j] for m,j in decision.items()}))
        evaluations.append((case['id'],training,test,decision))
    save(root/'ALL_CHOICES_SEALED333.json',dict(choices=choices,outer_predictions_started=False,
         protocol_sha256=hashlib.sha256((root/'PROTOCOL333.json').read_bytes()).hexdigest(),
         new_model_calls=0,provisional_mechanism_not_novelty_claim=True))
    results=[]
    for case,training,test,decision in evaluations:
        scores={name:s.evaluate(training,test,j) for j,name in enumerate(RECIPES)}
        lo=min(x['loss'] for x in scores.values());hi=max(x['loss'] for x in scores.values())
        for method,j in decision.items():
            value=scores[RECIPES[j]];old=scores[RECIPES[0]]['loss']
            results.append(dict(case=case,method=method,selected=RECIPES[j],loss=value['loss'],
                selection_regret=value['loss']-lo,harm=max(0,value['loss']-old),
                gain=max(0,old-value['loss']),opportunity=old-lo,bank_loss_range=hi-lo,
                selected_score=value))
        save(root/'audit'/(case+'_outer_scores.json'),scores)
    summary={}
    for method in ['keep','record_cv','condition_cv','worst_condition','validation_reliability']:
        rows=[r for r in results if r['method']==method]
        opportunity=np.mean([r['opportunity'] for r in rows]);gain=np.mean([r['gain'] for r in rows])
        summary[method]=dict(mean_record_error=float(np.mean([r['loss'] for r in rows])),
            mean_selection_regret=float(np.mean([r['selection_regret'] for r in rows])),
            mean_harm=float(np.mean([r['harm'] for r in rows])),
            useful_improvement_capture=float(gain/opportunity) if opportunity>EPS else None,
            decisions=len(rows),harmful_updates=sum(r['harm']>EPS for r in rows))
    save(root/'RESULTS333.json',dict(summary=summary,rows=results,
        genuine_PHM_fits=len(s.fits),checkpoint_cold_reload_checks=len(s.fits),
        model_predict_calls=s.forwards,new_agent_sessions=0,new_LLM_requests=0,
        formal_confirmation_scenarios=0,independent_development_studies=1,
        related_outer_tasks=4,all_decisions_sealed_before_outer_inference=True,
        novelty_claimed=False,seconds=time.monotonic()-s.started))
    print(json.dumps(summary))


if __name__=='__main__':main()
