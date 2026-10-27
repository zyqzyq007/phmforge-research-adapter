"""Optional common scaffold:measurable feature/model edits of a full predictor.

Both experimental arms may use or replace this file. recipe.json chooses axes
0/1. Axis0 is exactly raw supplied statistics plus StandardScaler/LDA. New
features and estimators are genuine Agent-written implementations,not a bank.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def recipe():
    value=json.loads(Path(__file__).with_name('recipe.json').read_text())
    assert set(value)=={'feature_mode','model_mode'}
    assert all(type(x) is int and x in [0,1] for x in value.values())
    return value


def features(data,mode):
    if mode==0:return np.asarray(data['features'],dtype=float)
    from feature_impl import new_features
    x=np.asarray(new_features(data),dtype=float)
    assert x.ndim==2 and len(x)==len(data['sample_ids']) and np.isfinite(x).all()
    return x


def estimator(mode):
    if mode==0:return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
    from model_impl import new_estimator
    return new_estimator()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','predict'])
    parser.add_argument('--train');parser.add_argument('--input');parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output')
    args=parser.parse_args();r=recipe()
    if args.action=='fit':
        with np.load(args.train,allow_pickle=False) as data:
            model=estimator(r['model_mode']);model.fit(features(data,r['feature_mode']),data['labels'])
        args.model_dir.mkdir(parents=True,exist_ok=True);joblib.dump(dict(recipe=r,model=model),args.model_dir/'model.joblib')
    else:
        checkpoint=joblib.load(args.model_dir/'model.joblib');assert checkpoint['recipe']==r
        with np.load(args.input,allow_pickle=False) as data:
            labels=checkpoint['model'].predict(features(data,r['feature_mode']))
            rows=[dict(sample_id=str(i),predicted_label=int(y)) for i,y in zip(data['sample_ids'],labels)]
        Path(args.output).write_text(json.dumps(dict(predictions=rows))+'\n')


if __name__=='__main__':main()
