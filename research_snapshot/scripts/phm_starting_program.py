"""Starting fitted PHM program;ordinary Agent may replace any part of it."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(data):return np.asarray(data['features'],dtype=float)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','predict'])
    parser.add_argument('--train');parser.add_argument('--input');parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output')
    args=parser.parse_args()
    if args.action=='fit':
        with np.load(args.train,allow_pickle=False) as data:
            model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
            model.fit(features(data),data['labels'])
        args.model_dir.mkdir(parents=True,exist_ok=True);joblib.dump(model,args.model_dir/'model.joblib')
    else:
        model=joblib.load(args.model_dir/'model.joblib')
        with np.load(args.input,allow_pickle=False) as data:
            labels=model.predict(features(data));rows=[dict(sample_id=str(i),predicted_label=int(y)) for i,y in zip(data['sample_ids'],labels)]
        Path(args.output).write_text(json.dumps(dict(predictions=rows))+'\n')


if __name__=='__main__':main()
