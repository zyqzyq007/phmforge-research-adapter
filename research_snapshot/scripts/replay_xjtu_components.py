"""Inference-only replay of an already fitted XJTU direct/fraction checkpoint."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoint.joblib'))
    p.add_argument('--component',choices=['direct','fraction','minimum'],required=True)
    a=p.parse_args();raw=dict(np.load(a.input,allow_pickle=False))
    assert 'labels' not in raw
    model=joblib.load(a.checkpoint)
    assert model['feature_transform']=='signed_log1p_first_31_columns'
    x=np.asarray(raw['features'],float).copy();assert x.shape[1]==34
    x[:,:31]=np.sign(x[:,:31])*np.log1p(np.abs(x[:,:31]))
    direct=np.maximum(0,model['direct'].predict(x))
    fraction=np.clip(model['fraction'].predict(x),0,1)
    fraction*=np.asarray([model['horizon_by_scenario_minutes'][int(s)] for s in raw['scenario_ids']])
    selected={'direct':direct,'fraction':fraction,'minimum':np.minimum(direct,fraction)}[a.component]
    assert np.isfinite(selected).all() and (selected>=0).all()
    result=dict(unit='minutes',predictions=[dict(sample_id=str(s),unit_id=int(u),scenario_id=int(c),predicted_rul=float(y))
        for s,u,c,y in zip(raw['sample_ids'],raw['unit_ids'],raw['scenario_ids'],selected)])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
