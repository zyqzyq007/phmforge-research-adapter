"""Inference-only ablation of B's already fitted XJTU blend components."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import solution


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=['tree','mlp','mlp035','blend'],required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    raw=dict(np.load(args.input,allow_pickle=False));assert 'labels' not in raw
    x=solution.feature_matrix(raw)
    if args.component=='tree':
        checkpoint=joblib.load(solution.MODEL_PATH);values=np.full(len(x),np.nan)
        for scenario,model in checkpoint['scenario_models'].items():
            mask=raw['scenario_ids']==scenario
            if mask.any():values[mask]=np.maximum(0,model.predict(x[mask]))**(1/checkpoint['et_target_power'])
    elif args.component=='blend':values=solution.predict(raw)
    else:
        path=solution.MLP_PATH if args.component=='mlp' else Path(__file__).resolve().parent/'artifacts/mlp035.pt'
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        values=solution.predict_mlp(checkpoint,x)
    assert np.isfinite(values).all() and (values>=0).all()
    result=dict(unit='minutes',predictions=[dict(sample_id=str(s),unit_id=int(u),scenario_id=int(c),predicted_rul=float(y))
        for s,u,c,y in zip(raw['sample_ids'],raw['unit_ids'],raw['scenario_ids'],values)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
