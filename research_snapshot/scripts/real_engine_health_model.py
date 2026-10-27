"""Genuine supervised prediction of publisher EngineMT-QA annotations.

Anonymous-channel numerical inference, not calibrated physical risk or causal
identification. No original-task source record is used for model fitting.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

HEADS=['Fan','LPC','HPC','HPT','LPT']
PARAMS=dict(n_estimators=500,max_features=1.0,min_samples_leaf=1,n_jobs=2,random_state=2026102309)


def fit_heads(data, supervision, component_model, output):
    data,output=Path(data),Path(output);sup=json.loads(Path(supervision).read_text())
    fit=dict(np.load(data/'train.npz',allow_pickle=False));dev=dict(np.load(data/'val.npz',allow_pickle=False))
    assert not set(fit['groups'])&set(dev['groups'])
    models={'components':joblib.load(component_model)};audit={'component_model_reused':str(component_model),'new_head_fits':0}
    for head in ['health_state','timing_causes']:
        targets=[sup[str(s)][head] for s in fit['sample_ids']]
        mask=np.asarray([v is not None for v in targets]);assert mask.any()
        model=ExtraTreesClassifier(**PARAMS)
        model.fit(fit['features'][mask],np.asarray([targets[i] for i in np.flatnonzero(mask)]))
        models[head]=model;audit['new_head_fits']+=1
        valid=[i for i,s in enumerate(dev['sample_ids']) if sup[str(s)][head] is not None]
        pred=model.predict(dev['features'][valid]);actual=[sup[str(dev['sample_ids'][i])][head] for i in valid]
        audit[head]=dict(fitting_rows=int(mask.sum()),development_rows=len(valid),
            development_annotation_accuracy=float(np.mean(pred==actual)))
    models.update(dataset='ENGINE_QA_HEALTH',feature_count=198,head_order=HEADS,
        scope='Publisher annotation agreement on N-CMAPSS-derived sequences; no calibrated physical risk or identified causality.')
    output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(models,output,compress=3)
    return audit


def infer(model_path,input_path,output_path):
    bundle=joblib.load(model_path)
    with np.load(input_path,allow_pickle=False) as data:
        if set(data.files)&{'labels','groups','actual_rul','health_state_labels','cause_labels'}:
            raise ValueError('Observation-only inference required')
        x=data['features'];ids=data['sample_ids'].astype(str)
    if x.shape!=(len(ids),bundle['feature_count']) or not np.isfinite(x).all() or len(set(ids))!=len(ids):
        raise ValueError('Invalid numerical observations/identities')
    bits=bundle['components'].predict(x).astype(int);states=bundle['health_state'].predict(x);causes=bundle['timing_causes'].predict(x)
    rows=[]
    for sid,b,state,cause in zip(ids,bits,states,causes):
        rows.append(dict(sample_id=str(sid),predicted_label=''.join(str(v) for v in b),
            component_health={h:('Fault' if v else 'Healthy') for h,v in zip(HEADS,b)},
            faulty_components=[h for h,v in zip(HEADS,b) if v],overall_health_state=str(state),
            timing_cause_annotation=str(cause).split('|'),physical_causal_conclusion=None))
    result=dict(predictions=rows,evidence_scope=bundle['scope'],physical_risk_certified=False)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
