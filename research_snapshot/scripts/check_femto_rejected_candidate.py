"""Replay r1's exact recorded, unselected scenario2 log-GBR candidate.

Same-test exploratory counterfactual; no tuning and no new Agent calls.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def model():
    return GradientBoostingRegressor(n_estimators=200,learning_rate=.04,max_depth=2,
        min_samples_leaf=4,loss='huber',random_state=253)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    train=np.load(a.data/'train.npz',allow_pickle=False);dev=np.load(a.data/'val.npz',allow_pickle=False)
    test=np.load(a.data/'test.npz',allow_pickle=False)
    fit=(train['scenario_ids']==2)&(train['unit_ids']==3);vm=dev['scenario_ids']==2
    selected=model().fit(train['features'][fit],np.log1p(train['labels'][fit]))
    predicted=np.maximum(0,np.expm1(selected.predict(dev['features'][vm])))
    residual=predicted-dev['labels'][vm]
    mae=float(np.abs(residual).mean());rmse=float(np.sqrt(np.mean(residual**2)))
    assert abs(mae-760.6601161287676)<1e-6 and abs(rmse-916.9309602397406)<1e-6
    allfit=train['scenario_ids']==2;tm=test['scenario_ids']==2
    final=model().fit(train['features'][allfit],np.log1p(train['labels'][allfit]))
    joblib.dump(final,a.output/'scenario2_log_gbr.joblib')
    predicted=np.maximum(0,np.expm1(final.predict(test['features'][tm])))
    rows=[{'sample_id':str(i),'unit_id':int(u),'scenario_id':2,'predicted_rul':float(v)}
        for i,u,v in zip(test['sample_ids'][tm],test['unit_ids'][tm],predicted)]
    result={'predictions':rows,'unit':'seconds','fit_units':sorted(set(train['unit_ids'][allfit].tolist())),
        'development_metrics_exactly_replayed':{'mae':mae,'rmse':rmse},
        'maximum_learning_age_seconds':float(train['time_sec'][allfit].max()),
        'query_ages_seconds':{str(u):float(t) for u,t in zip(test['unit_ids'][tm],test['time_sec'][tm])},
        'test_labels_read':False,'model_params':final.get_params(),
        'evidence':'r1 completed command at events.jsonl line88; no parameter change or new selection'}
    (a.output/'SEALED_ALTERNATIVE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'dev_metrics_reproduced':True,'test_predictions_sealed':len(rows),'new_LLM_calls':0}))


if __name__=='__main__':main()
