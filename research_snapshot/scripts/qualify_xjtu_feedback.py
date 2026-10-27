"""Independently replay real A278 training/CV, and test numerical MCP dispatch."""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from real_xjtu_rul_model import infer, matrix, predict_bank
from verified_phmforge import VerifiedForge


def independent_mae(p, y, units):
    groups = {}
    for a,b,u in zip(p,y,units): groups.setdefault(int(u),[]).append(abs(float(a)-float(b)))
    return sum(sum(v)/len(v) for v in groups.values())/len(groups)


def main(root):
    data = root/'public_data/datasets/XJTU'
    fit = dict(np.load(data/'train.npz',allow_pickle=False))
    dev = dict(np.load(data/'val.npz',allow_pickle=False))
    assert set(fit['unit_ids'].tolist()).isdisjoint(dev['unit_ids'].tolist())
    allocation = json.loads((root/'hidden/ALLOCATION278.json').read_text())
    rows = json.loads((root/'references/candidates/manifest.json').read_text())['models']
    checks = []
    for row in rows:
        oof = np.full(len(fit['labels']),np.nan)
        for number,held in enumerate(allocation['folds']):
            mask = np.isin(fit['unit_ids'],held)
            model = joblib.load(root/'references/candidates/cv_models'/(row['name']+'_fold'+str(number)+'.joblib'))
            assert set(model['fitting_units']) == set(fit['unit_ids'][~mask].tolist())
            assert set(model['fitting_units']).isdisjoint(held)
            a = {k:v if k=='dataset' else v[mask] for k,v in fit.items()}
            oof[mask] = predict_bank(model,a)
            estimator = model['model']
            if hasattr(estimator,'named_steps') and 'standardscaler' in estimator.named_steps:
                fitting = {k:v if k=='dataset' else v[~mask] for k,v in fit.items()}
                assert np.allclose(estimator.named_steps['standardscaler'].mean_,matrix(fitting).mean(0),rtol=1e-12,atol=1e-12)
        stored = np.load(root/'references/candidates'/(row['name']+'_OOF.npz'),allow_pickle=False)
        assert np.array_equal(stored['sample_ids'],fit['sample_ids'])
        assert np.allclose(stored['predictions'],oof,rtol=0,atol=1e-9)
        score = independent_mae(oof,fit['labels'],fit['unit_ids'])
        assert math.isclose(score,row['training_cv']['mean_bearing_mae_minutes'],abs_tol=1e-8)
        checks.append(dict(model=row['name'],folds=3,predictions_replayed=True,
            independent_bearing_mae_agrees=True,training_only_preprocessing=True))
    out = root/'qualification'; out.mkdir()
    observations = {k:v for k,v in dev.items() if k!='labels'}
    np.savez_compressed(out/'dev_observations.npz',**observations)
    a = infer(root/'references/mlp/model.pt',out/'dev_observations.npz',out/'trained_predictions.json')
    b = infer(root/'references/mlp/untrained_model.pt',out/'dev_observations.npz',out/'untrained_predictions.json')
    p = np.asarray([r['predicted_rul'] for r in a['predictions']])
    q = np.asarray([r['predicted_rul'] for r in b['predictions']])
    assert np.isfinite(p).all() and np.max(np.abs(p-q))>1e-6 and np.ptp(p)>1e-6
    training = json.loads((root/'references/mlp/training_audit.json').read_text())
    assert training['optimizer_steps']==training['nonzero_gradient_steps']>0
    assert training['changed_parameter_tensors']>0
    tool = VerifiedForge(root/'public_data',out/'mcp')
    blocked = tool.call('load_dataset',dict(dataset_name='XJTU',split='test'))
    gold = tool.call('load_ground_truth',dict(dataset_name='XJTU',file='test'))
    assert blocked['status']==gold['status']=='unavailable' and blocked['data_path'] is None
    loaded = tool.call('load_dataset',dict(dataset_name='XJTU',split='val'))
    observed = np.load(loaded['data_path'],allow_pickle=False)
    assert 'labels' not in observed and len(observed['sample_ids'])==3
    targets = tool.call('load_ground_truth',dict(dataset_name='XJTU',file='val'))
    target_rows = json.loads(Path(targets['artifact']).read_text())
    assert {r['sample_id']:r['actual_rul'] for r in target_rows}==dict(zip(dev['sample_ids'].tolist(),dev['labels'].tolist()))
    trained = tool.call('train_rul_model',dict(dataset='XJTU',model_type='mlp',epochs=1))
    assert trained['training_audit']['epochs_executed']==1
    assert trained['training_audit']['nonzero_gradient_steps']>0
    prediction = tool.call('predict_rul',dict(model_path=trained['model_path'],test_data=loaded['data_path'],unit_id=0))
    assert len(prediction['predictions'])==3
    unit = int(dev['unit_ids'][0])
    single = tool.call('predict_rul',dict(model_path=trained['model_path'],test_data=loaded['data_path'],unit_id=unit))
    assert len(single['predictions'])==1 and single['predictions'][0]['unit_id']==unit
    assert json.loads(Path(single['artifact']).read_text())['predictions']==single['predictions']
    for path in (root/'evaluation_data/queries').glob('*.npz'):
        v=np.load(path,allow_pickle=False)
        assert len(v['sample_ids'])==1 and 'labels' not in v
        assert len(v['history_features'])==int(v['elapsed_minutes'][0])
        assert v['history_elapsed_minutes'][-1]==v['elapsed_minutes'][0]
    report=dict(passed=True,bank_models=6,replayed_CV_fits=18,checks=checks,
        reference_nonzero_updates=training['nonzero_gradient_steps'],
        trained_untrained_predictions_differ=True,observed_input_dependence=True,
        real_MCP_training_prediction_and_unit_filter_passed=True,
        extra_qualification_training_jobs=1,
        qualification_nonzero_updates=trained['training_audit']['nonzero_gradient_steps'],
        test_access_denied=True,all_final_queries_causal_and_single=True,
        final_predictions_created=0,final_scores_read=False,LLM_calls=0,
        protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'))
    save(root/'audit/NUMERICAL_QUALIFICATION278.json',report)
    save(Path(__file__).resolve().parents[1]/'results/feedback278/NUMERICAL_QUALIFICATION278.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
