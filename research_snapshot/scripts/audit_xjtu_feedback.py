"""Reproduce the closed D submission's recorded CV without reading final data."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from acquire_forge_pdmb_sources244 import save, sha256


def metrics(prediction, truth, units):
    rows=[]
    for unit in sorted(set(units.tolist())):
        errors=[float(p)-float(y) for p,y,u in zip(prediction,truth,units) if u==unit]
        rows.append(dict(unit_id=int(unit),mae_minutes=sum(abs(e) for e in errors)/len(errors)))
    return dict(mean_bearing_mae_minutes=sum(x['mae_minutes'] for x in rows)/len(rows),per_bearing=rows)


def matrix(raw):
    x=np.asarray(raw['features'],float).copy()
    x[:,:31]=np.sign(x[:,:31])*np.log1p(np.abs(x[:,:31]))
    return x


def parts(model, raw):
    x=matrix(raw)
    direct=np.maximum(0,model['direct'].predict(x))
    fraction=np.clip(model['fraction'].predict(x),0,1)
    scaled=fraction*np.asarray([model['horizon_by_scenario_minutes'][int(s)] for s in raw['scenario_ids']])
    return direct,scaled,np.minimum(direct,scaled)


def main(root):
    assert not (root/'ALL_CHOICES_SEALED278.json').exists()
    ws=root/'d/workspaces/s278_adaptive_r2'
    status=json.loads((root/'d/primary/s278_adaptive_r2/s278_adaptive_r2/status.json').read_text())
    assert status['status']=='finished' and status['response_valid']
    out=root/'audit/closed_d_replay278';out.mkdir()
    train=dict(np.load(root/'public_data/datasets/XJTU/train.npz',allow_pickle=False))
    dev=dict(np.load(root/'public_data/datasets/XJTU/val.npz',allow_pickle=False))
    folds=json.loads((root/'references/training_cv_folds.json').read_text())['folds']
    x=matrix(train);y=train['labels'];units=train['unit_ids'];scenario=train['scenario_ids']
    life=y+train['elapsed_minutes'];rows=[];fit_count=0
    for name,checkpoint,oof in [
        ('selected50','artifacts/xjtu_et50_lower_direct_fraction.joblib','artifacts/selected_oof.npz'),
        ('alternative100','artifacts/alternatives/et100_leaf20_maxfeat25.joblib','artifacts/alternatives/et100_leaf20_oof.npz'),
        ('previous200','.runtime/rul-experiments/xjtu_et200_lower_direct_fraction.joblib',None)]:
        model=joblib.load(ws/checkpoint)
        assert model['fitting_unit_ids']==sorted(set(units.tolist()))
        expected_horizons={int(s):float(np.median(np.unique(life[scenario==s]))) for s in np.unique(scenario)}
        assert model['horizon_by_scenario_minutes']==expected_horizons
        row=dict(candidate=name,checkpoint=checkpoint,sha256=sha256(ws/checkpoint),
                 parameters=model['parameters'],training_only_horizons=expected_horizons,
                 development_components={k:metrics(p,dev['labels'],dev['unit_ids']) for k,p in zip(['direct','fraction','minimum'],parts(model,dev))})
        if oof:
            expected=dict(np.load(ws/oof,allow_pickle=False));assert np.array_equal(expected['sample_ids'],train['sample_ids'])
            predictions=np.full((len(y),3),np.nan);fold_rows=[]
            for fold in folds:
                fitting=np.isin(units,fold['fit_unit_ids']);held=np.isin(units,fold['heldout_unit_ids'])
                assert not np.any(fitting & held)
                direct=ExtraTreesRegressor(**dict(model['direct'].get_params(),random_state=278+fold['fold']))
                fraction=ExtraTreesRegressor(**dict(model['fraction'].get_params(),random_state=278+fold['fold']))
                direct.fit(x[fitting],y[fitting]);fraction.fit(x[fitting],y[fitting]/life[fitting]);fit_count+=2
                horizons={int(s):float(np.median(np.unique(life[fitting & (scenario==s)]))) for s in np.unique(scenario[fitting])}
                rebuilt=dict(direct=direct,fraction=fraction,horizon_by_scenario_minutes=horizons)
                raw={k:(v if k=='dataset' else v[held]) for k,v in train.items()}
                predictions[held]=np.column_stack(parts(rebuilt,raw))
                target=out/f'{name}_fold{fold["fold"]}.joblib';joblib.dump(rebuilt,target,compress=3)
                fold_rows.append(dict(fold=fold['fold'],fit_unit_ids=fold['fit_unit_ids'],heldout_unit_ids=fold['heldout_unit_ids'],checkpoint_sha256=sha256(target)))
            delta={}
            for column,key in enumerate(['direct_oof_predictions','fraction_oof_predictions','selected_oof_predictions']):
                delta[key]=float(np.max(np.abs(predictions[:,column]-expected[key])))
                assert np.allclose(predictions[:,column],expected[key],rtol=1e-12,atol=1e-9)
            row.update(cv_components={k:metrics(predictions[:,i],y,units) for i,k in enumerate(['direct','fraction','minimum'])},
                       oof_source_sha256=sha256(ws/oof),max_oof_absolute_difference=delta,folds=fold_rows)
        rows.append(row)
        print(json.dumps(dict(candidate=name,cv=row.get('cv_components',{}).get('minimum'),
                              dev=row['development_components']['minimum'])),flush=True)
    audit=dict(passed=True,final_inputs_or_labels_read=False,agent_files_changed=False,
               numerical_replay_fits=fit_count,not_new_agent_experiments=True,candidates=rows,
               feature_transform='signed log1p first 31 columns including elapsed age; matches training and inference',
               scope='Exact closed-run artifact checks and recorded CV reproduction; no newly selected model')
    save(root/'audit/CLOSED_ADAPTIVE_D_AUDIT278.json',audit)
    save(Path(__file__).resolve().parents[1]/'results/feedback278/CLOSED_ADAPTIVE_D_AUDIT278.json',audit)
    shutil.copy2(__file__,root/'code/audit_xjtu_feedback.py')


def audit_b(root):
    import torch
    assert not (root/'ALL_CHOICES_SEALED278.json').exists()
    ws=root/'b/workspaces/s278_adaptive_r1'
    status=json.loads((root/'b/primary/s278_adaptive_r1/s278_adaptive_r1/status.json').read_text())
    assert status['status']=='finished' and status['response_valid']
    out=root/'audit/closed_b_replay278';out.mkdir()
    sys.path.insert(0,str(root/'public_tools'))
    spec=importlib.util.spec_from_file_location('closed_b_training',ws/'train_custom.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    train=dict(np.load(root/'public_data/datasets/XJTU/train.npz',allow_pickle=False))
    dev=dict(np.load(root/'public_data/datasets/XJTU/val.npz',allow_pickle=False))
    config=json.loads((root/'public_data/datasets/XJTU/training_config.json').read_text())
    folds=json.loads((root/'references/training_cv_folds.json').read_text())['folds']
    prior=json.loads((ws/'artifacts/training_audit.json').read_text())
    oof=dict(np.load(ws/'artifacts/cv_oof.npz',allow_pickle=False))
    final=joblib.load(ws/'artifacts/custom_model.joblib')
    x=matrix(train);xd=matrix(dev);y=np.asarray(train['labels'],float)
    units=train['unit_ids'];scenarios=train['scenario_ids'];et=np.full(len(y),np.nan);mlp=et.copy()
    assert np.array_equal(oof['sample_ids'],train['sample_ids']) and np.array_equal(oof['actual_rul'],y)
    assert final['fitting_units']==sorted(set(units.tolist()))
    assert final['et_target_power']==.35 and final['mlp_blend_weight']==.5
    count=0;steps=0;fold_rows=[]
    for fold in folds:
        fit=np.isin(units,fold['fit_unit_ids']);held=np.isin(units,fold['heldout_unit_ids'])
        assert not np.any(fit & held)
        for scenario in [1,2,3]:
            a=fit & (scenarios==scenario);b=held & (scenarios==scenario)
            model=ExtraTreesRegressor(**final['scenario_models'][scenario].get_params()).fit(x[a],y[a]**.35)
            et[b]=np.maximum(0,model.predict(x[b]))**(1/.35);count+=1
            joblib.dump(model,out/f'fold{fold["fold"]}_scenario{scenario}.joblib',compress=3)
        checkpoint,audit=training.train_mlp(x[fit],y[fit],config,config['seed']);count+=1
        saved=next(v['mlp'] for v in prior['whole_bearing_folds'] if v['fold']==fold['fold'])
        assert audit==saved
        assert np.allclose(checkpoint['mean'].numpy(),x[fit].mean(0),rtol=0,atol=1e-12)
        mlp[held]=training.predict_mlp(checkpoint,x[held]);steps+=audit['nonzero_gradient_steps']
        torch.save(checkpoint,out/f'fold{fold["fold"]}_mlp.pt');fold_rows.append(dict(fold=fold['fold'],training_audit_matches=True,**audit))
    diffs={}
    for key,values in [('et_oof',et),('mlp_oof',mlp),('custom_oof',.5*(et+mlp))]:
        diffs[key]=float(np.max(np.abs(values-oof[key])))
        assert np.allclose(values,oof[key],rtol=1e-9,atol=1e-6)
    tree_predictions=np.full(len(dev['labels']),np.nan);full_matches=[]
    for scenario in [1,2,3]:
        a=scenarios==scenario;b=dev['scenario_ids']==scenario
        model=ExtraTreesRegressor(**final['scenario_models'][scenario].get_params()).fit(x[a],y[a]**.35);count+=1
        max_diff=max(float(np.max(np.abs(m.tree_.value-n.tree_.value))) for m,n in zip(model.estimators_,final['scenario_models'][scenario].estimators_))
        assert max_diff==0
        tree_predictions[b]=np.maximum(0,model.predict(xd[b]))**(1/.35)
        full_matches.append(dict(scenario=scenario,max_tree_value_difference=max_diff))
    checkpoint,audit=training.train_mlp(x,y,config,config['seed']);count+=1
    assert audit==prior['final_mlp_fit'];steps+=audit['nonzero_gradient_steps']
    original=torch.load(ws/'artifacts/xjtu_mlp.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,original['state_dict'][k]) for k,v in checkpoint['state_dict'].items())
    torch.save(checkpoint,out/'deployment_mlp_reproduced.pt')
    mlp_prediction=training.predict_mlp(checkpoint,xd)
    devmetrics={k:metrics(p,dev['labels'],dev['unit_ids']) for k,p in [('et',tree_predictions),('mlp',mlp_prediction),('custom_blend',.5*(tree_predictions+mlp_prediction))]}
    for k,m in devmetrics.items():assert abs(m['mean_bearing_mae_minutes']-prior['development_metrics'][k]['mean_bearing_mae_minutes'])<1e-5
    result=dict(passed=True,agent_files_changed=False,final_inputs_or_labels_read=False,
        numerical_replay_fits=count,nonzero_gradient_steps=steps,not_new_agent_experiments=True,
        independent_feature_matrix_matches=True,fold_mlp_audits=fold_rows,full_tree_matches=full_matches,
        max_oof_absolute_difference=diffs,development_components=devmetrics,
        final_mlp_audit=audit,final_mlp_state_identical=True,
        source_checkpoint_sha256=sha256(ws/'artifacts/custom_model.joblib'),source_mlp_sha256=sha256(ws/'artifacts/xjtu_mlp.pt'))
    save(root/'audit/CLOSED_ADAPTIVE_B_AUDIT278.json',result)
    save(Path(__file__).resolve().parents[1]/'results/feedback278/CLOSED_ADAPTIVE_B_AUDIT278.json',result)
    shutil.copy2(__file__,root/'code/audit_xjtu_feedback.py')
    print(json.dumps(dict(passed=True,fits=count,steps=steps,development=devmetrics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cohort',choices=['b','d'],default='d');args=parser.parse_args()
    (audit_b if args.cohort=='b' else main)(args.run.resolve())
