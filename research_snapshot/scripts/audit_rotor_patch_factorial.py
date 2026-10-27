"""Independent saved-tree traversal and arithmetic audit; no refitting."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np

from snu_order_control import read, save, sha, utc, packet


def forest_probability(model, values):
    values = values.astype(np.float32)
    votes = np.zeros((len(values), len(model.classes_)))
    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes = np.zeros(len(values), dtype=np.intp)
        while True:
            active = np.flatnonzero(tree.children_left[nodes] != -1)
            if not len(active):
                break
            current = nodes[active]
            left = values[active, tree.feature[current]] <= tree.threshold[current]
            nodes[active] = np.where(left, tree.children_left[current], tree.children_right[current])
        counts = tree.value[nodes, 0, :]
        votes += counts / counts.sum(axis=1, keepdims=True)
    return votes / len(model.estimators_)


def summary(y, probabilities, groups):
    pred = probabilities.argmax(axis=1)
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    per_record = [np.count_nonzero(pred[groups==g] != y[groups==g]) / np.count_nonzero(groups==g)
                  for g in sorted(set(groups))]
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2*np.diag(confusion), denominator, out=np.zeros(5), where=denominator>0)
    return dict(accuracy=float(np.trace(confusion)/confusion.sum()), macro_f1=float(f1.mean()),
                record_mean_error=float(np.mean(per_record)), record_errors=per_record)


def main(root):
    started = time.monotonic()
    cfg, choice, result = [read(root/name) for name in ['PROTOCOL424.json','CHOICES424.json','RESULT424.json']]
    checks = rows_checked = trees_checked = 0
    def check(condition):
        nonlocal checks
        assert condition, f'check {checks+1}'
        checks += 1
    def equal(a,b):
        check(bool(np.allclose(a,b,atol=1e-12,rtol=0)))
    check(sha(root/'code/rotor_patch_factorial.py') == cfg['code_sha256'])
    for name,digest in cfg['support_code_sha256'].items():check(sha(root/'code'/name)==digest)
    check(sha(Path(cfg['source'])/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256'])
    check(choice['utc']<read(root/'EXTERNAL_PREDICTIONS_SEALED424.json')['utc']<result['utc'])
    train,val = packet(root/'legal/train.npz'),packet(root/'legal/val.npz')
    external,truth = packet(root/'external/features.npz'),packet(root/'external/truth.npz')
    check(np.array_equal(external['sample_ids'],truth['sample_ids']))
    check(not set(train['groups'])&set(val['groups']) and not set(train['groups'])&set(truth['groups']))
    observed, finals = {}, {}
    for job,info in cfg['jobs'].items():
        oof=np.zeros((len(train['y']),5))
        for fold,load in enumerate([*cfg['loads'],None]):
            path=root/'models'/f'{job}_{fold}.joblib'
            check(sha(path)==choice['model_hashes'][path.name])
            model=joblib.load(path)
            report=read(root/'models'/f'{job}_{fold}_REPORT.json')
            check(model.get_params()==cfg['tree_recipes'][info['recipe']] or all(
                model.get_params()[k]==v for k,v in cfg['tree_recipes'][info['recipe']].items()))
            check(len(model.estimators_)==report['actual_trees'])
            check(sum(t.tree_.node_count for t in model.estimators_)==report['actual_nodes'])
            trees_checked+=len(model.estimators_)
            if load is not None:
                mask=train['loads']==load
                value=forest_probability(model,train[info['features']][mask])
                oof[mask]=value;rows_checked+=len(value)
                check(report['train_rows']==int((~mask).sum()))
            else:
                value=forest_probability(model,val[info['features']]);rows_checked+=len(value)
                equal(value,np.load(root/'predictions'/(job+'_val.npy')))
                dev=summary(val['y'],value,val['groups'])
                value=forest_probability(model,external[info['features']]);rows_checked+=len(value)
                equal(value,np.load(root/'external'/(job+'_probability.npy')))
                finals[job]=value
        equal(oof,np.load(root/'predictions'/(job+'_oof.npy')))
        observed[job]={'oof':summary(train['y'],oof,train['groups']),'development':dev}
    bank_info=cfg['files']['old_probability_bank'];bank_path=Path(cfg['source'])/bank_info['path']
    check(sha(bank_path)==bank_info['sha256']);bank=packet(bank_path)
    logistic_index=bank['names'].tolist().index('logreg_raw')
    observed['review_blend']={}
    for split,field,target,name in [('oof','oof',train,'oof'),('val','dev',val,'development')]:
        value=.9*np.load(root/'predictions'/('spectral_new_'+split+'.npy'))+.1*bank[field][:,logistic_index,:]
        equal(value,np.load(root/'predictions'/('review_blend_'+split+'.npy')))
        observed['review_blend'][name]=summary(target['y'],value,target['groups'])
    model_info=cfg['files']['logistic_model'];p=Path(cfg['source'])/model_info['path']
    check(sha(p)==model_info['sha256']);scaler,linear=joblib.load(p).steps[0][1],joblib.load(p).steps[1][1]
    # The first75 entries of the full representation are the untouched raw statistics.
    standardized=(external['all'][:,:75]-scaler.mean_)/scaler.scale_
    logits=standardized@linear.coef_.T+linear.intercept_
    logits-=logits.max(axis=1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
    finals['review_blend']=.9*finals['spectral_new']+.1*prob
    equal(finals['review_blend'],np.load(root/'external/review_blend_probability.npy'))
    for job,views in observed.items():
        for view,value in views.items():
            for key,number in value.items():equal(number,choice['all_legal'][job][view][key])
    order=[*cfg['jobs'],'review_blend']
    for metric,field in [('oof','ordinary_CV'),('development','ordinary_dev')]:
        expected=min(order,key=lambda j:(-observed[j][metric]['macro_f1'],order.index(j)))
        check(choice[field]==expected==result[field])
    independent={j:summary(truth['y'],p,truth['groups'])for j,p in finals.items()}
    losses={j:np.asarray(v['record_errors'])for j,v in independent.items()}
    base=losses['all_old'];oracle=np.stack(list(losses.values())).min(0);opportunity=base-oracle;valid=opportunity>1e-12
    for job,values in independent.items():
        errors=losses[job]
        values.update(selection_loss=float(np.mean(errors-oracle)),harmful_update=float(np.mean(np.maximum(errors-base,0))),
                      gain_capture=float(np.mean(np.maximum(base[valid]-errors[valid],0)/opportunity[valid])) if valid.any() else None,
                      gain_available_records=int(valid.sum()))
        for key,value in values.items():
            check(result['final'][job][key] is None) if value is None else equal(value,result['final'][job][key])
    feature={p:independent['spectral_'+p]['record_mean_error']-independent['all_'+p]['record_mean_error'] for p in ['old','new']}
    params={f:independent[f+'_new']['record_mean_error']-independent[f+'_old']['record_mean_error'] for f in ['all','spectral']}
    for key,value in feature.items():equal(value,result['feature_deletion_effect_at_each_recipe'][key])
    for key,value in params.items():equal(value,result['parameter_change_effect_at_each_features'][key])
    equal(feature['new']-feature['old'],result['interaction'])
    patch,old,review=independent['all_new'],independent['all_old'],independent['review_blend']
    gate=(patch['record_mean_error']<old['record_mean_error']-1e-12 and patch['harmful_update']<=review['harmful_update']+1e-12
          and patch['gain_capture'] is not None and patch['gain_capture']>=review['gain_capture']-1e-12)
    check(gate==result['primary_qualification_gate'])
    check(rows_checked==4*(3672+730+1468) and trees_checked==18000)
    save(root/'INDEPENDENT_AUDIT424.json',dict(utc=utc(),passed=True,checks=checks,parameter_forward_rows=rows_checked,
        actual_tree_states=trees_checked,classifier_fits_verified=24,audit_fits=0,seconds=time.monotonic()-started,
        result_sha256=sha(root/'RESULT424.json'),audit_code_sha256=sha(Path(__file__))))
    print({'passed':True,'checks':checks,'rows':rows_checked,'trees':trees_checked},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();main(args.run.resolve())
