"""Expose genuinely fitted candidate models; choose the starting one by train CV.

This is a controlled availability contrast alongside unrestricted coding, not
a restriction to menu selection. Never opens final observations or labels.
"""
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from acquire_forge_pdmb_sources244 import save, sha256
from frozen_classifier import estimators


def install(run, spec, source, reference, workspace):
    root=run.parent
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    case_spec=next(x for x in config['cases']
              if x['cohort']==run.name.removesuffix('_review') and x['dataset']==spec['dataset'])
    case=case_spec['case']
    bank_settings={**config.get('candidate_bank',{}),**case_spec.get('candidate_bank',{})}
    seed=config['numerical_seed']
    control=root/'controls'/case
    frozen=json.loads((control/'FROZEN_SELECTION.json').read_text())
    train=dict(np.load(source/'train.npz',allow_pickle=False))
    dev=dict(np.load(source/'val.npz',allow_pickle=False))
    cv_group_field=bank_settings.get('cv_group_field','groups')
    x,y=train['features'].astype(float),train['labels']
    source_groups=train['groups']
    g=train[cv_group_field]
    assert len(g)==len(y) and len(np.unique(g))>=3
    classes=json.loads((source/'dataset.json').read_text())['class_names']
    cache=root/'bank_selection'/bank_settings.get('cache_key',case)
    cached=cache/'TRAINING_CV.json'
    if not cached.exists():
        cache.mkdir(parents=True)
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
        folds=list(splitter.split(x,y,g))
        fold_records=[]
        for n,(fit,hold) in enumerate(folds):
            assert not set(g[fit]) & set(g[hold])
            assert not set(source_groups[fit]) & set(source_groups[hold])
            fold_records.append(dict(fold=n,fit_ids=train['sample_ids'][fit].tolist(),
                heldout_ids=train['sample_ids'][hold].tolist(),
                fitting_classes=np.unique(y[fit]).tolist(),heldout_classes=np.unique(y[hold]).tolist(),
                cv_group_field=cv_group_field,fit_cv_groups=np.unique(g[fit]).tolist(),
                heldout_cv_groups=np.unique(g[hold]).tolist()))
        rows=[]
        for order,(name,_) in enumerate(estimators(seed)):
            row=dict(name=name,order=order,valid=True)
            predicted=np.empty(len(y),dtype=y.dtype)
            assigned=np.zeros(len(y),dtype=int)
            errors=[]
            for n,(fit,hold) in enumerate(folds):
                model=dict(estimators(seed))[name]
                try:
                    model.fit(x[fit],y[fit])
                    predicted[hold]=model.predict(x[hold]);assigned[hold]+=1
                    dest=cache/'cv_models'/(name+'_fold'+str(n)+'.joblib')
                    dest.parent.mkdir(exist_ok=True)
                    joblib.dump(model,dest)
                except Exception as error:
                    errors.append(dict(fold=n,error_type=type(error).__name__,error=str(error)))
            if errors:
                row.update(valid=False,errors=errors)
            else:
                assert np.all(assigned==1)
                row.update(macro_f1=float(f1_score(y,predicted,labels=classes,average='macro',zero_division=0)),
                    accuracy=float(accuracy_score(y,predicted)),
                    mean_group_accuracy=float(np.mean([np.mean(predicted[g==v]==y[g==v]) for v in set(g)])))
                save(cache/(name+'_OOF.json'),dict(predictions=[dict(sample_id=str(i),predicted_label=p.item())
                    for i,p in zip(train['sample_ids'],predicted)]))
            rows.append(row)
        valid=[r for r in rows if r['valid']]
        assert valid,'No valid candidate training CV; stop before any Agent launch'
        selected=max(valid,key=lambda r:(r['macro_f1'],r['accuracy'],-r['order']))['name']
        save(cached,dict(case=case,rows=rows,folds=fold_records,selected=selected,
            selection_rule='Training '+cv_group_field+'-group OOF macro-F1, accuracy, then frozen order',
            cv_group_field=cv_group_field,numerical_seed=seed,
            train_sha256=sha256(source/'train.npz'),dev_sha256=sha256(source/'val.npz'),
            final_observations_or_answers_used=False,models_genuinely_fitted_per_successful_fold=True))
    cv=json.loads(cached.read_text())
    assert cv['train_sha256']==sha256(source/'train.npz')
    assert cv['dev_sha256']==sha256(source/'val.npz')
    assert cv.get('cv_group_field','groups')==cv_group_field
    assert cv.get('numerical_seed',seed)==seed
    starting=bank_settings.get('fixed_starting_model',cv['selected'])
    assert any(r['name']==starting and r['valid'] for r in frozen['candidates'])
    starting_rule=('Protocol-fixed identical starting model, independent of observed CV/development performance'
                   if 'fixed_starting_model' in bank_settings else cv['selection_rule'])
    bank=reference/'candidates';bank.mkdir()
    models=[]
    for record in frozen['candidates']:
        if not record['valid']:
            models.append(record);continue
        filename=record['name']+'_train.joblib'
        path=control/filename
        assert sha256(path)==frozen['model_hashes'][filename]
        shutil.copy2(path,bank/filename)
        # Independent development reproduction checks the actual saved model,
        # not a model description or a simulated metric.
        model=joblib.load(bank/filename)
        predicted=model.predict(dev['features'].astype(float))
        assert abs(float(accuracy_score(dev['labels'],predicted))-record['accuracy'])<1e-12
        assert abs(float(f1_score(dev['labels'],predicted,average='macro',zero_division=0))-record['macro_f1'])<1e-12
        models.append(dict(name=record['name'],checkpoint=filename,sha256=sha256(bank/filename),
            fitting_rows=len(y),fitting_source_groups=len(set(source_groups)),fit_scope='train.npz only',
            development_accuracy=record['accuracy'],development_macro_f1=record['macro_f1'],
            training_group_cv=next(r for r in cv['rows'] if r['name']==record['name'])))
    metadata=dict(dataset=spec['dataset'],class_names=classes,models=models,
        starting_model=starting,starting_choice_rule=starting_rule,
        training_cv_selected=cv['selected'],training_cv_selection_rule=cv['selection_rule'],
        fixed_development_selector=frozen['selected'],fitting_scope='Training only; no train+development refit',
        numerical_seed=seed,cv_group_field=cv_group_field,
        training_cv_scope=('Three folds of complete training source groups; normalization refitted inside every fold.'
            if cv_group_field=='groups' else 'Three folds of complete training '+cv_group_field+
            ' values; all source records at a held-out value excluded and normalization refitted inside every fold.'),
        final_observations_used=False,final_answers_used=False,
        source_train_sha256=cv['train_sha256'],source_dev_sha256=cv['dev_sha256'])
    save(bank/'manifest.json',metadata)
    save(bank/'training_cv_folds.json',cv['folds'])
    baseline='''"""Real fitted starting candidate; its selection rule is in the bank manifest."""
import argparse,sys
sys.path.insert(0,'/tools')
from frozen_classifier import infer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();infer('/reference/candidates/SELECTED_train.joblib',a.input,a.output)
'''.replace('SELECTED',starting)
    (workspace/'solution.py').write_text(baseline)
    return metadata
