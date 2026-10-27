"""Frozen, exploratory covariate-shift control on cached complete-program predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from controlled_feedback import save, sha, utc


def main(root):
    started = time.monotonic()
    plan = json.loads((root / 'PROTOCOL436.json').read_text())
    assert not (root / 'RESULT436.json').exists()
    for name, digest in plan['code_hashes'].items():
        assert sha(root / name) == digest
    for name, digest in plan['restored_hashes'].items():
        assert sha(root / name) == digest
    raw = Path(plan['unlabeled_queries']['path'])
    assert sha(raw) == plan['unlabeled_queries']['sha256']
    source = np.load(root / 'restored/inputs/development.npz', allow_pickle=False)['values']
    query_rows = np.loadtxt(raw)
    ids = sorted(set(query_rows[:, 0].astype(int)))
    target = np.stack([query_rows[query_rows[:, 0] == g][-1] for g in ids])
    assert len(ids) == 100 and source.shape[1] == target.shape[1] == 26
    # No engine identity, failure endpoint, RUL, or model output enters domain fitting.
    xs, xt = source[:, 1:], target[:, 1:]
    groups = source[:, 0].astype(int)
    source_ids = sorted(set(groups))
    source_fold = np.asarray([source_ids.index(g) % 5 for g in groups])
    target_fold = np.arange(len(target)) % 5
    ps, pt = np.empty(len(source)), np.empty(len(target))
    recovery_path = root / 'POSTFIT_RECOVERY436.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    (root / 'models').mkdir(exist_ok=recovery is not None)

    def one(fold):
        si, ti = source_fold != fold, target_fold != fold
        x = np.concatenate([xs[si], xt[ti]])
        domain = np.concatenate([np.zeros(si.sum()), np.ones(ti.sum())])
        fitted_path = root / 'models' / f'fold{fold}.pkl'
        if recovery is not None:
            assert sha(fitted_path) == recovery['fitted_sha256'][fitted_path.name]
            fitted = joblib.load(fitted_path)
            scale, model = fitted['scaler'], fitted['model']
        else:
            scale = StandardScaler().fit(x)
            model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000,
                                       solver='lbfgs', random_state=20261026).fit(scale.transform(x), domain)
        pv = model.predict_proba(scale.transform(xs[~si]))[:, 1]
        qt = model.predict_proba(scale.transform(xt[~ti]))[:, 1]
        z = scale.transform(xs[~si]) @ model.coef_[0] + model.intercept_[0]
        manual = np.asarray([1.0 / (1.0 + math.exp(-float(v))) for v in z])
        assert np.allclose(manual, pv, rtol=1e-11, atol=1e-11)
        assert set(groups[si]).isdisjoint(set(groups[~si]))
        if recovery is None:
            joblib.dump(dict(scaler=scale, model=model), fitted_path)
        return fold, pv, qt, dict(fold=fold, source_train_engines=sorted(int(g) for g in set(groups[si])),
            source_held_engines=sorted(int(g) for g in set(groups[~si])), target_train=int(ti.sum()),
            target_held=int((~ti).sum()), iterations=int(model.n_iter_[0]),
            manual_domain_probability_max_difference=float(np.max(np.abs(manual-pv))))

    with ThreadPoolExecutor(max_workers=4) as pool:
        folds = []
        for f, p, q, meta in pool.map(one, range(5)):
            ps[source_fold == f] = p; pt[target_fold == f] = q; folds.append(meta)
    odds = ps / np.maximum(1.0-ps, np.finfo(float).eps)
    weighted = np.minimum(odds, 20.0)
    weights = dict(uniform=np.ones(len(source)), importance=weighted,
                   importance_unclipped=odds,
                   engine_equal=np.asarray([1.0/(groups == g).sum() for g in groups]))
    truth = np.asarray([source[groups == g, 1].max()-c for g, c in zip(groups, source[:, 1])])
    keys = [(int(g), int(c)) for g, c in source[:, :2]]
    predictions = {}
    for ident, rel in plan['program_development'].items():
        rows = json.loads((root / rel).read_text())
        lookup = {(r['engine'], c):(y,p) for r in rows for c,y,p in zip(r['cycle'],r['truth'],r['prediction'])}
        assert set(keys) == set(lookup)
        assert np.array_equal(truth, [lookup[k][0] for k in keys])
        p = np.asarray([lookup[k][1] for k in keys])
        predictions[ident+'::original'] = p
        predictions[ident+'::cap125'] = np.minimum(p, 125.0)
    risks = {rule:{name:float(np.dot(w, np.abs(p-truth))/w.sum())
                   for name,p in predictions.items()} for rule,w in weights.items()}
    choices = {rule:min(scores, key=lambda k:(scores[k], k)) for rule,scores in risks.items()}
    np.savez_compressed(root / 'WEIGHTS436.npz', source_engine=groups, source_cycle=source[:, 1],
        domain_probability=ps, target_domain_probability=pt, source_fold=source_fold,
        target_fold=target_fold, **weights)
    save(root / 'CHOICES_SEALED436.json', dict(utc=utc(),choices=choices,risks=risks,
        weights_sha256=sha(root/'WEIGHTS436.npz'),final_labels_read=False))
    # External labels are read only after this fixed control's choices are sealed.
    labels = Path(plan['final_labels']['path'])
    assert sha(labels) == plan['final_labels']['sha256']
    y = np.loadtxt(labels); assert y.shape == (100,)
    final_predictions = {}
    for ident, rel in plan['program_final'].items():
        rows = json.loads((root / rel).read_text())['rows']
        by_id = {r['engine']:r['prediction'] for r in rows}
        p = np.asarray([by_id[g] for g in ids])
        final_predictions[ident+'::original'] = p
        final_predictions[ident+'::cap125'] = np.minimum(p, 125.0)
    base = np.abs(final_predictions['p00::original']-y)
    scores = {}
    for name,p in final_predictions.items():
        error = np.abs(p-y)
        mae = float(error.mean())
        assert math.isclose(mae,math.fsum(abs(float(a)-float(b)) for a,b in zip(p,y))/100,abs_tol=1e-10)
        scores[name] = dict(MAE=mae,harm=float(np.maximum(error-base,0).mean()),
                            gain=float(np.maximum(base-error,0).mean()))
    chosen = {rule:dict(program=choice,**scores[choice]) for rule,choice in choices.items()}
    group_mass = np.asarray([weighted[groups == g].sum() for g in source_ids])
    a,b = chosen['importance'],chosen['uniform']
    primary = a['MAE'] < b['MAE'] and a['MAE'] < scores['p00::original']['MAE']
    result = dict(utc=utc(),seconds=time.monotonic()-started,choices=chosen,scores=scores,
        primary_control_gate=primary,joint_descriptive_gate=primary and a['harm']<=b['harm'] and a['gain']>=b['gain'],
        folds=folds,domain_auc=float(roc_auc_score(np.r_[np.zeros(len(ps)),np.ones(len(pt))],np.r_[ps,pt])),
        row_ESS=float(weighted.sum()**2/np.dot(weighted,weighted)),
        engine_mass_ESS=float(group_mass.sum()**2/np.dot(group_mass,group_mass)),
        weight_min=float(weighted.min()),weight_max=float(weighted.max()),
        new_domain_fits=5,new_scaler_fits=5,new_PHM_fits=0,new_LLM_calls=0,
        postfit_serialization_recovery=recovery is not None,recovery_additional_fits=0,
        new_PHM_predictions=0,independent_confirmation=0,source_clusters=1,
        extra_unlabeled_query_information=True,original_Agent_comparison=False,
        novelty_claim=False,limitations=plan['limitations'])
    save(root/'RESULT436.json',result)
    print(json.dumps({k:result[k] for k in ['seconds','choices','primary_control_gate','joint_descriptive_gate','domain_auc','engine_mass_ESS']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
