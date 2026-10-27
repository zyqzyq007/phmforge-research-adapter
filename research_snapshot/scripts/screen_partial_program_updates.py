"""Exploratory acceptance controls for one frozen, actually trained RUL repair."""
import argparse
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from audit_program_update_capacity import arrays, predict
from controlled_feedback import save, sha, utc


SPEC = 'raw_input__mixed_54__residual__ridge10'


def inputs(root):
    cfg = json.loads((root/'PROTOCOL374.json').read_text())
    _, spaces, y, dy, groups, p, dp, names = arrays(root)
    archive = Path(cfg['capacity_archive'])
    assert sha(archive/'SOURCE_MANIFEST.json') == cfg['capacity_manifest_sha256']
    old = json.loads((archive/'RESULT373.json').read_text())
    outer = {r['held']: r for r in old['rows'][SPEC]}
    events = json.loads((archive/'FIT_EVENTS373.json').read_text())
    for event in [e for e in events if e['spec'] == SPEC]:
        assert sha(archive/event['model_path']) == event['model_sha256']
        np.testing.assert_allclose(outer[event['held']]['prediction'], event['warm_prediction'])
    return cfg, spaces['raw_input'], y, dy, groups, p, dp, names, outer


def fit(root):
    assert not (root/'HANDLE374.json').exists(), 'No implicit fit restart'
    cfg, (x, dx), y, dy, groups, p, dp, names, outer = inputs(root)
    save(root/'HANDLE374.json', dict(pid=os.getpid(), utc=utc(), state='fitting'))
    begin, events = time.monotonic(), []
    for held in range(8):
        available = [i for i in range(8) if i != held]
        # Deterministic split, independent of label/prediction values.
        ordered = sorted(available, key=lambda i: sha_text('A374split:'+str(i)))
        plans = [('loo', j, [i for i in available if i != j], [j, held]) for j in available]
        plans += [('split', -1, ordered[3:], ordered[:3]+[held])]
        for kind, probe, fitting, query in plans:
            assert time.monotonic()-begin < cfg['budget']['fit_wall_seconds']
            parent = int(np.argmin(np.mean(np.abs(dp[fitting]-dy[fitting, None]), axis=0)))
            tx = np.vstack([x, dx[fitting]])
            target = np.r_[y-p[:, parent], dy[fitting]-dp[fitting, parent]]
            scaler = StandardScaler()
            z = scaler.fit_transform(tx)
            mean = float(np.mean(target))
            model = Ridge(alpha=10)
            event = dict(kind=kind, held=held, probe=probe, fitting_development_cells=fitting,
                query_development_cells=query, fitting_source_cells=groups.tolist(), rows=len(target),
                parent=names[parent], actual_target_sha256=sha_bytes(np.ascontiguousarray(target).tobytes()),
                status='started')
            events.append(event)
            save(root/'FIT_EVENTS374.json', events)
            model.fit(z, target-mean)
            artifact = dict(scaler=scaler, model=model, target_mean=mean, update='residual',
                parent=names[parent], parent_index=parent, fitting_development_cells=fitting,
                fitting_source_cells=groups.tolist(), held=held, kind=kind, probe=probe)
            folder = root/'models'/('held_'+str(held))
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(kind+'_'+str(probe)+'.joblib')
            joblib.dump(artifact, path)
            warm = predict(artifact, dx[query], dp[query, parent])
            event.update(status='completed', model_path=str(path.relative_to(root)), model_sha256=sha(path),
                learned_coefficient_norm=float(np.linalg.norm(model.coef_)),
                warm_predictions=warm.tolist(), parent_predictions=dp[query, parent].tolist())
            save(root/'FIT_EVENTS374.json', events)
    assert len(events) == 64
    save(root/'FITS_TERMINAL374.json', dict(exit_code=0, actual_RUL_model_fits=64,
        learned_scalers=64, warm_predict_calls=64, warm_prediction_rows=144,
        reused_outer_models=8, fit_seconds=time.monotonic()-begin, new_LLM_calls=0))
    print(json.dumps(dict(actual_RUL_fits=64, new_LLM_calls=0)))


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def sha_text(value):
    return sha_bytes(value.encode())


def verify(root):
    assert not (root/'DECISIONS374.json').exists(), 'Do not rerun after decisions'
    cfg, (x, dx), y, dy, groups, p, dp, names, outer = inputs(root)
    assert json.loads((root/'FITS_TERMINAL374.json').read_text())['exit_code'] == 0
    events = json.loads((root/'FIT_EVENTS374.json').read_text())
    assert len(events) == 64
    cold = []
    for event in events:
        path = root/event['model_path']
        assert sha(path) == event['model_sha256'] and event['status'] == 'completed'
        artifact = joblib.load(path)
        fitting, query = event['fitting_development_cells'], event['query_development_cells']
        assert event['held'] not in fitting and not set(fitting) & set(query)
        parent = int(np.argmin(np.mean(np.abs(dp[fitting]-dy[fitting, None]), axis=0)))
        assert artifact['parent_index'] == parent
        target = np.r_[y-p[:, parent], dy[fitting]-dp[fitting, parent]]
        assert sha_bytes(np.ascontiguousarray(target).tobytes()) == event['actual_target_sha256']
        tx = np.vstack([x, dx[fitting]])
        np.testing.assert_allclose(artifact['scaler'].mean_, np.mean(tx, axis=0), rtol=2e-6, atol=1e-6)
        assert artifact['scaler'].n_samples_seen_ == len(target)
        value = predict(artifact, dx[query], dp[query, parent])
        np.testing.assert_allclose(value, event['warm_predictions'], rtol=1e-10, atol=1e-8)
        cold.append(dict(event, cold_predictions=value.tolist()))
    decisions = []
    for held in range(8):
        rows = [e for e in cold if e['held'] == held and e['kind'] == 'loo']
        assert len(rows) == 7
        available = np.array([e['probe'] for e in rows])
        old = np.array([e['parent_predictions'][0] for e in rows])
        proposal = np.array([e['cold_predictions'][0] for e in rows])
        query_proposals = np.array([e['cold_predictions'][1] for e in rows])
        truth = dy[available]
        residual = np.abs(proposal-truth)
        lower, upper = float(np.min(query_proposals-residual)), float(np.max(query_proposals+residual))
        parent = int(np.argmin(np.mean(np.abs(dp[available]-truth[:, None]), axis=0)))
        base, candidate = float(dp[held, parent]), outer[held]['prediction']
        delta = candidate-base
        alpha = np.array([0, .25, .5, .75, 1.])
        loss = np.abs(old[:, None]+alpha*(proposal-old)[:, None]-truth[:, None])
        mae, mse = np.mean(loss, axis=0), np.mean(loss**2, axis=0)
        best = int(np.argmin(mae))
        one_se = []
        for k in range(5):
            paired = loss[:, k]-loss[:, best]
            if np.mean(paired) <= np.std(paired, ddof=1)/np.sqrt(7)+1e-10:
                one_se.append(k)
        selected = dict(keep=0., full_update=1., fixed_quarter=.25, fixed_half=.5, fixed_three_quarters=.75,
            cv_binary_mse=float(mse[-1] < mse[0]), cv_binary_mae=float(mae[-1] < mae[0]),
            cv_mix_mae=float(alpha[best]), cv_mix_paired_1se=float(alpha[min(one_se)]))
        # Maximum accepted displacement with no worse absolute loss for ANY y in [L,U].
        signed_support = lower-base if delta > 0 else base-upper
        maximum_safe = min(abs(delta), 2*max(0., signed_support))
        minimax = min(abs(delta), max(0., signed_support))
        selected.update(interval_binary=float(abs(delta) <= 2*max(0., signed_support) and delta != 0),
            interval_partial_maxsafe=maximum_safe/abs(delta) if delta else 0.,
            interval_minimax=minimax/abs(delta) if delta else 0.)
        values = {key: base+weight*delta for key, weight in selected.items()}
        split = next(e for e in cold if e['held'] == held and e['kind'] == 'split')
        validation = split['query_development_cells'][:-1]
        sp, sc = np.array(split['parent_predictions']), np.array(split['cold_predictions'])
        use_split = np.mean((sc[:-1]-dy[validation])**2) < np.mean((sp[:-1]-dy[validation])**2)
        values['sre_split_mse'] = float(sc[-1] if use_split else sp[-1])
        for key in ['interval_binary', 'interval_partial_maxsafe', 'interval_minimax']:
            worst = max(abs(values[key]-bound)-abs(base-bound) for bound in [lower, upper])
            assert worst <= 1e-8, (held, key, worst)
        decisions.append(dict(held=held, parent=names[parent], base_prediction=base,
            candidate_prediction=candidate, interval_lower=lower, interval_upper=upper,
            signed_support=signed_support, weights=selected, predictions=values,
            cv_mae_by_fixed_alpha=mae.tolist(), cv_mse_by_fixed_alpha=mse.tolist(),
            split_SRE_accepted=bool(use_split)))
    save(root/'DECISIONS374.json', dict(utc=utc(), rows=decisions, held_truth_not_used_in_decision_rules=True,
        all8_already_used_in_design=True, no_conformal_or_conditional_safety_claim=True))
    scores = {}
    for row in decisions:
        actual, base_error = float(dy[row['held']]), abs(float(dy[row['held']])-row['base_prediction'])
        for method, value in row['predictions'].items():
            error = abs(value-actual)
            scores.setdefault(method, []).append(dict(held=row['held'], prediction=value, actual=actual,
                absolute_error=error, parent_error=base_error, harm=max(0, error-base_error), gain=max(0, base_error-error),
                interval_covers_actual=row['interval_lower'] <= actual <= row['interval_upper']))
    aggregate = {method: dict(MAE=float(np.mean([r['absolute_error'] for r in rows])),
        mean_harm=float(np.mean([r['harm'] for r in rows])), mean_gain=float(np.mean([r['gain'] for r in rows])),
        wins=sum(r['absolute_error'] < r['parent_error']-1e-9 for r in rows),
        losses=sum(r['absolute_error'] > r['parent_error']+1e-9 for r in rows)) for method, rows in scores.items()}
    controls = cfg['primary_strong_controls']
    gates = {}
    for method in cfg['mechanism_candidates']:
        m = aggregate[method]
        gates[method] = all(m['MAE'] < aggregate[c]['MAE']-1e-9 and
            m['mean_harm'] <= aggregate[c]['mean_harm']+1e-9 and
            m['mean_gain'] >= aggregate[c]['mean_gain']-1e-9 for c in controls)
    result = dict(utc=utc(), aggregate=aggregate, rows=scores, joint_numeric_screen=gates,
        interval_coverage_cells=int(sum(r['interval_lower'] <= dy[r['held']] <= r['interval_upper'] for r in decisions)),
        study_role='Exploratory mechanism screen; ordinary uncertainty-control components, not established innovation.',
        actual_RUL_fits=64, independent_cold_predict_calls=64, reused_outer_predictions=8, new_LLM_calls=0,
        same_information_among_current_rules=True, fair_comparison_to_original_A367_Agent=False,
        all8_seen_before_design=True, independent_confirmation_scenarios=0, final22_opened=False,
        novelty_confirmed=False, majority_complete=False)
    save(root/'RESULT374.json', result)
    save(root/'TERMINAL374.json', dict(exit_code=0, utc=utc()))
    print(json.dumps(dict(aggregate=aggregate, numeric_gates=gates, interval_coverage=result['interval_coverage_cells'])))


def expected_loss(root):
    """One fixed empirical predictive-loss control; no new checkpoint calls."""
    assert not (root/'DECISIONS382.json').exists(), 'Never retune/restart after choices'
    cfg=json.loads((root/'PROTOCOL382.json').read_text())
    src={k:Path(v['path']) for k,v in cfg['sources'].items()}
    for k,p in src.items():assert sha(p/'SOURCE_MANIFEST.json')==cfg['sources'][k]['manifest_sha256']
    events=json.loads((src['partial']/'FIT_EVENTS374.json').read_text())
    full={e['held']:e for e in json.loads((src['capacity']/'FIT_EVENTS373.json').read_text()) if e['spec']==SPEC}
    with np.load(src['source']/'private_inputs/discovery_truth.npz',allow_pickle=False) as z:dy=z['y']
    decisions=[]
    for held in range(8):
        history=[e for e in events if e['held']==held and e['kind']=='loo']
        assert len(history)==7 and {e['probe'] for e in history}==set(range(8))-{held}
        current=[];residual=[]
        for e in history:
            assert held not in e['fitting_development_cells'] and e['probe'] not in e['fitting_development_cells']
            assert sha(src['partial']/e['model_path'])==e['model_sha256']
            current.append(e['warm_predictions'][1])
            residual.append(float(dy[e['probe']])-e['warm_predictions'][0])
        current,residual=np.array(current),np.array(residual)
        b,c=full[held]['parent_prediction'],full[held]['warm_prediction']
        proxy=current+residual
        centers=dict(expected_absolute_loss=float(np.median(proxy)),
            no_residual_adjustment=float(np.median(current)),
            full_model_plus_median_residual=float(c+np.median(residual)))
        predictions={key:float(np.clip(center,min(b,c),max(b,c))) for key,center in centers.items()}
        # A median minimizes empirical absolute loss; projection handles the
        # common bounded action space. Verify against every breakpoint.
        candidates=np.clip(np.r_[proxy,b,c],min(b,c),max(b,c))
        losses=np.mean(abs(proxy[:,None]-candidates[None,:]),axis=0)
        empirical=float(np.mean(abs(proxy-predictions['expected_absolute_loss'])))
        assert empirical <= float(np.min(losses))+1e-9
        decisions.append(dict(held=held,parent=b,candidate=c,probes=[e['probe'] for e in history],
            signed_residuals=residual.tolist(),current_leaveout_forecasts=current.tolist(),
            pseudo_outcomes=proxy.tolist(),centers=centers,predictions=predictions,
            weights={k:(v-b)/(c-b) if c!=b else 0. for k,v in predictions.items()},empirical_absolute_loss=empirical))
    save(root/'DECISIONS382.json',dict(rows=decisions,current_held_truth_used_by_rules=False,
        probabilistic_calibration_guaranteed=False,all8_design_used=True))
    records={};aggregate={}
    for method in decisions[0]['predictions']:
        rr=[]
        for d in decisions:
            held=d['held'];y=float(dy[held]);b,c=d['parent'],d['candidate'];p=d['predictions'][method]
            old,error=abs(b-y),abs(p-y);oracle_error=abs(np.clip(y,min(b,c),max(b,c))-y)
            opportunity=old-oracle_error;gain=max(0.,old-error)
            rr.append(dict(held=held,prediction=p,MAE=error,regret=error-oracle_error,
                harm=max(0.,error-old),gain=gain,gain_capture=gain/opportunity if opportunity>1e-9 else None))
        records[method]=rr
        aggregate[method]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ['MAE','regret','harm','gain','gain_capture']}
    native=json.loads((src['native']/'RESULT375.json').read_text())
    for name,value in native['aggregate'].items():aggregate['A375_'+name]=value
    primary=aggregate['expected_absolute_loss'];ordinary=aggregate['A375_ordinary'];fixed=aggregate['A375_fixed_partial_cap']
    joint=(primary['MAE']<ordinary['MAE']-1e-9 and primary['regret']<ordinary['regret']-1e-9 and
           primary['harm']<=ordinary['harm']+1e-9 and primary['gain_capture']>=ordinary['gain_capture']-1e-9)
    differences=[]
    for i in range(8):
        errors=[r['methods']['ordinary']['MAE'] for r in native['records'] if r['held']==i]
        assert len(errors)==2
        differences.append(records['expected_absolute_loss'][i]['MAE']-float(np.mean(errors)))
    result=dict(utc=utc(),aggregate=aggregate,rows=records,ordinary_joint_numeric_gate=joint,
        beats_fixed_control_MAE=primary['MAE']<fixed['MAE']-1e-9,
        paired_cell_wins=sum(d<-1e-9 for d in differences),paired_cell_ties=sum(abs(d)<=1e-9 for d in differences),
        paired_cell_losses=sum(d>1e-9 for d in differences),same_action_segment=True,same_existing_legal_evidence=True,
        new_fits=0,new_checkpoint_predictions=0,new_LLM_calls=0,cached_pseudo_outcomes=56,
        preserved_A375_deviation='16validsubmissionsbut15zero-fitcompliant;oneordinarysession4extraauxiliaryfits,all16retained.',
        calibration_not_proven=True,novelty_confirmed=False,independent_confirmation=False,
        final22_opened=False,A33872_opened=False,majority_goal_achieved=False)
    save(root/'RESULT382.json',result);save(root/'TERMINAL382.json',dict(exit_code=0,utc=utc()))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fit', 'verify', 'expected_loss'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run.resolve())
