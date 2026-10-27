"""A344 prospective, source-disjoint, real calibration-query comparison.

Reuses genuinely trained, actual Agent-written programs and the native CLI
harness. Only calibration labels can be queried; final arrays open post-seal.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_free_program_comparison import independent_score
from program_revision_runtime import Runtime, load, subset, observations, score
from program_condition_queries import PROGRAMS, route, evidence_policy, policy_family
from run_frozen_candidate_review import source_context

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
OUT = PROJECT / 'results/query344'


def read(path):
    return json.loads(Path(path).read_text())


def state(root, **changes):
    p = root / 'STATUS344.json'
    value = read(p) if p.exists() else {}
    value.update(updated_utc=utc(), **changes)
    save(p, value)
    save(OUT / 'current_run.json', dict(value, run=str(root)))


def check_code(root):
    for name, digest in read(root / 'CODE_FREEZE.json').items():
        assert sha(root / 'code' / name) == digest
    for name, digest in read(root / 'SHARED_CODE_FREEZE344.json').items():
        assert sha(root / 'shared_tools' / name) == digest


def programs(root):
    cfg = read(root / 'QUERY_PROTOCOL344.json')
    archive = Path(cfg['model_archive'])
    inventory, receipt = source_context(archive)
    assert receipt['manifest_sha256'] == cfg['model_archive_manifest_sha256']
    result = {}
    for ident, item in cfg['model_bindings'].items():
        p = archive / item['relative_path']
        assert sha(p / 'REPORT.json') == item['report_sha256']
        for filename, digest in item['artifacts'].items():
            assert sha(p / filename) == digest == inventory[item['relative_path']+'/'+filename]['sha256']
        result[ident] = p
    return result


def setup(root):
    assert not (root / 'CODE_FREEZE.json').exists()
    cfg = read(root / 'QUERY_PROTOCOL344.json')
    assert not (root / 'CALIBRATION_PREDICTION_STARTED344.json').exists()
    from screen_program_updates import CODE
    names = set(CODE) | {'query_program_update.py', 'program_condition_queries.py',
        'program_revision_runtime.py', 'evaluate_free_program_comparison.py'}
    for name in names:
        p = root / 'code' / name
        if p.exists():
            assert sha(p) == sha(PROJECT / 'scripts' / name)
        else:
            shutil.copy2(PROJECT / 'scripts' / name, p)
    for name in ['shared_tools', 'public/rows', 'private', 'predictions', 'evaluation']:
        (root / name).mkdir(parents=True, exist_ok=True)
    archive = Path(cfg['model_archive'])
    inventory,_=source_context(archive)
    for name in ['program_update_tools.py', 'program_condition_queries.py', 'controlled_feedback.py',
                 'isolated_model_gateway.py', 'prepare_fault_extensions.py', 'acquire_forge_pdmb_sources244.py']:
        shutil.copy2(PROJECT / 'scripts' / name, root / 'shared_tools' / name)
    for name in ['phm_program_features.py', 'program_training_entry.py']:
        assert sha(archive/'common_tools'/name)==inventory['common_tools/'+name]['sha256']
        shutil.copy2(archive / 'common_tools' / name, root / 'shared_tools' / name)
    for ident, p in programs(root).items():
        out = root / 'public/rows/programs' / ident
        out.mkdir(parents=True)
        shutil.copytree(p / 'source', out / 'source')
        shutil.copytree(p / 'full_fit/artifacts', out / 'full_fit/artifacts')
        shutil.copy2(p / 'REPORT.json', out / 'REPORT.json')
    for original, target in [('train', 'training'), ('development', 'old_development'), ('independent_check', 'old_check')]:
        assert sha(archive/'private_inputs'/(original+'.npz'))==inventory['private_inputs/'+original+'.npz']['sha256']
        shutil.copy2(archive / 'private_inputs' / (original + '.npz'), root / 'public/rows' / (target + '.npz'))
    save(root / 'ENGINE_CONFIG.json', dict(input_root=str(root / 'qualification'), common_tools=str(archive / 'common_tools'),
         fit_seconds=0, predict_seconds=45, program_seconds=900))
    campaign = dict(cfg, result_key='query344', status_file='STATUS344.json', preparation_file='PREPARATION344.json',
        stage='calibration_query_selection', qualify_submission_chain=True,
        qualification_query_slot='calib_1', reset_qualification_transport=True,
        expected_mcp_actions=['cases', 'evidence', 'query', 'select'],
        broker_config_extra=dict(query_private=str(root / 'private/CALIBRATION_PACKETS344.json'), max_queries=2),
        researcher_previously_knows_final_scores=False)
    save(root / 'CAMPAIGN_CONFIG.json', campaign)
    save(root / 'CODE_FREEZE.json', {p.name: sha(p) for p in (root / 'code').glob('*.py')})
    save(root / 'SHARED_CODE_FREEZE344.json', {p.name: sha(p) for p in (root / 'shared_tools').glob('*.py')})
    state(root, status='implementation_frozen_before_model_inference', new_LLM_calls=0, new_fits=0,
          final_data_read=False, method_spec_sha256=sha(root / 'QUERY_PROTOCOL344.json'))
    print(json.dumps(dict(status='implementation_frozen', files=len(read(root / 'CODE_FREEZE.json')), new_predictions=0)))


def vote_predictions(data, pred, include_labels):
    records = []
    for group in sorted(set(data['groups'].tolist())):
        mask = data['groups'] == group
        counts = np.bincount(pred[mask], minlength=4)
        guessed = int(np.flatnonzero(counts == counts.max())[0])
        record = dict(group=group, rotation_hz=float(data['rotation_hz'][mask].mean()), predicted=guessed,
                      window_predictions=pred[mask].astype(int).tolist(), window_class_counts=counts.tolist())
        if include_labels:
            ys = set(data['labels'][mask].tolist())
            assert len(ys) == 1
            record['actual'] = int(next(iter(ys)))
        records.append(record)
    return records


def packet(data, predictions, condition_hz, include_labels):
    records = {p: vote_predictions(data, value, include_labels) for p, value in predictions.items()}
    ids = sorted(r['group'] for r in records['p00'])
    rows = []
    for group in ids:
        by = {p: next(r for r in value if r['group'] == group) for p, value in records.items()}
        row = dict(group=group, rotation_hz=by['p00']['rotation_hz'],
                   programs={p: {k: v for k, v in r.items() if k not in ['group', 'rotation_hz', 'actual']} for p, r in by.items()})
        if include_labels:
            row['actual'] = by['p00']['actual']
        rows.append(row)
    value = dict(condition_hz=condition_hz, records=rows, full_checkpoints=True, real_measured_predictions=True)
    if include_labels:
        value['errors'] = {p: sum(r['actual'] != r['programs'][p]['predicted'] for r in rows) / len(rows) for p in PROGRAMS}
    else:
        entropies = []
        for r in rows:
            counts = np.bincount([r['programs'][p]['predicted'] for p in PROGRAMS], minlength=4)
            probabilities = counts[counts > 0] / len(PROGRAMS)
            entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
        value['mean_record_vote_entropy'] = float(np.mean(entropies))
    return value


def cold_bank(root, data, folder):
    catalog = programs(root)
    conditions=sorted(set(data['condition_id'].tolist()))
    def one(ident,condition):
        mask=data['condition_id']==condition;part=subset(data,mask)
        engine = Runtime(root)
        engine.deadline = time.monotonic() + 900
        out = root / folder / ident / ('context_'+str(int(condition)))
        out.mkdir(parents=True)
        source = catalog[ident]
        first, a = engine.predict(out / 'forward', source / 'source', source / 'full_fit/artifacts', part)
        second, b = engine.predict(out / 'cold_repeat', source / 'source', source / 'full_fit/artifacts', part)
        assert np.array_equal(first, second)
        assert all(x['completed_estimator_fits'] == x['optimizer_steps'] == 0 for x in a + b)
        independent = independent_score(part, first)
        result = score(part, first)
        assert result['record_error'] == independent['record_error'] and result['confusion'] == independent['confusion']
        save(out / 'AUDIT.json', dict(cold_identical=True, independent_votes_verified=True, prediction_processes=len(a + b)))
        return ident,condition,mask,first
    values = {p:np.full(len(data['labels']),-1,dtype=int) for p in PROGRAMS};done=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(one,p,c) for p in PROGRAMS for c in conditions]):
            key,condition,mask,value = future.result();values[key][mask]=value;done.append([key,int(condition)])
            save(root / folder / 'PROGRESS.json', dict(completed=done, planned=3*len(conditions)))
    assert all(np.all(v>=0) for v in values.values())
    for key,value in values.items():np.savez_compressed(root/folder/key/'PREDICTIONS.npz',sample_ids=data['sample_ids'],prediction=value)
    return values


def calibration(root):
    check_code(root)
    assert not (root / 'CALIBRATION_PREDICTION_STARTED344.json').exists()
    source = read(root / 'QUALIFICATION344.json')
    assert source['passed'] and source['whole_acquisitions'] == 28
    for filename, value in source['files'].items():
        assert sha(root / 'qualification' / filename) == value['sha256']
    # Read only identity hashes for the final role,never its numerical array.
    source_cfg=read(root/'PROTOCOL344.json');prior=set()
    for entry in source_cfg['prior_bindings']+[dict(archive=source_cfg['source_archive'])]:
        old=Path(entry['archive']);inventory,_=source_context(old)
        for relative,item in inventory.items():
            if relative.startswith('workspace_snapshot/'):continue
            if 'IDENTITIES' in relative and relative.endswith('.json'):
                assert sha(old/relative)==item['sha256']
                prior.update(x.get('raw_sha256',x.get('source_sha256')) for x in read(old/relative))
            elif relative.startswith('records/') and relative.endswith('/fetch/RECEIPT.json'):
                assert sha(old/relative)==item['sha256']
                row=read(old/relative)
                if row.get('passed') and row.get('complete_record_CRC_verified'):prior.add(row['sha256'])
    prior.discard(None);identities=read(root/'qualification/IDENTITIES344.json')
    current=[x['raw_sha256'] for x in identities];assert len(set(current))==len(current)==28
    assert not set(current)&prior
    save(root/'audit/RAW_SOURCE_DISJOINT344.json',dict(passed=True,prior_unique_raw=len(prior),current_unique_raw=28,overlap=0))
    cfg = read(root / 'QUERY_PROTOCOL344.json')
    save(root / 'CALIBRATION_PREDICTION_STARTED344.json', dict(utc=utc(), protocol_sha256=sha(root / 'QUERY_PROTOCOL344.json'),
         code_freeze_sha256=sha(root / 'CODE_FREEZE.json'), final_data_read=False))
    data = load(root / 'qualification/calibration.npz')
    predictions = cold_bank(root, data, 'predictions/calibration')
    np.savez_compressed(root / 'public/rows/calibration_observations.npz', **observations(data))
    labeled = {}; unlabeled = []
    for index, condition in enumerate(sorted(set(data['condition_id'].tolist())), 1):
        mask = data['condition_id'] == condition
        selected = subset(data, mask)
        pp = {p: values[mask] for p, values in predictions.items()}
        slot = 'calib_' + str(index)
        labeled[slot] = packet(selected, pp, float(condition), True)
        unlabeled.append(dict(slot=slot, **packet(selected, pp, float(condition), False)))
    archive = Path(cfg['model_archive'])
    inventory,_=source_context(archive)
    catalog = programs(root)
    base = {}
    for role, center, condition in [('development', 'base38', 38), ('independent_check', 'base54', 54)]:
        old_data = load(archive / 'private_inputs' / (role + '.npz'))
        assert sha(archive/'private_inputs'/(role+'.npz'))==inventory['private_inputs/'+role+'.npz']['sha256']
        pp = {}
        for p in PROGRAMS:
            file = catalog[p] / 'development/PREDICTIONS.npz' if role == 'development' else archive / 'subjects/controlled_edits_2/common_check' / p / 'PREDICTIONS.npz'
            assert sha(file)==inventory[str(file.relative_to(archive))]['sha256']
            with np.load(file, allow_pickle=False) as z:
                assert np.array_equal(z['sample_ids'], old_data['sample_ids'])
                pp[p] = z['prediction']
        base[center] = packet(old_data, pp, condition, True)
    private = dict(base_centers={k: p['condition_hz'] for k, p in base.items()}, base_packets=base, calibration=labeled)
    save(root / 'private/CALIBRATION_PACKETS344.json', private)
    case = dict(id='update', candidate_ids=PROGRAMS, incumbent_ids=['p00'], source_paths={p: '/data/rows/programs/' + p for p in PROGRAMS},
                old_full_checkpoint_packets=base, calibration_slots=unlabeled,
                query_budget=2, final_query_budget=0, real_training_reused=True,
                CV_checkpoints_are_not_deployed_full_checkpoint=True,
                candidate_training_reports={p: {k: v for k, v in read(folder / 'REPORT.json').items() if k not in ['artifacts']} for p, folder in catalog.items()})
    save(root / 'public/cases.json', [case])
    save(root / 'public/rows/update.json', case)
    save(root / 'public/contract.json', dict(allow_validation_queries=True, candidate_ids=PROGRAMS,
        population='Futuremeasuredoperationin10to60Hzenvelope,equalplanningweighttoeachthird;exactfutureconditionsandoutcomesunavailable.',
        actions='Global oneprogram,or nearestnominalcenter assignments over base38/base54 and all actuallyqueriedslots. Routeusingmeasuredrecordrotation_hz;ties smallercenter.',
        centers_are_validation_metadata_not_test_IDs=True, windows_per_record=9,
        final_class_counts_hidden=True, base_models_immutable=True, no_new_fitting=True,
        primary='Equalwhole-acquisitionerror;reportlocal/typedharmseparately. No invented maintenancecosts.',
        calibration_observations='/data/rows/calibration_observations.npz',
        programs='/data/rows/programs', query_budget=2, label_cost_per_group=4,
        nearest_policy_example=dict(kind='nearest', assignments=dict(base38='p00', base54='p04'))))
    controls = []
    coverage = [unlabeled[0]['slot'], unlabeled[-1]['slot']]
    disagreement = [x['slot'] for x in sorted(unlabeled, key=lambda r: (-r['mean_record_vote_entropy'], r['condition_hz']))[:2]]
    controls.append(dict(method='keep_p00', policy=dict(kind='global', program='p00'), centers=private['base_centers'], queried=[]))
    for name, queries in [('old_full_checkpoint', []), ('uniform_coverage_query', coverage), ('QBC_disagreement_query', disagreement), ('all_calibration_information_upper_control', sorted(labeled))]:
        packets = dict(base, **{k: labeled[k] for k in queries})
        for kind in ['global', 'nearest']:
            policy, centers = evidence_policy(packets, kind)
            controls.append(dict(method=name + '_' + kind, policy=policy, centers=centers, queried=queries,
                                 labels_consumed=4 * len(queries), privileged_over_budget=len(queries) > 2))
    save(root / 'FIXED_CHOICES.json', dict(utc=utc(), choices=controls, final_scores_joined=False))
    save(root / 'PREPARATION344.json', dict(passed=True, actual_calibration_prediction_processes=96,
         old_full_checkpoint_predictions_reused=True, independent_cold_votes_verified=True,
         final_data_read=False, code_freeze_sha256=sha(root / 'CODE_FREEZE.json')))
    state(root, status='calibration_and_fixed_controls_ready', new_fits=0, final_data_read=False, scientific_model_calls=0)
    print(json.dumps(dict(status='calibration_ready', slots=4, final_data_read=False, prediction_processes=96)))


def agents(root):
    check_code(root)
    from screen_program_updates import agents as native
    native(root)
    state(root, status='all_native_choices_sealed', final_data_read=False)


def routed_prediction(data, bank, policy, centers):
    prediction = np.empty(len(data['labels']), dtype=int)
    routes = {}
    for group in sorted(set(data['groups'].tolist())):
        mask = data['groups'] == group
        p = route(policy, centers, float(data['rotation_hz'][mask].mean()))
        prediction[mask] = bank[p][mask]
        routes[group] = p
    return prediction, routes


def evaluate(root):
    check_code(root)
    assert (root / 'ALL_CHOICES_SEALED.json').exists() and not (root / 'FINAL_EVALUATION_STARTED344.json').exists()
    for item in read(root / 'ALL_CHOICES_SEALED.json')['files']:
        assert sha(root / item['path']) == item['sha256']
    save(root / 'FINAL_EVALUATION_STARTED344.json', dict(utc=utc(), all_choices_sha256=sha(root / 'ALL_CHOICES_SEALED.json'), final_first_open=True))
    # FIRST read of final numerical arrays:all4native decisions are terminal/sealed.
    data = load(root / 'qualification/final.npz')
    bank = cold_bank(root, data, 'evaluation/bank')
    private = read(root / 'private/CALIBRATION_PACKETS344.json')
    family = policy_family(private['base_centers'], {k: v['condition_hz'] for k, v in private['calibration'].items()}, 2)
    oracle = 1.; family_count = 0
    for policy, centers in family:
        p, _ = routed_prediction(data, bank, policy, centers)
        oracle = min(oracle, score(data, p)['record_error'])
        family_count += 1
    baseline = score(data, bank['p00'])['record_error']
    def scored(policy, centers):
        prediction, routes = routed_prediction(data, bank, policy, centers)
        result = score(data, prediction)
        err = result['record_error']; opportunity = baseline - oracle
        result.update(selection_regret=err - oracle, harm=max(0., err - baseline),
                      useful_gain_capture=min(1., max(0., baseline - err) / opportunity) if opportunity > 0 else None,
                      routes=routes, policy=policy, centers=centers)
        return result
    fixed = []
    for row in read(root/'FIXED_CHOICES.json')['choices']:
        value=scored(row['policy'],row['centers'])
        if row.get('privileged_over_budget',False):
            value['selection_regret']=None;value['useful_gain_capture']=None
        fixed.append(dict(method=row['method'],labels_consumed=row.get('labels_consumed',0),
                          privileged_over_budget=row.get('privileged_over_budget',False),**value))
    episodes = read(root / 'AGENT_CHOICES.json')['episodes']; rows = []; catalog = programs(root)
    replay_processes = 0
    for ep in episodes:
        result = dict(id=ep['id'], condition=ep['condition'], completed=ep['completed'])
        if ep['completed']:
            choice = ep['choice']; prediction, routes = routed_prediction(data, bank, choice['policy'], choice['centers'])
            result.update(scored(choice['policy'], choice['centers']), queried_slots=choice['queried_slots'], labels_consumed=choice['labels_consumed'])
            engine = Runtime(root); engine.deadline = time.monotonic() + 900
            for group, ident in routes.items():
                mask = data['groups'] == group; folder = catalog[ident]
                actual, audits = engine.predict(root / 'evaluation/selected' / ep['id'] / group,
                    folder / 'source', folder / 'full_fit/artifacts', subset(data, mask))
                assert np.array_equal(actual, prediction[mask])
                assert all(a['completed_estimator_fits'] == a['optimizer_steps'] == 0 for a in audits)
                replay_processes += len(audits)
            result['selected_policy_actual_cold_replay_verified'] = True
        rows.append(result)
        save(root / 'FINAL_PROGRESS344.json', rows)
    summary = {}
    for arm in ['ordinary', 'code_guided']:
        rr = [r for r in rows if r['condition'] == arm]; good = [r for r in rr if r['completed']]
        summary[arm] = dict(planned=2, valid=len(good),
            **{key: float(np.mean([r[key] for r in good])) if len(good) == 2 and good[0][key] is not None else None
               for key in ['record_error', 'selection_regret', 'harm', 'useful_gain_capture']})
    result = dict(iteration='A344', utc=utc(), status='complete', native_rows=rows, native_summary=summary,
        fixed_controls=fixed, baseline_error=baseline, common_action_oracle=oracle, oracle_family_enumerated=family_count,
        fixed_global_oracle=min(score(data, p)['record_error'] for p in bank.values()),
        final_bank_scores={k: score(data, p) for k, p in bank.items()},
        physical_development_cohorts=1, independent_confirmation_scenarios=0, new_base_fits=0,
        calibration_prediction_processes=96, final_bank_prediction_processes=72, selected_policy_replay_processes=replay_processes,
        majority_goal_complete=False, novelty_established=False, reserved338_performance_unopened=True)
    save(root / 'RESULT344.json', result); save(OUT / 'RESULT344.json', result)
    state(root, status='evaluation_complete', final_data_read=True, controller_status='terminal')
    print(json.dumps(dict(summary=summary, common_oracle=oracle, selected_replay_processes=replay_processes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'calibration', 'agents', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run.resolve())
