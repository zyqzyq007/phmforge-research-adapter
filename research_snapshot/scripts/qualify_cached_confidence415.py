"""Frozen confidence controls on already-trained PHM programs; never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
from datetime import datetime, timezone

import numpy as np


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def normalize(p):
    p = np.asarray(p, dtype=float)
    assert p.ndim == 2 and np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)
    return p / p.sum(1, keepdims=True)


def confidence(p):
    p = normalize(p)
    return dict(average_confidence=float(p.max(1).mean()),
                negative_entropy=float(np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=1).mean()))


def rank(rows, key):
    top = max(r[key] for r in rows)
    return next(r['id'] for r in rows if r[key] >= top - 1e-12)


def pair_probs(path, classes):
    obj = load(path)
    rows = obj['predictions']
    ids = [str(x['sample_id']) for x in rows]
    assert len(ids) == len(set(ids))
    values = []
    for row in rows:
        raw = row.get('probabilities', row.get('class_probabilities'))
        assert raw is not None
        if isinstance(raw, dict):
            assert set(raw) == {str(c) for c in classes}
            vector = [raw[str(c)] for c in classes]
        else:
            assert len(raw) == len(classes)
            vector = raw
        assert str(classes[int(np.argmax(vector))]) == str(row['predicted_label']), 'Probability argmax differs from full emitted class'
        values.append(vector)
    return ids, normalize(values)


def choose(run):
    start = time.monotonic()
    rows = load(run / 'inputs/a414/PAIR_PATH_ADMISSION414.json')
    admitted = load(run / 'inputs/a414/ADMISSION_RESULT414.json')['rows']
    ids = {r['id'] for r in admitted if r['normal_formal'] and r['admission'] == 'vector_present_semantics_pending'}
    assert len(ids) == 3
    pairs = []
    for row in rows:
        if row['id'] not in ids:
            continue
        source = 'rotor' if row['dataset'] == 'ROTORBROKENBAR' else 'cwru'
        audits = list((run / 'inputs' / source).rglob('training_audit.json'))
        assert len(audits) == 1
        classes = load(audits[0])['class_names']
        a = run / 'inputs/a414/inputs' / row['source_folder']
        ref_ids, ref_p = pair_probs(a / row['reference_path'], classes)
        candidate_ids, candidate_p = pair_probs(a / row['final_path'], classes)
        assert ref_ids == candidate_ids
        values = [dict(id='reference', **confidence(ref_p)), dict(id='Agent', **confidence(candidate_p))]
        pairs.append(dict(id=row['id'], dataset=row['dataset'], source_cluster=source, n=len(ref_ids),
                          classes=classes, scores=values, selected={key:rank(values,key) for key in values[0] if key!='id'},
                          paths=dict(reference=str((a/row['reference_path']).relative_to(run)), Agent=str((a/row['final_path']).relative_to(run))),
                          file_sha256=dict(reference=sha(a/row['reference_path']), Agent=sha(a/row['final_path'])),
                          scope='Benchmark original reference, not necessarily immediate review incumbent; window classification only, not safety/cost-action certification.'))
    g2 = run / 'inputs/g2'
    bank = dict(np.load(g2 / 'bank/probabilities.npz', allow_pickle=False))
    classes = bank['classes'].tolist()
    proof = load(g2 / 'BASIS_QUALIFICATION.json')
    assert proof['passed'] and proof['bank_sha256'] == sha(g2 / 'bank/probabilities.npz')
    names = bank['basis_names'].tolist()
    assert names == [m['name'] for m in proof['models']]
    stacks = []
    ids = None
    checks = []
    for split in ['val','test']:
        models = []
        for name in names:
            p = load(g2 / 'evaluation282/basis' / name / split / 'probabilities.json')
            assert p['classes'] == classes
            sample_ids = [str(x) for x in p['sample_ids']]
            assert len(sample_ids) == len(set(sample_ids))
            if models:
                assert sample_ids == current_ids
            current_ids = sample_ids
            models.append(normalize(p['probabilities']))
        stacked = np.stack(models, axis=1)
        if split == 'val':
            assert current_ids == bank['development_ids'].tolist()
            assert np.allclose(stacked, bank['development'], rtol=0, atol=1e-12)
        else:
            ids = current_ids
            stacks = stacked
        checks.append(dict(split=split, models=len(models), n=len(current_ids), passed=True))
    cases = []
    cached = {}
    for phase_path in ['PHASE_SEALED.json','review/PHASE_SEALED.json']:
        phase = load(g2 / phase_path)
        assert phase['all_native_processes_terminal'] and not phase['final_scores_opened']
        for case in phase['cases']:
            assert case['status'] == 'finished' and case['delivered'] and case['selection']
            score_rows = []
            for trial in case['trials']:
                w = tuple(trial['weights'])
                assert len(w) == len(names) and min(w)>=0 and sum(w)==12
                if w not in cached:
                    p = np.einsum('nbc,b->nc', stacks, np.asarray(w, dtype=float)) / 12
                    cached[w] = confidence(p)
                score_rows.append(dict(id=trial['id'], weights=list(w), **cached[w]))
            chosen = case['selection']['candidate_id']
            assert next(t['weights'] for t in score_rows if t['id']==chosen) == case['selection']['weights']
            selected = {key:rank(score_rows,key) for key in ['average_confidence','negative_entropy']}
            cases.append(dict(stage=phase['stage'], case=case['id'], ordinary_choice=chosen,
                              ordinary_weights=case['selection']['weights'], candidates=score_rows, selected=selected))
    assert len(cases) == 12
    result = dict(utc=utc(), protocol_sha256=sha(run/'PROTOCOL415.json'), pairs=pairs, banks=cases,
                  qualification=checks, bank_names=names, bank_classes=classes, bank_rows=len(ids),
                  unique_weight_vectors_evaluated=len(cached), final_result_metrics_read=False,
                  seconds=time.monotonic()-start, new_model_forwards=0, new_fits=0, new_LLM=0)
    assert result['seconds'] < 60
    save(run / 'CHOICES_SEALED415.json',result)
    print(dict(pairs=[dict(id=p['id'], scores=p['scores'], selected=p['selected']) for p in pairs],
               bank_stages=len(cases), unique_vectors=len(cached), seconds=result['seconds']))


def read_bound_result(archive, member):
    receipt = load(archive / 'ARCHIVE_VERIFIED.json')
    assert sha(archive / 'SOURCE_MANIFEST.json') == receipt['manifest_sha256']
    entry = next(x for x in load(archive / 'SOURCE_MANIFEST.json')['files'] if x['path']==member)
    if (archive/member).is_file():
        raw = (archive/member).read_bytes()
    else:
        with tarfile.open(archive/receipt['bundle_filename'], 'r|gz') as handle:
            for item in handle:
                if item.name==member:
                    raw = handle.extractfile(item).read()
                    break
            else:
                raise FileNotFoundError(member)
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    return json.loads(raw), dict(archive=str(archive),member=member,bytes=entry['bytes'],sha256=entry['sha256'],
                                 retained='Explicit scalar metric projection only; original full JSON remains in verified archive.')


def score(run, workspace):
    start=time.monotonic()
    seal=load(run/'CHOICES_SEALED415.json')
    assert not seal['final_result_metrics_read']
    admissions=load(run/'inputs/a414/ADMISSION_PLAN414.json')
    sources={}
    for name in ['results/safety314/NUMERICAL_RESULTS314.json','results/target413/RESULT413.json']:
        f=workspace/name
        if name in admissions['sources']:
            assert sha(f)==admissions['sources'][name]['sha256']
        else:
            assert name.startswith('results/target413/')
            archive=Path(load(workspace/'results/target413/ARCHIVE_VERIFIED.json')['archive_path'])
            entry=next(x for x in load(archive/'SOURCE_MANIFEST.json')['files'] if x['path']=='RESULT413.json')
            assert sha(f)==entry['sha256']
        sources[name]=dict(path=name,sha256=sha(f))
        save(run/'source'/f.name,load(f))
    rotor=next(r for r in load(workspace/'results/safety314/NUMERICAL_RESULTS314.json')['rows'] if r['cohort']=='r')
    cwru=load(workspace/'results/target413/RESULT413.json')
    pair_results=[]
    for row in seal['pairs']:
        if row['source_cluster']=='rotor':
            metrics={'reference':rotor['actual_starting_reference'],'Agent':rotor['final']}
            quality={name:dict(accuracy=m['accuracy'],macro_f1=m['macro_f1'], n=m['n'],
                               record_groups=m.get('groups')) for name,m in metrics.items()}
        else:
            quality={name:dict(accuracy=cwru['models'][key]['window_correct']/cwru['models'][key]['window_total'],
                               n=cwru['models'][key]['window_total'],record_accuracy=cwru['models'][key]['record_correct']/cwru['models'][key]['record_total'])
                     for name,key in [('reference','mlp'),('Agent','ratio')]}
        assert all(v['n']==row['n'] for v in quality.values())
        selections=dict(ordinary='Agent',keep_original='reference',**row['selected'])
        outcomes={k:quality[v]['accuracy'] for k,v in selections.items()}
        regret={k:max(v['accuracy'] for v in quality.values())-acc for k,acc in outcomes.items()}
        ref=quality['reference']['accuracy']
        harmful={k:max(0.,ref-v) for k,v in outcomes.items()}
        opportunity=max(v['accuracy'] for v in quality.values())-ref
        utilization={k:(max(0.,v-ref)/opportunity if opportunity>1e-12 else None) for k,v in outcomes.items()}
        pair_results.append(dict(id=row['id'],source_cluster=row['source_cluster'],quality=quality,selections=selections,
                                 outcomes=outcomes,selection_loss=regret,harmful_change_loss=harmful,
                                 available_gain=opportunity,gain_capture=utilization,
                                 original_metrics_inherited_from_verified_archives=True))
    full,binding=read_bound_result(Path('/data/yyf-zyq/PHM_20261010/20261023T002126Z_ensemble298'),'ENSEMBLE_RESULTS.json')
    # The 115 MB original JSON is verified in memory; only auditable scalar projections are retained.
    original=full.get('cases',full.get('rows'))
    assert isinstance(original,list)
    bank_results=[]
    projections=[]
    for row in seal['banks']:
        old=next(r for r in original if r['stage']==row['stage'] and r['id']==row['case'])
        assert len(row['candidates'])==len(old['candidates'])
        byid={r['id']:r for r in old['candidates']}
        assert old['chosen']['id']==row['ordinary_choice']
        for candidate in row['candidates']:
            assert candidate['weights']==byid[candidate['id']]['weights']
        projection=dict(stage=row['stage'],case=row['case'],candidates=[dict(id=r['id'],weights=r['weights'],final=r['final']) for r in old['candidates']])
        projections.append(projection)
        selections=dict(ordinary=row['ordinary_choice'],**row['selected'],
                        fixed_CV=old['fixed_selectors_same_actual_pool']['cv'],
                        fixed_development=old['fixed_selectors_same_actual_pool']['development'])
        outcomes={k:dict(id=v,weights=byid[v]['weights'],accuracy=byid[v]['final']['accuracy'],macro_f1=byid[v]['final']['macro_f1']) for k,v in selections.items()}
        oracle=max(r['final']['macro_f1'] for r in old['candidates'])
        regrets={k:oracle-v['macro_f1'] for k,v in outcomes.items()}
        bank_results.append(dict(stage=row['stage'],case=row['case'],outcomes=outcomes,macro_f1_selection_loss=regrets,
                                 candidates=len(byid),pool_oracle_macro_f1=oracle,
                                 actual_original_reference=None,gain_capture=None,harmful_change_loss=None))
    save(run/'SOURCE_METRIC_PROJECTION415.json',dict(binding=binding,cases=projections))
    aggregate={}
    for phase in ['search','review']:
        rows=[r for r in bank_results if r['stage']==phase]
        aggregate[phase]={key:dict(mean_F1=float(np.mean([r['outcomes'][key]['macro_f1'] for r in rows])),
                                   mean_accuracy=float(np.mean([r['outcomes'][key]['accuracy'] for r in rows])),
                                   mean_selection_loss=float(np.mean([r['macro_f1_selection_loss'][key] for r in rows])))
                          for key in rows[0]['outcomes']}
        ordinary=[r['outcomes']['ordinary']['macro_f1'] for r in rows]
        for key in aggregate[phase]:
            diffs=[r['outcomes'][key]['macro_f1']-o for r,o in zip(rows,ordinary)]
            aggregate[phase][key]['win_tie_loss']=[sum(d>1e-12 for d in diffs),sum(abs(d)<=1e-12 for d in diffs),sum(d < -1e-12 for d in diffs)]
    result=dict(utc=utc(),seal_sha256=sha(run/'CHOICES_SEALED415.json'),pairs=pair_results,banks=bank_results,
                bank_aggregate=aggregate,source_metric_bindings=sources,source_full_result_read_bytes=binding['bytes'],
                computed_predictions='Only deterministic arithmetic on saved real probabilities; no new model inference.',
                new_LLM=0,new_fits=0,new_model_forwards=0,new_independent_scenarios=0,confirmation_opened=False,
                seconds=time.monotonic()-start,majority_or_method_success=False,
                scope='Exploratory transductive controls on eligibility-limited old corpus, 2 paired source labels + 1 separate bank source. No new Agent session, no full original-task safety/cost score.')
    assert result['seconds']+seal['seconds']<60
    save(run/'RESULT415.json',result)
    print(json.dumps(dict(pair_outcomes=[dict(id=r['id'],outcomes=r['outcomes'],gain_capture=r['gain_capture']) for r in pair_results],bank_aggregate=aggregate,seconds=result['seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['choose','score'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,default=Path.cwd())
    args=parser.parse_args()
    if args.mode=='choose':
        choose(args.run)
    else:
        score(args.run,args.workspace)
