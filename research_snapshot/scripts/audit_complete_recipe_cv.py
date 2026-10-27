"""Independent stdlib recount of the frozen full-recipe validation measurement."""
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def score(rows,window):
    matrix=[[0]*7 for _ in range(7)]
    for r in rows:matrix[r['label']][r['prediction']]+=r['windows'] if window else 1
    total=sum(map(sum,matrix));correct=sum(matrix[i][i] for i in range(7));f1=[]
    for i in range(7):
        den=sum(matrix[i])+sum(row[i] for row in matrix)
        f1.append(Fraction(2*matrix[i][i],den) if den else Fraction(0))
    return dict(total=total,correct=correct,errors=total-correct,error=float(Fraction(total-correct,total)),macro_F1_fixed7=float(sum(f1)/7),confusion_matrix=matrix)


def main(root):
    protocol=json.loads((root/'PROTOCOL390.json').read_text());freeze=json.loads((root/'FREEZE390.json').read_text())
    assert sha(root/'PROTOCOL390.json')==freeze['protocol_sha256']
    assert sha(root/'code/check_complete_recipe_cv.py')==freeze['code_sha256']
    assert json.loads((root/'TERMINAL390.json').read_text())['exit_code']==0
    assert sha(root/'EVIDENCE390.json')==json.loads((root/'EVIDENCE_SEAL390.json').read_text())['sha256']
    for b in protocol['bindings']:assert sha(b['source_path'])==b['sha256']
    e=json.loads((root/'EVIDENCE390.json').read_text());metrics={};checks=[]
    for side, v in e.items():
        rows=sorted(v['rows'],key=lambda r:r['record_index'])
        assert [r['record_index'] for r in rows]==list(range(57)) and len({r['signature'] for r in rows})==57
        assert sum(r['windows'] for r in rows)==11030 and v['classifier_fits']==3 and v['pipeline_predictions']==4
        assert v['checkpoint_replay_matches'] and v['original_scaler_mean_max_abs']<1e-12
        for l in v['lineage']:
            test=[r['record_index'] for r in rows if r['family']==l['held']]
            fit=[r['record_index'] for r in rows if r['family']!=l['held'] and (side=='before' or r['family'] in ['6205','6206'])]
            assert test==l['held_record_indices'] and fit==l['fit_record_indices']
            assert set(test).isdisjoint(fit) and len(fit)==l['scaler_n_samples']==l['fitting_records']
            assert l['scaler_mean_max_abs']<1e-12 and l['normalization_rebuild_differences']==[0.0,0.0] and l['coefficient_predict_agreement']
            checks.append(dict(side=side,held=l['held'],train_records=len(fit),held_records=len(test),passed=True))
        metrics[side]=dict(windows=score(rows,True),records=score(rows,False),by_family={f:dict(windows=score([r for r in rows if r['family']==f],True),records=score([r for r in rows if r['family']==f],False)) for f in ['6204','6205','6206']})
    b={r['record_index']:r for r in e['before']['rows']};a={r['record_index']:r for r in e['after']['rows']}
    fixes=[];breaks=[]
    for i in range(57):
        assert all(b[i][k]==a[i][k] for k in ['signature','family','load','windows','label'])
        if b[i]['prediction']!=b[i]['label'] and a[i]['prediction']==a[i]['label']:fixes.append(i)
        if b[i]['prediction']==b[i]['label'] and a[i]['prediction']!=a[i]['label']:breaks.append(i)
    source=json.loads(Path(protocol['inputs']['PROBE.json']).read_text())['public_scores']['val']
    dev={s:1-source[name]['samples']['accuracy'] for s,name in protocol['recipes'].items()}
    choice='after' if metrics['after']['windows']['errors']<=metrics['before']['windows']['errors'] and dev['after']<dev['before'] else 'before'
    result=dict(scope='One already-opened HUST training/development source; no new Agent or final evaluation',metrics=metrics,dev_error_reused=dev,paired=dict(fixed_records=fixes,broken_records=breaks,fixed_windows=sum(b[i]['windows'] for i in fixes),broken_windows=sum(b[i]['windows'] for i in breaks)),frozen_fixed_choice=choice,complete_CV_avoids_known_bad_update=(choice=='before'),new_classifier_fits=6,new_scaler_fits=6,new_pipeline_predictions=8,new_LLM_calls=0,new_final_predictions=0,seconds=json.loads((root/'TERMINAL390.json').read_text())['seconds'],novelty_established=False,independent_confirmation=False,majority_goal_complete=False)
    save(root/'RESULT390.json',result)
    save(root/'RECOUNT390.json',dict(passed=True,source_bindings=len(protocol['bindings']),record_pairs=57,lineage_checks=checks,full_train_predictions_replayed=22060,normalization_fold_batches=12,manual_coefficient_prediction_checks=6,method='Independent stdlib Fraction/confusion recount plus original worker coefficient and preprocessing identity checks; no new fitting',result_sha256=sha(root/'RESULT390.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ['metrics']}))
    for side in metrics:print(side,json.dumps({k:metrics[side][k] for k in ['windows','records']}))
    # This supplementary comparison was added after the primary measurement.
    # Keep its timing and identity separate; it never changes the frozen rule.
    p=Path('/data/yyf-zyq/PHM_20261010/20261022T023131Z_validation274/d/workspaces/s274_hust_geometry_r2_initial/validation.json')
    source=p.parents[3]
    index={v['path']:v for v in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']}
    entry=index[str(p.relative_to(source))];assert sha(p)==entry['sha256']
    reported=[]
    for family,v in json.loads(p.read_text())['training_family_holdout_metrics'].items():
        cm=v['confusion_matrix'];n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(7))
        assert n==v['rows'] and abs(correct/n-v['accuracy'])<1e-12
        reported.append(dict(family=family,n=n,correct=correct,errors=n-correct))
    reported_error=sum(v['errors'] for v in reported)/sum(v['n'] for v in reported)
    assert reported_error < metrics['before']['windows']['error'] < metrics['after']['windows']['error']
    save(root/'REPORTED_VS_COMPLETE390.json',dict(scope='Post-primary-recount comparison to already archived reported CV; no refitting, not an extra preregistered endpoint',source=str(p),sha256=entry['sha256'],reported_after_folds=reported,reported_after_error=reported_error,corrected_after_error=metrics['after']['windows']['error'],newly_measured_before_error=metrics['before']['windows']['error'],rank_with_reported_after='after',rank_with_complete_after='before',interpretation='Reported after CV reintroduced excluded 6204 records in two folds. Applying the original after fitting rule reverses this paired aggregate ranking. Before/after recipes also differ in normalization; this is not a pure cause attribution of the final 352-window harm.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
