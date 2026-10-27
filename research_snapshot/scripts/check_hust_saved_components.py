"""Localize a closed HUST submission using its exact saved components.

No retraining or new parameter selection; same-test exploratory diagnostics.
Prediction code never sees final labels. This does not confirm an Agent issue.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import subprocess

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import files, namespace, terminal
from run_full_benchmark_campaign import clean_env
from verified_benchmark_scoring import classification


INNER = r'''
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/app')
import solution as s
d=np.load('/data/datasets/HUST/test.npz',allow_pickle=False)
assert 'labels' not in d and 'groups' not in d
a=s.load_artifact();x=d['features'];w=d['waveforms']
original,groups=s.predict_observations(x,w,a)
assert min(groups)>=0
assignment,prob=s._group_profile_predictions(groups,x,a)
assert np.array_equal(original,assignment[groups])
modes={'original':original,'without_one_to_one_assignment':prob.argmax(1)[groups],
 'saved_per_window_fallback':s._sample_fallback(x,a)}
payload={'predictions':{mode:[{'sample_id':str(i),'predicted_label':int(a['class_names'][v])}
 for i,v in zip(d['sample_ids'],pred)] for mode,pred in modes.items()},
 'inferred_groups':[{'sample_id':str(i),'cluster':int(g)} for i,g in zip(d['sample_ids'],groups)],
 'cluster_probabilities':prob.tolist(),'class_names':a['class_names'].tolist(),
 'final_labels_read':False,'retraining':False}
Path('/output/SEALED_COMPONENT_PREDICTIONS.json').write_text(json.dumps(payload)+'\n')
'''


def main(run):
    jobs=terminal(run)
    score=json.loads((run/'CODING_SCORE.json').read_text())
    audit=json.loads((run/'CODING_CHOICE_AUDIT.json').read_text())
    candidates=[j for j in jobs if j['dataset']=='HUST' and
        any(x['job']==j['id'] and x['formal_submission'] for x in score['rows'])]
    assert len(candidates)==1
    job=candidates[0];workspace=run/job['workspace']
    expected=next(x['files'] for x in audit['rows'] if x['job']==job['id'])
    assert files(workspace)==expected
    root=run/'saved_component_diagnostic';root.mkdir()
    app=root/'app';output=root/'output';app.mkdir();output.mkdir()
    for n in ['.runtime/tmp','.runtime/mpl','artifacts']:(app/n).mkdir(parents=True,exist_ok=True)
    for name in ['solution.py','artifacts/hust_group_model.npz']:
        shutil.copy2(workspace/name,app/name)
    save(root/'PLAN.json',dict(selected_job=job['id'],modes=['original','without_one_to_one_assignment','saved_per_window_fallback'],
        reason='Separate source-group discovery,profile class mapping and saved per-window alternative',
        source_labels_in_predictor=False,new_model_calls=0,retraining=False,
        posthoc_same_test_not_confirmation=True,no_original_failure_rule_changed=True))
    command=namespace(run,job,app,output,['/opt/python312/bin/python3.12','-B','-c',INNER])
    save(root/'command.json',command)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (root/'stdout').open('wb') as out,(root/'stderr').open('wb') as err:
        result=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=180)
    assert result.returncode==0
    path=output/'SEALED_COMPONENT_PREDICTIONS.json';seal=sha256(path)
    saved=json.loads(path.read_text())
    # Final target access starts only after this label-free process exits.
    gold=json.loads((run/'hidden/HUST/labels.json').read_text());lookup={x['sample_id']:x for x in gold}
    measures={mode:classification(rows,gold) for mode,rows in saved['predictions'].items()}
    original=json.loads((run/'evaluation'/job['id']/'output/predictions.json').read_text())['predictions']
    assert {x['sample_id']:x['predicted_label'] for x in original}=={x['sample_id']:x['predicted_label'] for x in saved['predictions']['original']}
    clusters=defaultdict(Counter)
    for x in saved['inferred_groups']:clusters[x['cluster']][lookup[x['sample_id']]['group']]+=1
    purity=sum(max(v.values()) for v in clusters.values())/len(gold)
    group_metrics={}
    for mode,predictions in saved['predictions'].items():
        by_group=defaultdict(list)
        for x in predictions:
            truth=lookup[x['sample_id']];by_group[truth['group']].append(x['predicted_label']==truth['label'])
        group_metrics[mode]={k:dict(n=len(v),correct=sum(v),accuracy=sum(v)/len(v)) for k,v in sorted(by_group.items())}
    assert sha256(path)==seal
    report=dict(job=job['id'],same_test_exploratory_only=True,new_model_calls=0,new_parameters_fitted=False,
        prediction_seal=seal,original_prediction_reproduced=True,metrics=measures,per_group=group_metrics,
        inferred_cluster_to_source_group_counts={str(k):dict(v) for k,v in clusters.items()},
        inferred_group_purity=purity,automatic_Agent_problem_claim=False)
    save(run/'SAVED_COMPONENT_DIAGNOSTIC.json',report)
    print(json.dumps(dict(group_purity=purity,metrics={k:{m:v[m] for m in ['accuracy','macro_f1','balanced_accuracy']} for k,v in measures.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve())
