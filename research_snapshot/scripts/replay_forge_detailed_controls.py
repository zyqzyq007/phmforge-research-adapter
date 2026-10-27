"""Replay preserved A306 components; no retraining or new model choices."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import files, namespace, terminal
from run_full_benchmark_campaign import clean_env
from score_forge_detailed_tasks import detailed

ENTRY = '''import argparse,json
from pathlib import Path
import joblib,numpy as np
p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
with np.load(a.input,allow_pickle=False) as data:
 assert not set(data.files)&{'labels','groups','origin_labels'}
 ids=[str(x) for x in data['sample_ids']]
 if a.case=='h':
  import hust_pipeline as hp
  model=joblib.load('/app/artifacts/hust_classifier.joblib')
  features=hp.extract_features(data);scaled=model['scaler'].transform(features)
  values={'per_window_ensemble':np.asarray(model['class_names'])[hp.model_probabilities(model,features).argmax(1)],
   'full_LDA_component':model['full_model'].predict(scaled),
   'selected_LDA_component':model['selected_model'].predict(scaled[:,model['selected_indices']])}
  result={k:[dict(sample_id=i,predicted_label=int(v)) for i,v in zip(ids,y)] for k,y in values.items()}
 else:
  model=joblib.load('/reference/origin_reference.joblib')
  result={'provided_origin_RF':[dict(sample_id=i,predicted_origin=str(v)) for i,v in zip(ids,model.predict(data['features']))]}
Path(a.output).write_text(json.dumps(result)+'\\n')
'''


def main(root):
    registration=json.loads((root/'audit/ADDITIONAL_CONTROL_REGISTRATION306.json').read_text())
    assert registration['registered_before_final_scores']
    assert registration['script_sha256']==sha256(Path(__file__))
    for path, expected in registration['preserved_sources_sha256'].items():
        assert sha256(Path(path))==expected, 'A registered fitted source changed'
    assert (root/'DETAILED_RESULTS306.json').is_file()
    all_results=[]
    for case,name in [('h','HUST'),('p','PADERBORN')]:
        cohort=root/case;job,=terminal(cohort)
        workspace=cohort/job['workspace'];frozen=files(workspace)
        assert frozen==json.loads((cohort/'CODING_CHOICE_AUDIT.json').read_text())['rows'][0]['files']
        dest=cohort/'component_replay';dest.mkdir(exist_ok=False)
        app=dest/'app';shutil.copytree(workspace,app,symlinks=True)
        (app/'component_entry.py').write_text(ENTRY)
        for split in ['val','test']:
            output=dest/split;output.mkdir()
            query=dict(job)
            if split=='val':query['evaluation_mounts']=str(cohort/'detailed_evaluation/development/mounts.json')
            argv=namespace(cohort,query,app,output,['/opt/python312/bin/python3.12','-B','/app/component_entry.py',
                '--case',case,'--input',f'/data/datasets/{name}/{split}.npz','--output','/output/controls.json'])
            save(output/'command.json',argv)
            env=clean_env(cohort);env.update(PYTHONPATH='/opt/phm_python/extra:/app',PYTHONDONTWRITEBYTECODE='1',
                TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
                MPLCONFIGDIR='/app/.runtime/mpl',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (output/'stdout').open('wb') as stdout,(output/'stderr').open('wb') as stderr:
                try:code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=180).returncode
                except subprocess.TimeoutExpired:code='timeout'
            row=dict(case=case,split=split,exit_code=code,new_fits=0,new_model_calls=0)
            try:
                assert code==0
                predictions=json.loads((output/'controls.json').read_text())
                if split=='test':truth=json.loads((cohort/'hidden'/name/'labels.json').read_text())
                else:
                    with np.load(cohort/'public_data'/job['id']/'datasets'/name/'val.npz',allow_pickle=False) as data:
                        truth=[dict(sample_id=str(ident),label=int(data['labels'][i]),group=str(data['groups'][i])) for i,ident in enumerate(data['sample_ids'])]
                        for i,t in enumerate(truth):
                            for field in ['bearing_type','load_w','shaft_frequency_hz']:
                                if field in data:t[field]=data[field][i].item()
                            if 'origin_labels' in data:t['origin_label']=str(data['origin_labels'][i])
                if case=='p':
                    # This comparator scores origin independently; fault predictions
                    # are deliberately omitted, so no invented joint score is emitted.
                    pred=predictions['provided_origin_RF'];index={x['sample_id']:x for x in truth}
                    assert len({x['sample_id'] for x in pred})==len(pred)==len(index)
                    assert {x['sample_id'] for x in pred}==set(index)
                    hits=[x['predicted_origin']==index[x['sample_id']]['origin_label'] for x in pred]
                    damaged=[v for x,v in zip(pred,hits) if index[x['sample_id']]['label']!=0]
                    row.update(valid=True,n=len(pred),origin_accuracy=float(np.mean(hits)),damaged_only_origin_accuracy=float(np.mean(damaged)))
                else:row.update(valid=True,controls={k:detailed(v,truth,case) for k,v in predictions.items()})
                row['prediction_sha256']=sha256(output/'controls.json')
            except (AssertionError,KeyError,TypeError,ValueError,OSError) as error:
                row.update(valid=False,error_type=type(error).__name__,error=str(error))
            save(output/'SCORE.json',row);all_results.append(row)
        assert files(workspace)==frozen
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=all_results,new_fits=0,new_model_calls=0,
        scope='Preserved fitted components and provided reference; no new independent Agent cases.')
    save(root/'ADDITIONAL_CONTROLS306.json',result)
    save(Path(__file__).resolve().parents[1]/'results/coverage306/ADDITIONAL_CONTROLS306.json',result)
    print(json.dumps([dict(case=r['case'],split=r['split'],valid=r['valid']) for r in all_results]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
