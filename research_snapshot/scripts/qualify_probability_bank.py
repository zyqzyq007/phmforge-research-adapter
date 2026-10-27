"""Check source rejection and genuine historical cold outputs without fitting.

The replay is restricted by the run protocol to one saved development model.
Neither test arrays nor new recipes/seeds/weights are searched here.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np

from controlled_feedback import save,sha,utc,transform
from evaluate_controlled_feedback import cold,independent
from prepare_forge_coding_baseline import PYTHON,RUNTIME
from run_full_benchmark_campaign import clean_env


def qualify(root):
    config=json.loads((root/'PROTOCOL293.json').read_text())
    source=Path(config['historical_source_archive']);manifest=source/'SOURCE_MANIFEST.json'
    assert sha(manifest)==config['source_manifest_sha256']
    index={r['path']:r for r in json.loads(manifest.read_text())['files']};bindings=[]
    for relative in config['source_files']:
        path=source/relative;entry=index[relative]
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        bindings.append(dict(path=str(path),bytes=entry['bytes'],sha256=entry['sha256']))
    def function(path,name):
        return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
    scripts=Path(__file__).parent
    assert function(source/'code/controlled_feedback.py','transform')==function(scripts/'controlled_feedback.py','transform')
    assert sha(source/'workspace_snapshot/scripts/prepare_fault_extensions.py')==sha(scripts/'prepare_fault_extensions.py')
    save(root/'audit/SOURCE_BINDINGS293.json',dict(utc=utc(),files=bindings,transform_ast_equal=True,feature_module_bytes_equal=True))
    for name in ['controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(scripts/name,root/'shared_tools'/name)
    env=clean_env(root);env.update(PYTHONPATH=str(RUNTIME/'deps')+':'+str(scripts),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[str(PYTHON/'bin/python3.12'),'-B',str(scripts/'prepare_probability_bank.py'),'--run',str(root/'unadmitted_source')]
    denied=subprocess.run(command,env=env,text=True,capture_output=True,timeout=45)
    (root/'logs/unadmitted_source.stdout').write_text(denied.stdout);(root/'logs/unadmitted_source.stderr').write_text(denied.stderr)
    save(root/'audit/SOURCE_REJECTION_COMMAND293.json',command)
    checks=dict(actual_unadmitted_source_rejected=denied.returncode!=0 and 'Source gate incomplete' in denied.stderr,
        no_fitting_start_marker=not (root/'unadmitted_source/BASIS_FITTING_STARTED.json').exists(),
        no_fitting_directory=not (root/'unadmitted_source/basis').exists())
    data=np.load(source/config['development_file'],allow_pickle=False)
    old=np.load(source/config['historical_predictions'],allow_pickle=False)
    report=json.loads((source/config['historical_fit_report']).read_text())
    checkpoint=source/config['historical_checkpoint']
    observations=root/'evaluation_inputs/development';observations.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(observations/'observations.npz',waveforms=data['waveforms'],sample_ids=data['sample_ids'])
    before=sha(checkpoint);model=joblib.load(checkpoint)
    actual=model['model'].predict_proba(transform(data['waveforms'],model['recipe']['features']))
    # One actual host prediction above; two real isolated inference processes below.
    probability=cold(root,'cold_probability_fixture','lda','development',checkpoint=checkpoint,probability_output=True)
    local_model=root/'legacy_cold_fixture/broker/lda/checkpoint.joblib';local_model.parent.mkdir(parents=True)
    shutil.copy2(checkpoint,local_model)
    hard=cold(root,'legacy_cold_fixture','lda','development')
    classes=model['model'].classes_.tolist();ids=data['sample_ids'].tolist()
    checks.update(cold_classes_equal=probability['classes']==classes,
        cold_ids_equal=probability['sample_ids']==ids,
        cold_native_probabilities_equal=bool(np.allclose(actual,probability['probabilities'],rtol=0,atol=1e-12)),
        original_development_predictions_equal=bool(np.array_equal(np.asarray(classes)[actual.argmax(axis=1)],old['development'])),
        legacy_cold_output_preserved=[x['sample_id'] for x in hard]==ids and np.array_equal([x['predicted_label'] for x in hard],old['development']),
        source_checkpoint_unchanged=sha(checkpoint)==before==report['checkpoint_sha256'])
    computed=independent(data['labels'],np.asarray(classes)[actual.argmax(axis=1)],classes)
    checks['original_development_metrics_equal']=computed['confusion']==report['development']['confusion'] and abs(computed['macro_f1']-report['development']['macro_f1'])<1e-12
    assert all(sha(Path(x['path']))==x['sha256'] for x in bindings)
    for name in ['prepare_probability_bank.py','qualify_probability_bank.py','evaluate_controlled_feedback.py']:
        shutil.copy2(scripts/name,root/'code'/name)
    result=dict(passed=bool(all(checks.values())),utc=utc(),checks={k:bool(v) for k,v in checks.items()},
        genuine_historical_development_observations=len(ids),isolated_inference_processes=2,host_inference_passes=1,
        new_validation_inference_passes=3,new_fits=0,new_LLM_calls=0,new_Agent_cases=0,new_final_scores=0,
        historical_development_metrics=computed,G2_real_probability_fitting_qualified=False,
        scope='Infrastructure source-rejection and fixed old development replay only;no new overfitting evidence.')
    save(root/'RESULT293.json',result);print(json.dumps(result));assert result['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();qualify(args.run.resolve())
