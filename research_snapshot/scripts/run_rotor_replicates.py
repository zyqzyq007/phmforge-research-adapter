"""Execute paired new-record coding/review subjects after real-training gates."""
import argparse
import json
import os
from pathlib import Path

from acquire_forge_pdmb_sources244 import save, sha256
from run_validation_transfer import campaign, prepare, execute_wave, state


def public_view(run):
    job,=json.loads((run/'PLAN187.json').read_text())['jobs']
    assert job['test_inputs_visible_during_development'] is False
    mounts=json.loads(Path(job['mounts']).read_text())['mounts']
    hashes={}
    for label in ['/data','/reference']:
        source=Path(next(x['source'] for x in mounts if x['destination']==label))
        hashes[label]={str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()}
        if label=='/data':assert not any(p.name=='test.npz' for p in source.rglob('*'))
    protocol=json.loads((run/'PROTOCOL187.json').read_text())
    assert protocol['cli_version_online']=='0.164.0'
    assert sha256(Path(protocol['cli']))==protocol['cli_sha256']
    solution=run/job['workspace']/'solution.py'
    return dict(input_files=hashes,solution=sha256(solution) if solution.is_file() else None,
                prompt=Path(job['prompt_file']).read_text(),CLI_sha256=protocol['cli_sha256'])


def main(root):
    config=campaign(root)
    assert not (root/'CAMPAIGN_STATUS.json').exists(), 'No automatic subject restart'
    receipt=json.loads((root/'PREPARATION305.json').read_text())
    assert receipt['passed'] and receipt['actual_neural_training_runs']==2
    assert receipt['actual_fixed_estimator_fits']==12 and receipt['actual_fold_estimator_fits']==36
    assert receipt['saved_fold_models_independently_replayed']==36
    assert receipt['protocol_sha256']==sha256(root/'CAMPAIGN_CONFIG.json')
    state(root,status='preparing_initial',supervisor_pid=os.getpid(),planned_model_sessions=8,
          physical_split_blocks=2,task_count=2,automatic_retries=False,hidden_scores_read=False,
          actual_preparation_neural_fits=2,actual_preparation_estimator_fits=48)
    first=[root/name for name in config['cohorts']]
    try:
        for run in first:prepare(root,run,'classification','initial')
        pairs=[]
        for a,b in [('a','b'),('c','d')]:
            av,bv=public_view(root/a),public_view(root/b)
            assert av['solution'] is not None and bv['solution'] is not None
            for view,arm in [(av,'ordinary'),(bv,'full_program')]:
                suffix=config['arm_prompt_addenda'][arm]
                assert view['prompt'].endswith(suffix)
                view['prompt']=view['prompt'][:-len(suffix)]
            assert av==bv, 'Unintended difference beyond validation instruction'
            pairs.append(dict(cohorts=[a,b],data_references_starting_weights_and_base_prompt_identical=True))
        gate=dict(passed=True,pairs=pairs,actual_new_neural_fits=2,actual_new_estimator_fits=48,
                  saved_fold_models_replayed=36,final_scores_opened=False)
        save(root/'audit/PAIRED_INPUT_GATE305.json',gate)
        save(Path(config['project'])/config['result_directory']/'PAIRED_INPUT_GATE305.json',gate)
        execute_wave(root,'initial',first)
        second=[root/(p.name+'_review') for p in first]
        for run,previous in zip(second,first):
            prepare(root,run,'classification','review',previous)
            assert public_view(run)['input_files']==public_view(previous)['input_files']
        execute_wave(root,'review',second)
        state(root,status='all8_subjects_terminal_waiting_choice_registration_and_external_scoring',
              completed_model_sessions=8,no_final_scores_opened=True,
              next_action='Register all actual initial/review choices and component contrasts,seal all files,then independently cold-evaluate.No test-informed fitting.')
    except BaseException as error:
        state(root,status='controller_error_inspect_before_any_resume',error_type=type(error).__name__,
              error=str(error),automatic_retries=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
