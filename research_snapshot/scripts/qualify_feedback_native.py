"""Execute real train-only fitting through the actual native CLI command tool.

The local response fixture requests a command; it does not simulate PHM data,
fitting, predictions, scores or a model result. No provider credentials are read.
"""
import argparse
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

from controlled_feedback import save, utc
from isolated_model_gateway import UnixHTTPServer
from prepare_forge_coding_baseline import CLI, PYTHON
from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env
from run_controlled_feedback import numerical_env


def qualify(root,ensemble_bank=None,source_template=None,fixture_name=None,ensemble_stage='search',prior_pool=None,gate_file=None):
    assert ensemble_stage in ['search','review']
    assert ensemble_stage=='search' or (ensemble_bank and prior_pool)
    name=fixture_name or ('native_ensemble_transport' if ensemble_bank else 'native_feedback_fit')
    assert Path(name).name==name and name not in ['.','..']
    fixture=root/'audit'/name;fixture.mkdir(exist_ok=False)
    app=fixture/'app'
    for name in ['.runtime/tmp','.runtime/mpl','.runtime/feedback_requests']:(app/name).mkdir(parents=True)
    for name in ['feedback','broker','runtime/tmp','runtime/mpl','logs']:(fixture/name).mkdir(parents=True)
    config=dict(condition='delayed',train=str(root/'audit/qualification_train.npz'),
        development=str(root/'audit/qualification_dev.npz'),request_directory='app/.runtime/feedback_requests')
    if ensemble_bank:
        config=dict(condition='delayed',stage=ensemble_stage,bank=str(ensemble_bank),candidate_budget=6 if ensemble_stage=='review' else 1,weight_denominator=12,
                    request_directory='app/.runtime/feedback_requests')
        if ensemble_stage=='review':
            destination=fixture/'references/prior_pool.json';destination.parent.mkdir()
            shutil.copy2(prior_pool,destination);config['prior_pool']='references/prior_pool.json'
    elif (root/'FITTING_SETTINGS.json').exists():
        config.update(fitting_settings=str(root/'FITTING_SETTINGS.json'),publish_validation_evidence=True)
    save(fixture/'BROKER_CONFIG.json',config)
    script=Path(__file__).with_name('controlled_ensemble_feedback.py' if ensemble_bank else 'controlled_feedback.py')
    log=(fixture/'logs/broker.log').open('wb')
    broker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B',str(script),'serve','--run',str(fixture)],env=numerical_env(fixture),stdout=log,stderr=log)
    if source_template:source=Path(source_template)
    else:
        campaign=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());source=root/campaign['cases'][0]['id']
    job=json.loads((source/'PLAN187.json').read_text())['jobs'][0]
    mount_path=Path(job['mounts'])
    if source_template and not mount_path.exists():mount_path=source/'plans'/mount_path.name
    mounts=json.loads(mount_path.read_text())
    mounts['mounts']=[m for m in mounts['mounts'] if m['destination'] in ['/opt/python312','/opt/phm_python/extra','/tools']]
    if ensemble_bank:
        for mount in mounts['mounts']:
            if mount['destination']=='/tools':mount['source']=str(root/'code')
    mounts['mounts'].append(dict(source=str(fixture/'feedback'),destination='/feedback'))
    save(fixture/'mounts.json',mounts)
    (fixture/'config.toml').write_text((source/'cli_config187.toml').read_text().split('[mcp_servers.phmforge]')[0])
    if ensemble_bank:
        operation=(dict(action='select',candidate_id='base_3',reason='Predetermined numeric transport fixture;not an Agent choice.')
                   if ensemble_stage=='review' else dict(action='evaluate',vectors=[[2]*6]))
        request=json.dumps(operation,separators=(',',':'))
        cmd="python3 /tools/controlled_ensemble_feedback.py client --request '"+request+"'"
    else:
        recipe=json.loads((root/'audit/qualification_recipe.json').read_text())
        request=json.dumps(dict(action='fit',recipe=recipe),separators=(',',':'))
        cmd="python3 /tools/controlled_feedback.py client --request '"+request+"'"
    captured=[]
    class Capture(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(captured)==1:
                item=dict(type='function_call',id='fc_feedback_fixture',call_id='call_feedback_fixture',name='exec_command',
                    arguments=json.dumps(dict(cmd=cmd,yield_time_ms=30000,max_output_tokens=6000)),status='completed')
                base=dict(id='resp_feedback_fixture',object='response',model='deepseek-flash',created_at=0,status='completed',output=[item],
                    usage=dict(input_tokens=0,output_tokens=0,total_tokens=0))
                events=[dict(type='response.created',response={**base,'status':'in_progress','output':[]}),
                    dict(type='response.output_item.added',output_index=0,item=item),dict(type='response.output_item.done',output_index=0,item=item),
                    dict(type='response.completed',response=base)]
                body=''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode();self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
            else:
                body=b'{"error":{"message":"Offline transport qualification complete; no model contacted"}}';self.send_response(400)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    socket=root/'native_feedback_probe.sock';server=UnixHTTPServer(str(socket),Capture)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        for _ in range(100):
            if (fixture/'BROKER_READY.json').exists():break
            assert broker.poll() is None
            time.sleep(.05)
        ns=SimpleNamespace(cli=CLI,model_gateway_socket=socket,runtime_mounts_file=fixture/'mounts.json',credential_env_file=None,
            isolation_config=fixture/'config.toml',model_catalog=source/'model_catalog187.json',permission_profile='iot-offline',workspace_alias='/app')
        out=fixture/'native';out.mkdir()
        command=isolate_filesystem([str(CLI),'exec','--json','--skip-git-repo-check','--model','deepseek-flash','-C','/app','-'],ns,app,out,None)
        save(fixture/'command.json',command)
        env=clean_env(fixture);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
        result=subprocess.run(command,input='Offline infrastructure qualification. Execute the requested real numeric operation.',env=env,text=True,capture_output=True,timeout=210)
        (fixture/'stdout').write_text(result.stdout);(fixture/'stderr').write_text(result.stderr)
        for i,payload in enumerate(captured):save(fixture/f'request_{i+1}.json',payload)
        state=json.loads((fixture/'broker/STATE.json').read_text())
        report=json.loads((fixture/'broker/candidate_1/fit_report.json').read_text()) if state['trials'] and not ensemble_bank else {}
        outputs=[i.get('output','') for p in captured for i in p.get('input',[]) if isinstance(i,dict) and i.get('type')=='function_call_output']
        checks=dict(actual_native_tool_returned_fitting_result=any('"real_fits": 4' in str(x) and '"remaining": 5' in str(x) for x in outputs),
            one_candidate=len(state['trials'])==1,real_fit=report.get('real_fits')==4,cold_reload=report.get('cold_reload_equal') is True,
            development_feedback_withheld=all(not isinstance(t.get('development'),dict) for p in (fixture/'feedback/responses').glob('*.json') for t in json.loads(p.read_text()).get('result',{}).get('trials',[])),
            final_inputs_absent=all('/hidden/' not in str(v) for v in json.loads((fixture/'BROKER_CONFIG.json').read_text()).values()))
        if ensemble_bank:
            import numpy as np
            from controlled_feedback import metrics
            bank=np.load(ensemble_bank,allow_pickle=False)
            weights=np.asarray([0,0,12,0,0,0] if ensemble_stage=='review' else [2]*6)
            predicted=bank['classes'][(bank['development']*weights[None,:,None]).sum(axis=1).argmax(axis=1)]
            expected=metrics(bank['development_labels'],predicted,bank['classes'].tolist())
            chosen=next(x for x in state['trials'] if x['id']==('base_3' if ensemble_stage=='review' else 'ensemble_1'))
            checks=dict(actual_native_tool_returned_real_ensemble_feedback=any(
                    ('"ensemble_candidates": 0' in str(x) and '"remaining": 6' in str(x) and '"final_choice": {' in str(x))
                    if ensemble_stage=='review' else ('"ensemble_candidates": 1' in str(x) and '"remaining": 0' in str(x)) for x in outputs),
                exact_requested_weight=chosen['weights']==weights.tolist(),
                permitted_raw_development_available=(fixture/'feedback/development_probabilities.npz').exists(),
                no_model_fits=not (fixture/'broker/candidate_1').exists(),
                genuine_probability_score_equal=chosen['development']==expected,
                no_final_arrays_in_bank=not any('final' in k or 'test' in k for k in bank.files))
            if ensemble_stage=='review':
                checks['native_retention_without_new_candidate']=state['final_choice']['candidate_id']=='base_3' and not any(x['kind']=='ensemble' for x in state['trials'])
        if config.get('publish_validation_evidence'):
            evidence=fixture/'feedback/evidence/candidate_1'
            checks['row_evidence_real_and_withheld']=all((evidence/(part+'.json')).exists() for part in ['train','cv']) and not (evidence/'development.json').exists()
            checks['cv_condition_isolation']=all(not set(x['fitting_conditions'])&set(x['checking_conditions']) for x in report['cv_folds'])
        final=dict(passed=all(checks.values()),checks=checks,utc=utc(),new_real_qualification_estimator_fits=report.get('real_fits',0),
            real_model_calls=0,provider_credentials_read=False,fixture_response_is_not_a_model_or_benchmark_outcome=True,native_exit_code=result.returncode,ensemble_stage=ensemble_stage if ensemble_bank else None)
        filename=gate_file or ('NATIVE_ENSEMBLE_GATE290.json' if ensemble_bank else 'NATIVE_FEEDBACK_GATE282.json')
        assert Path(filename).name==filename
        save(fixture/'RESULT.json',final);save(root/filename,final)
        assert final['passed'],final
        print(json.dumps(final))
    finally:
        broker.terminate();broker.wait(timeout=30);log.close()
        server.shutdown();server.server_close();thread.join();socket.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--ensemble-bank',type=Path);p.add_argument('--source-template',type=Path)
    p.add_argument('--fixture-name')
    p.add_argument('--ensemble-stage',choices=['search','review'],default='search');p.add_argument('--prior-pool',type=Path)
    p.add_argument('--gate-file')
    a=p.parse_args();qualify(a.run.resolve(),a.ensemble_bank,a.source_template,a.fixture_name,a.ensemble_stage,a.prior_pool,a.gate_file)
