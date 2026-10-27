"""Offline native-request capture and real HTTP gateway budget propagation.

Only protocol fixtures are synthesized. No real model, credential, PHM data,
prediction or benchmark score is used. These are infrastructure checks.
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import isolated_model_gateway as gateway
from acquire_forge_pdmb_sources244 import save
from prepare_forge_coding_baseline import CLI,PYTHON,RUNTIME
from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env


def unix_post(path,payload):
    client=http.client.HTTPConnection('localhost',timeout=20)
    client.sock=socket.socket(socket.AF_UNIX);client.sock.connect(str(path))
    client.request('POST','/v1/responses',json.dumps(payload),{'Content-Type':'application/json'})
    response=client.getresponse();code=response.status;response.read();client.close();return code


def main(run,template,attempt=1):
    root=run/'audit'/('offline_output_budget_probe_'+str(attempt));root.mkdir()
    captured=[]
    class Capture(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body=b'{"error":{"message":"Offline request capture only; no model was contacted"}}'
            self.send_response(400);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    source_socket=run/('probe_native_'+str(attempt)+'.sock')
    server=gateway.UnixHTTPServer(str(source_socket),Capture)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config=run/('offline_budget_config_'+str(attempt)+'.toml');config.write_text((template/'cli_config187.toml').read_text().split('[mcp_servers.phmforge]')[0])
    catalog=run/('offline_budget_catalog_'+str(attempt)+'.json');shutil.copy2(template/'model_catalog187.json',catalog)
    jobs=json.loads((template/'PLAN187.json').read_text())['jobs']
    declared=json.loads(Path(jobs[0]['mounts']).read_text())['mounts']
    tool_mounts=[m for m in declared if m['destination']=='/tools']
    assert len(tool_mounts)==1 and Path(tool_mounts[0]['source']).is_dir()
    mounts=run/('offline_budget_mounts_'+str(attempt)+'.json');save(mounts,{'mounts':[
        {'source':str(PYTHON),'destination':'/opt/python312'},
        {'source':str(RUNTIME/'deps'),'destination':'/opt/phm_python/extra'},
        tool_mounts[0]]})
    app=root/'app';app.mkdir();folder=root/('native_cli_'+str(attempt));folder.mkdir()
    for name in ['.runtime/tmp','.runtime/mpl']:(app/name).mkdir(parents=True)
    cli=Path(json.loads((template/'PROTOCOL187.json').read_text()).get('cli',str(CLI)))
    ns=SimpleNamespace(cli=cli,model_gateway_socket=source_socket,runtime_mounts_file=mounts,
        credential_env_file=None,isolation_config=config,model_catalog=catalog,permission_profile='iot-offline',workspace_alias='/app')
    inner=[str(cli),'exec','--json','--skip-git-repo-check','--model','deepseek-flash','-C','/app','-']
    command=isolate_filesystem(inner,ns,app,folder,None);save(root/'native_command.json',command)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
        TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
    result=subprocess.run(command,input='This is an offline request-format check. Return READY; do not execute tools.',
        text=True,capture_output=True,env=env,timeout=45)
    (root/'native_stdout').write_text(result.stdout);(root/'native_stderr').write_text(result.stderr)
    server.shutdown();server.server_close();thread.join();source_socket.unlink()
    assert len(captured)==1,'Native CLI did not emit exactly one expected request'
    save(root/'native_client_request.json',captured[0])
    native_limit=captured[0].get('max_output_tokens')
    assert native_limit is None,'Native CLI explicitly supplies an additional limit; investigate before launching'
    received=[]
    class Provider(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body=json.dumps({'model':'deepseek-flash','status':'completed','usage':{'input_tokens':0,'output_tokens':0},'output':[]}).encode()+b'\n'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    provider=ThreadingHTTPServer(('127.0.0.1',0),Provider)
    provider_thread=threading.Thread(target=provider.serve_forever,daemon=True);provider_thread.start()
    # The real credential reader is replaced before the local gateway starts.
    original_reader=gateway.credential;gateway.credential=lambda path:'protocol-fixture-not-a-real-credential'
    tests=[]
    try:
        for mode,total in [('per_request',262144),('total_budget',4096)]:
            directory=root/mode;directory.mkdir();route=run/('probe_'+str(attempt)+'_'+mode+'.sock')
            save(directory/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job='offline_fixture',socket=str(route))],
                base_url='http://127.0.0.1:'+str(provider.server_port),max_requests=8,max_input_tokens=10000,
                max_output_tokens=total,max_output_per_request=32768))
            worker=threading.Thread(target=gateway.gateway,args=(directory,),daemon=True);worker.start()
            for _ in range(100):
                if (directory/'MODEL_GATEWAY_READY.json').exists():break
                time.sleep(.02)
            else:raise TimeoutError('Offline gateway not ready')
            cases=[('omitted',{},min(32768,total))]
            if mode=='per_request':cases += [('lower',{'max_output_tokens':8192},8192),('higher',{'max_output_tokens':65536},32768),('null',{'max_output_tokens':None},32768)]
            try:
                for name,extra,expected in cases:
                    payload=dict(model='deepseek-flash',input='Protocolfixture,noPHMdata',stream=False,**extra)
                    assert unix_post(route,payload)==200
                    actual=received[-1]['max_output_tokens'];assert actual==expected
                    tests.append(dict(mode=mode,case=name,expected=expected,actual=actual,passed=True))
            finally:
                (directory/'MODEL_GATEWAY_STOP').touch();worker.join(timeout=10);assert not worker.is_alive()
                if route.exists():route.unlink()
    finally:
        gateway.credential=original_reader;provider.shutdown();provider.server_close();provider_thread.join()
    report=dict(passed=all(x['passed'] for x in tests),native_cli_request_omits_output_limit=native_limit is None,
        native_probe_exit_code=result.returncode,native_probe_expected_HTTP400=True,tests=tests,
        real_model_calls=0,real_credentials_read=False,PHM_data_used=False,benchmark_result=False,
        interpretation='Native CLI omits max_output_tokens; gateway now forwards declared32768 rather than old8192default,while preservingexplicitlowerlimits and remainingtotalbudget')
    save(run/'audit/OUTPUT_BUDGET_QUALIFICATION.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
    p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();main(a.run,a.template,a.attempt)
