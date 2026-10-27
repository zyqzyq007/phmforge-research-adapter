"""Check the MCP catalog in a real CLI request without contacting any model.

The private HTTP endpoint only captures requests and returns an explicit 400.
This is an infrastructure test, never a simulated PHM/Agent result.
"""
import argparse
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

from acquire_forge_pdmb_sources244 import save
from isolated_model_gateway import UnixHTTPServer
from prepare_forge_coding_baseline import CLI
from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env


MCP_ENV = '''required = true
env = { PYTHONPATH = "/opt/phm_python/extra", PYTHONDONTWRITEBYTECODE = "1", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", MPLCONFIGDIR = "/app/.runtime/mpl" }
'''


def mcp_config(text, environment=True):
    assert text.count('[mcp_servers.phmforge]') == 1
    addition = MCP_ENV if environment else 'required = true\n'
    before,block=text.split('[mcp_servers.phmforge]\n',1)
    block='\n'.join(line for line in block.splitlines()
        if not line.startswith(('required = ','env = ')))+'\n'
    return before+'[mcp_servers.phmforge]\n'+addition+block


def probe(run, template, job, name, config_text):
    root=run/'audit/native_mcp'/job['id']/name
    root.mkdir(parents=True, exist_ok=False)
    config=root/'config.toml';config.write_text(config_text)
    app=root/'app';app.mkdir()
    for relative in ['.runtime/tmp','.runtime/mpl','artifacts']:
        (app/relative).mkdir(parents=True)
    cache=run/'clients-cache'/('native_mcp_'+job['id']+'_'+name)
    cache.mkdir(parents=True)
    captured=[]
    class Capture(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if name=='fixed_discovery' and len(captured)==1:
                item=dict(type='tool_search_call',id='ts_offline_probe',call_id='offline_search',
                    execution='client',status='completed',arguments=dict(query='phmforge',limit=30))
                base=dict(id='resp_offline_probe',object='response',model='deepseek-flash',
                    created_at=0,status='completed',output=[item],
                    usage=dict(input_tokens=0,output_tokens=0,total_tokens=0))
                events=[dict(type='response.created',response={**base,'status':'in_progress','output':[]}),
                    dict(type='response.output_item.added',output_index=0,item=item),
                    dict(type='response.output_item.done',output_index=0,item=item),
                    dict(type='response.completed',response=base)]
                body=''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode()
                self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                return
            body=b'{"error":{"message":"Offline MCP registration probe; no model contacted"}}'
            self.send_response(400);self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
    socket_path=run/'ipc'/('mcp'+str(len(list((run/'audit/native_mcp').glob('*/*'))))+'.sock')
    socket_path.parent.mkdir(parents=True,exist_ok=True)
    server=UnixHTTPServer(str(socket_path),Capture)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    cli=Path(json.loads((template/'PROTOCOL187.json').read_text()).get('cli',str(CLI)))
    ns=SimpleNamespace(cli=cli,model_gateway_socket=socket_path,
        runtime_mounts_file=Path(job['mounts']),credential_env_file=None,
        isolation_config=config,model_catalog=template/'model_catalog187.json',
        permission_profile='iot-offline',workspace_alias='/app')
    command=isolate_filesystem([str(cli),'exec','--json','--skip-git-repo-check',
        '--model','deepseek-flash','-C','/app','-'],ns,app,cache,None)
    save(root/'command.json',command)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',
        PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',
        TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
    try:
        result=subprocess.run(command,input='Offline tool registration probe. Return READY.',
            text=True,capture_output=True,env=env,timeout=90)
    finally:
        server.shutdown();server.server_close();thread.join();socket_path.unlink()
    (root/'stdout').write_text(result.stdout);(root/'stderr').write_text(result.stderr)
    names=[]
    def collect(tools,prefix=''):
        for tool in tools:
            if tool.get('type')=='namespace':collect(tool.get('tools',[]),tool['name']+'__')
            else:names.append(prefix+tool.get('name',tool.get('function',{}).get('name',tool.get('type'))))
    for index,payload in enumerate(captured,1):
        save(root/('request_'+str(index)+'.json'),payload)
        collect(payload.get('tools',[]))
        for value in payload.get('input',[]):
            if isinstance(value,dict) and value.get('type') in ['tool_search_output','additional_tools']:
                collect(value.get('tools',[]))
    catalog=[x for x in names if isinstance(x,str) and x.startswith('mcp__phmforge__')]
    expected=json.loads((template/'PROTOCOL187.json').read_text()).get('expected_mcp_tools')
    catalog_passed=(set(catalog)==set(expected)) if expected is not None else len(set(catalog))==22
    report=dict(job=job['id'],condition=name,cli_exit_code=result.returncode,
        offline_HTTP400_expected=True,requests_captured=len(captured),tool_names=names,
        phmforge_tools=catalog,expected_mcp_tools=expected,passed=catalog_passed,
        real_model_calls=0,real_credentials_read=False,benchmark_result=False)
    save(root/'RESULT.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--jobs',nargs='+')
    p.add_argument('--conditions',nargs='+',choices=['original','required_only','fixed','fixed_static','fixed_discovery'],
                   default=['original','required_only','fixed'])
    p.add_argument('--report',default='NATIVE_CLI_MCP_QUALIFICATION.json')
    a=p.parse_args();run=a.run.resolve();template=a.template.resolve()
    jobs=json.loads((template/'PLAN187.json').read_text())['jobs']
    if a.jobs:jobs=[j for j in jobs if j['id'] in a.jobs];assert len(jobs)==len(a.jobs)
    original=(template/'cli_config187.toml').read_text();results=[]
    for job in jobs:
        for condition in a.conditions:
            text=original if condition=='original' else mcp_config(original,condition in ['fixed','fixed_static','fixed_discovery'])
            if condition=='fixed_static':
                text=text.replace('[features]\n','[features]\ntool_search = false\n')
            row=probe(run,template,job,condition,text);results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='tool_names'}),flush=True)
    report=dict(passed=all(r['passed'] for r in results if r['condition'] in ['fixed','fixed_static','fixed_discovery']),
        real_model_calls=0,real_credentials_read=False,benchmark_result=False,
        fixed_passed=all(r['passed'] for r in results if r['condition'] in ['fixed','fixed_static','fixed_discovery']),rows=results)
    save(run/'audit'/a.report,report)
    return 0 if report['fixed_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
