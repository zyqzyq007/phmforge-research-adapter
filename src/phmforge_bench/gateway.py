"""Budgeted Responses relay: API credentials stay outside the Agent namespace.

Each task receives one Unix socket bound to a single model and fixed upstream.
The inner TCP relay is only a transport for the native coding CLI. It cannot
fetch URLs, read host files, or select another model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)



class UnixHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True


def restrict_tools(tools, namespaces, names):
    """Restrict orchestration capabilities, keeping PHM/code tools unchanged."""
    kept=[];removed=[]
    for tool in tools:
        if tool.get('type')=='namespace' and tool.get('name') in namespaces:
            removed.append(tool['name']);continue
        if tool.get('name') in names:
            removed.append(tool['name']);continue
        if tool.get('type')=='namespace':
            child,dropped=restrict_tools(tool.get('tools',[]),namespaces,names)
            kept.append(dict(tool,tools=child));removed.extend(dropped)
        else:kept.append(tool)
    return kept,removed


def prohibited_calls(value,namespaces,names):
    if isinstance(value,list):return [x for v in value for x in prohibited_calls(v,namespaces,names)]
    if not isinstance(value,dict):return []
    if value.get('type') in ['function_call','custom_tool_call'] and (value.get('namespace') in namespaces or value.get('name') in names):
        return [dict(namespace=value.get('namespace'),name=value.get('name'))]
    return [x for v in value.values() for x in prohibited_calls(v,namespaces,names)]


def gateway(run):
    run = Path(run).resolve()
    config = json.loads((run/'MODEL_GATEWAY_CONFIG.json').read_text())
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise ValueError('DEEPSEEK_API_KEY is required in the gateway environment')
    upstream = config['base_url'].rstrip('/') + '/responses'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    servers = []
    for route in config['routes']:
        folder = run/'model_gateway'/route['job']
        folder.mkdir(parents=True, exist_ok=True)
        state = dict(job=route['job'], requests=0, input_tokens=0, output_tokens=0, inflight=False,
                     inflight_count=0, reserved_output_tokens=0, peak_inflight=0,
                     active=True, returned_model_ids=[], credential_in_agent=False)
        lock = threading.Lock()
        def make_handler(state, lock, folder):
            completed = threading.Condition(lock)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def reject(self, code, message):
                    body = json.dumps({'error': {'message': message}}).encode()
                    self.send_response(code)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    self.reject(403,'Only the pinned Responses endpoint is available')
                def do_POST(self):
                    if self.path not in ['/responses','/v1/responses']:
                        return self.reject(403,'Endpoint outside the task model capability')
                    length = int(self.headers.get('Content-Length','0'))
                    if not 0 < length <= 8*1024**2:
                        return self.reject(413,'Invalid model request size')
                    raw = self.rfile.read(length)
                    if key.encode() in raw:
                        return self.reject(403,'Credential-like material is not a task input')
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        return self.reject(400,'Invalid JSON')
                    if payload.get('model') != 'deepseek-flash':
                        return self.reject(403,'Only deepseek-flash is authorized')
                    denied_namespaces=config.get('denied_tool_namespaces',[])
                    denied_names=config.get('denied_tool_names',[])
                    payload['tools'],removed_tools=restrict_tools(payload.get('tools',[]),denied_namespaces,denied_names)
                    client_output_limit = payload.get('max_output_tokens')
                    try:
                        requested_output_limit = (int(client_output_limit) if client_output_limit is not None
                                                  else int(config['max_output_per_request']))
                    except (TypeError,ValueError):
                        return self.reject(400,'Invalid output token limit')
                    if requested_output_limit < 1:
                        return self.reject(400,'Output token limit must be positive')
                    with completed:
                        # The client can see response.completed before the
                        # previous HTTP handler has finished its accounting.
                        # Serialize that hand-off instead of rejecting a valid
                        # sequential native CLI request as parallel traffic.
                        if not completed.wait_for(lambda: state['inflight_count'] < config.get('max_inflight',1),timeout=config.get('concurrent_queue_seconds',10)):
                            return self.reject(409,'Task concurrent request limit reached')
                        if (not state['active'] or state['requests'] >= config['max_requests']
                                or state['input_tokens'] >= config['max_input_tokens']
                                or state['output_tokens'] >= config['max_output_tokens']):
                            return self.reject(429,'Frozen task model budget exhausted')
                        available = config['max_output_tokens']-state['output_tokens']-state['reserved_output_tokens']
                        if available < 1:
                            return self.reject(429,'Task output budget already reserved')
                        state['requests'] += 1
                        state['inflight_count'] += 1
                        state['peak_inflight'] = max(state['peak_inflight'],state['inflight_count'])
                        state['inflight'] = True
                        serial = state['requests']
                        payload['max_output_tokens'] = min(requested_output_limit,
                            config['max_output_per_request'],available)
                        state['reserved_output_tokens'] += payload['max_output_tokens']
                        save(folder/'status.json',state)
                    safe = json.dumps(payload, ensure_ascii=False).encode()
                    (folder/f'{serial:03d}_request.json').write_bytes(safe)
                    started = time.monotonic()
                    receipt = dict(job=state['job'],sequence=serial,model_requested='deepseek-flash',
                                   request_sha256=hashlib.sha256(safe).hexdigest(),
                                   client_max_output_tokens=client_output_limit,
                                   configured_max_output_per_request=config['max_output_per_request'],
                                   effective_max_output_tokens=payload['max_output_tokens'],
                                   removed_tool_catalog_entries=removed_tools,
                                   started_utc=datetime.now(timezone.utc).isoformat())
                    request = urllib.request.Request(upstream,data=safe,method='POST',headers={
                        'Authorization':'Bearer '+key,'Content-Type':'application/json',
                        'Accept':'text/event-stream' if payload.get('stream') else 'application/json'})
                    headers_started = False
                    try:
                        with opener.open(request,timeout=600) as response:
                            receipt['upstream_status']=response.status
                            buffered=config.get('buffer_verified_responses',False)
                            def send_headers():
                                self.send_response(response.status)
                                self.send_header('Content-Type',response.headers.get('Content-Type','application/json'))
                                self.send_header('Connection','close')
                                self.end_headers()
                            if not buffered:
                                try:send_headers();headers_started=True
                                except (BrokenPipeError,ConnectionResetError):receipt['client_disconnected']=True
                            response_parts = []
                            with (folder/f'{serial:03d}_response.sse').open('wb') as output:
                                while True:
                                    chunk = response.readline()
                                    if not chunk:
                                        break
                                    chunk = chunk.replace(key.encode(),b'[REDACTED]')
                                    output.write(chunk)
                                    response_parts.append(chunk)
                                    if not buffered and not receipt.get('client_disconnected'):
                                        try:
                                            self.wfile.write(chunk)
                                            self.wfile.flush()
                                        except (BrokenPipeError,ConnectionResetError):
                                            receipt['client_disconnected']=True
                            content = b''.join(response_parts)
                            events = []
                            for line in content.splitlines():
                                if line.startswith(b'data: '):
                                    try: events.append(json.loads(line[6:]))
                                    except ValueError: pass
                            if not events:
                                try: events=[json.loads(content)]
                                except ValueError: pass
                            usage={}
                            returned=[]
                            for event in events:
                                value=event.get('response',event)
                                if value.get('model'): returned.append(value['model'])
                                if value.get('usage'): usage=value['usage']
                            receipt.update(returned_model_ids=sorted(set(returned)),usage=usage,
                                           response_sha256=hashlib.sha256(content).hexdigest())
                            with lock:
                                state['input_tokens'] += int(usage.get('input_tokens',0))
                                state['output_tokens'] += int(usage.get('output_tokens',0))
                                state['returned_model_ids']=sorted(set(state['returned_model_ids']+returned))
                                if not {'input_tokens','output_tokens'} <= usage.keys():
                                    state['active'] = False
                                    receipt['accounting_error'] = 'Provider usage missing; further requests disabled'
                            if buffered:
                                forbidden=prohibited_calls(events,denied_namespaces,denied_names)
                                if forbidden:
                                    receipt['blocked_response_calls']=forbidden
                                    with lock:state['active']=False
                                    try:self.reject(502,'Response attempted a disabled orchestration capability')
                                    except (BrokenPipeError,ConnectionResetError):receipt['client_disconnected']=True
                                else:
                                    try:
                                        send_headers();headers_started=True
                                        self.wfile.write(content);self.wfile.flush()
                                    except (BrokenPipeError,ConnectionResetError):receipt['client_disconnected']=True
                    except Exception as error:
                        receipt.update(error_type=type(error).__name__,error=str(error).replace(key,'[REDACTED]'))
                        if isinstance(error,urllib.error.HTTPError):
                            body=error.read().replace(key.encode(),b'[REDACTED]')
                            (folder/f'{serial:03d}_provider_error.txt').write_bytes(body)
                            receipt['upstream_status']=error.code
                        if not headers_started:
                            try: self.reject(502,'Model provider request failed; see private transport receipt')
                            except (BrokenPipeError,ConnectionResetError): pass
                    finally:
                        receipt['elapsed_seconds']=time.monotonic()-started
                        save(folder/f'{serial:03d}_receipt.json',receipt)
                        with completed:
                            state['inflight_count'] -= 1
                            state['reserved_output_tokens'] -= payload['max_output_tokens']
                            state['inflight']=bool(state['inflight_count'])
                            save(folder/'status.json',state)
                            completed.notify_all()
                    self.close_connection=True
            return Handler
        path = Path(route['socket'])
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        server = UnixHTTPServer(str(path),make_handler(state,lock,folder))
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        servers.append((server,state,lock,folder))
        save(folder/'status.json',state)
    save(run/'MODEL_GATEWAY_READY.json',dict(pid=os.getpid(),routes=len(servers),
         model='deepseek-flash',credential_logged=False,upstream=config['base_url']))
    try:
        while not (run/'MODEL_GATEWAY_STOP').exists(): time.sleep(.5)
    finally:
        for server,state,lock,folder in servers:
            with lock:
                state['active']=False
                save(folder/'status.json',state)
            server.shutdown()
            server.server_close()
            # Do not seal or archive while a provider handler still writes.
            deadline=time.monotonic()+610
            while state['inflight_count'] and time.monotonic()<deadline:
                time.sleep(.1)
            if state['inflight_count']:
                raise RuntimeError('Model handlers failed to drain')
            Path(server.server_address).unlink(missing_ok=True)


def relay(args):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            upstream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            upstream.connect(args.socket)
            with upstream:
                sockets=[self.request,upstream]
                while True:
                    ready,_,_=select.select(sockets,[],[],650)
                    if not ready: return
                    for source in ready:
                        data=source.recv(65536)
                        if not data: return
                        (upstream if source is self.request else self.request).sendall(data)
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=True
        daemon_threads=True
    with Server(('127.0.0.1',args.port),Handler) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        try: return subprocess.call(command)
        finally: server.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='mode',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--run',type=Path,required=True)
    inner=sub.add_parser('relay');inner.add_argument('--socket',required=True)
    inner.add_argument('--port',type=int,default=47777);inner.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.mode=='serve': gateway(a.run)
    else: raise SystemExit(relay(a))
