"""Offline engineering checks; no real LLM call and no synthetic PHM scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import socket
import threading
import time
from unittest.mock import patch

from controlled_feedback import save, utc
from controlled_program_checks import Broker, replay_check
import isolated_model_gateway as gateway_module


def audit(root):
    folder=root/'audit/contracts';folder.mkdir(exist_ok=False)
    route=folder/'gateway';route.mkdir()
    endpoint=route/'s.sock';barrier=threading.Barrier(3)
    class Response(io.BytesIO):
        status=200
        headers={'Content-Type':'application/json'}
    class Opener:
        def open(self,request,timeout):
            payload=json.loads(request.data);assert payload['max_output_tokens']==20
            assert not gateway_module.prohibited_calls(payload.get('tools',[]),['collaboration'],['spawn_agent'])
            assert all(t.get('name')!='collaboration' for t in payload.get('tools',[]))
            barrier.wait(timeout=10)
            return Response(json.dumps(dict(model='deepseek-flash',usage={'input_tokens':20,'output_tokens':10})).encode())
    save(route/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job='offline',socket=str(endpoint))],
        base_url='https://invalid.example',max_requests=3,max_input_tokens=1000,max_output_tokens=60,max_output_per_request=20,max_inflight=3,
        denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent'],buffer_verified_responses=True))
    def request():
        data=json.dumps(dict(model='deepseek-flash',max_output_tokens=20,tools=[
            dict(type='namespace',name='collaboration',tools=[dict(type='function',name='spawn_agent')]),
            dict(type='namespace',name='mcp__phmforge',tools=[dict(type='function',name='check')])])).encode()
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
            s.settimeout(15);s.connect(str(endpoint))
            s.sendall(b'POST /responses HTTP/1.0\r\nHost: local\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data)
            chunks=[]
            while True:
                part=s.recv(65536)
                if not part:break
                chunks.append(part)
        return b''.join(chunks)
    with patch.object(gateway_module,'credential',return_value='offline-fixture-no-real-key'),patch.object(gateway_module.urllib.request,'build_opener',return_value=Opener()):
        thread=threading.Thread(target=gateway_module.gateway,args=(route,));thread.start()
        for _ in range(100):
            if (route/'MODEL_GATEWAY_READY.json').exists():break
            time.sleep(.05)
        with ThreadPoolExecutor(max_workers=3) as pool:responses=list(pool.map(lambda _:request(),range(3)))
        assert all(b'200 OK' in r for r in responses)
        for _ in range(100):
            state=json.loads((route/'model_gateway/offline/status.json').read_text())
            if not state['inflight']:break
            time.sleep(.05)
        assert state['requests']==3 and state['input_tokens']==60 and state['output_tokens']==30
        assert state['reserved_output_tokens']==0 and state['peak_inflight']==3
        assert b'429' in request()
        (route/'MODEL_GATEWAY_STOP').touch();thread.join(timeout=20)
        assert not thread.is_alive() and not endpoint.exists()
    # A model cannot bypass the restricted catalog by emitting a forbidden call.
    route=folder/'response_guard';route.mkdir();endpoint=route/'s.sock'
    save(route/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job='offline',socket=str(endpoint))],
        base_url='https://invalid.example',max_requests=1,max_input_tokens=1000,max_output_tokens=60,max_output_per_request=20,
        denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent'],buffer_verified_responses=True))
    class ForbiddenOpener:
        def open(self,request,timeout):
            return Response(json.dumps(dict(model='deepseek-flash',usage={'input_tokens':20,'output_tokens':10},
                output=[dict(type='function_call',namespace='collaboration',name='spawn_agent',arguments='{}')])).encode())
    with patch.object(gateway_module,'credential',return_value='offline-fixture-no-real-key'),patch.object(gateway_module.urllib.request,'build_opener',return_value=ForbiddenOpener()):
        thread=threading.Thread(target=gateway_module.gateway,args=(route,));thread.start()
        for _ in range(100):
            if (route/'MODEL_GATEWAY_READY.json').exists():break
            time.sleep(.05)
        response=request();assert b'502' in response and b'function_call' not in response
        (route/'MODEL_GATEWAY_STOP').touch();thread.join(timeout=20);assert not thread.is_alive()
    receipt=json.loads((route/'model_gateway/offline/001_receipt.json').read_text())
    assert receipt['blocked_response_calls'] and receipt['usage']['output_tokens']==10
    # Exercise actual permission logic with real frozen validation reports.
    case=folder/'broker';case.mkdir()
    save(case/'BROKER_CONFIG.json',dict(campaign=str(root),block='block78',arm='C',request_directory='app/requests'))
    broker=Broker(case);rejections=[]
    for phase,action in [('investigation','check'),('investigation','hypotheses'),('investigation','select'),('coordinator_plan','check'),('builder_checks','select')]:
        save(case/'PHASE.json',dict(role=phase))
        try:broker.dispatch(dict(action=action,candidate_id='P00'));raise RuntimeError('Phase unexpectedly allowed')
        except AssertionError:rejections.append(dict(phase=phase,action=action))
    save(case/'PHASE.json',dict(role='coordinator_plan'))
    broker.dispatch(dict(action='hypotheses',hypotheses=[dict(id='fixture',explanation='Engineering registration test only',prediction='No PHM claim',contradiction='No PHM claim')]))
    broker.dispatch(dict(action='plan',candidate_ids=['P00'],contrasts='Engineering only',fixed_factors='Frozen',interpretation='No PHM claim'))
    # One genuine cold validation replay checks the reused numerical assets.
    report=replay_check(root,'block78','P11B',folder/'real_cold_validation')
    assert report['complete_program_matched'] and report['actual_new_fits']==0 and not report['final_input_read']
    result=dict(passed=True,utc=utc(),real_model_calls=0,real_credentials_read=False,
        offline_transport_fixture=True,synthetic_PHM_results=False,shared_concurrency_peak=state['peak_inflight'],
        native_delegation_removed_from_upstream_catalog=True,forbidden_response_rejected_before_forwarding=True,
        budget_request_stop_passed=True,output_reservation_released=True,phase_rejections=rejections,
        actual_cold_model_replays=report['actual_cold_models_replayed'],new_fits=0)
    save(root/'audit/ENGINEERING_AUDIT.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
