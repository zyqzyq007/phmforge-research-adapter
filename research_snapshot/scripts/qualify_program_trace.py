"""Replay frozen PHM predictions with source tracing; no fitting or new labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from program_revision_runtime import Runtime

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
root=parser.parse_args().run.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
cfg=read(root/'PROTOCOL352.json')
for name,digest in read(root/'CODE_FREEZE352.json').items():assert sha(root/name)==digest,name
assert not (root/'STARTED352.json').exists()
save(root/'STARTED352.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=__import__('os').getpid(),new_fits=0,new_LLM_calls=0))
archive=Path(cfg['archive']);manifest=read(archive/'SOURCE_MANIFEST.json')
assert sha(archive/'SOURCE_MANIFEST.json')==cfg['manifest_sha256']
lookup={x['path']:x for x in manifest['files']}
for name,digest in cfg['source_bindings'].items():
    p=archive/name;assert sha(p)==digest==lookup[name]['sha256'] and p.stat().st_size==lookup[name]['bytes']
for program in cfg['programs']:
    for packet in program['packets']:
        with np.load(archive/packet['input'],allow_pickle=False) as x:
            assert 'labels' not in x.files and len(x['sample_ids'])==9 and len(set(x['groups']))==1
save(root/'INPUT_QUALIFICATION352.json',dict(status='passed',all_source_bindings_verified=True,single_acquisition_no_labels=True,new_fits=0,new_predictions=0))
started=time.monotonic();deadline=started+cfg['budget']['prediction_wall_seconds'];values={}
def run_packet(program,index):
    packet=program['packets'][index];folder=root/'predictions'/program['id']/str(index)
    runtime=Runtime(root);runtime.deadline=deadline
    result,audit=runtime.command(archive/program['source'],archive/program['artifacts'],archive/packet['input'],folder,'predict')
    expected=read(archive/packet['expected'])['predictions'];actual=read(folder/'predictions.json')['predictions']
    assert {x['sample_id']:x['predicted_label'] for x in expected}=={x['sample_id']:x['predicted_label'] for x in actual}
    trace=read(folder/'execution_trace.json');assert trace['lines'] and trace['arcs']
    return dict(status='passed',source_packet=packet['input'],all9_predictions_equal=True,seconds=result['seconds'],
        Python_lines=len(trace['lines']),Python_arcs=len(trace['arcs']),prediction_fits=audit['completed_estimator_fits'])
def one(program):
    rows=[]
    for i in range(1 if program['id']=='baseline' else 0,len(program['packets'])):rows.append(run_packet(program,i))
    return program['id'],rows
try:
    first=cfg['programs'][0];assert first['id']=='baseline'
    preflight=run_packet(first,0);save(root/'PREFLIGHT352.json',preflight)
    (root/'predictions').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=cfg['budget']['parallel_programs']) as pool:
        for future in as_completed([pool.submit(one,p) for p in cfg['programs']]):
            key,rows=future.result();values[key]=([preflight] if key=='baseline' else [])+rows
            save(root/'PROGRESS352.json',dict(programs=values))
    assert sum(map(len,values.values()))==48
    result=dict(status='complete',programs=values,actual_prediction_processes=48,all_predictions_match_archived_untraced=True,
        fits_during_prediction=0,new_fits=0,new_LLM_calls=0,new_independent_scenarios=0,
        method_superiority_test=False,seconds=time.monotonic()-started)
except Exception as error:
    result=dict(status='qualification_error',error_type=type(error).__name__,error=str(error),programs=values,
        actual_prediction_processes=len(list((root/'predictions').rglob('PROCESS.json'))),new_fits=0,new_LLM_calls=0)
save(root/'RESULT352.json',result);print(json.dumps({k:v for k,v in result.items() if k!='programs'}))
if result['status']!='complete':raise SystemExit(1)
