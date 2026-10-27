"""Independently recount prediction equivalence and trace information gate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
root=parser.parse_args().run.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signature(trace,counts=False):
    values=[(x['file'],x['function'],x['start'],x['end'],x['count'] if counts else 0) for x in trace['arcs']]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()
cfg=read(root/'PROTOCOL352.json');result=read(root/'RESULT352.json');assert result['status']=='complete'
archive=Path(cfg['archive']);assert sha(archive/'SOURCE_MANIFEST.json')==cfg['manifest_sha256']
for name,digest in read(root/'CODE_FREEZE352.json').items():assert sha(root/name)==digest
for name,digest in cfg['source_bindings'].items():assert sha(archive/name)==digest
assert not Path('/proc',str(read(root/'STARTED352.json')['pid'])).exists()
programs={p['id']:p for p in cfg['programs']};rows={};all_count=0
for pid,program in programs.items():
    paths=[];counted=[];times=[];predictions=[]
    for i,packet in enumerate(program['packets']):
        folder=root/'predictions'/pid/str(i);process=read(folder/'PROCESS.json');audit=read(folder/'training_audit.json')
        assert process['returncode']==0 and '--unshare-net' in process['command']
        assert audit['mode']=='predict' and audit['completed_estimator_fits']==audit['optimizer_steps']==0
        expected=read(archive/packet['expected'])['predictions'];actual=read(folder/'predictions.json')['predictions']
        mapping={x['sample_id']:x['predicted_label'] for x in actual}
        assert mapping=={x['sample_id']:x['predicted_label'] for x in expected} and len(mapping)==9
        trace=read(folder/'execution_trace.json');paths.append(signature(trace));counted.append(signature(trace,True))
        times.append(process['seconds']);predictions.append(mapping);all_count+=1
    rows[pid]=dict(replays=len(paths),unique_arc_sets=len(set(paths)),unique_arc_count_profiles=len(set(counted)),
                   mean_process_seconds=sum(times)/len(times),max_process_seconds=max(times),arc_set_hashes=paths)
    rows[pid]['prediction_maps']=predictions
guard=programs['guarded_lda'];source=archive/guard['source']/'program.py'
nodes={ast.unparse(n.test):n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.If)}
tests=['predicted == 1','supplied_mean[3] > 0.8','supplied_mean[2] > 0.35']
assert all(t in nodes for t in tests)
observed=[]
for i in range(12):
    old=set(rows['dual_lda']['prediction_maps'][i].values());new=set(rows['guarded_lda']['prediction_maps'][i].values())
    assert len(old)==len(new)==1
    old,new=next(iter(old)),next(iter(new))
    assert set(rows['dual_lda']['prediction_maps'][i])==set(rows['guarded_lda']['prediction_maps'][i])
    if old!=1:assert new==old;expected=[False,None,None]
    elif new==3:expected=[True,True,None]
    elif new==2:expected=[True,False,True]
    else:assert new==1;expected=[True,False,False]
    arcs=read(root/'predictions/guarded_lda'/str(i)/'execution_trace.json')['arcs'];states=[]
    for text in tests:
        node=nodes[text];ends={a['end'] for a in arcs if a['file']=='program.py' and a['function']=='predict_acquisition' and a['start']==node.lineno}
        states.append(None if not ends else node.body[0].lineno in ends)
    assert states==expected,(i,states,expected)
    observed.append(dict(packet=i,role=guard['packets'][i]['role'],old_prediction=old,new_prediction=new,
                         observed_guard_states=states,states_determined_by_pair_predictions=True))
assert all_count==48 and read(root/'STATIC_GUARD_PROOF352.json')['model_artifact_SHA_identical']
constant=all(rows[p]['unique_arc_sets']==1 for p in ['baseline','global_update','dual_lda'])
new_gate=not constant
for value in rows.values():value.pop('prediction_maps')
report=dict(status='passed',all48_prediction_replays_independently_recounted=True,all432_window_predictions_match=True,
    all_frozen_code_input_source_weight_bindings_unchanged=True,prediction_fit_events=0,new_fits=0,new_predictions=0,new_LLM_calls=0,
    programs=rows,guard_observations=observed,guard_states_all_recoverable_from_pair_predictions=True,
    new_measurement_gate_passed=new_gate,advance_branch_coverage_as_main_mechanism=new_gate,
    gate_scope='Currenttwoqualificationchangesonly;notallcode-awaretestsorallPHMprograms.',
    independent_confirmation_scenarios=0,majority_goal_complete=False)
(root/'TRACE_INFORMATION_GATE352.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['programs','guard_observations']}))
