"""Seal stronger full-checkpoint controls from legal evidence before final read."""
import argparse
import json
from pathlib import Path
import shutil

from controlled_feedback import save, sha, utc

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')


def read(path):return json.loads(Path(path).read_text())


def freeze(root):
    assert not (root/'FINAL_STARTED347.json').exists()
    assert not (root/'CONDITION_CONTROL_CHOICES.json').exists()
    seal=read(root/'ALL_CHOICES_SEALED337.json');assert len(seal['episodes'])==4
    spec=root/'audit/ADDITIONAL_CONDITION_CONTROLS347.json';assert read(spec)['no_primary_change']
    rows=[]
    for episode in seal['episodes']:
        state=read(root/'subjects'/episode['id']/'broker/STATE.json');packet=state.get('probe')
        check=packet['reports'] if packet and packet['status']=='complete' else {};risks={};bindings={}
        for ident,v in state['programs'].items():
            if ident not in check or check[ident]['status']!='complete':continue
            report=v['report'];file=Path(v['path'])/'REPORT.json';assert read(file)==report
            records=[*report['development']['records'],*check[ident]['records']]
            conditions=sorted({r['condition_hz'] for r in records})
            risks[ident]=[sum(r['actual']!=r['predicted'] for r in records if r['condition_hz']==hz)/sum(r['condition_hz']==hz for r in records) for hz in conditions]
            bindings[ident]=dict(report_sha256=sha(file),conditions_hz=conditions,record_errors=risks[ident])
        assert len({tuple(v['conditions_hz']) for v in bindings.values()})<=1
        choice={k:None for k in ['full_checkpoint_equal_condition','full_checkpoint_worst_condition']}
        if risks:
            mean=lambda k:sum(risks[k])/len(risks[k])
            choice['full_checkpoint_equal_condition']=min(risks,key=lambda k:(mean(k),k!='p00',k))
            choice['full_checkpoint_worst_condition']=min(risks,key=lambda k:(max(risks[k]),mean(k),k!='p00',k))
        routing=[]
        if risks:
            centers=next(iter(bindings.values()))['conditions_hz']
            for j,center in enumerate(centers):
                selected=min(risks,key=lambda k:(risks[k][j],k!='p00',k))
                routing.append(dict(center_hz=center,program_id=selected,legal_record_error=risks[selected][j]))
        rows.append(dict(episode=episode['id'],arm=episode['arm'],eligible_programs=list(risks),bindings=bindings,choices=choice,
            nearest_full_checkpoint_condition=routing))
    shutil.copy2(Path(__file__),root/'audit/FROZEN_CONDITION_CONTROL.py')
    value=dict(utc=utc(),source_spec_sha256=sha(spec),code_sha256=sha(root/'audit/FROZEN_CONDITION_CONTROL.py'),
        native_choices_seal_sha256=sha(root/'ALL_CHOICES_SEALED337.json'),choices=rows,final_data_read=False,
        original_control_rows_not_replaced=True,independent_method_confirmation=0)
    save(root/'CONDITION_CONTROL_CHOICES.json',value);save(PROJECT/'results/local347/CONDITION_CONTROL_CHOICES.json',value)
    print(json.dumps(dict(status='additional_controls_sealed_before_final',episodes=len(rows))))


def score(root):
    import numpy as np
    choices=read(root/'CONDITION_CONTROL_CHOICES.json');result=read(root/'RESULT347.json');rows=[]
    assert choices['final_data_read'] is False
    assert sha(root/'ALL_CHOICES_SEALED337.json')==choices['native_choices_seal_sha256']
    for episode in choices['choices']:
        values={}
        for name,ident in episode['choices'].items():
            if ident is None:values[name]=dict(status='no_common_check_evidence');continue
            key='common_p00' if ident=='p00' else episode['episode']+'__'+ident
            value=result['all_program_results'][key];row=dict(program_id=ident,status=value['status'])
            if value['status']=='complete':row.update(record_error=value['record_error'],harm=max(0.,value['record_error']-result['baseline_error']),gain=result['baseline_error']-value['record_error'])
            values[name]=row
        route=episode.get('nearest_full_checkpoint_condition',[])
        if route:
            source=Path(read(root/'LOCAL_REVISION_PROTOCOL347.json')['source_archive']);file=source/'qualification/final.npz'
            assert sha(file)==read(source/'QUALIFICATION346.json')['files']['final.npz']['sha256']
            with np.load(file,allow_pickle=False) as data:
                groups=data['groups'].copy();rotation=data['rotation_hz'].copy();sample_ids=data['sample_ids'].copy();actual=data['labels'].copy()
            predictions={};missing=[]
            for ident in {r['program_id'] for r in route}:
                key='common_p00' if ident=='p00' else episode['episode']+'__'+ident
                value=result['all_program_results'][key]
                if value['status']!='complete':missing.append(ident);continue
                with np.load(root/'evaluation'/key/'forward/PREDICTIONS.npz',allow_pickle=False) as data:
                    assert np.array_equal(data['sample_ids'],sample_ids);predictions[ident]=data['prediction'].copy()
            policy_rows=[]
            if not missing:
                for group in np.unique(groups):
                    mask=groups==group;hz=float(np.mean(rotation[mask]));item=min(route,key=lambda r:(abs(hz-r['center_hz']),r['center_hz']))
                    counts=np.bincount(predictions[item['program_id']][mask],minlength=4);predicted=int(np.flatnonzero(counts==counts.max())[0]);truth=set(actual[mask].tolist());assert len(truth)==1
                    policy_rows.append(dict(group=str(group),rotation_hz=hz,selected_program=item['program_id'],selected_center_hz=item['center_hz'],predicted=predicted,actual=int(truth.pop())))
                error=sum(r['predicted']!=r['actual'] for r in policy_rows)/len(policy_rows)
                values['nearest_full_checkpoint_condition']=dict(status='complete_cached_actual_cold_predictions',record_error=error,
                    harm=max(0.,error-result['baseline_error']),gain=result['baseline_error']-error,records=policy_rows)
            else:values['nearest_full_checkpoint_condition']=dict(status='evaluation_error',missing_programs=missing)
        else:values['nearest_full_checkpoint_condition']=dict(status='no_common_check_evidence')
        rows.append(dict(episode=episode['episode'],arm=episode['arm'],methods=values))
    value=dict(utc=utc(),choices_sha256=sha(root/'CONDITION_CONTROL_CHOICES.json'),real_prediction_result_sha256=sha(root/'RESULT347.json'),
        rows=rows,new_fits=0,new_predictions=0,new_LLM_calls=0,descriptive_per_own_pool_not_crossarm_oracle_advantage=True)
    save(root/'CONDITION_CONTROL_RESULTS.json',value);save(PROJECT/'results/local347/CONDITION_CONTROL_RESULTS.json',value)
    print(json.dumps(dict(status='additional_controls_scored_from_frozen_actual_predictions',episodes=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','score']);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();globals()[args.action](args.run.resolve())
