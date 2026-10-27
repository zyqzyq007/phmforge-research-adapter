"""Actual MCP training/inference checks inside a data-only numerical namespace."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from real_bearing_rul_model import infer
from real_rul_model import write_json


async def main():
    checks={};output=Path('/output');record=[]
    params=StdioServerParameters(command='/opt/python312/bin/python3.12',args=[
        '-B','/tools/verified_benchmark_worker.py','--family','forge','--data','/data',
        '--output',str(output),'--transport','mcp'],env=dict(os.environ))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize();catalog=await session.list_tools()
            checks['actual22_mcp_tools']=len(catalog.tools)==22
            async def call(name,args):
                response=await session.call_tool(name,args,read_timeout_seconds=__import__('datetime').timedelta(seconds=180))
                value=json.loads(response.content[0].text)
                record.append({'tool':name,'args':args,'is_error':response.isError,'result':value})
                assert not response.isError,(name,value)
                return value
            fit=await call('load_dataset',{'dataset_name':'FEMTO','split':'train'})
            dev=await call('load_dataset',{'dataset_name':'FEMTO','split':'val'})
            checks['load_real_training_observations']=Path(fit['data_path']).is_file()
            observed=np.load(dev['data_path'],allow_pickle=False)
            checks['development_input_has_no_labels']='labels' not in observed
            truth=await call('load_ground_truth',{'dataset_name':'FEMTO','file':'val'})
            checks['development_targets_are_explicit']=truth['rows']==9
            for split in ['','test']:
                denied=await call('load_ground_truth',{'dataset_name':'FEMTO','file':split})
                checks['final_truth_denied_'+(split or 'empty')]=denied['status']=='unavailable'
            trained=await call('train_rul_model',{'dataset':'FEMTO','model_type':'mlp','epochs':30})
            audit=trained['training_audit'];model=Path(trained['model_path'])
            checks['real_weight_updates']=audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
            checks['three_actual_scenario_models']=len(audit['scenarios'])==3
            checks['fitting_and_validation_bearings_disjoint']=all(not set(x['fit_unit_ids'])&set(x['validation_unit_ids']) for x in audit['scenarios'])
            checks['no_training_test_labels']=audit['test_labels_accessed'] is False
            test=await call('predict_rul',{'model_path':str(model),'test_data':'FEMTO','unit_id':0})
            checks['all11_test_bearings_predicted']=len(test['predictions'])==len({x['unit_id'] for x in test['predictions']})==11
            sealed=output/'SEALED_QUALIFICATION_PREDICTIONS.json';write_json(sealed,{'predictions':test['predictions']})
            cold=output/'cold.json'
            subprocess.run(['/opt/python312/bin/python3.12','-B','/tools/real_bearing_rul_model.py',
                'infer','--model',str(model),'--input','/data/datasets/FEMTO/test.npz','--output',str(cold)],check=True)
            checks['cold_reload_exactly_reproduces']=json.loads(cold.read_text())['predictions']==test['predictions']
            single=await call('predict_rul',{'model_path':str(model),'test_data':'FEMTO','unit_id':7})
            checks['single_unit_artifact_scope']=single['predictions']==[x for x in test['predictions'] if x['unit_id']==7]==json.loads(Path(single['artifact']).read_text())['predictions']
            random=infer(model.with_name('untrained_model.pt'),Path('/data/datasets/FEMTO/test.npz'),output/'untrained.json')
            checks['prediction_depends_on_weights']=not np.allclose([x['predicted_rul'] for x in random['predictions']],[x['predicted_rul'] for x in test['predictions']])
            data=dict(np.load('/data/datasets/FEMTO/test.npz',allow_pickle=False));data['features']=data['features']*1.03+.1
            np.savez_compressed(output/'perturbed.npz',**data)
            perturbed=infer(model,output/'perturbed.npz',output/'perturbed.json')
            checks['prediction_depends_on_input']=not np.allclose([x['predicted_rul'] for x in perturbed['predictions']],[x['predicted_rul'] for x in test['predictions']])
    write_json(output/'MCP_CALLS.json',record)
    result={'passed':all(checks.values()),'checks':checks,'model_path':str(model),'audit':audit,
        'new_LLM_calls':0,'external_test_scores_not_read':True,'scope':'real MCP data/training/inference and numerical controls,not an Agent result'}
    write_json(output/'MCP_QUALIFICATION.json',result)
    print(json.dumps({'passed':result['passed'],'checks':checks,'optimizer_steps':audit['optimizer_steps']}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':asyncio.run(main())
