"""Adversarial MCP contracts and independent recomputation of real runs."""
import argparse
import asyncio
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import shutil

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from acquire_forge_pdmb_sources244 import save
from launch_verified_environment import command


async def check(run,runtime):
    checks=[]
    output=run/"outputs/repair_contract_checks"
    output.mkdir(exist_ok=False)
    model=next((run/"outputs/CWRU").glob("CWRU_mlp_*/model.pt"))
    shutil.copy2(model,output/"cross_dataset_model.pt")
    argv=command(runtime,output,"forge","mcp")
    with (run/"logs/repair_contracts.stderr").open("w") as err:
        async with stdio_client(StdioServerParameters(command=argv[0],args=argv[1:]),errlog=err) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                catalog=await session.list_tools()
                assert len(catalog.tools)==22
                checks.append(dict(test="real_MCP_22_tools",passed=True))
                async def call(label,tool,args,reject=False):
                    result=await session.call_tool(tool,args,read_timeout_seconds=timedelta(seconds=60))
                    if reject:
                        assert result.isError, (label,result)
                        if label == "host_labels_inaccessible":
                            assert "outside public data/artifact roots" in str(result.content)
                        checks.append(dict(test=label,passed=True,error=str(result.content)))
                        return
                    assert not result.isError,(label,result)
                    value=json.loads(next(x.text for x in result.content if x.type=="text"))
                    checks.append(dict(test=label,passed=True,result=value))
                    return value
                gt=[dict(unit_id=1,actual_rul=4),dict(unit_id=2,actual_rul=7)]
                pred=[dict(unit_id=2,predicted_rul=7),dict(unit_id=1,predicted_rul=4)]
                value=await call("identity_alignment_ignores_list_order","verify_ground_truth",
                                 dict(predictions=json.dumps(pred),ground_truth=json.dumps(gt)))
                assert value["metrics"]["mae"]==0 and value["metrics"]["positive_error_lower_bound"] is None
                for label,rows in [
                    ("missing_identity",pred[:1]),
                    ("duplicate_identity",[pred[0],pred[0]]),
                    ("extra_identity",pred+[dict(unit_id=3,predicted_rul=0)]),
                    ("nonfinite_prediction",[dict(unit_id=1,predicted_rul=float("nan")),pred[0]]),
                    ("positional_arrays_not_certified",[4,7]),
                ]:
                    await call(label,"verify_ground_truth",dict(predictions=json.dumps(rows),ground_truth=json.dumps(gt)),True)
                for dataset in ["MFPT","CMAPSS_FD001"]:
                    for file in ["","test","../external_labels/labels.json","RUL_FD001.txt"]:
                        value=await call(dataset+"_hidden_labels_"+file,"load_ground_truth",
                                         dict(dataset_name=dataset,file=file))
                        assert value["status"]=="unavailable"
                await call("host_labels_inaccessible","verify_classification",
                           dict(ground_truth=str(run/"external_labels/MFPT/labels.json"),
                                predictions='[{"sample_id":"identity-check","predicted_label":0}]'),True)
                await call("cross_dataset_checkpoint_rejected","classify_faults",
                           dict(model_path="/output/cross_dataset_model.pt",test_data="MFPT",unit_id=0),True)
                await call("fault_labels_not_lifetime_labels","train_rul_model",
                           dict(dataset="MFPT",model_type="mlp",epochs=1),True)
                value=await call("engine_actual_input_catalog","load_dataset",
                                 dict(dataset_name="EngineMTQA",split="test"))
                assert value["status"]=="partial" and not value["provenance"]["physical_health_contract_qualified"]
                inputs=json.loads((runtime/"public/engine/task_inputs.json").read_text())
                paths=inputs[0]["sequence_files"]
                value=await call("engine_actual_sequence_statistics","analyze_engine_signals",
                                 dict(sensor_data=json.dumps(dict(sequence_files=paths))))
                source=np.load(runtime/"public/engine/signals"/Path(paths[0]).name,allow_pickle=False)
                assert np.array_equal(value["sequences"][0]["channel_means"],source.mean(0))
                assert value["physical_conclusion"] is None and not value["channel_semantics_verified"]
    independent=[]
    for name in ["CWRU","MFPT","ELECTRICMOTORVIBRATIONS","ROTORBROKENBAR"]:
        report=json.loads((run/"audit"/(name+"_QUALIFICATION246.json")).read_text())
        predictions=json.loads((run/"outputs"/name/"sealed_test_predictions.json").read_text())["predictions"]
        gold=json.loads((run/"external_labels"/name/"labels.json").read_text())
        p={row["sample_id"]:row["predicted_label"] for row in predictions}
        g={row["sample_id"]:row["label"] for row in gold}
        assert len(p)==len(predictions) and p.keys()==g.keys()
        counts=Counter((actual,p[key]) for key,actual in g.items())
        classes=sorted(set(g.values()))
        accuracy=sum(a==p[k] for k,a in g.items())/len(g)
        recalls=[]
        f1s=[]
        for c in classes:
            tp=counts[(c,c)]
            actual=sum(n for (a,b),n in counts.items() if a==c)
            predicted=sum(n for (a,b),n in counts.items() if b==c)
            recalls.append(tp/actual)
            f1s.append(2*tp/max(actual+predicted,1))
        score=report["external_score"]
        assert abs(score["accuracy"]-accuracy)<1e-12
        assert abs(score["balanced_accuracy"]-sum(recalls)/len(classes))<1e-12
        assert abs(score["macro_f1"]-sum(f1s)/len(classes))<1e-12
        assert score["confusion_matrix"]==[[counts[(a,b)] for b in classes] for a in classes]
        assert report["training_audit"]["epochs_executed"]==50
        assert report["qualified"] and all(report["checks"].values())
        independent.append(dict(dataset=name,n=len(g),accuracy=accuracy,metrics_recomputed=True,
                                matrix_recomputed=True,epochs=50,passed=True))
    save(run/"audit/INDEPENDENT_REPAIR_CHECKS246.json",dict(
        mcp_contract_checks=checks, independent_numeric_checks=independent, all_passed=True))
    print(json.dumps(dict(mcp_checks=len(checks),numeric_tasks=len(independent),all_passed=True)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True)
    a=p.parse_args()
    asyncio.run(check(a.run,a.runtime))
