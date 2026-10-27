"""Real MCP training, cold reload, dependence controls, and external scoring.

All model calls run in the existing no-network filesystem sandbox. Final
labels are read only by the host after prediction artifacts are sealed.
"""
import argparse
import asyncio
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from acquire_forge_pdmb_sources244 import save
from launch_verified_environment import command
from verified_benchmark_scoring import classification


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def qualify(run, runtime, name, gold_path, resume_training=False, defer_score=False):
    workspace = run/"outputs"/name
    log = run/"logs"/(name+".stderr")
    log.parent.mkdir(exist_ok=True)
    report = dict(dataset=name, calls=[], label_path_provided_to_model=False,
                  new_LLM_calls=0, official_benchmark_score=False)
    previous = None
    report_path = run/"audit"/(name+"_QUALIFICATION246.json")
    if report_path.exists():
        if not resume_training:
            raise FileExistsError("Existing qualification report; explicit training reuse required")
        previous = json.loads(report_path.read_text())
        if previous.get("external_score") is not None:
            raise ValueError("No re-selection after reading final labels")
        old = report_path.with_name(name+"_FAILED_ATTEMPT1.json")
        if old.exists():
            raise FileExistsError(old)
        report_path.rename(old)
        report["reused_training_from"] = dict(path=str(old), sha256=sha(old))
        for filename in ["sealed_test_predictions.json", "qualification_calls.json"]:
            path = workspace/filename
            path.rename(workspace/("failed_attempt1_"+filename))
    async def session_block(mode, model=None):
        argv = command(runtime, workspace, "forge", "mcp")
        report.setdefault("sandbox_commands", []).append(argv)
        with log.open("a") as errors:
            async with stdio_client(StdioServerParameters(command=argv[0], args=argv[1:]), errlog=errors) as (read,write):
                async with ClientSession(read,write) as session:
                    await session.initialize()
                    catalog=await session.list_tools()
                    assert len(catalog.tools)==22
                    async def call(tool, args):
                        result=await session.call_tool(tool, args, read_timeout_seconds=timedelta(seconds=900))
                        if result.isError:
                            raise RuntimeError(str(result.content))
                        value=json.loads(next(item.text for item in result.content if item.type=="text"))
                        report["calls"].append(dict(phase=mode,tool=tool,arguments=args,result=value))
                        save(workspace/"qualification_calls.json",report["calls"])
                        return value
                    if mode=="train":
                        await call("load_dataset",dict(dataset_name=name,split="train"))
                        test=await call("load_dataset",dict(dataset_name=name,split="test"))
                        if previous is None:
                            fit=await call("train_fault_classifier",dict(dataset=name,model_type="mlp",epochs=50))
                        else:
                            fit=next(row["result"] for row in previous["calls"] if row["tool"]=="train_fault_classifier")
                            original=next(row["result"] for row in previous["calls"] if row["tool"]=="classify_faults")
                            checkpoint=workspace/Path(fit["model_path"]).parent.name/"model.pt"
                            assert sha(checkpoint)==original["checkpoint_sha256"]
                        pred=await call("classify_faults",dict(model_path=fit["model_path"],test_data=name,unit_id=0))
                        save(workspace/"sealed_test_predictions.json",pred)
                        report["prediction_seal_sha256"]=sha(workspace/"sealed_test_predictions.json")
                        report["training_audit"]=fit["training_audit"]
                        # Real development review, rather than an oracle final-label tool.
                        val=await call("load_dataset",dict(dataset_name=name,split="val"))
                        truth=await call("load_ground_truth",dict(dataset_name=name,file="val"))
                        dev=await call("classify_faults",dict(model_path=fit["model_path"],test_data=val["data_path"],unit_id=0))
                        check=await call("verify_classification",dict(predictions=dev["artifact"],ground_truth=truth["artifact"]))
                        assert check["identity_coverage_verified"] and not check["final_benchmark_certified"]
                        assert abs(check["metrics"]["accuracy"]-fit["training_audit"]["selected_validation_accuracy"])<1e-12
                        final=await call("load_ground_truth",dict(dataset_name=name,file="test"))
                        assert final["status"]=="unavailable"
                        return fit["model_path"],pred
                    repeat=await call("classify_faults",dict(model_path=model,test_data=name,unit_id=0))
                    initial=await call("classify_faults",dict(model_path=str(Path(model).with_name("untrained_model.pt")),
                                                            test_data=name,unit_id=0))
                    changed=await call("classify_faults",dict(model_path=model,
                        test_data=f"/app/datasets/{name}/changed_input.npz",unit_id=0))
                    return repeat,initial,changed
    try:
        model,pred=await session_block("train")
        repeated,initial,changed=await session_block("cold_reload",model)
        checks=dict(
            real_optimizer_updates=report["training_audit"]["optimizer_steps"]>0,
            real_nonzero_gradients=report["training_audit"]["nonzero_gradient_steps"]>0,
            trained_weights_changed=report["training_audit"]["changed_parameter_tensors"]>0,
            exact_cold_reload=pred["predictions"]==repeated["predictions"],
            weight_dependent_probabilities=pred["predictions"]!=initial["predictions"],
            input_dependent_probabilities=pred["predictions"]!=changed["predictions"],
            prediction_seal_unchanged=report["prediction_seal_sha256"]==sha(workspace/"sealed_test_predictions.json"),
        )
        # First final-label access occurs here, after both inference processes
        # exited and after the prediction seal and controls were recorded.
        save(workspace/"PRE_SCORE_CHECKS.json",checks)
        if defer_score:
            observed=np.load(runtime/'public/datasets'/name/'test.npz',allow_pickle=False)
            expected=observed['sample_ids'].tolist()
            actual=[row['sample_id'] for row in pred['predictions']]
            checks['complete_unique_identity_coverage']=len(actual)==len(set(actual))==len(expected) and set(actual)==set(expected)
            report['external_scores_deferred']=True
        else:
            gold=json.loads(gold_path.read_text())
            report["external_score"]=classification(pred["predictions"],gold)
            report["gold_sha256"]=sha(gold_path)
            checks["complete_unique_identity_coverage"]=report["external_score"]["n"]==len(gold)
        report["checks"]=checks
        report["qualified"]=all(checks.values())
        report["status"]="qualified" if report["qualified"] else "control_failed"
        if not report["qualified"]:
            raise ValueError("Scientific qualification control failed")
    except Exception as error:
        report.update(qualified=False,status="failed",error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc())
    save(run/"audit"/(name+"_QUALIFICATION246.json"),report)
    summary={key:report[key] for key in ["dataset","status","qualified"]}
    summary.update(external_score=report.get("external_score"), checks=report.get("checks"))
    print(json.dumps(summary),flush=True)
    return summary


async def main(args):
    names=args.datasets
    golds={}
    for name in names:
        path=args.run/"external_labels"/name/"labels.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        golds[name]=path
    results=await asyncio.gather(*(qualify(args.run,args.runtime,name,golds[name],args.resume_training,args.defer_score) for name in names))
    save(args.run/"audit"/("QUALIFICATION_BATCH246_"+("_".join(names))+".json"),results)
    return 0 if all(row["qualified"] for row in results) else 1


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--datasets",nargs="+",required=True)
    p.add_argument("--resume-training",action="store_true")
    p.add_argument("--defer-score",action="store_true",help="Check coverage using observations only; do not read final labels or scores")
    raise SystemExit(asyncio.run(main(p.parse_args())))
