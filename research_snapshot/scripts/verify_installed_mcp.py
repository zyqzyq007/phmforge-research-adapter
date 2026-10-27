"""Cold-start the installed profiles through the actual MCP protocol."""
import asyncio
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from launch_verified_environment import command


async def verify(run, runtime, families=None, suffix="240"):
    reports=[]
    for family in families or ["forge","asset_legacy","asset_current"]:
        output=run/"outputs"/("installed_"+family)
        argv=command(runtime,output,family,"mcp",run/"audit/COUCH_RESOURCE240.json" if family=="asset_current" else None)
        report=dict(family=family,command=argv,calls=[])
        with (run/"logs"/("installed_"+family+".stderr")).open("w") as errlog:
            async with stdio_client(StdioServerParameters(command=argv[0],args=argv[1:]),errlog=errlog) as (read,write):
                async with ClientSession(read,write) as session:
                    await session.initialize()
                    catalog=await session.list_tools()
                    report["tool_count"]=len(catalog.tools)
                    catalog_by_name={tool.name:tool for tool in catalog.tools}
                    async def call(name,arguments):
                        response=await session.call_tool(name,arguments,read_timeout_seconds=__import__("datetime").timedelta(seconds=600))
                        if response.isError:
                            raise RuntimeError(str(response.content))
                        raw=json.loads(next(x.text for x in response.content if x.type=="text"))
                        report["calls"].append(dict(tool=name,arguments=arguments,result=raw))
                        return raw
                    if family=="forge":
                        assert report["tool_count"]==22
                        for name in ["train_rul_model","train_fault_classifier"]:
                            assert catalog_by_name[name].inputSchema["properties"]["epochs"]["default"]==50
                        fit=await call("train_rul_model",dict(dataset="CMAPSS_FD001",model_type="mlp",epochs=1))
                        assert fit["training_audit"]["optimizer_steps"]>0
                        pred=await call("predict_rul",dict(model_path=fit["model_path"],test_data="CMAPSS_FD001",unit_id=0))
                        assert len(pred["predictions"])==100
                        selected=await call("predict_rul",dict(model_path=fit["model_path"],test_data="CMAPSS_FD001",unit_id=1))
                        assert selected["prediction_scope"]=="single_unit"
                        assert len(selected["predictions"])==1
                        assert selected["predictions"][0]["unit_id"]==1
                        saved=json.loads((output/Path(selected["artifact"]).name).read_text())
                        assert saved["prediction_scope"]=="single_unit"
                        assert saved["predictions"]==selected["predictions"]
                        unavailable=await call("load_ground_truth",dict(dataset_name="CMAPSS_FD001"))
                        assert unavailable["status"]=="unavailable"
                        value=await call("calculate_mae",dict(ground_truth="[1,2]",predictions="[1,3]"))
                        assert value["value"]==.5
                    elif family=="asset_legacy":
                        value=await call("tsfm.run_tsfm_forecasting",dict(dataset_path="/data/context.csv",
                            timestamp_column="timestamp",target_columns=["Chiller 6 Power Input"],
                            model_checkpoint="ttm_96_28",forecast_horizon=28))
                        assert "success" in json.dumps(value)
                    else:
                        await call("wo.list_workorders",dict(page_size=0))
                        await call("wo.list_workorders",dict(site_id="MAIN",page_size=0))
                        await call("tsfm.find_models",dict(task_id="tsfm_forecasting"))
        report["passed"]=True
        (run/"audit"/("INSTALLED_MCP_"+family.upper()+suffix+".json")).write_text(json.dumps(report,indent=2))
        reports.append(dict(family=family,tools=report["tool_count"],passed=True))
    (run/"audit"/("INSTALLED_MCP"+suffix+".json")).write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("runtime",type=Path)
    parser.add_argument("--families",nargs="+",choices=["forge","asset_legacy","asset_current"])
    parser.add_argument("--suffix",default="240")
    args=parser.parse_args()
    asyncio.run(verify(args.run.resolve(),args.runtime.resolve(),args.families,args.suffix))
