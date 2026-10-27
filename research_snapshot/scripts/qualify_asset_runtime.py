"""Exercise actual MCP calls in one persistent loop inside an isolated container."""
import argparse
import asyncio
import json
from pathlib import Path

import torch


def json_value(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, (list, tuple)):
        return [json_value(a) for a in x]
    if isinstance(x, dict):
        return {k: json_value(v) for k, v in x.items()}
    if hasattr(x, "content"):
        return dict(content=json_value(x.content), structured_content=getattr(x, "structured_content", None))
    return x


def payload(result):
    raw = json_value(result)
    if isinstance(raw, dict) and raw.get("structured_content"):
        return raw["structured_content"]
    blocks = raw["content"] if isinstance(raw, dict) else raw[0]
    for b in blocks:
        if b.get("type") == "text":
            return json.loads(b["text"])
    raise ValueError(f"Unrecognized MCP result shape: {str(raw)[:200]}")


async def execute(args):
    from servers.tsfm.main import mcp
    torch.set_num_threads(2)
    out = Path("/output")
    spec = json.loads(Path("/data/dataset.json").read_text())
    assert not Path("/hidden").exists()
    catalog = [x.model_dump(mode="json") for x in await mcp.list_tools()]
    (out / f"{args.mode}_catalog.json").write_text(json.dumps(catalog, indent=2))
    common = dict(timestamp_column="timestamp", target_columns=[spec["target"]],
                  forecast_horizon=28, frequency_sampling="oov")
    if args.mode == "train":
        name = "run_tsfm_finetuning"
        kwargs = dict(common, dataset_path="/data/train.csv", model_checkpoint="ttm_96_28",
                      save_model_dir="/output/tuned", n_finetune=1., n_test=.1)
    else:
        name = "run_tsfm_forecasting"
        kwargs = dict(common, dataset_path=args.input, model_checkpoint=args.model)
    try:
        raw = await mcp.call_tool(name, kwargs)
        result = payload(raw)
        record = dict(tool=name, arguments=kwargs, raw=json_value(raw), result=result)
        if result.get("results_file"):
            artifact = Path(result["results_file"])
            record["artifact"] = json.loads(artifact.read_text())
    except Exception as exc:
        record = dict(tool=name, arguments=kwargs, error_type=type(exc).__name__, error=str(exc))
    (out / f"{args.name}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k in ["result", "error_type", "error"]}))
    if record.get("result", {}).get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("--name", default="native_train")
    p.add_argument("--input", default="/data/context.csv")
    p.add_argument("--model", default="/output/tuned/selected_model")
    asyncio.run(execute(p.parse_args()))
