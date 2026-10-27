"""Latest-main native MCP + actual CouchDB + instrumented, unmodified Trainer calls."""
import asyncio
import csv
import importlib
import json
from pathlib import Path
import sys

import torch

from qualify_asset_runtime import json_value, payload
from verified_asset_audit import digest, snapshot


async def main(mode):
    from transformers import Trainer
    torch.set_num_threads(2)
    root = Path("/output")
    training_calls = []
    original = Trainer.train
    def audited_train(self, *args, **kwargs):
        before = snapshot(self.model)
        result = original(self, *args, **kwargs)
        after = snapshot(self.model)
        training_calls.append(dict(global_step=self.state.global_step,
                                    changed_parameter_tensors=sum(not torch.equal(before[k], after[k]) for k in before),
                                    before_sha256=digest(before), after_sha256=digest(after),
                                    log_history=self.state.log_history))
        (root / "trainer_audit.json").write_text(json.dumps(training_calls, indent=2))
        return result
    Trainer.train = audited_train
    servers = {name: importlib.import_module(f"servers.{name}.main").mcp
               for name in ["wo", "iot", "tsfm", "utilities", "fmsr", "vibration"]}
    catalog = {}
    for name, mcp in servers.items():
        catalog.update({name+"."+t.name: t.model_dump(mode="json") for t in await mcp.list_tools()})
    (root / "current_tool_catalog.json").write_text(json.dumps(catalog, indent=2))
    calls = []
    async def call(name, **kwargs):
        server, tool = name.split(".", 1)
        raw = await servers[server].call_tool(tool, kwargs)
        result = payload(raw)
        calls.append(dict(tool=name, arguments=kwargs, raw=json_value(raw), result=result))
        (root / f"calls_{mode}.json").write_text(json.dumps(calls, indent=2))
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(result["error"])
        return result
    spec = json.loads(Path("/data/dataset.json").read_text())
    with Path("/data/train.csv").open() as stream:
        row_count = sum(1 for _ in csv.DictReader(stream))
    initial_window = max(128, int(row_count * 0.6))
    step = max(1, int(row_count * 0.1))
    if mode == "forecast":
        recipe = dict(
            estimator=dict(
                sktime_class="sktime.forecasting.ttm.TinyTimeMixerForecaster",
                params=dict(
                    model_path="/output/current_model",
                    fit_strategy="zero-shot",
                    use_source_package=True,
                    device="cpu",
                ),
            ),
            fh=list(range(1, 29)),
            eval=dict(initial_window=96, step=28, metrics=["mae"]),
        )
        result = await call(
            "tsfm.run_recipe",
            dataset_path="/data/serving_context.csv",
            timestamp_column="timestamp",
            target_columns=[spec["target"]],
            recipe=recipe,
            asset_id="Chiller 6",
        )
        artifact = Path(result["results_file"].removeprefix("file://"))
        assert artifact.is_relative_to(root) and artifact.is_file()
        (root / "artifact_forecast.json").write_bytes(artifact.read_bytes())
        (root / "result_forecast.json").write_text(json.dumps(result, indent=2))
        forecast = result.get("forecast", result.get("results", {}).get("forecast", []))
        print(json.dumps(dict(mode=mode, status=result.get("status"),
                              forecasts=len(forecast),
                              tool_count=len(catalog))))
        return
    if mode == "train":
        await call("wo.list_workorders", page_size=0)
        # Reusing the same loop checks the former client-loop mismatch.
        await call("wo.list_workorders", asset_num="CWC04009", page_size=0)
        await call("iot.sites")
        await call("iot.history", site_name="MAIN", asset_id="Chiller 6", limit=3,
                   sensors=[spec["target"]])
    recipe = dict(estimator=dict(sktime_class="sktime.forecasting.ttm.TinyTimeMixerForecaster",
                                  params=dict(model_path="/models/ttm_96_28" if mode == "train" else "/output/current_model",
                                              fit_strategy="full" if mode == "train" else "zero-shot",
                                              use_source_package=True, device="cpu",
                                              training_args=dict(output_dir="/output/trainer",
                                                                 num_train_epochs=2, per_device_train_batch_size=64,
                                                                 learning_rate=.0001, report_to=[], use_cpu=True,
                                                                 save_strategy="no", eval_strategy="no",
                                                                 dataloader_num_workers=0, seed=240))),
                  fh=list(range(1, 29)), eval=dict(initial_window=2800, step=1000, metrics=["mae"]))
    recipe["eval"].update(initial_window=initial_window, step=step,
                          row_count_basis=row_count,
                          sizing_rule="60% initial window and 10% step; fixed before training outcome")
    if mode == "train":
        recipe["save_to"] = "/output/current_model"
    result = await call("tsfm.run_recipe", dataset_path="/data/train.csv",
                        timestamp_column="timestamp", target_columns=[spec["target"]],
                        recipe=recipe, asset_id="Chiller 6")
    artifact = Path(result["results_file"].removeprefix("file://"))
    assert artifact.is_relative_to(root) and artifact.is_file()
    (root / f"artifact_{mode}.json").write_bytes(artifact.read_bytes())
    (root / f"result_{mode}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(mode=mode, status=result.get("status"), training_calls=len(training_calls),
                         tool_count=len(catalog))))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
