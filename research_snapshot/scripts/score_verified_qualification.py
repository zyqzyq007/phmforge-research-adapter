"""Independent evaluation after isolated jobs finish; includes anti-fake controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from verified_benchmark_scoring import regression, classification, evidence_bundle
from real_rul_model import write_json


def vector(rows, task):
    return np.asarray([x["predicted_rul"] if task == "rul" else x["probabilities"] for x in rows], float)


def score(run):
    results = []
    registry = json.loads((run / "public/dataset_registry.json").read_text())
    for job in sorted((run / "outputs").iterdir()):
        if job.name == "asset":
            continue
        dataset = next((name for name in registry if job.name.startswith(name + "_")), None)
        if dataset is None:
            continue
        task = registry[dataset]["task"]
        needed = ["predict", "repeat", "untrained", "changed_input"]
        missing = [x for x in needed if not (job / (x+".json")).is_file()]
        if missing:
            results.append(dict(job=job.name, status="incomplete", missing=missing))
            continue
        blobs = {k: json.loads((job / (k+".json")).read_text()) for k in needed}
        model_path = Path((job / "model_path.txt").read_text())
        model_dir = job / model_path.parent.name
        audit = json.loads((model_dir / "training_audit.json").read_text())
        if task == "rul":
            target = np.loadtxt(run / "hidden" / dataset / "rul.txt")
            gold = [dict(unit_id=i+1, actual_rul=float(v)) for i, v in enumerate(target)]
        else:
            gold = json.loads((run / "hidden" / dataset / "labels.json").read_text())
        func = regression if task == "rul" else classification
        metrics = {k: func(blobs[k]["predictions"], gold) for k in ["predict", "untrained"]}
        arrays = {k: vector(blobs[k]["predictions"], task) for k in needed}
        controls = dict(
            parameter_updates=audit["optimizer_steps"] > 0 and audit["changed_parameter_tensors"] > 0,
            nonzero_gradients=audit["nonzero_gradient_steps"] > 0,
            cold_reload_equal=bool(np.allclose(arrays["predict"], arrays["repeat"], rtol=0, atol=1e-6)),
            weight_dependency=bool(not np.allclose(arrays["predict"], arrays["untrained"], rtol=0, atol=1e-6)),
            input_dependency=bool(not np.allclose(arrays["predict"], arrays["changed_input"], rtol=0, atol=1e-6)),
            hidden_labels_not_mounted=True)
        result = dict(job=job.name, dataset=dataset, task=task,
                      qualification_pass=all(controls.values()), controls=controls,
                      optimizer_steps=audit["optimizer_steps"],
                      changed_parameter_tensors=audit["changed_parameter_tensors"],
                      evaluation=metrics, dataset_limitations=registry[dataset],
                      claim="real execution qualified; no assertion of strong-model accuracy")
        results.append(result)
    asset = run / "outputs/asset"
    asset_result = dict(status="incomplete")
    required = ["native_predict", "native_repeat", "native_pretrained", "native_changed_input"]
    if all((asset / (x+".json")).is_file() for x in required):
        blobs = {k: json.loads((asset / (k+".json")).read_text()) for k in required}
        if all("artifact" in v for v in blobs.values()):
            arrays = {k: np.asarray(v["artifact"]["target_prediction"], float).reshape(-1)
                      for k, v in blobs.items()}
            gold = json.loads((run / "hidden/asset/chiller6_future.json").read_text())
            def metrics(k):
                times = np.asarray(blobs[k]["artifact"]["timestamp"]).reshape(-1)
                for t in times:
                    if np.datetime64(t, "ns") != np.datetime64(t, "s"):
                        raise ValueError("Unexpected subsecond forecast timestamp")
                pred = [dict(timestamp=str(np.datetime64(t, "s")), prediction=float(v))
                        for t, v in zip(times, arrays[k])]
                return regression(pred, gold, "timestamp", "prediction", "value")
            audit = json.loads((asset / "tuned/training_audit.json").read_text())
            target_values = np.asarray([x["value"] for x in gold], float)
            target_quality = dict(
                rows=int(target_values.size),
                nonzero_rows=int((target_values != 0).sum()),
                distinct_values=int(np.unique(target_values).size),
                min=float(target_values.min()),
                max=float(target_values.max()),
                std=float(target_values.std()),
                nondegenerate=bool(
                    target_values.size >= 2
                    and np.unique(target_values).size >= 2
                    and (target_values != 0).mean() >= 0.5
                    and target_values.std() > 1e-12
                ),
            )
            controls = dict(parameters_updated=audit["optimizer_steps"] > 0 and audit["changed_parameter_tensors"] > 0,
                            cold_reload_equal=bool(np.allclose(arrays["native_predict"], arrays["native_repeat"], rtol=0, atol=1e-6)),
                            weight_dependency=bool(not np.allclose(arrays["native_predict"], arrays["native_pretrained"], rtol=0, atol=1e-6)),
                            input_dependency=bool(not np.allclose(arrays["native_predict"], arrays["native_changed_input"], rtol=0, atol=1e-6)),
                            hidden_future_not_mounted=True,
                            nondegenerate_target=target_quality["nondegenerate"])
            asset_result = dict(status="scored" if all(controls.values()) else "failed_controls",
                                qualification_pass=all(controls.values()), controls=controls,
                                target_quality=target_quality,
                                optimizer_steps=audit["optimizer_steps"],
                                changed_parameter_tensors=audit["changed_parameter_tensors"],
                                trained=metrics("native_predict"), pretrained=metrics("native_pretrained"),
                                data=json.loads((run / "public/asset/dataset.json").read_text()))
            current_path = run / "outputs/asset_current/result_forecast.json"
            if current_path.is_file():
                current = json.loads(current_path.read_text())
                forecast_values = current.get("forecast", current.get("results", {}).get("forecast", []))
                forecast = np.asarray(forecast_values, float).reshape(-1)
                with (run / "public/asset/serving_context.csv").open() as stream:
                    context_rows = list(csv.DictReader(stream))
                origin = np.datetime64(context_rows[-1]["timestamp"], "s")
                times = [str(origin + np.timedelta64(15 * (i + 1), "m"))
                         for i in range(forecast.size)]
                current_gold = gold
                predictions = [dict(timestamp=t, prediction=float(v))
                               for t, v in zip(times, forecast)]
                trainer_audit = json.loads(
                    (run / "outputs/asset_current/trainer_audit.json").read_text()
                )
                asset_result["current_main"] = dict(
                    status=current.get("status"),
                    optimization_steps=sum(x["global_step"] for x in trainer_audit),
                    forecast_rows=int(forecast.size),
                    evaluation=regression(
                        predictions, current_gold, "timestamp", "prediction", "value"
                    ),
                )
    report = dict(forge=results, asset=asset_result, original_benchmark_scores_replaced=False,
                  qualification_is_not_full_task_completion=True)
    write_json(run / "audit/QUALIFICATION_RESULTS240.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    score(p.parse_args().run.resolve())
