"""PHMForge-compatible tool names with real numerical backends and explicit limits.

This is a research adaptation. Original source and original scores remain separate.
The caller must mount only public data and the writable artifact directory.
"""
import json
import math
from pathlib import Path
import uuid

import numpy as np

import real_rul_model as rul
import real_fault_model as fault
from verified_benchmark_scoring import classification, regression


DEFAULT_EPOCHS = 50

BLOCKED = {
    "assess_component_health": "No calibrated component/operating-condition health reference supplied",
    "diagnose_timing_issues": "Two deviations alone do not identify a physical fault mechanism",
    "optimize_maintenance_schedule": "No action-dependent failure distribution, horizon, constraints or cost model",
    "check_compliance": "No applicable standard clauses, operating mode or auditable compliance evidence",
    "generate_safety_recommendations": "No validated maintenance/safety policy or diagnostic evidence",
    "web_search": "No configured search service in this network-isolated numerical profile",
}


def vector_pair(predictions, actual):
    def parse(value):
        if isinstance(value, str):
            return json.loads(value) if value.strip().startswith("[") else value.split(",")
        return value
    predictions, actual = parse(predictions), parse(actual)
    p, a = np.asarray(predictions, dtype=float), np.asarray(actual, dtype=float)
    if p.ndim != 1 or p.shape != a.shape or not len(p):
        raise ValueError("Expected equal, nonempty one-dimensional vectors; broadcasting prohibited")
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite metric input")
    return p, a


class VerifiedForge:
    def __init__(self, data_root, output):
        self.root, self.output = Path(data_root).resolve(), Path(output).resolve()
        self.registry = json.loads((self.root / "dataset_registry.json").read_text())
        self.output.mkdir(parents=True, exist_ok=True)

    def dataset(self, name, task=None):
        name = name.upper().replace(" ", "_")
        if name not in self.registry:
            aliases = {"PADERBORN": "PADEBORN", "PADEBORN": "PADERBORN",
                       "GEARBOXUOC": "UOC", "ROTOR_BROKEN_BAR": "ROTORBROKENBAR"}
            name = aliases.get(name, name)
        if name.startswith("FD00"):
            name = "CMAPSS_" + name
        if name not in self.registry:
            raise FileNotFoundError(f"Source-bound dataset unavailable: {name}")
        item = self.registry[name]
        if task and item["task"] != task:
            raise ValueError(f"{name} has {item['task']} targets, not {task} targets")
        return name, item, self.root / item["path"]

    def artifact(self, value):
        path = Path(value).resolve()
        if not any(path.is_relative_to(root) for root in [self.root, self.output]):
            raise PermissionError("Path lies outside public data/artifact roots")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def call(self, tool, args):
        policy_path = self.root / "execution_policy.json"
        if tool in {"train_rul_model", "train_fault_classifier"} and policy_path.exists():
            policy = json.loads(policy_path.read_text())
            if policy.get("allow_new_fitting") is False:
                return dict(status="unavailable", tool=tool,
                            reason=policy.get('reason', "This frozen-candidate review phase permits inference and verification only; new fitting is disabled."),
                            training_performed=False)
        if tool in BLOCKED:
            return dict(status="unavailable", tool=tool, reason=BLOCKED[tool],
                        physical_conclusion=None)
        if tool == "load_ground_truth":
            name, item, directory = self.dataset(args["dataset_name"])
            if item["task"] not in ["classification","rul"]:
                return dict(status="unavailable", tool=tool, physical_conclusion=None,
                            reason="This source has no authenticated development-target contract")
            # Original empty/test-file requests remain blocked. No default that
            # silently returns training targets in place of final test labels.
            split = args.get("file", "").removesuffix(".npz").removesuffix(".txt")
            if 'available_splits' in item and split not in item['available_splits']:
                return dict(status='unavailable', tool=tool,
                            reason='This split is not mounted in the current information-access condition.')
            if split not in ["train", "val"]:
                return dict(status="unavailable", tool=tool, physical_conclusion=None,
                            reason="Final test labels are external. Explicit file='train' or 'val' is required.")
            if item["task"] == "classification":
                data = np.load(directory / (split + ".npz"), allow_pickle=False)
                rows = [dict(sample_id=str(i), label=y, group=str(g))
                        for i, y, g in zip(data["sample_ids"], data["labels"].tolist(), data["groups"])]
            elif item.get('format') == 'cmapss_npz':
                data = np.load(directory / (split + '.npz'), allow_pickle=False)
                rows = [dict(sample_id=str(i), unit_id=int(u), endpoint_cycle=int(t), actual_rul=float(y))
                        for i,u,t,y in zip(data['sample_ids'],data['unit_ids'],data['endpoint_cycles'],data['labels'])]
            elif item.get('format') in ['femto_npz','xjtu_npz','measured_endpoint_npz']:
                data=np.load(directory/(split+'.npz'),allow_pickle=False)
                rows=[dict(sample_id=str(i),unit_id=int(u),scenario_id=int(s),actual_rul=float(y))
                      for i,u,s,y in zip(data['sample_ids'],data['unit_ids'],data['scenario_ids'],data['labels'])]
            else:
                config = json.loads((directory / "training_config.json").read_text())
                raw = rul.read_series(directory / "train.txt")
                _, ys, ids = rul.development_windows(
                    raw, config["validation_engine_ids" if split == "val" else "train_engine_ids"],
                    config, validation=split == "val")
                rows = [dict(sample_id=f"{r['unit_id']}:{r['endpoint_cycle']}",
                             actual_rul=float(y), **r) for r, y in zip(ids, ys)]
            target = self.output / f"development_labels_{uuid.uuid4().hex[:12]}.json"
            rul.write_json(target, rows)
            return dict(status="success", dataset=name, split=split, artifact=str(target),
                        rows=len(rows), scope="development only; not final benchmark truth",
                        final_evaluation_labels_available=False)
        if tool in ["verify_ground_truth", "verify_classification"]:
            def records(value):
                if isinstance(value, str):
                    value = (json.loads(value) if value.lstrip().startswith(("[", "{"))
                             else json.loads(self.artifact(value).read_text()))
                if isinstance(value, dict):
                    value = value.get("predictions", value.get("records"))
                if not isinstance(value, list) or not value or not all(isinstance(r, dict) for r in value):
                    raise ValueError("Explicit identity-bearing records required; positional arrays are not verified")
                return value
            predicted, actual = records(args["predictions"]), records(args["ground_truth"])
            if tool == "verify_ground_truth":
                identity = "sample_id" if "sample_id" in actual[0] else "unit_id"
                metrics = regression(predicted, actual, identity=identity)
            else:
                # Group statistics require genuine source groups. Caller arrays
                # without these can support sample metrics only.
                groups_supplied = all("group" in row for row in actual)
                metrics = classification(predicted, [{"group": "unspecified", **row} for row in actual])
                if not groups_supplied:
                    for field in ["groups", "mean_recording_accuracy"]:
                        metrics.pop(field)
            metrics["source"] = "identity-aligned computation over caller-provided records"
            return dict(status="success", identity_coverage_verified=True, metrics=metrics,
                        final_benchmark_certified=False, physical_conclusion=None,
                        scope="Verifies coverage and arithmetic; does not authenticate caller-supplied truth")
        if tool == "load_dataset":
            name, item, directory = self.dataset(args["dataset_name"])
            split = args.get("split", "train")
            if 'available_splits' in item and split not in item['available_splits']:
                return dict(status='unavailable', dataset=name, split=split, data_path=None,
                            reason='This split is not mounted in the current information-access condition.')
            if split == 'test' and item.get('test_inputs_visible',True) is False:
                return dict(status='unavailable',dataset=name,split=split,data_path=None,
                    reason='Final observations are sealed during authoring/review; external evaluation releases them after program freezing.')
            if item["task"] == "sequence_qa":
                if split != "test":
                    raise ValueError("Recovered engine source supports only split='test'")
                path = self.artifact(directory/"task_inputs.json")
                return dict(status="partial", dataset=name, split=split, data_path=str(path),
                            provenance=item, evaluation_labels_available=False,
                            physical_conclusion=None,
                            reason="Actual sequences recovered; physical channel mapping and health contracts unresolved")
            bearing_format=item.get('format') in ['femto_npz','xjtu_npz','cmapss_npz','measured_endpoint_npz']
            if split not in (["train", "test"] if item["task"] == "rul" and not bearing_format else ["train", "val", "test"]):
                raise ValueError("Requested split unavailable; no fallback to training data")
            path = directory / (split + (".txt" if item["task"] == "rul" and not bearing_format else ".npz"))
            self.artifact(path)
            if (item["task"] == "classification" or bearing_format) and split == "val":
                # Development inference uses the same no-label input contract
                # as testing. Its labels remain explicitly available through
                # load_ground_truth(file='val').
                data = dict(np.load(path, allow_pickle=False))
                observations = {k:v for k,v in data.items() if k not in ["labels","groups"]}
                observations["dataset"] = np.asarray(name)
                path = self.output / f"development_observations_{uuid.uuid4().hex[:12]}.npz"
                np.savez_compressed(path, **observations)
            return dict(status="success", dataset=name, split=split, data_path=str(path),
                        provenance=item, evaluation_labels_available=False)
        if tool in ["train_rul_model", "train_fault_classifier"]:
            task = "rul" if tool == "train_rul_model" else "classification"
            name, info, directory = self.dataset(args["dataset"], task)
            arch = args.get("model_type", "mlp")
            epochs = args.get("epochs", DEFAULT_EPOCHS)
            if arch not in info["architectures"] or not isinstance(epochs, int) or epochs < 1:
                raise ValueError("Invalid architecture or epoch count for this dataset")
            dest = self.output / f"{name}_{arch}_{uuid.uuid4().hex[:12]}"
            if info.get('format')=='measured_endpoint_npz':
                import real_gap_rul_model
                audit=real_gap_rul_model.train(directory,dest,arch,epochs_override=epochs)
            elif info.get('format')=='xjtu_npz':
                import real_xjtu_rul_model
                audit=real_xjtu_rul_model.train(directory,dest,arch,epochs_override=epochs)
            elif info.get('format')=='femto_npz':
                import real_bearing_rul_model
                audit=real_bearing_rul_model.train(directory,dest,arch,epochs_override=epochs)
            else:
                audit = (rul.train(directory, dest, arch, epochs_override=epochs)
                         if task == "rul" else fault.train(directory, dest, arch, epochs))
            return dict(status="success", model_path=str(dest / "model.pt"),
                        audit_path=str(dest / "training_audit.json"), training_audit=audit)
        if tool in ["predict_rul", "classify_faults"]:
            task = "rul" if tool == "predict_rul" else "classification"
            unit_id = args.get("unit_id")
            if task == "rul" and (
                    not isinstance(unit_id, int) or isinstance(unit_id, bool)
                    or unit_id < 0):
                raise ValueError("RUL unit_id must be a nonnegative integer")
            if task == "classification" and unit_id > 0:
                raise ValueError("Use explicit sample IDs in prediction artifact; sample is not a physical unit")
            model = self.artifact(args["model_path"])
            try:
                expected_dataset, source_info, directory = self.dataset(args["test_data"], task)
                source = directory / ("test.txt" if task == "rul" and source_info.get('format') not in ['femto_npz','xjtu_npz','cmapss_npz','measured_endpoint_npz'] else "test.npz")
            except FileNotFoundError:
                source = self.artifact(args["test_data"])
                expected_dataset = None
                metadata = source.parent / "dataset.json"
                if task == "classification" and metadata.is_file():
                    expected_dataset = json.loads(metadata.read_text())["dataset"]
            if task == "classification" and expected_dataset is not None:
                checkpoint = fault.torch.load(model, weights_only=True, map_location="cpu")
                if checkpoint["dataset"] != expected_dataset:
                    raise ValueError("Model and observation dataset identities differ")
            output = self.output / f"predictions_{uuid.uuid4().hex[:12]}.json"
            if task=='rul' and source.suffix=='.npz':
                with np.load(source,allow_pickle=False) as observations:
                    measured_dataset=observations['dataset'].item()
                if measured_dataset in ['LDM_RTF','IMS_ENDPOINT']:
                    import real_gap_rul_model
                    real_gap_rul_model.infer(model,source,output)
                elif measured_dataset.startswith('CMAPSS_'):
                    rul.infer(model,source,output)
                elif measured_dataset=='XJTU':
                    import real_xjtu_rul_model
                    real_xjtu_rul_model.infer(model,source,output)
                else:
                    import real_bearing_rul_model
                    real_bearing_rul_model.infer(model,source,output)
            else:
                (rul.infer if task == "rul" else fault.infer)(model, source, output)
            value = json.loads(output.read_text())
            if task == "rul":
                value["requested_unit_id"] = unit_id
                value["prediction_scope"] = "all_units" if unit_id == 0 else "single_unit"
                if unit_id > 0:
                    value["predictions"] = [
                        row for row in value["predictions"] if row["unit_id"] == unit_id
                    ]
                    if not value["predictions"]:
                        raise ValueError("Unit ID absent from observation file")
                    rul.write_json(output, value)
            return dict(status="success", artifact=str(output), **value)
        if tool in ["calculate_mae", "calculate_rmse", "calculate_accuracy"]:
            if tool == "calculate_accuracy":
                def labels(x):
                    return json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x.split(",") if isinstance(x, str) else x
                p, a = labels(args["predictions"]), labels(args["ground_truth"])
                if not isinstance(p, list) or not isinstance(a, list) or not p or len(p) != len(a):
                    raise ValueError("Equal nonempty label vectors required")
                return dict(status="success", metric="accuracy", value=sum(x == y for x, y in zip(p, a))/len(p),
                            count=len(p), evidence="caller-supplied labels; no external verification")
            p, a = vector_pair(args["predictions"], args.get("ground_truth", args.get("actual")))
            metric = {"calculate_mae": lambda: np.abs(p-a).mean(),
                      "calculate_rmse": lambda: np.sqrt(np.square(p-a).mean()),
                      "calculate_accuracy": lambda: (p == a).mean()}[tool]
            return dict(status="success", metric=tool.removeprefix("calculate_"),
                        value=float(metric()), count=len(p), evidence="caller-supplied vectors, not external verification")
        if tool == "analyze_engine_signals":
            values = args["sensor_data"]
            values = json.loads(values) if isinstance(values, str) else values
            if isinstance(values, dict) and "sequence_files" in values:
                paths = values["sequence_files"]
                if not isinstance(paths, list) or not paths or len(paths) > 10:
                    raise ValueError("Provide one to ten public sequence paths")
                records = []
                for filename in paths:
                    path = self.artifact(filename)
                    data = np.load(path, allow_pickle=False)
                    if data.shape != (600, 33) or not np.isfinite(data).all():
                        raise ValueError("Expected a finite recovered 600-by-33 source sequence")
                    records.append(dict(source_id=path.stem, shape=list(data.shape),
                                        channel_means=data.mean(0).tolist(),
                                        channel_std=data.std(0).tolist()))
                return dict(status="partial", sequences=records, physical_conclusion=None,
                            channel_semantics_verified=False,
                            scope="Actual array statistics; not health, component efficiency or failure probability")
            if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
                raise ValueError("Finite named sensor readings required")
            return dict(status="partial", readings=values, physical_conclusion=None,
                        reason="Sensor units, operating conditions and reference limits required for diagnosis")
        if tool == "detect_degradation_trend":
            data = args["cycle_data"]
            data = json.loads(data) if isinstance(data, str) else data
            x, y = vector_pair([a["cycle"] for a in data], [a["value"] for a in data])
            if len(x) < 2 or not np.all(np.diff(x) > 0):
                raise ValueError("At least two strictly increasing observation cycles required")
            slope, intercept = np.polyfit(x, y, 1)
            return dict(status="success", slope=float(slope), intercept=float(intercept),
                        change=float(y[-1]-y[0]), count=len(x),
                        physical_conclusion=None, scope="descriptive linear trend, not health certification")
        if tool in ["calculate_maintenance_cost", "calculate_failure_cost"]:
            a = dict(args)
            defaults = dict(downtime_hours=4. if tool == "calculate_maintenance_cost" else 48.,
                            hourly_rate=500., consequential_cost=0.)
            a = dict(defaults, **a)
            if any(not math.isfinite(float(v)) or v < 0 for v in a.values()):
                raise ValueError("Costs, durations and counts must be finite and nonnegative")
            if tool == "calculate_maintenance_cost":
                result = a["frequency_per_year"] * (a["preventive_cost"] + a["downtime_hours"] * a["hourly_rate"])
            else:
                if a["failure_probability"] > 1:
                    raise ValueError("Probability must lie in [0,1]")
                result = a["failure_probability"] * (a["repair_cost"] + a["downtime_hours"] * a["hourly_rate"] + a["consequential_cost"])
            return dict(status="success", expected_annual_cost=result, supplied_assumptions=a,
                        scope="conditional arithmetic; failure probability is not estimated")
        if tool == "assess_safety_risk":
            a = [args[k] for k in ["severity", "probability", "detectability"]]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in a):
                raise ValueError("Finite nonnegative ratings required")
            return dict(status="partial", rpn=math.prod(a), physical_conclusion=None,
                        scope="Product of caller-supplied ordinal ratings; no validated risk class")
        raise KeyError(tool)
