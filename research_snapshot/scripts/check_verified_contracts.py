"""Export compatible schemas and run negative controls that matter for validity."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import verified_phmforge as verified
from verified_phmforge import BLOCKED, DEFAULT_EPOCHS, VerifiedForge, vector_pair
from verified_benchmark_scoring import classification, regression, evidence_bundle


def catalog(source):
    inputs, tools = {}, {}
    for file in source.glob("*.py"):
        for node in ast.parse(file.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(isinstance(b, ast.Name) and b.id == "BaseModel" for b in node.bases):
                fields, required = {}, []
                for a in node.body:
                    if not isinstance(a, ast.AnnAssign):
                        continue
                    kind = {"str":"string", "float":"number", "int":"integer", "bool":"boolean"}[a.annotation.id]
                    value = dict(type=kind)
                    options = {k.arg: ast.literal_eval(k.value) for k in a.value.keywords}
                    value.update(options)
                    fields[a.target.id] = value
                    if "default" not in options:
                        required.append(a.target.id)
                inputs[node.name] = dict(type="object", properties=fields, required=required, additionalProperties=False)
            if any(isinstance(b, ast.Name) and b.id == "BaseTool" for b in node.bases):
                values = {a.target.id:a.value for a in node.body if isinstance(a, ast.AnnAssign)}
                tools[ast.literal_eval(values["name"])] = values["args_schema"].id
    return {name:dict(description=(
                "UNAVAILABLE in verified profile: "+BLOCKED[name] if name in BLOCKED else
                "Research-adapted real computation. "+name+
                ". Data and model must exist; no hidden labels, synthetic returns or physical certification."),
                inputSchema=inputs[schema], original_schema_name=schema)
            for name, schema in tools.items()}


def check(run):
    schemas = catalog(run/"sources/forge_tools")
    assert len(schemas) == 22
    for name in ["train_rul_model", "train_fault_classifier"]:
        assert schemas[name]["inputSchema"]["properties"]["epochs"]["default"] == DEFAULT_EPOCHS
    (run/"public/tool_catalog.json").write_text(json.dumps(schemas, indent=2))
    out = run/"outputs/contract_checks"
    out.mkdir(exist_ok=True)
    tools = VerifiedForge(run/"public", out)
    results = []
    def reject(name, fn):
        try:
            fn()
        except (ValueError, FileNotFoundError, PermissionError, TypeError, KeyError) as exc:
            results.append(dict(test=name, passed=True, rejection=type(exc).__name__, message=str(exc)))
        else:
            raise AssertionError("Expected rejection: "+name)
    reject("missing_model", lambda: tools.call("predict_rul", dict(
        model_path=str(out/"absent.pt"), test_data="CMAPSS_FD001", unit_id=0)))
    reject("missing_dataset", lambda: tools.call("load_dataset", dict(dataset_name="NOT_PRESENT")))
    reject("no_train_as_test_fallback", lambda: tools.call("load_dataset", dict(dataset_name="CWRU",split="unknown")))
    reject("fault_data_has_no_RUL", lambda: tools.call("train_rul_model", dict(dataset="MFPT")))
    reject("missing_raw_waveform", lambda: tools.call("train_fault_classifier", dict(dataset="CWRU",model_type="lstm")))
    reject("metric_broadcast", lambda: vector_pair([1,2],[1]))
    reject("metric_NaN", lambda: vector_pair([float("nan")],[1]))
    gold = [dict(unit_id=1,actual_rul=1),dict(unit_id=2,actual_rul=2)]
    reject("partial_submission", lambda: regression([dict(unit_id=1,predicted_rul=1)],gold))
    reject("duplicate_submission", lambda: regression([dict(unit_id=1,predicted_rul=1)]*2,gold))
    reject("fabricated_class", lambda: classification([dict(sample_id="a",predicted_label=7)],
                                                      [dict(sample_id="a",label=1,group="g")]))
    assert tools.call("calculate_mae",dict(predictions="[1,3]",ground_truth="[1,2]"))["value"] == .5
    assert tools.call("calculate_accuracy",dict(predictions='["healthy","fault"]',ground_truth='["healthy","healthy"]'))["value"] == .5
    for name in BLOCKED:
        result = tools.call(name,{})
        assert result["status"] == "unavailable" and result["physical_conclusion"] is None
        results.append(dict(test="unsupported_not_fabricated:"+name,passed=True))
    original_train = verified.rul.train
    try:
        verified.rul.train = lambda directory, dest, arch, epochs_override: {
            "epochs_executed": epochs_override
        }
        fit = tools.call("train_rul_model", dict(dataset="CMAPSS_FD001"))
        assert fit["training_audit"]["epochs_executed"] == DEFAULT_EPOCHS
        results.append(dict(test="training_default_matches_catalog",passed=True))
    finally:
        verified.rul.train = original_train
    dummy_model = out/"dummy_model.pt"
    dummy_model.write_bytes(b"contract-check-only")
    original_infer = verified.rul.infer
    try:
        def fake_infer(model_path, input_path, output_path):
            verified.rul.write_json(output_path, dict(
                predictions=[
                    dict(unit_id=1,predicted_rul=10.0),
                    dict(unit_id=2,predicted_rul=20.0),
                    dict(unit_id=3,predicted_rul=30.0),
                ],
                checkpoint_sha256="fixture",
                input_sha256="fixture",
            ))
        verified.rul.infer = fake_infer
        selected = tools.call("predict_rul", dict(
            model_path=str(dummy_model),test_data="CMAPSS_FD001",unit_id=2))
        saved = json.loads(Path(selected["artifact"]).read_text())
        assert selected["prediction_scope"] == "single_unit"
        assert selected["predictions"] == [dict(unit_id=2,predicted_rul=20.0)]
        assert saved["requested_unit_id"] == 2
        assert saved["prediction_scope"] == "single_unit"
        assert saved["predictions"] == [dict(unit_id=2,predicted_rul=20.0)]
        results.append(dict(test="single_unit_result_matches_artifact",passed=True))
    finally:
        verified.rul.infer = original_infer
    reject("invalid_rul_unit_id", lambda: tools.call("predict_rul", dict(
        model_path=str(dummy_model),test_data="CMAPSS_FD001",unit_id=-1)))
    calls=[dict(tool="list_tools",result="catalog"*10000),
           dict(tool="actual_business_tool",arguments={"asset":"x"},result={"measured_value":4})]
    pack=evidence_bundle(calls,"answer",out/"complete_evidence.json")
    assert pack["business_calls"] == 1 and pack["judge_ready"] and not pack["evidence_truncated"]
    fail=evidence_bundle(calls,"answer",out/"capacity_checked_evidence.json",max_prompt_characters=1)
    assert not fail["judge_ready"] and fail["complete_evidence_available"]
    results += [dict(test="business_evidence_not_hidden_by_tool_discovery",passed=True),
                dict(test="over_budget_judge_blocks_instead_of_truncating",passed=True)]
    report=dict(negative_controls=results, all_passed=True, forge_tool_schema_count=len(schemas),
                no_new_LLM_calls=True)
    (run/"audit/CONTRACT_CHECKS240.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=len(results),schemas=len(schemas))))


if __name__=="__main__":
    check(Path(sys.argv[1]).resolve())
