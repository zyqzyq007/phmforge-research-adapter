"""Verify the A243 PHMForge interface fixes without external labels or an LLM."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(runtime, run):
    runtime, run = runtime.resolve(), run.resolve()
    if (run / "audit/INTERFACE_CHECKS243.json").exists():
        raise FileExistsError(run / "audit/INTERFACE_CHECKS243.json")
    for directory in ["audit", "outputs", "logs", "tmp", "mpl", "xdg", "hf"]:
        (run / directory).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(runtime / "code"))
    import verified_phmforge as verified
    from verified_phmforge import BLOCKED, DEFAULT_EPOCHS, VerifiedForge

    output = run / "outputs/contract"
    output.mkdir()
    tools = VerifiedForge(runtime / "public", output)
    catalog = json.loads((runtime / "public/tool_catalog.json").read_text())
    checks = []

    def passed(name, **evidence):
        checks.append(dict(test=name, passed=True, **evidence))

    assert len(catalog) == 22
    assert len(BLOCKED) == 9
    for name in ["train_rul_model", "train_fault_classifier"]:
        default = catalog[name]["inputSchema"]["properties"]["epochs"]["default"]
        assert default == DEFAULT_EPOCHS == 50
        passed("catalog_default_epochs", tool=name, default=default)

    original_train = verified.rul.train
    try:
        verified.rul.train = lambda directory, dest, arch, epochs_override: {
            "dataset": directory.name,
            "architecture": arch,
            "epochs_executed": epochs_override,
        }
        fit = tools.call("train_rul_model", dict(dataset="CMAPSS_FD001"))
        assert fit["training_audit"]["epochs_executed"] == 50
        passed("omitted_epochs_execute_schema_default",
               tool="train_rul_model", epochs=50)
    finally:
        verified.rul.train = original_train

    dummy_model = output / "dummy_model.pt"
    dummy_model.write_bytes(b"interface-check-only")
    original_infer = verified.rul.infer
    try:
        def fake_infer(model_path, input_path, output_path):
            verified.rul.write_json(output_path, dict(
                predictions=[
                    dict(unit_id=1, predicted_rul=10.0),
                    dict(unit_id=2, predicted_rul=20.0),
                    dict(unit_id=3, predicted_rul=30.0),
                ],
                checkpoint_sha256="fixture",
                input_sha256="fixture",
            ))
        verified.rul.infer = fake_infer
        all_units = tools.call("predict_rul", dict(
            model_path=str(dummy_model),
            test_data="CMAPSS_FD001",
            unit_id=0,
        ))
        assert all_units["prediction_scope"] == "all_units"
        assert len(all_units["predictions"]) == 3
        saved_all = json.loads(Path(all_units["artifact"]).read_text())
        assert saved_all["predictions"] == all_units["predictions"]
        passed("all_unit_artifact_matches_result", rows=3)

        selected = tools.call("predict_rul", dict(
            model_path=str(dummy_model),
            test_data="CMAPSS_FD001",
            unit_id=2,
        ))
        assert selected["prediction_scope"] == "single_unit"
        assert selected["predictions"] == [dict(unit_id=2, predicted_rul=20.0)]
        saved = json.loads(Path(selected["artifact"]).read_text())
        assert saved["requested_unit_id"] == 2
        assert saved["prediction_scope"] == "single_unit"
        assert saved["predictions"] == selected["predictions"]
        passed("single_unit_artifact_matches_result", unit_id=2, rows=1)
    finally:
        verified.rul.infer = original_infer

    try:
        tools.call("predict_rul", dict(
            model_path=str(dummy_model),
            test_data="CMAPSS_FD001",
            unit_id=-1,
        ))
    except ValueError as exc:
        passed("invalid_unit_rejected_before_inference",
               rejection=type(exc).__name__, message=str(exc))
    else:
        raise AssertionError("Negative unit_id was accepted")

    result = dict(
        status="passed",
        runtime=str(runtime),
        runtime_manifest_sha256=sha256(runtime / "FILE_MANIFEST.json"),
        source_adapter_sha256=sha256(runtime / "code/verified_phmforge.py"),
        public_adapter_sha256=sha256(runtime / "public/code/verified_phmforge.py"),
        checks=checks,
        new_LLM_calls=0,
        new_training_jobs=0,
        external_labels_accessed=False,
    )
    write_json(run / "audit/INTERFACE_CHECKS243.json", result)
    print(json.dumps(dict(status="passed", checks=len(checks))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    check(args.runtime, args.run)
