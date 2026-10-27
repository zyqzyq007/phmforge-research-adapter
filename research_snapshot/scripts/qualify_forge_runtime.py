"""Exercise the adapted tool interface with no access to hidden labels."""
import argparse
import json
from pathlib import Path

from verified_phmforge import VerifiedForge


def main(args):
    assert not Path("/hidden").exists()
    tools = VerifiedForge("/app", "/output")
    calls = []
    def call(name, **kw):
        value = tools.call(name, kw)
        calls.append(dict(tool=name, arguments=kw, result=value))
        Path("/output/calls.json").write_text(json.dumps(calls, indent=2))
        return value
    if args.mode == "train":
        name, item, _ = tools.dataset(args.dataset)
        call("load_dataset", dataset_name=name, split="train")
        fit = call("train_rul_model" if item["task"] == "rul" else "train_fault_classifier",
                   dataset=name, model_type=args.arch, epochs=args.epochs)
        Path("/output/model_path.txt").write_text(fit["model_path"])
        print(json.dumps(dict(dataset=name, architecture=args.arch,
                              optimizer_steps=fit["training_audit"]["optimizer_steps"])))
    else:
        _, item, _ = tools.dataset(args.dataset)
        model = Path("/output/model_path.txt").read_text()
        if args.mode == "untrained":
            model = str(Path(model).with_name("untrained_model.pt"))
        result = tools.call("predict_rul" if item["task"] == "rul" else "classify_faults",
                            dict(model_path=model, test_data=args.input or args.dataset, unit_id=0))
        Path(f"/output/{args.mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(rows=len(result["predictions"]), model=model)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict", "repeat", "untrained", "changed_input"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--arch", default="mlp")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--input")
    main(p.parse_args())
