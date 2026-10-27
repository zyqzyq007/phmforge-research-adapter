"""Training provenance for the adapted native AssetOps TTM implementation."""
import hashlib
import json
import os
from pathlib import Path

import torch


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def record_training(initial, trainer, directory, source, df, train, val):
    selected = snapshot(trainer.model)
    changed = sum(not torch.equal(initial[k], selected[k]) for k in initial)
    steps = trainer.state.global_step
    if steps <= 0 or changed <= 0:
        raise RuntimeError("Training did not update model parameters")
    audit = dict(optimizer_steps=steps, changed_parameter_tensors=changed,
                 initial_state_sha256=digest(initial), selected_state_sha256=digest(selected),
                 pretrained_source=source, selected_by="held-out chronological development loss",
                 best_model_checkpoint=trainer.state.best_model_checkpoint,
                 train_windows=len(train), validation_windows=len(val), input_rows=len(df),
                 input_dataframe_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                 log_history=trainer.state.log_history,
                 actual_training=True, hidden_future_mounted=False)
    Path(directory, "training_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def available_models():
    root = Path(os.environ["PATH_TO_MODELS_DIR"])
    result = []
    for path in sorted(root.glob("*/config.json")):
        if not any((path.parent / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]):
            continue
        cfg = json.loads(path.read_text())
        # Some distributed folders have names inconsistent with their actual config.
        if path.parent.name == "ttm_512_96" and cfg["context_length"] != 512:
            continue
        result.append(dict(model_id=path.parent.name, model_checkpoint=str(path.parent),
                           model_description=f"Shipped weights; actual context {cfg['context_length']}, horizon {cfg['prediction_length']}. Domain tuning unverified.",
                           context_length=cfg["context_length"],
                           prediction_length=cfg["prediction_length"],
                           provenance="checkpoint shipped with original public AssetOps source"))
    return result
