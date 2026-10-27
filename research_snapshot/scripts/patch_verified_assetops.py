"""Apply minimal, recorded patches to a separate AssetOps native source copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def patch(run):
    source = run / "restored/legacy/runtime/asset_public"
    target = run / "adapted/asset"
    shutil.copytree(source, target)
    records = []
    def change(relative, old, new, count=1):
        path = target / relative
        text = path.read_text()
        assert text.count(old) == count, (relative, old[:100], text.count(old))
        original = hashlib.sha256(text.encode()).hexdigest()
        text = text.replace(old, new)
        path.write_text(text)
        records.append(dict(file=relative, before=original,
                            after=hashlib.sha256(text.encode()).hexdigest(),
                            removed=old, added=new))
    f = "servers/tsfm/forecasting.py"
    change(f, '"evaluation_strategy": "epoch",', '"eval_strategy": "epoch",')
    change(f, '"num_workers": 4,', '"num_workers": 0,')
    change(f, '"lr": 0.0,', '"lr": 0.0001,')
    change(f, '"epochs_warmup": 5,', '"epochs_warmup": 0,')
    change(f, '"greater_is_better": False,',
           '"greater_is_better": False,\n    "report_to": [],\n    "use_cpu": True,')
    change(f, "y_gt[:, :ix_fhorizon, ix_target],",
           "y_gt[:, :ix_fhorizon + 1, ix_target],")
    change(f, "y_pred[:, :ix_fhorizon, ix_target],",
           "y_pred[:, :ix_fhorizon + 1, ix_target],")
    # Do not silently train a random replacement if a requested checkpoint is absent.
    change(f, "    if os.path.exists(model_checkpoint):\n",
           '    if not os.path.isfile(os.path.join(model_checkpoint, "config.json")):\n'
           '        raise FileNotFoundError("Requested pretrained checkpoint is unavailable")\n'
           '    if os.path.exists(model_checkpoint):\n')
    change(f, "    start_time = time.time()\n    if n_finetune > 0:\n",
           "    from verified_asset_audit import snapshot, record_training\n"
           "    initial_state = snapshot(finetune_forecast_model)\n"
           "    start_time = time.time()\n    if n_finetune > 0:\n")
    change(f, "    train_time = time.time() - start_time\n",
           "    train_time = time.time() - start_time\n"
           '    selected_dir = os.path.join(save_model_dir, "selected_model")\n'
           "    finetune_forecast_trainer.save_model(selected_dir)\n"
           "    record_training(initial_state, finetune_forecast_trainer, save_model_dir,\n"
           "                    model_checkpoint, df_dataframe, train_dataset, valid_dataset)\n")
    main = "servers/tsfm/main.py"
    change(main,
           '            _find_largest_tsfm_checkpoint_directory(fewshot_dir) or abs_save_dir\n',
           '            os.path.join(abs_save_dir, "selected_model")\n')
    # Read-only test fixtures cannot be mistaken for distinct fine-tuned checkpoints.
    change(main, "    return TSFMModelsResult(models=[TSFMModelEntry(**m) for m in _TSFM_MODELS])\n",
           "    from verified_asset_audit import available_models\n"
           "    return TSFMModelsResult(models=[TSFMModelEntry(**m) for m in available_models()])\n")
    change(main, "            output = _get_ttm_hf_inference(\n",
           "            from verified_asset_forecast import forecast\n"
           "            output = forecast(\n")
    (run / "audit/ASSET_PATCHES240.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(patches=len(records), destination=str(target))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    patch(p.parse_args().run.resolve())
