"""Future-only TTM inference; never use a supervised train-window dataset to forecast."""
import hashlib

import numpy as np
import pandas as pd
import torch
from tsfm_public import TinyTimeMixerForPrediction


def forecast(df_dataframe, dataset_config_dictionary, model_config, model_checkpoint,
             scaling=False, tsp=None, forecast_horizon=-1):
    if scaling:
        raise ValueError("External scaling requires the fitted training preprocessor; no inference refitting")
    spec = dataset_config_dictionary["column_specifiers"]
    targets = spec["target_columns"]
    if spec.get("conditional_columns"):
        raise ValueError("Conditional-channel forecasts are not yet qualified")
    length = model_config["context_length"]
    horizon = model_config["prediction_length"] if forecast_horizon == -1 else forecast_horizon
    if not 1 <= horizon <= model_config["prediction_length"]:
        raise ValueError("Requested horizon lies outside model output contract")
    if len(targets) != model_config["num_input_channels"]:
        raise ValueError("Target columns do not match the saved model channel contract")
    model = TinyTimeMixerForPrediction.from_pretrained(model_checkpoint)
    model.eval()
    ids = dataset_config_dictionary.get("id_columns") or spec.get("id_columns")
    groups = df_dataframe.groupby(ids, sort=False) if ids else [(None, df_dataframe)]
    predictions, times, origins, audit = [], [], [], []
    for group, df in groups:
        if len(df) < length:
            raise ValueError("Insufficient observed history")
        df = df.sort_values(spec["timestamp_column"]).iloc[-length:]
        t = pd.to_datetime(df[spec["timestamp_column"]])
        dt = t.diff().dropna()
        if dt.empty or dt.iloc[0].total_seconds() <= 0 or not (dt == dt.iloc[0]).all():
            raise ValueError("Forecast context must have a unique, regular time grid")
        raw = df[targets].to_numpy(np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite observations; explicit causal preprocessing required")
        x = torch.from_numpy(raw).unsqueeze(0)
        with torch.no_grad():
            p = model(past_values=x, past_observed_mask=torch.ones_like(x, dtype=torch.bool),
                      return_dict=True).prediction_outputs[0, :horizon].numpy()
        if p.shape != (horizon, len(targets)) or not np.isfinite(p).all():
            raise ValueError("Malformed model prediction")
        origin = t.iloc[-1]
        predictions.append(p)
        origins.append(origin)
        times.append([origin + dt.iloc[0]*(i+1) for i in range(horizon)])
        audit.append(dict(group=str(group), observed_rows=len(raw), forecast_origin=origin.isoformat(),
                          input_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                          future_values_passed_to_model=False))
    return dict(target_columns=targets, target_prediction=np.asarray(predictions),
                timestamp=origins, timestamp_prediction=np.asarray(times, dtype="datetime64[ns]"),
                inference_audit=audit)
