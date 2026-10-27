"""Train and run an actual CPU RUL predictor; inference never reads RUL labels.

Research replacement for PHMForge's simplified model backend, not its original
implementation. NASA's simulated trajectories and uncapped evaluation labels
remain unchanged. Train/validation engines and all fitting choices are explicit.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
COLUMNS = [4 + sensor for sensor in SENSORS]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest_state(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def read_series(path):
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 26 or not np.isfinite(raw).all():
        raise ValueError("Expected finite original 26-column C-MAPSS observations")
    for unit in np.unique(raw[:, 0]).astype(int):
        cycles = raw[raw[:, 0] == unit, 1]
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise ValueError(f"Invalid cycle ordering for unit {unit}")
    return raw


def window_at(series, endpoint, window):
    result = series[max(0, endpoint - window):endpoint, COLUMNS]
    if len(result) < window:
        result = np.concatenate(
            [np.repeat(result[:1], window - len(result), axis=0), result], axis=0
        )
    return result


def development_windows(raw, ids, config, validation=False):
    xs, ys, records = [], [], []
    for unit in ids:
        series = raw[raw[:, 0] == unit]
        if len(series) < config["window"]:
            raise ValueError("Development trajectory shorter than the window")
        if validation:
            endpoints = sorted(set(
                max(config["window"], min(len(series) - 1, int(len(series) * f)))
                for f in config["validation_prefix_fractions"]
            ))
        else:
            endpoints = list(range(config["window"], len(series) + 1,
                                   config["training_stride"]))
        for endpoint in endpoints:
            xs.append(window_at(series, endpoint, config["window"]))
            ys.append(float(len(series) - endpoint))
            records.append({"unit_id": int(unit), "endpoint_cycle": int(endpoint)})
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), records


class RULNetwork(nn.Module):
    def __init__(self, architecture, window, channels):
        super().__init__()
        self.architecture = architecture
        if architecture == "mlp":
            self.network = nn.Sequential(
                nn.Flatten(), nn.Linear(window * channels, 96), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(96, 48), nn.ReLU(),
                nn.Linear(48, 1),
            )
        elif architecture == "lstm":
            self.recurrent = nn.LSTM(channels, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        elif architecture == "transformer":
            self.projection = nn.Linear(channels, 32)
            self.position = nn.Parameter(torch.zeros(1, window, 32))
            layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64,
                                               dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(32, 1)
        else:
            raise ValueError(architecture)

    def forward(self, x):
        if self.architecture == "mlp":
            return self.network(x).squeeze(-1)
        if self.architecture == "transformer":
            return self.head(self.encoder(self.projection(x) + self.position)[:, -1]).squeeze(-1)
        output, _ = self.recurrent(x)
        return self.head(output[:, -1]).squeeze(-1)


def predict_array(model, windows, mean, scale, cap):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(((windows - mean) / scale).astype(np.float32))
        result = torch.cat([model(batch) for batch in x.split(256)]).numpy()
    return np.clip(result * cap, 0.0, cap)


def errors(predicted, observed):
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {"mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.mean(residual ** 2)))}


def train(data, output, architecture, shuffle_labels=False, epochs_override=None):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((data / "training_config.json").read_text())
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    raw = read_series(data / "train.txt")
    train_ids, validation_ids = config["train_engine_ids"], config["validation_engine_ids"]
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids + validation_ids) == set(raw[:, 0].astype(int))
    train_x, train_y, train_records = development_windows(raw, train_ids, config)
    val_x, val_y, val_records = development_windows(raw, validation_ids, config, True)
    # Fit normalization only on the training engines, never validation/test.
    training_raw = raw[np.isin(raw[:, 0], train_ids)][:, COLUMNS]
    mean = training_raw.mean(axis=0).astype(np.float32)
    scale = training_raw.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    cap = float(config["training_rul_cap"])
    targets = np.minimum(train_y, cap) / cap
    if shuffle_labels:
        targets = np.random.default_rng(seed + 1).permutation(targets)
    x = torch.from_numpy(((train_x - mean) / scale).astype(np.float32))
    y = torch.from_numpy(targets.astype(np.float32))
    model = RULNetwork(architecture, config["window"], len(SENSORS))
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_hash = digest_state(initial_state)
    initial_pred = predict_array(model, val_x, mean, scale, cap)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, updates, nonzero_gradients = [], 0, 0
    best_error, best_state, best_epoch = float("inf"), None, None
    epochs = config["epochs"] if epochs_override is None else epochs_override
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total, count, grad_max = 0.0, 0, 0.0
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[indices])
            loss = torch.mean((predicted - y[indices]) ** 2)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            nonzero_gradients += int(np.isfinite(gradient) and gradient > 0)
            grad_max = max(grad_max, gradient)
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        val_prediction = predict_array(model, val_x, mean, scale, cap)
        metrics = errors(val_prediction, val_y)
        row = {"epoch": epoch, "training_mse_normalized": total / count,
               "validation_uncapped": metrics, "updates": updates,
               "largest_gradient_norm": grad_max}
        history.append(row)
        with (output / "training_history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if metrics["mae"] < best_error:
            best_error, best_epoch = metrics["mae"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    final_hash = digest_state(best_state)
    changed = sum(not torch.equal(best_state[k], initial_state[k]) for k in best_state)
    checkpoint = {"state_dict": best_state, "mean": torch.from_numpy(mean),
                  "scale": torch.from_numpy(scale), "architecture": architecture,
                  "window": config["window"], "sensors": SENSORS, "rul_cap": cap}
    torch.save(checkpoint, output / "model.pt")
    torch.save(dict(checkpoint, state_dict=initial_state), output / "untrained_model.pt")
    metadata = {
        "architecture": architecture, "seed": seed, "torch_version": str(torch.__version__),
        "training_source_sha256": hashlib.sha256((data / "train.txt").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((data / "training_config.json").read_bytes()).hexdigest(),
        "training_engine_ids": train_ids, "validation_engine_ids": validation_ids,
        "training_windows": len(train_x), "validation_windows": len(val_x),
        "train_validation_engine_overlap": sorted(set(train_ids) & set(validation_ids)),
        "optimizer": "Adam", "optimizer_steps": updates,
        "nonzero_gradient_steps": nonzero_gradients, "changed_parameter_tensors": changed,
        "initial_state_sha256": initial_hash, "selected_state_sha256": final_hash,
        "selected_epoch": best_epoch, "epochs_executed": epochs,
        "checkpoint_selection": "minimum uncapped MAE on held-out training engines",
        "initial_validation": errors(initial_pred, val_y),
        "selected_validation": errors(predict_array(model, val_x, mean, scale, cap), val_y),
        "constant_validation": errors(np.full(len(val_y), np.median(train_y)), val_y),
        "labels_shuffled_control": shuffle_labels,
        "evaluation_target": "uncapped remaining cycles; training target capped at 125",
        "real_test_labels_accessed": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    write_json(output / "training_audit.json", metadata)
    write_json(output / "development_predictions.json", [
        dict(record, actual_rul=float(actual), predicted_rul=float(prediction))
        for record, actual, prediction in zip(
            val_records, val_y, predict_array(model, val_x, mean, scale, cap))
    ])
    return metadata


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["sensors"] != SENSORS:
        raise ValueError("Checkpoint feature contract differs")
    model = RULNetwork(checkpoint["architecture"], checkpoint["window"], len(SENSORS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def infer(checkpoint_path, input_path, output_path):
    torch.set_num_threads(2)
    checkpoint_path, input_path = Path(checkpoint_path), Path(input_path)
    model, checkpoint = load_model(checkpoint_path)
    identities = None
    if input_path.suffix == '.npz':
        with np.load(input_path, allow_pickle=False) as data:
            if set(data.files) & {'labels', 'actual_rul', 'groups'}:
                raise ValueError('Inference requires observation-only windows')
            windows = data['windows'].astype(np.float32)
            units = data['unit_ids'].astype(int).tolist()
            identities = data['sample_ids'].astype(str).tolist()
            endpoints = data['endpoint_cycles'].astype(int).tolist()
        if (windows.shape != (len(units), checkpoint['window'], len(SENSORS))
                or not np.isfinite(windows).all() or not len(units)
                or len(set(identities)) != len(units) or len(endpoints) != len(units)):
            raise ValueError('Invalid C-MAPSS window identities/dimensions')
    else:
        raw = read_series(input_path)
        units = sorted(np.unique(raw[:, 0]).astype(int))
        windows = np.asarray([
            window_at(raw[raw[:, 0] == unit], int((raw[:, 0] == unit).sum()),
                      checkpoint["window"]) for unit in units
        ], dtype=np.float32)
    predicted = predict_array(model, windows, checkpoint["mean"].numpy(),
                              checkpoint["scale"].numpy(), checkpoint["rul_cap"])
    rows = [{"unit_id": int(unit), "predicted_rul": float(value)}
            for unit, value in zip(units, predicted)]
    if identities is not None:
        for row, identity, endpoint in zip(rows, identities, endpoints):
            row.update(sample_id=identity, endpoint_cycle=endpoint)
    payload = {"predictions": rows,
                            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
    write_json(output_path, payload)
    return dict(payload, units=len(rows), minimum=float(predicted.min()),
                maximum=float(predicted.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--out", required=True, type=Path)
    fit.add_argument("--arch", choices=["mlp", "lstm", "transformer"], required=True)
    fit.add_argument("--shuffle-labels", action="store_true")
    fit.add_argument("--epochs", type=int)
    pred = sub.add_parser("predict")
    pred.add_argument("--model", required=True, type=Path)
    pred.add_argument("--input", required=True, type=Path)
    pred.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.data, args.out, args.arch, args.shuffle_labels, args.epochs)
    else:
        result = infer(args.model, args.input, args.out)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
