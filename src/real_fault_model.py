"""Real fault classifiers. No labels are loaded by the inference path."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real_rul_model import digest_state, write_json


class FaultNetwork(nn.Module):
    def __init__(self, architecture, features, classes):
        super().__init__()
        self.architecture = architecture
        if architecture == "mlp":
            self.net = nn.Sequential(nn.Linear(features, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes))
        elif architecture == "lstm":
            self.rnn = nn.LSTM(1, 24, batch_first=True)
            self.head = nn.Linear(24, classes)
        elif architecture == "transformer":
            self.project = nn.Linear(1, 24)
            self.position = nn.Parameter(torch.zeros(1, features, 24))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(24, 4, 48, dropout=0.1, batch_first=True),
                1, enable_nested_tensor=False)
            self.head = nn.Linear(24, classes)
        else:
            raise ValueError("Unsupported fault architecture")

    def forward(self, x):
        if self.architecture == "mlp":
            return self.net(x)
        if self.architecture == "lstm":
            x, _ = self.rnn(x.unsqueeze(-1))
            return self.head(x[:, -1])
        x = self.encoder(self.project(x.unsqueeze(-1)) + self.position)
        return self.head(x.mean(dim=1))


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(b) for b in x.split(128)]).argmax(dim=1).numpy()
    return float(np.mean(p == y))


def train(data, output, architecture="mlp", epochs=30):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    data, output = Path(data), Path(output)
    info = json.loads((data / "dataset.json").read_text())
    if architecture not in info["architectures"]:
        raise ValueError(f"{architecture} requires measured waveforms; available: {info['architectures']}")
    key = "features" if architecture == "mlp" else "signals"
    fit = np.load(data / "train.npz", allow_pickle=False)
    dev = np.load(data / "val.npz", allow_pickle=False)
    a, b = fit[key].astype(np.float32), dev[key].astype(np.float32)
    # Published labels may be names or non-contiguous integers. Encode only for
    # cross entropy; predictions retain the original labels in class_names.
    classes = info["class_names"]
    encoding = {value: index for index, value in enumerate(classes)}
    if len(encoding) != len(classes):
        raise ValueError("Duplicate class mapping")
    y = np.asarray([encoding[value] for value in fit["labels"].tolist()], dtype=np.int64)
    vy = np.asarray([encoding[value] for value in dev["labels"].tolist()], dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b) or a.shape[1] != b.shape[1]:
        raise ValueError("Empty or inconsistent feature matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite observations")
    if set(fit["groups"]) & set(dev["groups"]):
        raise ValueError("Train/validation recording groups overlap")
    mean, scale = a.mean(axis=0), np.maximum(a.std(axis=0), 1e-6)
    x = torch.from_numpy((a - mean) / scale)
    vx = torch.from_numpy((b - mean) / scale)
    targets = torch.from_numpy(y).long()
    torch.set_num_threads(2)
    torch.manual_seed(info["seed"])
    model = FaultNetwork(architecture, a.shape[1], len(info["class_names"]))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_accuracy = evaluate(model, vx, vy)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, state, history, updates, gradients, best_epoch = -1, None, [], 0, 0, None
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for indices in torch.randperm(len(x)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), targets[indices])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            gradients += int(bool(torch.isfinite(norm)) and float(norm) > 0)
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        accuracy = evaluate(model, vx, vy)
        history.append(dict(epoch=epoch, train_loss=total/count,
                            validation_accuracy=accuracy, updates=updates))
        if accuracy > best:
            best = accuracy
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
    checkpoint = dict(state_dict=state, architecture=architecture, input_key=key,
                      features=a.shape[1], classes=len(info["class_names"]),
                      class_names=info["class_names"], mean=torch.from_numpy(mean),
                      scale=torch.from_numpy(scale), dataset=info["dataset"])
    torch.save(checkpoint, output / "model.pt")
    torch.save(dict(checkpoint, state_dict=initial), output / "untrained_model.pt")
    audit = dict(dataset=info["dataset"], epochs_executed=epochs, optimizer_steps=updates,
                 nonzero_gradient_steps=gradients, initial_state_sha256=digest_state(initial),
                 selected_state_sha256=digest_state(state),
                 changed_parameter_tensors=sum(not torch.equal(state[k], initial[k]) for k in state),
                 training_file_sha256=hashlib.sha256((data/"train.npz").read_bytes()).hexdigest(),
                 initial_validation_accuracy=initial_accuracy, selected_validation_accuracy=best,
                 selected_epoch=best_epoch, class_names=classes,
                 train_rows=len(a), validation_rows=len(b), selection="development accuracy only",
                 group_overlap=[], training_only_normalization=True, test_labels_accessed=False,
                 group_definition=info["group_definition"], history=history)
    write_json(output / "training_audit.json", audit)
    return audit


def infer(checkpoint_path, input_path, output_path):
    torch.set_num_threads(2)
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = FaultNetwork(ckpt["architecture"], ckpt["features"], ckpt["classes"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    data = np.load(input_path, allow_pickle=False)
    if "labels" in data:
        raise ValueError("Inference input must contain observations only, not labels")
    raw = data[ckpt["input_key"]].astype(np.float32)
    identities = data["sample_ids"]
    if (raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all()
            or raw.shape[1] != ckpt["features"] or len(identities) != len(raw)
            or len(set(identities.tolist())) != len(identities)):
        raise ValueError("Invalid observation dimensions/values")
    if "dataset" in data and data["dataset"].item() != ckpt["dataset"]:
        raise ValueError("Model and observation dataset identities differ")
    x = torch.from_numpy((raw - ckpt["mean"].numpy()) / ckpt["scale"].numpy())
    model.eval()
    with torch.no_grad():
        prob = torch.cat([model(b).softmax(-1) for b in x.split(128)]).numpy()
    rows = [dict(sample_id=str(i), predicted_label=ckpt["class_names"][int(j)],
                 probabilities=p.tolist()) for i, j, p in zip(data["sample_ids"], prob.argmax(1), prob)]
    value = dict(predictions=rows, checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                 input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest())
    write_json(output_path, value)
    return value
