"""Genuine train-only reference fits, with grouped validation and cold replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from author_net import CRNN


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metrics(y, predicted, groups):
    assert np.isfinite(predicted).all() and len(y) == len(predicted)
    rows = [dict(cell=str(g), n=int(np.sum(groups == g)),
        MAE=float(np.mean(np.abs(y[groups == g] - predicted[groups == g])))) for g in np.unique(groups)]
    mae = float(np.mean([r['MAE'] for r in rows]))
    independent = np.mean([mean_absolute_error(y[groups == g], predicted[groups == g]) for g in np.unique(groups)])
    assert abs(mae - independent) < 1e-8
    return dict(mean_cell_MAE=mae, pooled_RMSE=float(np.sqrt(np.mean((y - predicted)**2))), cells=rows)


def network_predictions(model, values):
    rul, capacity = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 128):
            batch = torch.from_numpy(values[start:start + 128])
            a, b = model(batch)
            rul.append(a.reshape(-1).numpy() * 3000.0)
            capacity.append(b.reshape(len(batch), 10).numpy() * 1000.0)
    return np.maximum(np.concatenate(rul), 0), np.concatenate(capacity)


def fit_job(root, job_id):
    read = lambda p: json.loads(Path(p).read_text())
    save = lambda p, d: Path(p).write_text(json.dumps(d, indent=2) + '\n')
    config, job = read(root / 'PROTOCOL366.json'), read(root / 'JOBS366.json')[job_id]
    out = root / 'models' / job_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    save(out / 'STARTED.json', dict(utc=datetime.now(timezone.utc).isoformat(), job=job,
        source_final_accessed=False, actual_fit_entries=1))
    with np.load(root / 'clients-cache/train.npz', allow_pickle=False) as source:
        train = {k: source[k][job['fit_indices']] for k in source.files}
        if job_id != 'full':
            valid = {k: source[k][job['validation_indices']] for k in source.files}
    if job_id == 'full':
        with np.load(root / 'clients-cache/development.npz', allow_pickle=False) as source:
            valid = {k: source[k] for k in source.files}
    assert sorted(set(train['groups'].tolist())) == job['fitting_cells']
    assert not set(train['groups']) & set(valid['groups'])
    assert len(train['x']) > 0
    threads = config['budget']['threads_per_job']
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    seed = config['neural']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    models = dict(
        ridge10=make_pipeline(StandardScaler(), Ridge(alpha=10)),
        lifetime_ridge10=make_pipeline(StandardScaler(), Ridge(alpha=10)),
        knn5=make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance')),
        rf200=RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=threads),
        et200=ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=threads),
        hgb100=HistGradientBoostingRegressor(max_iter=100, random_state=seed))
    events, predictions, capacity_predictions = [], {}, {}
    for name, model in models.items():
        target = train['y'] + train['age'] if name == 'lifetime_ridge10' else train['y']
        event = dict(model=name, actual_estimator=type(model).__name__, rows=len(target),
            fitting_cells=job['fitting_cells'], target='lifetime' if name == 'lifetime_ridge10' else 'RUL', status='started')
        events.append(event)
        save(out / 'FIT_EVENTS.json', events)
        tick = time.monotonic()
        model.fit(train['x'], target)
        warm = model.predict(valid['x'])
        if name == 'lifetime_ridge10':
            warm -= valid['age']
        warm = np.maximum(warm, 0)
        path = out / (name + '.joblib')
        joblib.dump(dict(model=model, fitting_cells=job['fitting_cells'], model_id=name), path)
        cold = joblib.load(path)['model'].predict(valid['x'])
        if name == 'lifetime_ridge10':
            cold -= valid['age']
        np.testing.assert_allclose(warm, np.maximum(cold, 0), rtol=1e-10, atol=1e-8)
        predictions[name] = warm
        event.update(status='completed', seconds=time.monotonic() - tick,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cold_replay_passed=True)
        save(out / 'FIT_EVENTS.json', events)
    net = CRNN(ni=100, nc=4, no=64, nh=64)
    initial_sha = state_sha(net)
    torch.save(dict(state_dict=net.state_dict(), fitting_cells=job['fitting_cells']), out / 'initial.pt')
    initial_rul, initial_cap = network_predictions(net, train['nn'])
    initial_loss = float(np.mean(((initial_rul - train['y']) / 3000)**2) +
                         np.mean(((initial_cap - train['cap']) / 1000)**2))
    loader = DataLoader(TensorDataset(torch.from_numpy(train['nn']),
        torch.from_numpy((train['y'] / 3000).astype(np.float32)),
        torch.from_numpy((train['cap'] / 1000).astype(np.float32))),
        batch_size=config['neural']['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(net.parameters(), lr=config['neural']['learning_rate'])
    history, steps, changed, best_loss = [], 0, 0, float('inf')
    params = list(net.parameters())
    for epoch in range(config['neural']['epochs']):
        assert time.monotonic() - started < config['budget']['job_seconds'], 'Whole fit job wall budget'
        net.train()
        losses = []
        for x, rul, cap in loader:
            optimizer.zero_grad()
            r, q = net(x)
            loss = torch.mean((r.reshape(-1) - rul)**2) + torch.mean((q.reshape(len(x), 10) - cap)**2)
            assert torch.isfinite(loss)
            loss.backward()
            before = [p.detach().reshape(-1)[:8].clone() for p in params]
            optimizer.step()
            steps += 1
            changed += any(not torch.equal(a, p.detach().reshape(-1)[:8]) for a, p in zip(before, params))
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        save(out / 'NEURAL_PROGRESS.json', dict(epoch=epoch + 1, loss=epoch_loss, optimizer_steps=steps,
            sampled_nonzero_update_steps=changed, elapsed_seconds=time.monotonic() - started))
        assert steps <= config['budget']['neural_steps_max_per_job']
    net.load_state_dict(best_state)
    selected_sha = state_sha(net)
    assert steps > 0 and changed > 0 and initial_sha != selected_sha
    torch.save(dict(state_dict=net.state_dict(), fitting_cells=job['fitting_cells'],
        initial_state_sha256=initial_sha, selected_state_sha256=selected_sha,
        optimizer_steps=steps, loss_history=history, epoch_selected=best_epoch), out / 'author_crnn.pt')
    warm_rul, warm_cap = network_predictions(net, valid['nn'])
    reload = CRNN(ni=100, nc=4, no=64, nh=64)
    reload.load_state_dict(torch.load(out / 'author_crnn.pt', weights_only=True)['state_dict'])
    cold_rul, cold_cap = network_predictions(reload, valid['nn'])
    np.testing.assert_allclose(warm_rul, cold_rul, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(warm_cap, cold_cap, rtol=1e-6, atol=1e-4)
    predictions['author_crnn'] = warm_rul
    capacity_predictions['author_crnn'] = warm_cap
    report = dict(job_id=job_id, fitting_cells=job['fitting_cells'], validation_cells=sorted(set(valid['groups'].tolist())),
        fitting_rows=len(train['y']), validation_rows=len(valid['y']), completed_sklearn_fits=6,
        completed_neural_fits=1, optimizer_steps=steps, sampled_nonzero_update_steps=changed,
        initial_state_sha256=initial_sha, selected_state_sha256=selected_sha, loss_history=history,
        initial_full_training_combined_loss=initial_loss, selected_training_epoch=best_epoch,
        in_process_checkpoint_reload_predictions_agree=True, final_source_values_accessed=False,
        metrics={name: metrics(valid['y'], p, valid['groups']) for name, p in predictions.items()},
        capacity_MAE_mAh=float(np.mean(np.abs(warm_cap - valid['cap']))), seconds=time.monotonic() - started)
    np.savez_compressed(out / 'PREDICTIONS.npz', **predictions, groups=valid['groups'], cycle=valid['cycle'],
        actual_rul=valid['y'], cap_pred=warm_cap, actual_cap=valid['cap'])
    save(out / 'REPORT.json', report)
    save(out / 'TERMINAL.json', dict(exit_code=0, completed_models=7, optimizer_steps=steps))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    report = fit_job(args.run.resolve(), args.job)
    print(json.dumps(dict(job=args.job, completed_models=7, optimizer_steps=report['optimizer_steps'])))
