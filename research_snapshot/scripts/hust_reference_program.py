"""Complete, editable train/predict interface for the qualified battery models."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

MODEL_KIND = 'author_crnn'


def train(path, output):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from author_net import CRNN
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    torch.set_num_threads(2)
    torch.manual_seed(366)
    np.random.seed(366)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    output.mkdir(parents=True, exist_ok=False)
    models = dict(ridge10=lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        lifetime_ridge10=lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        knn5=lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance')),
        rf200=lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=366, n_jobs=2),
        et200=lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=366, n_jobs=2),
        hgb100=lambda: HistGradientBoostingRegressor(max_iter=100, random_state=366))
    if MODEL_KIND in models:
        model = models[MODEL_KIND]()
        model.fit(d['x'], d['y'] + d['age'] if MODEL_KIND == 'lifetime_ridge10' else d['y'])
        joblib.dump(dict(model=model), output / (MODEL_KIND + '.joblib'))
        return
    assert MODEL_KIND == 'author_crnn'
    model = CRNN(ni=100, nc=4, no=64, nh=64)
    loader = DataLoader(TensorDataset(torch.from_numpy(d['nn']), torch.from_numpy((d['y']/3000).astype('float32')),
        torch.from_numpy((d['cap']/1000).astype('float32'))), batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(366))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    best = float('inf')
    for epoch in range(120):
        losses = []
        model.train()
        for x, rul, cap in loader:
            opt.zero_grad()
            r, q = model(x)
            loss = ((r.reshape(-1)-rul)**2).mean() + ((q.reshape(len(x), 10)-cap)**2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if np.mean(losses) < best:
            best = np.mean(losses)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=state), output / 'author_crnn.pt')


def predict(path, model_dir, output):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if MODEL_KIND == 'author_crnn':
        import torch
        from author_net import CRNN
        torch.set_num_threads(2)
        model = CRNN(ni=100, nc=4, no=64, nh=64)
        model.load_state_dict(torch.load(model_dir / 'author_crnn.pt', weights_only=True)['state_dict'])
        model.eval()
        with torch.no_grad():
            r, q = model(torch.from_numpy(d['nn']))
            p = r.reshape(-1).numpy() * 3000
    else:
        p = joblib.load(model_dir / (MODEL_KIND + '.joblib'))['model'].predict(d['x'])
        if MODEL_KIND == 'lifetime_ridge10':
            p -= d['age']
    output.write_text(json.dumps(dict(rul=np.maximum(p, 0).tolist())) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'predict'])
    p.add_argument('--train', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--model-dir', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'fit':
        train(a.train, a.model_dir)
    else:
        predict(a.input, a.model_dir, a.output)
