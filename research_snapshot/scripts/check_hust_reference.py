"""Fresh-process replay: no fitting or optimizer is used in this verifier."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from author_net import CRNN

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--job', required=True)
args = parser.parse_args()
root, job_id = args.run.resolve(), args.job
job = json.loads((root / 'JOBS366.json').read_text())[job_id]
out = root / 'models' / job_id
path = root / 'clients-cache' / ('development.npz' if job_id == 'full' else 'train.npz')
with np.load(path, allow_pickle=False) as data:
    index = np.arange(len(data['y'])) if job_id == 'full' else np.asarray(job['validation_indices'])
    values = {k: data[k][index] for k in data.files}
with np.load(out / 'PREDICTIONS.npz', allow_pickle=False) as data:
    cached = {k: data[k] for k in data.files}
torch.set_num_threads(4)
count = 0
for name in ['ridge10', 'lifetime_ridge10', 'knn5', 'rf200', 'et200', 'hgb100']:
    checkpoint = joblib.load(out / (name + '.joblib'))
    assert checkpoint['fitting_cells'] == job['fitting_cells']
    pred = checkpoint['model'].predict(values['x'])
    if name == 'lifetime_ridge10':
        pred -= values['age']
    np.testing.assert_allclose(np.maximum(pred, 0), cached[name], rtol=1e-10, atol=1e-8)
    count += 1
net = CRNN(ni=100, nc=4, no=64, nh=64)
net.load_state_dict(torch.load(out / 'author_crnn.pt', weights_only=True)['state_dict'])
net.eval()
rul, capacity = [], []
with torch.no_grad():
    for start in range(0, len(values['nn']), 128):
        x = torch.from_numpy(values['nn'][start:start + 128])
        a, b = net(x)
        rul.append(a.reshape(-1).numpy() * 3000)
        capacity.append(b.reshape(len(x), 10).numpy() * 1000)
np.testing.assert_allclose(np.maximum(np.concatenate(rul), 0), cached['author_crnn'], rtol=1e-6, atol=1e-4)
np.testing.assert_allclose(np.concatenate(capacity), cached['cap_pred'], rtol=1e-6, atol=1e-4)
assert np.array_equal(values['y'], cached['actual_rul'])
print(json.dumps(dict(passed=True, models=count + 1, new_fits=0, new_optimizer_steps=0)))
