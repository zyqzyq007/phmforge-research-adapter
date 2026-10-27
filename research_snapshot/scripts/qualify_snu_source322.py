"""Recover the official v5 cutouts and run the predeclared measured-data check.

This is source qualification, not an Agent experiment or the original three-fault
component PHMForge task. No ring gear was damaged in the published dataset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def recover(root):
    prefix = root / 'source/SNU_Figshare_code.zip'
    rest = root / 'source/snu_remaining.bin'
    target = root / 'source/SNU_Figshare_code_full.zip'
    assert prefix.stat().st_size == 356515840
    assert rest.stat().st_size == 903449845
    with target.open('xb') as out:
        for part in (prefix, rest):
            with part.open('rb') as f:
                shutil.copyfileobj(f, out, 4 * 1024 * 1024)
    assert target.stat().st_size == 1259965685
    md5 = digest(target, 'md5')
    assert md5 == 'd755e1d3855b7daecac8a80d89093fde', 'Publisher checksum mismatch; no fitting allowed'
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        members = [dict(path=x.filename, bytes=x.file_size, CRC32=f'{x.CRC:08x}') for x in z.infolist()]
        for member in z.namelist():
            if member.endswith(('.mlx', '.m', '.txt')):
                (root / 'source/author_code' / Path(member).name).write_bytes(z.read(member))
    save(root / 'audit/WHOLE_PUBLISHER_ARCHIVE_VERIFIED322.json', dict(
        source='https://figshare.com/articles/dataset/32025606/5',
        bytes=target.stat().st_size, md5=md5, sha256=digest(target),
        publisher_MD5_matches=True, every_member_CRC_matches=True, members=members,
        reused_prefix_sha256=digest(prefix), remainder_sha256=digest(rest)))


def features(root):
    import h5py
    import numpy as np
    from scipy.io import loadmat
    archive = root / 'source/SNU_Figshare_code_full.zip'
    assert json.loads((root / 'audit/WHOLE_PUBLISHER_ARCHIVE_VERIFIED322.json').read_text())['publisher_MD5_matches']
    # Author Figure9 explicitly maps column 1 to vibration, 2 to angular
    # position, and 3 to rotational speed. Only vibration is used here.
    rows, audit = [], []
    feature_names = ['mean', 'std', 'rms', 'peak_abs', 'peak_to_peak',
                     'mean_abs', 'pearson_kurtosis', 'skewness', 'crest_factor',
                     'shape_factor', 'impulse_factor', 'zero_crossing_rate']
    feature_names += [f'log_power_band_{i:02}' for i in range(32)]
    window = 5 * 25600
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith('5m/') and n.endswith('.mat')]
        assert len(names) == 14
        for name in names:
            p = root / 'tmp/current_member.mat'
            p.write_bytes(z.read(name))
            key = Path(name).stem
            if h5py.is_hdf5(p):
                with h5py.File(p) as f:
                    a = np.asarray(f[key]).T
                decoder = 'h5py_matlab73_transposed'
            else:
                a = np.asarray(loadmat(p)[key])
                decoder = 'scipy_matlab5'
            assert a.shape == (7680001, 3) and np.isfinite(a).all(), (name, a.shape)
            # Fault location is publisher metadata, not inferred from amplitude.
            fault = key[3:]
            label = 0 if fault == 'n' else 2 if fault.startswith('s_') else 1
            stationary = key.startswith('xc_')
            for i in range(60):
                x = a[i*window:(i+1)*window, 0].astype('float64')
                mu, std = x.mean(), x.std()
                rms = float(np.sqrt(np.mean(x*x)))
                peak, absolute = float(np.max(np.abs(x))), float(np.mean(np.abs(x)))
                centered = x - mu
                value = [float(mu), float(std), rms, peak, float(np.ptp(x)), absolute,
                         float(np.mean(centered**4) / max(std**4, 1e-30)),
                         float(np.mean(centered**3) / max(std**3, 1e-30)),
                         peak/max(rms, 1e-15), rms/max(absolute, 1e-15),
                         peak/max(absolute, 1e-15), float(np.mean(x[1:]*x[:-1] < 0))]
                power = np.abs(np.fft.rfft(centered))**2 / (window*window)
                # Constant frequency bins, fixed before accessing any score.
                value += [float(np.log1p(v.sum())) for v in np.array_split(power, 32)]
                split = ('fit' if i < 36 else 'development' if i >= 48 else 'gap') if stationary else 'final'
                rows.append(dict(member=name, index=i, label=label, split=split, X=value))
            audit.append(dict(member=name, sha256=digest(p), decoder=decoder,
                              shape=list(a.shape), all_values_finite=True,
                              complete_windows=60, trailing_samples_excluded=1,
                              faulty_component=['healthy','planet_gear','sun_gear'][label]))
            del a
            p.unlink()
    for split in ['fit', 'development', 'final']:
        part = [r for r in rows if r['split'] == split]
        np.savez_compressed(root / ('features_' + split + '.npz'),
                            X=np.array([r['X'] for r in part]), y=np.array([r['label'] for r in part]),
                            source=np.array([r['member'] for r in part]), window=np.array([r['index'] for r in part]))
    save(root / 'audit/MEASURED_FEATURE_QUALIFICATION322.json', dict(
        members=audit, feature_names=feature_names, feature_columns=44,
        column_semantics_source='Author Figure9.mlx: vibration in first column; Figure9.txt retained',
        sampling_frequency_Hz=25600, disjoint_nonoverlapping_seconds_per_window=5,
        features_finite=all(np.isfinite(r['X']).all() for r in rows),
        counts={s:sum(r['split']==s for r in rows) for s in ['fit','development','gap','final']},
        source_labels_are_machined_damage=True, simulated_signals=False,
        independent_physical_replication=False, ring_gear_class_available=False,
        h5py_version=h5py.__version__, numpy_version=np.__version__))


def train(root):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
    fit = np.load(root / 'features_fit.npz', allow_pickle=False)
    dev = np.load(root / 'features_development.npz', allow_pickle=False)
    models = {
        'lda': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        'et': ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=322, n_jobs=2),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=322))}
    def metric(data, pred):
        return dict(macro_f1=float(f1_score(data['y'], pred, average='macro', labels=[0,1,2], zero_division=0)),
                    balanced_accuracy=float(balanced_accuracy_score(data['y'], pred)),
                    accuracy=float(accuracy_score(data['y'], pred)),
                    confusion_matrix=confusion_matrix(data['y'], pred, labels=[0,1,2]).tolist(),
                    per_source=[dict(source=str(s), n=int((data['source']==s).sum()),
                       accuracy=float(accuracy_score(data['y'][data['source']==s], pred[data['source']==s])))
                       for s in sorted(set(data['source']))])
    out = root / 'models'; out.mkdir(exist_ok=False)
    development, training = {}, {}
    for name, model in models.items():
        model.fit(fit['X'], fit['y'])
        joblib.dump(model, out / (name + '.joblib'))
        pred = model.predict(dev['X']); np.save(root / (name + '_development_prediction.npy'), pred)
        development[name] = metric(dev, pred)
        training[name] = dict(fit_samples=len(fit['y']), artifact_sha256=digest(out / (name + '.joblib')))
        if name == 'mlp':
            nn = model[-1]
            assert nn.n_iter_ > 1 and np.isfinite(nn.loss_curve_).all()
            training[name].update(executed_epochs=int(nn.n_iter_), samples_seen=int(nn.t_),
                                  loss_curve=[float(v) for v in nn.loss_curve_],
                                  parameter_tensors=len(nn.coefs_)+len(nn.intercepts_))
        if name == 'et': training[name]['actual_fitted_trees'] = len(model.estimators_)
    chosen = sorted(models, key=lambda n:(-development[n]['macro_f1'], n))[0]
    seal = dict(utc=datetime.now(timezone.utc).isoformat(), chosen_by_development_macro_f1=chosen,
                development=development, training=training, sklearn_version=sklearn.__version__,
                final_predictions_or_metrics_opened=False,
                final_features_sha256=digest(root/'features_final.npz'),
                source_scope='three classes including healthy, two actual faulty components; no ring-gear fault',
                new_agent_sessions=0)
    save(root / 'CHOICES_SEALED322.json', seal)
    final = np.load(root / 'features_final.npz', allow_pickle=False)
    result = {}
    for name, model in models.items():
        pred = model.predict(final['X']); np.save(root / (name+'_final_prediction.npy'), pred)
        result[name] = metric(final, pred)
    save(root / 'SOURCE_BASELINE_RESULTS322.json', dict(
        chosen=chosen, development=development, final=result, training=training,
        source_qualification_complete=True, original_task_complete=False,
        new_agent_experiments=0, new_confirmed_agent_failures=0,
        limitations=['No damaged ring-gear class', 'Only one recording per fault/condition configuration',
                     'Development uses a separated segment of the training recording',
                     'No risk, lifetime, or compliance labels',
                     'Test changes speed and load together; their effects are not separable'],
        interpretation='Qualification of measured source and real train/infer pipeline only; no Agent-specific claim.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--phase',choices=['recover','features','train'],required=True)
    args=p.parse_args();globals()[args.phase](args.run.resolve())
