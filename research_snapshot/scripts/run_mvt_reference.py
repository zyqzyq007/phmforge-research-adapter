"""Execute the immutable official MVT-Flow tests, training and score replay."""
import argparse
import builtins
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
import traceback
from types import MethodType


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(body):
    return hashlib.sha256(body).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight','preflight_seed177','train','replay'], required=True)
    args = parser.parse_args()
    R = args.run.resolve()
    protocol = read(R/'protocol_initial59.json')
    dep = read(R/'model_dependency59.json')
    source = Path(protocol['data']['original_code'])
    assert sha(Path(dep['directory'])/'DEPENDENCY_MANIFEST.json') == dep['manifest_SHA256']
    assert sha(Path(dep['pyarrow']['directory'])/'DEPENDENCY_MANIFEST.json') == dep['pyarrow']['manifest_SHA256']
    sys.path[:0] = [dep['directory'],dep['pyarrow']['directory'],str(source)]
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    import numpy as np
    import pandas as pd
    import torch
    from sklearn import metrics
    import train
    import voraus_ad

    torch.set_num_threads(protocol['CPU_threads'])
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '1.12.1+cpu' and np.__version__ == '1.23.4'
    train.DATASET_PATH = Path(protocol['data']['path'])
    train.DEVICE = torch.device('cpu')
    is_preflight = args.mode.startswith('preflight')
    train.MODEL_PATH = None if is_preflight else R/'model_final.pth'
    start = time.monotonic()
    state = dict(status='running', pid=os.getpid(), mode=args.mode,
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 CPU_threads=torch.get_num_threads(), GPU_used=False, new_LLM_calls=0)
    status_path = R/f'{args.mode}_status.json'
    save(status_path, state)
    captured = {}
    budget = protocol['preflight_budget_s'] if is_preflight else protocol['training_time_budget_s']

    def timeout(_signum, _frame):
        raise TimeoutError('Fixed '+args.mode+' wall-clock budget expired')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(budget)
    original_print = builtins.print

    def capture_print(*items, **kwargs):
        original_print(*items, **kwargs)
        text = ' '.join(str(x) for x in items)
        match = re.match(r'Epoch (\d+): auroc\(mean\)=([\d.]+), loss=([-\d.]+)',text)
        if args.mode == 'train' and match:
            state.update(completed_epochs=int(match[1])+1, elapsed_seconds=time.monotonic()-start,
                         latest_original_epoch_print=text, updated_utc=datetime.now(timezone.utc).isoformat())
            save(status_path,state)
            with (R/'training_progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(state))+'\n')

    def export_scaler(obj):
        return dict(columns=list(voraus_ad.Signals.machine()),mean=obj.mean_.tolist(),
                    scale=obj.scale_.tolist(),var=obj.var_.tolist(),
                    n_samples_seen=int(obj.n_samples_seen_),fit_scope='All unpadded PRE_A training rows only, official loader')

    original_scaler = voraus_ad.StandardScaler

    def track_scaler(*items, **kwargs):
        obj = original_scaler(*items, **kwargs)
        captured['scaler'] = obj
        return obj

    original_loader = train.load_torch_dataloaders

    def track_loader(*items, **kwargs):
        result = original_loader(*items, **kwargs)
        captured['data'] = result
        tr, te, _, _ = result
        assert len(tr) == 948 and len(te) == 1174
        assert sum(bool(x['anomaly']) for x in te.labels) == 755
        assert all(not x['anomaly'] and x['variant'] == 'PRE_A' for x in tr.labels)
        assert len(tr.columns) == 130 and all(tuple(x.shape)==(1164,130) for x in tr.tensors+te.tensors)
        suffix = '_replay59' if args.mode == 'replay' else '59'
        save(R/f'split_and_shape{suffix}.json',dict(training_ids=[x['sample'] for x in tr.labels],
             test_ids=[x['sample'] for x in te.labels],train=948,test=1174,train_normal=948,
             test_normal=419,test_anomalous=755,columns=tr.columns,padded_record_shape=[1164,130]))
        save(R/f'standard_scaler{suffix}.json',export_scaler(captured['scaler']))
        if args.mode == 'replay':
            assert sha(R/'standard_scaler_replay59.json') == sha(R/'standard_scaler59.json')
            assert sha(R/'split_and_shape_replay59.json') == sha(R/'split_and_shape59.json')
        return result

    def score(model, dataloader):
        rows = []
        model.eval()
        with torch.no_grad():
            for tensors, labels in dataloader:
                z,jac = model.forward(tensors.float().transpose(2,1))
                jac = torch.sum(jac,dim=tuple(range(1,jac.dim())))
                scores = train.get_loss_per_sample(z,jac)
                assert bool(torch.isfinite(scores).all())
                for j in range(len(scores)):
                    label = {k:v[j].item() if isinstance(v,torch.Tensor) else v[j] for k,v in labels.items()}
                    rows.append(dict(label,score=scores[j].item()))
        return rows

    def evaluate(model, data, suffix):
        tr,te,_,test_loader = data
        train_loader = torch.utils.data.DataLoader(tr,batch_size=train.configuration.batchsize,shuffle=False)
        normal_rows = score(model,train_loader)
        rows = score(model,test_loader)
        threshold = float(np.percentile([x['score'] for x in normal_rows],99))
        for kind, values in [('train',normal_rows),('test',rows)]:
            fields = list(values[0]) + ['prediction']
            with (R/f'{kind}_scores_{suffix}.csv').open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(dict(x,prediction=x['score']>threshold) for x in values)
        df = pd.DataFrame(rows)
        categories = []
        for cat in voraus_ad.ANOMALY_CATEGORIES:
            selected = df[(df['category']==cat.name) | ~df['anomaly']]
            actual = df[df['category']==cat.name]
            categories.append(dict(category=cat.name,anomaly_records=len(actual),healthy_controls=419,
                AUROC=float(metrics.roc_auc_score(selected['anomaly'],selected['score'])),
                true_positives=int((actual['score']>threshold).sum()),
                false_negatives=int((actual['score']<=threshold).sum())))
        macro = float(np.mean([x['AUROC'] for x in categories]))
        adequacy = macro>=0.85 and all(x['AUROC']>=0.5 for x in categories)
        result = dict(status='finished',selected_epoch=70,epochs_selected_using_test=False,
            macro_AUROC=macro,pooled_AUROC=float(metrics.roc_auc_score(df['anomaly'],df['score'])),
            categories=categories,reference_adequacy_gate_passed=adequacy,
            train99percentile_threshold=threshold,threshold_scope='In-sample normal training only; exploratory calibration',
            healthy_test=419,anomalous_test=755,training=948,
            false_positives=int(((~df['anomaly']) & (df['score']>threshold)).sum()),
            true_positives=int(((df['anomaly']) & (df['score']>threshold)).sum()),
            physical_robots=1,source_families=1,labels_are_record_level=True,
            fault_onset_latency_or_maintenance_damage_measured=False,new_LLM_calls=0,E3=0,
            model_SHA256=sha(R/'model_final.pth'))
        save(R/f'reference_metrics_{suffix}.json',result)
        return result

    try:
        save(R/f'{args.mode}_environment59.json',dict(Python=sys.version,torch=torch.__version__,numpy=np.__version__,
             pandas=pd.__version__,official_code=str(source),official_code_commit=protocol['data']['code_commit'],
             original_train_source_SHA256=sha(source/'train.py'),configuration=train.configuration.dict(by_alias=True),
             allowed_GPU_count=0,threads=torch.get_num_threads()))
        if is_preflight:
            import pytest
            plugins = []
            if args.mode == 'preflight_seed177':
                assert read(R/'preflight_status.json')['status'] == 'failed'

                class RestoreTrainingSeed:
                    def pytest_runtest_setup(self, item):
                        if item.name == 'test_train':
                            before = sha_bytes(torch.get_rng_state().numpy().tobytes())
                            torch.manual_seed(177)
                            torch.cuda.manual_seed_all(177)
                            np.random.seed(177)
                            random.seed(177)
                            save(R/'training_test_seed_control59.json',dict(seed=177,
                                 torch_rng_before_SHA256=before,
                                 torch_rng_after_SHA256=sha_bytes(torch.get_rng_state().numpy().tobytes()),
                                 changed_only='Restore original configured RNG seeds at test_train entry',
                                 original_test_source_unmodified=True))

                plugins = [RestoreTrainingSeed()]
            junit = 'author_pytest59.xml' if args.mode == 'preflight' else 'author_pytest_seed177_59.xml'
            exit_code = pytest.main([str(source/'tests'),'-o','addopts=','-o',f'cache_dir={R}/clients-cache/pytest',
                                    '--junitxml',str(R/'logs'/junit),'-q'],plugins=plugins)
            assert exit_code == 0, 'Author numerical tests failed; inspect retained outputs'
        else:
            voraus_ad.StandardScaler = track_scaler
            train.load_torch_dataloaders = track_loader
            if args.mode == 'train':
                first = read(R/'preflight_status.json')['status']
                assert first == 'finished' or read(R/'preflight_seed177_status.json')['status'] == 'finished'
                original_model = train.NormalizingFlow

                def track_model(*items, **kwargs):
                    save(R/'production_initial_rng59.json',dict(seed=train.configuration.seed,
                         torch_rng_SHA256=sha_bytes(torch.get_rng_state().numpy().tobytes()),
                         numpy_rng_algorithm=np.random.get_state()[0],
                         numpy_rng_state_SHA256=sha_bytes(np.random.get_state()[1].tobytes()),
                         shape=list(items[0])))
                    model = original_model(*items, **kwargs)
                    captured['model'] = model
                    return model

                train.NormalizingFlow = track_model
                original_adam = torch.optim.Adam

                def track_adam(*items, **kwargs):
                    optimizer = original_adam(*items, **kwargs)
                    captured['optimizer'] = optimizer
                    original_step = optimizer.step

                    def step(_optimizer, *values, **options):
                        output = original_step(*values, **options)
                        state['optimizer_steps'] = state.get('optimizer_steps',0)+1
                        if state['optimizer_steps'] == 1 or state['optimizer_steps']%5 == 0:
                            state.update(updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
                            save(status_path,state)
                        return output

                    optimizer.step = MethodType(step,optimizer)
                    return optimizer

                torch.optim.Adam = track_adam
                builtins.print = capture_print
                results = train.train()
                builtins.print = original_print
                assert len(results) == 70 and state.get('completed_epochs') == 70
                save(R/'original_training_results59.json',results)
                signal.alarm(1800)
                result = evaluate(captured['model'],captured['data'],'final')
                original_print(json.dumps(dict(final_macro_AUROC=result['macro_AUROC'],adequacy=result['reference_adequacy_gate_passed'])))
            else:
                assert read(R/'train_status.json')['status'] == 'finished'
                data = track_loader(dataset=train.DATASET_PATH,batch_size=train.configuration.batchsize,
                    columns=voraus_ad.Signals.machine(),seed=177,frequency_divider=1,train_gain=1.0,normalize=True,pad=True)
                model = train.NormalizingFlow((130,1164),train.configuration).float()
                model.load_state_dict(torch.load(R/'model_final.pth',map_location='cpu'))
                evaluate(model,data,'replay')
                prior = pd.read_csv(R/'test_scores_final.csv').sort_values('sample')
                current = pd.read_csv(R/'test_scores_replay.csv').sort_values('sample')
                assert prior[['sample','anomaly','category','variant']].equals(current[['sample','anomaly','category','variant']])
                difference = float(np.max(np.abs(prior['score'].values-current['score'].values)))
                assert difference<=1e-4
                save(R/'cold_replay59.json',dict(status='passed',records=1174,
                     score_max_absolute_difference=difference,score_tolerance=1e-4,
                     predictions_identical=bool(prior['prediction'].equals(current['prediction'])),
                     original_model_SHA256=sha(R/'model_final.pth')))
                original_print(json.dumps(dict(replay='passed',records=1174,max_score_difference=difference)))
        state.update(status='finished',ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save(status_path,state)
    except BaseException as error:
        builtins.print = original_print
        signal.alarm(0)
        state.update(status='failed',error_type=type(error).__name__,error=str(error),
                     ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save(status_path,state)
        if args.mode == 'train' and 'model' in captured:
            torch.save(captured['model'].state_dict(),R/'model_partial_error.pth')
        traceback.print_exc()
        raise
    finally:
        builtins.print = original_print
        signal.alarm(0)


if __name__ == '__main__':
    main()
