"""Past-only capacity checks of real, held-cell CRNN updates; no Agent claims."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from author_net import CRNN
from snu_order_control import read, save, sha, utc, packet


def state_hash(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_from(path):
    model = CRNN(ni=100, nc=4, no=64, nh=64)
    model.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
    model.eval()
    return model


def observe(model, x, past):
    with torch.no_grad():
        r, q = model(x)
    q = q.reshape(10).cpu().numpy()
    target = past.cpu().numpy()
    return dict(rul=max(0., float(r.reshape(-1)[0])*3000), capacity_scaled=q.tolist(),
        fit6_MSE=float(np.mean((q[:6]-target[:6])**2)),
        hold3_MSE=float(np.mean((q[6:9]-target[6:9])**2)),
        all9_MSE=float(np.mean((q[:9]-target)**2)))


def worker(root, cfg, context):
    import sys
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)) and '/private/' in str(args[0]):
            raise PermissionError('Current RUL and current capacity are not adaptation inputs')
    sys.addaudithook(guard)
    started = time.monotonic(); data = packet(root/'legal/observations.npz')
    job = cfg['contexts'][context]; initial_path = Path(job['checkpoint'])
    assert sha(initial_path) == job['checkpoint_sha256']
    model = model_from(initial_path); initial = copy.deepcopy(model.state_dict())
    baseline_hash = state_hash(model); rows = []
    for index in job['query_indices']:
        folder = root/'models'/f'q{index:02d}'; folder.mkdir()
        x = torch.from_numpy(data['nn'][index:index+1])
        past = torch.from_numpy(data['past_cap'][index]/1000)
        model.load_state_dict(initial); model.eval()
        baseline = observe(model, x, past)
        weight = model.rul.weight.detach().cpu().numpy().reshape(10)
        bias = float(model.rul.bias.detach().cpu()[0])
        q_exact = np.asarray(baseline['capacity_scaled'], dtype=np.float32)
        q_exact[:9] = past.numpy()
        exact = max(0., float(q_exact@weight+bias)*3000)
        paths = {}
        for count in [6, 9]:
            model.load_state_dict(initial); model.eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
            observations = [dict(step=0, **baseline, state_hash=baseline_hash,
                                 checkpoint=str(initial_path), checkpoint_sha256=job['checkpoint_sha256'])]
            updates = []
            for step in range(1, cfg['steps']+1):
                optimizer.zero_grad(set_to_none=True)
                _, q = model(x)
                loss = torch.mean((q.reshape(10)[:count]-past[:count])**2)
                assert torch.isfinite(loss)
                loss.backward()
                assert model.rul.weight.grad is None and model.rul.bias.grad is None
                gradient = float(sum(torch.sum(p.grad*p.grad).detach() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                updates.append(dict(step=step, fit_loss_before=float(loss.detach()), gradient_squared_norm=gradient))
                if step in cfg['snapshots']:
                    checkpoint = folder/f'fit{count}_step{step}.pt'
                    torch.save(dict(state_dict=model.state_dict()), checkpoint)
                    observations.append(dict(step=step, **observe(model, x, past), state_hash=state_hash(model),
                        checkpoint=str(checkpoint.relative_to(root)), checkpoint_sha256=sha(checkpoint)))
            assert state_hash(model) != baseline_hash
            assert torch.equal(model.rul.weight, initial['rul.weight']) and torch.equal(model.rul.bias, initial['rul.bias'])
            paths[str(count)] = dict(snapshots=observations, optimizer_steps=updates)
        self6 = min(paths['6']['snapshots'], key=lambda z: (z['fit6_MSE'], z['step']))
        hold3 = min(paths['6']['snapshots'], key=lambda z: (z['hold3_MSE'], z['step']))
        self9 = min(paths['9']['snapshots'], key=lambda z: (z['all9_MSE'], z['step']))
        row = dict(index=index, sample_id=str(data['sample_ids'][index]), context=context,
            source_group=str(data['groups'][index]), role=str(data['roles'][index]),
            baseline=baseline, exact_past9_rul=exact, head_weight=weight.tolist(), head_bias=bias,
            paths=paths, decisions=dict(keep=dict(step=0,rul=baseline['rul']),
            exact9=dict(step=None,rul=exact), self6=dict(step=self6['step'],rul=self6['rul']),
            hold3=dict(step=hold3['step'],rul=hold3['rul']), self9=dict(step=self9['step'],rul=self9['rul'])),
            actual_online_fit_entries=2, actual_optimizer_steps=2*cfg['steps'],
            current_capacity_or_RUL_used=False, rul_head_unchanged=True)
        save(root/'reports'/f'q{index:02d}.json', row); rows.append(index)
    save(root/'logs'/(context+'_TRAINED.json'), dict(utc=utc(), queries=rows, seconds=time.monotonic()-started,
        online_fit_entries=2*len(rows), optimizer_steps=2*len(rows)*cfg['steps']))


def seal(root, cfg):
    rows = [read(root/'reports'/f'q{i:02d}.json') for i in range(cfg['queries'])]
    save(root/'CHOICES423.json', dict(utc=utc(), decisions={str(r['index']):r['decisions'] for r in rows},
         report_hashes={str(i):sha(root/'reports'/f'q{i:02d}.json') for i in range(cfg['queries'])},
         external_scoring_started=False))


def effects(errors, base, oracle):
    opportunities = base-oracle; valid = opportunities > 1e-6
    return dict(MAE=float(np.mean(errors)), selection_loss=float(np.mean(errors-oracle)),
        harmful_update=float(np.mean(np.maximum(errors-base,0))),
        gain_capture=float(np.mean(np.maximum(base[valid]-errors[valid],0)/opportunities[valid])) if valid.any() else None,
        gain_available_cells=int(valid.sum()), absolute_errors=errors.tolist())


def score(root, cfg):
    seal = read(root/'CHOICES423.json'); truth = packet(root/'private/targets.npz')
    rows = [read(root/'reports'/f'q{i:02d}.json') for i in range(cfg['queries'])]
    for i in range(cfg['queries']): assert sha(root/'reports'/f'q{i:02d}.json') == seal['report_hashes'][str(i)]
    predictions = {policy:np.asarray([r['decisions'][policy]['rul'] for r in rows]) for policy in cfg['policies']}
    summary = {}
    for role in ['training_oof','development']:
        mask = truth['roles'] == role; y = truth['y'][mask]
        base = abs(predictions['keep'][mask]-y)
        oracle = np.asarray([min(abs(s['rul']-truth['y'][i]) for s in row['paths']['6']['snapshots'])
                             for i,row in enumerate(rows) if mask[i]])
        summary[role] = {policy:effects(abs(predictions[policy][mask]-y),base,oracle)
                         for policy in ['keep','self6','hold3']}
        for policy in ['exact9','self9']:
            summary[role][policy] = dict(MAE=float(np.mean(abs(predictions[policy][mask]-y))),
                absolute_errors=abs(predictions[policy][mask]-y).tolist())
        for policy, count, key in [('self6','6','fit6_MSE'),('hold3','6','hold3_MSE'),('self9','9','all9_MSE')]:
            flags=[]
            for i,row in enumerate(rows):
                if not mask[i]:continue
                selected = next(s for s in row['paths'][count]['snapshots'] if s['step']==row['decisions'][policy]['step'])
                improved=selected[key] < row['baseline'][key]-1e-12
                delta=abs(selected['rul']-truth['y'][i])-abs(row['baseline']['rul']-truth['y'][i])
                flags.append(dict(index=i,proxy_improved=improved,RUL_error_change=float(delta),
                                  proxy_improved_RUL_worsened=bool(improved and delta>1e-3)))
            summary[role][policy]['proxy_primary_disagreement']=flags
    global_policy = min(cfg['policies'],key=lambda p:(summary['training_oof'][p]['MAE'],cfg['policies'].index(p)))
    a,b = summary['development']['hold3'],summary['development']['self6']
    gate = (a['MAE']<b['MAE']-1e-6 and a['harmful_update']<=b['harmful_update']+1e-6 and
            a['gain_capture'] is not None and a['gain_capture']>=b['gain_capture']-1e-12)
    result=dict(utc=utc(),cohorts=summary,source_CV_selected_policy=global_policy,
        source_CV_selected_development_MAE=summary['development'][global_policy]['MAE'],
        primary_gate=gate,selected_pool_definition='Six frozen checkpoints of the same fit6 trajectory;exact9/self9 reported separately.',
        observed_disagreement_is_not_an_Agent_failure=True,source_clusters=1,
        new_Agent_calls=0,new_LLM_calls=0,independent_confirmation=0,
        formal22_accessed=False,online_adaptation_entries=110,actual_Adam_updates=2200)
    save(root/'RESULT423.json',result)
    print(json.dumps(dict(primary_gate=gate,source_CV_policy=global_policy,
        training_MAE={k:v['MAE'] for k,v in summary['training_oof'].items()},
        development_MAE={k:v['MAE'] for k,v in summary['development'].items()})),flush=True)


def audit(root,cfg):
    started=time.monotonic(); obs=packet(root/'legal/observations.npz'); truth=packet(root/'private/targets.npz')
    rows=[read(root/'reports'/f'q{i:02d}.json') for i in range(cfg['queries'])]
    checked=updates=changed=0; max_head_error=0.; recalc={p:[] for p in cfg['policies']}
    for i,row in enumerate(rows):
        x=torch.from_numpy(obs['nn'][i:i+1]);past=obs['past_cap'][i]/1000
        initial=model_from(cfg['contexts'][row['context']]['checkpoint'])
        for count in ['6','9']:
            gradient_steps=row['paths'][count]['optimizer_steps'];assert len(gradient_steps)==20
            assert all(s['gradient_squared_norm']>0 for s in gradient_steps);updates+=len(gradient_steps)
            for entry in row['paths'][count]['snapshots']:
                path=Path(entry['checkpoint']); path=path if path.is_absolute()else root/path
                assert sha(path)==entry['checkpoint_sha256'];model=model_from(path);assert state_hash(model)==entry['state_hash']
                assert torch.equal(model.rul.weight,initial.rul.weight)and torch.equal(model.rul.bias,initial.rul.bias)
                with torch.no_grad():r,q=model(x)
                actual=max(float(r.reshape(-1)[0])*3000,0);q=q.reshape(10).numpy()
                np.testing.assert_allclose(actual,entry['rul'],atol=1e-3,rtol=0)
                np.testing.assert_allclose(q,entry['capacity_scaled'],atol=1e-7,rtol=0)
                linear=max(float(np.asarray(row['head_weight'])@q+row['head_bias'])*3000,0)
                max_head_error=max(max_head_error,abs(linear-actual));assert abs(linear-actual)<1e-3
                for key,indices in [('fit6_MSE',slice(0,6)),('hold3_MSE',slice(6,9)),('all9_MSE',slice(0,9))]:
                    residual=q[indices].astype(float)-past[indices].astype(float)
                    np.testing.assert_allclose(float(residual@residual/len(residual)),entry[key],atol=1e-10,rtol=1e-5)
                if entry['step']>0:assert entry['state_hash']!=row['paths'][count]['snapshots'][0]['state_hash'];changed+=1
                checked+=1
        for policy,count,key in [('self6','6','fit6_MSE'),('hold3','6','hold3_MSE'),('self9','9','all9_MSE')]:
            chosen=min(row['paths'][count]['snapshots'],key=lambda e:(e[key],e['step']))
            assert chosen['step']==row['decisions'][policy]['step']
        for policy in cfg['policies']:recalc[policy].append(abs(row['decisions'][policy]['rul']-truth['y'][i]))
        exact=np.asarray(row['baseline']['capacity_scaled'],dtype=np.float32);exact[:9]=past
        expected=max(float(exact@np.asarray(row['head_weight'],dtype=np.float32)+np.float32(row['head_bias']))*3000,0)
        assert abs(expected-row['decisions']['exact9']['rul'])<1e-3
    result=read(root/'RESULT423.json')
    for role in ['training_oof','development']:
        mask=truth['roles']==role
        for policy,errors in recalc.items():
            np.testing.assert_allclose(np.asarray(errors)[mask],result['cohorts'][role][policy]['absolute_errors'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(np.mean(np.asarray(errors)[mask]),result['cohorts'][role][policy]['MAE'],rtol=0,atol=1e-8)
    assert checked==660 and changed==550 and updates==2200
    save(root/'INDEPENDENT_AUDIT423.json',dict(utc=utc(),passed=True,seconds=time.monotonic()-started,
        cold_checkpoint_replays=checked,changed_saved_states=changed,actual_optimizer_steps=updates,
        actual_online_adaptation_entries=110,analytic_head_max_absolute_error_cycles=max_head_error,
        no_new_audit_fit=True,formal22_accessed=False,result_sha256=sha(root/'RESULT423.json')))
    print(json.dumps(dict(passed=True,cold_replays=checked,optimizer_steps=updates)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','seal','score','audit']);p.add_argument('--run',type=Path,required=True);p.add_argument('--context');args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(423);np.random.seed(423)
    root=args.run.resolve();cfg=read(root/'PROTOCOL423.json');assert sha(Path(__file__))==cfg['code_sha256']
    if args.phase=='worker':worker(root,cfg,args.context)
    else:globals()[args.phase](root,cfg)
