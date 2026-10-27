"""A448: frozen anchored score fitting on complete cached PHM program pools.

This is a conventional linear residual-control qualification, not a new Agent
or a claimed reproduction of AlphaEdit/SRE. Only legal full-model predictions
enter fitting; external outcomes are joined after parameter sealing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import json
import hashlib
import tarfile
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
FITTED = ['anchored_ls', 'unanchored_ls', 'anchored_ridge']
RULES = FITTED + ['keep', 'equal_vote', 'legal_best', 'cv_best']


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(root):
    cfg = read(root/'PROTOCOL448.json'); source = Path(cfg['source'])
    receipt = read(source/'ARCHIVE_VERIFIED.json')
    assert sha(source/'SOURCE_MANIFEST.json') == cfg['manifest_sha256'] == receipt['manifest_sha256']
    entry = next(r for r in read(source/'SOURCE_MANIFEST.json')['files'] if r['path'] == 'INPUTS403.json')
    if (source/'INPUTS403.json').exists(): raw = (source/'INPUTS403.json').read_bytes()
    else:
        with tarfile.open(source/receipt['bundle_filename'], 'r:gz') as tf:
            raw = tf.extractfile('INPUTS403.json').read()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    (root/'INPUTS403.json').write_bytes(raw)
    original = json.loads(raw); assert len(original['pools']) == 20
    cases = []
    for i, pool in enumerate(original['pools']):
        names = sorted(pool['members'], key=lambda n: (n != 'p00', n))
        groups = sorted(pool['members']['p00']['legal']); assert names[0] == 'p00'
        for n in names:
            assert sorted(pool['members'][n]['legal']) == groups
            assert all(pool['members'][n]['legal'][g][0] == pool['members']['p00']['legal'][g][0] for g in groups)
        y = np.asarray([pool['members']['p00']['legal'][g][0] for g in groups])
        pred = np.asarray([[pool['members'][n]['legal'][g][1] for n in names] for g in groups])
        assert np.isin(y, range(4)).all() and np.isin(pred, range(4)).all()
        # One row per original acquisition. No window pseudoreplication or CV/full-model pooling.
        p = root/'legal'/f'{i:02d}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p, y=y, predictions=pred, groups=np.asarray(groups))
        legal_scores = {n: sum(pool['members'][n]['legal'][g][0] != pool['members'][n]['legal'][g][1] for g in groups) / len(groups) for n in names}
        cv_scores = {n: sum(a != b for a, b in pool['members'][n]['cv'].values()) / len(pool['members'][n]['cv']) for n in names}
        cases.append(dict(id=f'{i:02d}', batch=pool['batch'], cohort=pool['cohort'], subject=pool['subject'], names=names,
                          complete=pool['complete'], registry_deviation=pool['registry_deviation'],
                          check_coverage=pool['check_coverage'], legal_records=len(groups),
                          legal_best=min(names, key=lambda n: (legal_scores[n], names.index(n))),
                          cv_best=min(names, key=lambda n: (cv_scores[n], names.index(n)))))
    save(root/'CASES448.json', cases)
    save(root/'BINDING448.json', dict(source=str(source), **entry))
    paths = [root/'PROTOCOL448.json', root/'CASES448.json', root/'BINDING448.json', *sorted((root/'legal').glob('*.npz')), *sorted((root/'code').glob('*.py'))]
    save(root/'FREEZE448.json', {str(p.relative_to(root)):sha(p) for p in paths})
    print(json.dumps(dict(prepared_pools=len(cases), candidate_memberships=sum(len(c['names']) for c in cases), legal_records=[c['legal_records'] for c in cases])), flush=True)


def design(pred):
    # Shared scalar coefficient per complete program, four class score channels.
    return np.eye(4)[pred.astype(int)].transpose(0, 2, 1).reshape(-1, pred.shape[1])


def fit_one(root, input_path):
    with np.load(input_path, allow_pickle=False) as z:
        a = design(z['predictions']); y = np.eye(4)[z['y'].astype(int)].reshape(-1)
    m = a.shape[1]; e = np.eye(m)[0]; u, s, vt = np.linalg.svd(a, full_matrices=False)
    good = s > s[0] * 1e-10
    inverse = (vt[good].T / s[good]) @ u[:, good].T
    projector = inverse @ a
    zero = inverse @ y
    anchored = e + inverse @ (y - a @ e)
    ridge = e + np.linalg.solve(a.T @ a + np.eye(m), a.T @ (y - a @ e))
    weights = dict(anchored_ls=anchored, unanchored_ls=zero, anchored_ridge=ridge)
    save(root/'MODELS.json', dict(utc=utc(), input_sha256=sha(input_path), fits=3,
        rank=int(good.sum()), columns=m, singular_values=s.tolist(),
        incumbent_null_component=((np.eye(m)-projector) @ e).tolist(),
        weights={k:v.tolist() for k,v in weights.items()},
        training_squared_error={k:float(np.sum((a @ v-y)**2)) for k,v in weights.items()},
        changed_from_incumbent={k:bool(np.linalg.norm(v-e)>1e-10) for k,v in weights.items()},
        unseen_safety_guarantee=False))


def fit(root):
    cfg = read(root/'PROTOCOL448.json')
    for p, digest in read(root/'FREEZE448.json').items(): assert sha(root/p) == digest
    assert not (root/'STARTED448.json').exists(); save(root/'STARTED448.json',dict(utc=utc()))
    def one(case):
        output = root/'fits'/case['id']; (output/'tmp/mpl').mkdir(parents=True)
        cmd = ['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
            if Path(p).exists(): cmd += ['--ro-bind',p,p]
        for p,q in [(PYTHON,'/opt/python'),(DEPS,'/opt/deps'),(root/'code','/code'),(root/'legal'/(case['id']+'.npz'),'/input.npz')]: cmd += ['--ro-bind',str(p),q]
        cmd += ['--bind',str(output),'/output','--proc','/proc','--dev','/dev','--chdir','/code','--remount-ro','/',
                '/opt/python/bin/python3.12','-B','/code/screen_anchored_score_updates.py','--stage','fit_one','--run','/output','--input','/input.npz']
        env=dict(PATH='/opt/python/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/tmp/mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        r=subprocess.run(cmd, env=env, capture_output=True,text=True,timeout=60)
        save(output/'PROCESS.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        assert r.returncode==0,r.stderr
        return dict(case=case['id'],models_sha256=sha(output/'MODELS.json'),fits=3)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: seals=list(pool.map(one,read(root/'CASES448.json')))
    save(root/'SEALED448.json',dict(utc=utc(),models=seals,fit_seconds=time.monotonic()-started,supervised_combiners_fitted=60,base_fits=0,LLM_calls=0))
    print(json.dumps(dict(fits=60,seconds=time.monotonic()-started)),flush=True)


def choose(scores, anchor):
    # Numerical equality is defined before fitting; incumbent first, then lowest class.
    return [int(old if np.isclose(row[old],row.max(),atol=1e-10,rtol=0) else np.flatnonzero(np.isclose(row,row.max(),atol=1e-10,rtol=0))[0]) for row,old in zip(scores,anchor)]


def metrics(pred,y,base):
    n=len(y)
    return {k:str(Fraction(v,n)) for k,v in dict(error=sum(a!=b for a,b in zip(pred,y)),harm=sum(p!=t and b==t for p,t,b in zip(pred,y,base)),gain=sum(p==t and b!=t for p,t,b in zip(pred,y,base))).items()}


def aggregate(rows, score_key='scores'):
    rules=list(rows[0][score_key]);cohorts={}
    for c in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==c]
        cohorts[c]={rule:{key:str(sum((Fraction(r[score_key][rule][key]) for r in rr),Fraction())/len(rr)) for key in ['error','harm','gain']} for rule in rules}
    mean={rule:{key:str(sum((Fraction(c[rule][key]) for c in cohorts.values()),Fraction())/len(cohorts)) for key in ['error','harm','gain']} for rule in rules}
    return dict(cohorts=cohorts,means=mean)


def evaluate(root):
    sealed=read(root/'SEALED448.json');cases=read(root/'CASES448.json');original=read(root/'INPUTS403.json')
    assert sha(root/'INPUTS403.json')==read(root/'BINDING448.json')['sha256']
    for r in sealed['models']: assert sha(root/'fits'/r['case']/'MODELS.json')==r['models_sha256']
    rows=[];native=[];a342_index=0
    for c,pool in zip(cases,original['pools']):
        names=c['names'];groups=sorted(pool['members']['p00']['final'])
        y=[pool['members']['p00']['final'][g][0] for g in groups]
        predictions=np.asarray([[pool['members'][n]['final'][g][1] for n in names] for g in groups]);base=predictions[:,0]
        assert all(pool['members'][n]['final'][g][0]==y[j] for n in names for j,g in enumerate(groups))
        a=design(predictions);model=read(root/'fits'/c['id']/'MODELS.json')
        raw={r:(a@np.asarray(model['weights'][r])).reshape(len(groups),4) for r in FITTED}
        output={r:choose(score,base) for r,score in raw.items()}
        output.update(keep=base.tolist(),equal_vote=choose((a@np.ones(len(names))).reshape(len(groups),4),base),
                      legal_best=predictions[:,names.index(c['legal_best'])].tolist(),cv_best=predictions[:,names.index(c['cv_best'])].tolist())
        scores={r:metrics(output[r],y,base) for r in RULES}
        rows.append(dict(**c,groups=groups,truth=y,base=base.tolist(),predictions=output,scores=scores,
                         rank=model['rank'],null_norm=float(np.linalg.norm(model['incumbent_null_component'])),
                         anchored_vs_unanchored_changes=sum(a!=b for a,b in zip(output['anchored_ls'],output['unanchored_ls']))))
        actual=[]
        if c['batch']=='A342':
            a342_index+=1
            actual=[x for x in original['native_context'] if x['pool']==f'pool_{a342_index}' and x['arm']=='ordinary' and x['completed']]
        elif c['complete'] and not c['registry_deviation'] and c['subject'].startswith('ordinary'):
            actual=[dict(selector=c['subject'],selected=pool['native_submission']['program_id'])]
        for item in actual:
            pred=[pool['members'][item['selected']]['final'][g][1] for g in groups]
            native.append(dict(case=c['id'],cohort=c['cohort'],selector=item['selector'],selected=item['selected'],
                               scores=dict(native=metrics(pred,y,base),**scores)))
    all_agg=aggregate(rows); eligible=[r for r in rows if r['complete'] and not r['registry_deviation']]
    native_agg=aggregate(native);gates={}
    for comparison,agg in [('native',native_agg),('unanchored_ls',all_agg),('anchored_ridge',all_agg),('equal_vote',all_agg),('legal_best',all_agg),('cv_best',all_agg)]:
        new=agg['means']['anchored_ls'];old=agg['means'][comparison]
        wins=sum(Fraction(c['anchored_ls']['error'])<Fraction(c[comparison]['error']) for c in agg['cohorts'].values())
        ties=sum(Fraction(c['anchored_ls']['error'])==Fraction(c[comparison]['error']) for c in agg['cohorts'].values())
        gates[comparison]=dict(wins=wins,ties=ties,losses=len(agg['cohorts'])-wins-ties,passed=wins>=2 and Fraction(new['error'])<Fraction(old['error']) and Fraction(new['harm'])<=Fraction(old['harm']) and Fraction(new['gain'])>=Fraction(old['gain']))
    passed=all(g['passed'] for g in gates.values())
    result=dict(utc=utc(),rows=rows,native=native,all=all_agg,eligible=aggregate(eligible),native_aggregate=native_agg,
                gates=gates,primary_passed=passed,complete_non_deviating_pools=len(eligible),native_pairs=len(native),
                source_clusters=1,related_training_cohorts=3,independent_confirmation=False,new_base_fits=0,
                new_LLM_calls=0,new_Agent_stages=0,new_supervised_combiners=60,
                fair_native_intervention=False,novelty_established=False,goal_complete=False)
    save(root/'RESULT448.json',result)
    print(json.dumps(dict(primary_passed=passed,gates=gates,all=all_agg['means'],native=native_agg['means'],anchored_output_changes=sum(r['anchored_vs_unanchored_changes'] for r in rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--input',type=Path);a=p.parse_args()
    if a.stage=='fit_one':fit_one(a.run,a.input)
    else:globals()[a.stage](a.run)
