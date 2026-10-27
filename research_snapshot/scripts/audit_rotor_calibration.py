"""Independent raw-value, prediction, metric and selection audit for A417."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(root):
    started = time.monotonic()
    cfg, prepared, result = [read(root/n) for n in ['PROTOCOL417.json','PREPARED417.json','RESULT417.json']]
    checks = 0

    def check(value):
        nonlocal checks
        assert value
        checks += 1

    for path, expected in read(root/'FREEZE417.json').items():
        check(digest(path) == expected)
    bindings = read(root/'SOURCE_BINDINGS417.json')
    for row in bindings['files']:
        check(digest(Path(bindings['archive'])/row['path']) == row['sha256'])
    metrics, source_wave_checks, feature_checks = {}, 0, 0
    for batch in prepared['batches']:
        tag = batch['tag']
        gold = read(root/'private'/tag/'gold.json')
        provenance = {r['sample_id']:r for r in read(root/'private'/tag/'lineage.json')}
        source_by_id = {r['sample_id']:r for r in gold}
        source_records = {r['export_path']:r for r in cfg['source_records']}
        inp = root/'observations'/tag/'datasets/ROTORBROKENBAR/test.npz'
        check(digest(inp)==batch['input_sha256'])
        with np.load(inp,allow_pickle=False) as archive_input:
            observed = {k:archive_input[k] for k in archive_input.files}
            check(set(observed)=={'features','waveforms','sample_ids','sampling_hz','operating_load_Nm','dataset'})
            loaded = {}
            sampled = set()
            for i, identity in enumerate(observed['sample_ids'].astype(str)):
                row = provenance[identity]
                name = row['source']
                native = source_records[name]
                if name not in loaded:
                    path = Path(cfg['data_archive'])/name
                    check(digest(path)==native['export_sha256'])
                    with np.load(path,allow_pickle=False) as saved:
                        loaded[name]=saved['signals'].copy()
                        check(int(saved['broken_bars'])==native['broken_bars'])
                wave = loaded[name][row['start']:row['start']+2048]
                check(np.array_equal(observed['waveforms'][i],wave.T))
                check(source_by_id[identity]['label']==native['broken_bars'])
                source_wave_checks += 1
                # One independently expressed feature calculation per complete record.
                if name not in sampled:
                    sampled.add(name)
                    centered = wave-np.mean(wave,axis=0)
                    v = np.mean(centered*centered,axis=0)
                    sd = np.sqrt(v)
                    f = [np.mean(wave,axis=0),sd,np.sqrt(np.mean(wave*wave,axis=0)),
                         np.amax(np.abs(wave),axis=0),np.amax(wave,axis=0)-np.amin(wave,axis=0),
                         np.mean(centered**3,axis=0)/(np.maximum(sd,1e-12)**3),
                         np.mean(centered**4,axis=0)/(np.maximum(v,1e-12)**2)]
                    power = np.abs(np.fft.rfft(centered,axis=0))**2/(2048**2)
                    f += [np.log1p(np.sum(power[1+j*128:1+(j+1)*128],axis=0)) for j in range(8)]
                    check(np.array_equal(np.concatenate(f).astype(np.float32),observed['features'][i]))
                    feature_checks += 1
        for cohort in cfg['cohorts']:
            command = read(root/'replays'/tag/cohort/'0/COMMAND.json')
            check('--unshare-net' in command and '/tmp' not in command)
            check(str(root/'observations'/tag) in command)
            check(not any('/private/' in s for s in command))
            for variant in cfg['variants'][cohort]:
                name = variant['name']
                maps = []
                for repeat in [0,1]:
                    directory = root/'replays'/tag/cohort/str(repeat)
                    check(read(directory/'STATUS.json')['status']=='complete')
                    check(read(directory/'output/FIT_GUARD417.json')['observed_fit_calls']==0)
                    p = read(directory/'output'/(name+'.json'))['predictions']
                    mapping = {v['sample_id']:int(v['predicted_label']) for v in p}
                    check(len(mapping)==len(gold)==len(p) and set(mapping)==set(source_by_id))
                    maps.append(mapping)
                check(maps[0]==maps[1])
                grouped = defaultdict(list)
                confusion = [[0]*5 for _ in range(5)]
                for row in gold:
                    p = maps[0][row['sample_id']]
                    confusion[row['label']][p] += 1
                    grouped[row['group']].append((row['label'],p))
                loss = sum((Fraction(sum(y!=p for y,p in group),len(group)) for group in grouped.values()),Fraction())/len(grouped)
                row, = [r for r in result['rows'] if (r['batch'],r['cohort'],r['name'])==(tag,cohort,name)]
                check(row['confusion']==confusion and Fraction(row['record_mean_error'])==loss)
                acc = sum(confusion[i][i] for i in range(5))/len(gold)
                f1 = sum(2*confusion[k][k]/den if (den:=sum(confusion[k])+sum(c[k] for c in confusion)) else 0 for k in range(5))/5
                check(abs(row['accuracy']-acc)<1e-12 and abs(row['macro_f1']-f1)<1e-12)
                metrics[tag,cohort,name] = loss
    aggregates = {}
    decisions = read(root/'CHOICES417.json')['choices']
    for rule in ['same_load_validation','broader_load_validation','ordinary_review']:
        losses, regrets, harms, captured, available = [],[],[],[],[]
        for held in [7,8]:
            other = 15-held
            for family in 'abcd':
                candidates = [family,family+'_review']
                if rule=='ordinary_review':
                    chosen = candidates[1]
                else:
                    loads = [30] if rule=='same_load_validation' else [30,35]
                    risks = {c:sum((metrics[f'a{other}_l{v}',c,'complete'] for v in loads),Fraction())/len(loads) for c in candidates}
                    chosen = candidates[1] if risks[candidates[1]]<risks[candidates[0]] else candidates[0]
                    item, = [d for d in decisions if d['held_acquisition']==held and d['family']==family and d['rule']==rule]
                    check(item['selected']==chosen and {c:Fraction(x) for c,x in item['risks'].items()}==risks)
                initial, reviewed = [metrics[f'a{held}_l40',c,'complete'] for c in candidates]
                selected = metrics[f'a{held}_l40',chosen,'complete']
                losses.append(selected);regrets.append(selected-min(initial,reviewed))
                harms.append(max(selected-initial,0));captured.append(max(initial-selected,0))
                available.append(initial-min(initial,reviewed))
        aggregates[rule] = dict(mean_error=float(sum(losses)/8),mean_selection_loss=float(sum(regrets)/8),
            mean_harm=float(sum(harms)/8),captured_gain=float(sum(captured)),available_gain=float(sum(available)),
            gain_capture=float(sum(captured)/sum(available)) if sum(available)>0 else None)
    a_harm = [float(metrics[f'a{a}_l40','a_review','complete']-metrics[f'a{a}_l40','a','complete']) for a in [7,8]]
    old, new = aggregates['ordinary_review'], aggregates['broader_load_validation']
    passed = all(x>=0.05 for x in a_harm) and new['mean_harm']<old['mean_harm'] and new['mean_selection_loss']<old['mean_selection_loss'] and new['captured_gain']>=old['captured_gain']
    execution = read(root/'EXECUTION417.json')
    total_seconds = execution.get('cumulative_seconds',execution['seconds']) + read(root/'attempts/interface_failure/EXECUTION417.json')['seconds']
    check(total_seconds<=cfg['execution_wall_seconds'])
    summary = dict(passed=True,checks=checks,waveform_windows_checked=source_wave_checks,
        independent_feature_checks=feature_checks,prediction_pairs=len(metrics),
        fixed_validation_controls=aggregates,a_review_harm_at4Nm=a_harm,
        numerical_feasibility_gate=passed,formal_method_confirmation=False,
        protocol_deviations=read(root/'RECOVERY_GUARD417.json') if (root/'RECOVERY_GUARD417.json').exists() else None,
        cold_execution_seconds_including_failures=total_seconds,audit_seconds=time.monotonic()-started)
    (root/'AUDIT417.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
