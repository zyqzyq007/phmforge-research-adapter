"""Complete missing ordinary checks on immutable historical CMAPSS programs.

No feature, code, training, threshold or final outcome changes are permitted.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from screen_score_blinding import restore


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def prepare(run):
    cfg = read(run/'PROTOCOL453.json')
    for key in ['corpus', 'source']:
        assert sha(Path(cfg[key+'_archive'])/'SOURCE_MANIFEST.json') == cfg[key+'_manifest_sha256']
    bindings = restore(Path(cfg['corpus_archive']), run/'corpus', lambda p: p == 'analysis/LEGAL_EVIDENCE450.json')
    prefix = cfg['missing_program']+'/'
    bindings += restore(Path(cfg['source_archive']), run/'original', lambda p:
        p in ['inputs/development.npz', 'inputs/metadata.json', 'ENGINE_CONFIG.json']
        or p.startswith('common_tools/')
        or p.startswith('subjects/') and '/checks/' in p and p.endswith('/CHECK.json')
        or p.startswith(prefix) and (p.endswith('/REPORT.json') or '/source/' in p
            or '/full_fit/artifacts/' in p or p == prefix+'development/PREDICTIONS.json'))
    legal = [r for r in read(run/'corpus/analysis/LEGAL_EVIDENCE450.json') if r['source_family'] == 'CMAPSS_FD001']
    assert len(legal) == 23 and len({r['content_identity'] for r in legal}) == 14
    original_manifest = {r['path']: r for r in read(Path(cfg['source_archive'])/'SOURCE_MANIFEST.json')['files']}
    grouped = defaultdict(list)
    for row in legal:
        grouped[row['content_identity']].append(row)
    available, missing, checks = {}, [], 0
    for content, members in grouped.items():
        reports = []
        for row in members:
            subject = row['episode'].split('/')[1]
            path = run/'original/subjects'/subject/'checks'/row['program_id']/'CHECK.json'
            if path.exists():
                check = read(path)
                relative = row['report_path'].split('inputs/435/', 1)[1]
                assert check['code_and_weights_sha256'] == original_manifest[relative]['sha256']
                assert check == row['additional_legal_check']
                reports.append((row['id'], check, str(path.relative_to(run))))
                checks += 2
        if not reports:
            missing.append(dict(content=content,members=[r['id'] for r in members])); continue
        canonical = reports[0][1]
        for _, copy, _ in reports[1:]:
            assert copy['rows'] == canonical['rows'] and copy['engine_errors'] == canonical['engine_errors']
            checks += 1
        assert len(canonical['rows']) == 60 and len(canonical['engine_errors']) == 20
        available[content] = dict(check=canonical, binding=reports[0][2], copied_from=reports[0][0],
                                  duplicate_records=[r[0] for r in reports])
    assert len(available) == 13 and len(missing) == 1
    assert missing[0]['members'] == ['A435/causal_repair_1/p03']
    program = run/'original'/cfg['missing_program']
    report = read(program/'REPORT.json')
    for p, expected in report['artifacts'].items():
        assert sha(program/p) == expected; checks += 1
    engine = read(run/'original/ENGINE_CONFIG.json')
    assert engine['prefix_fractions'] == cfg['prefix_fractions']
    engine.update(input_root=str(run/'original/inputs'),common_tools=str(run/'original/common_tools'),
        prediction_workers=cfg['workers'],predict_seconds=cfg['predict_seconds'],program_seconds=cfg['program_seconds'])
    save(run/'ENGINE_CONFIG.json', engine)
    save(run/'REUSED_CHECKS453.json', available)
    save(run/'LEGAL_MEMBERS453.json', legal)
    save(run/'PREPARATION453.json', dict(bindings=bindings,checks=checks,missing=missing,reused_distinct=13,
        development_sha256=sha(run/'original/inputs/development.npz'),
        new_prediction_processes=60,final_outcomes_restored=False))
    print(json.dumps(dict(prepared=True,reused=13,missing=missing,checks=checks)))


def run_check(run):
    from cmapss_program_runtime import Runtime
    cfg = read(run/'PROTOCOL453.json'); start = time.monotonic()
    assert not (run/'NEW_CHECK_TERMINAL453.json').exists()
    program = run/'original'/cfg['missing_program']
    report = read(program/'REPORT.json')
    for p, expected in report['artifacts'].items():
        assert sha(program/p) == expected
    try:
        output = Runtime(run).check(program, run/'new_check')
        for p, expected in report['artifacts'].items():
            assert sha(program/p) == expected
        save(run/'NEW_CHECK_TERMINAL453.json',dict(status='complete',seconds=time.monotonic()-start,
            new_predictions=len(output['rows']),new_fits=0,check_sha256=sha(run/'new_check/CHECK.json')))
    except Exception as e:
        save(run/'NEW_CHECK_TERMINAL453.json',dict(status='failed',seconds=time.monotonic()-start,
            error_type=type(e).__name__,error=str(e),preserve_partial=True))
        raise
    print(json.dumps(read(run/'NEW_CHECK_TERMINAL453.json')))


def choose(run):
    assert read(run/'NEW_CHECK_TERMINAL453.json')['status'] == 'complete'
    assert not (run/'outcomes').exists() and not (run/'DECISIONS453.json').exists()
    legal = read(run/'LEGAL_MEMBERS453.json'); available = read(run/'REUSED_CHECKS453.json')
    missing = read(run/'PREPARATION453.json')['missing'][0]['content']
    available[missing] = dict(check=read(run/'new_check/CHECK.json'),binding='new_check/CHECK.json')
    decisions = []; pools = []
    for case in sorted({r['episode'] for r in legal}):
        members = [r for r in legal if r['episode'] == case]
        ids = [r['program_id'] for r in members]; old = ids.index('p00')
        full = [np.mean([e['MAE'] for e in r['legal_report']['development']['engine_errors']]) for r in members]
        complete = [np.mean([e['MAE'] for e in available[r['content_identity']]['check']['engine_errors']]) for r in members]
        partial = [r['additional_legal_check'] for r in members]
        common = all(bool(x) for x in partial)
        previous = [np.mean([e['MAE'] for e in x['engine_errors']]) for x in partial] if common else full
        rules = dict(old_shared_secondary=previous,complete_prefix_mean=complete,full_development_mean=full)
        for rule, values in rules.items():
            tied = [i for i, x in enumerate(values) if x <= min(values)+1e-12]
            index = old if old in tied else tied[0]
            decisions.append(dict(case=case,method=rule,selected=ids[index]))
        decisions.append(dict(case=case,method='keep',selected='p00'))
        pools.append(dict(case=case,ids=ids,rules=rules,previous_full_fallback=not common))
    save(run/'COMPLETE_EVIDENCE453.json',available)
    save(run/'VISIBLE_SELECTION453.json',pools)
    save(run/'DECISIONS453.json',decisions)
    save(run/'ALL_CHOICES_SEALED453.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        decision_sha256=sha(run/'DECISIONS453.json'),evidence_sha256=sha(run/'COMPLETE_EVIDENCE453.json'),
        programs_unchanged=True,all4pools=True,final_outcomes_restored=False))
    print(json.dumps(decisions))


def score(run):
    assert sha(run/'DECISIONS453.json') == read(run/'ALL_CHOICES_SEALED453.json')['decision_sha256']
    cfg = read(run/'PROTOCOL453.json')
    bindings = restore(Path(cfg['corpus_archive']),run/'outcomes',lambda p:p in ['analysis/HISTORICAL_OUTCOMES450.json','analysis/EPISODES450.json'])
    gold = {r['id']:r['final_loss'] for r in read(run/'outcomes/analysis/HISTORICAL_OUTCOMES450.json')}
    pools = {r['case']:r for r in read(run/'VISIBLE_SELECTION453.json')}
    rows = []
    for d in read(run/'DECISIONS453.json'):
        p = pools[d['case']];losses = [gold[d['case']+'/'+x] for x in p['ids']]
        span = max(losses)-min(losses)
        scaled = [(x-min(losses))/span for x in losses] if span>1e-12 else [0.]*len(losses)
        i,old = p['ids'].index(d['selected']),p['ids'].index('p00')
        rows.append(dict(**d,MAE=losses[i],pool_best_MAE=min(losses),raw_selection_loss=losses[i]-min(losses),
            regret=scaled[i],harm=max(0,scaled[i]-scaled[old]),gain=max(0,scaled[old]-scaled[i])))
    summary = {m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ['MAE','raw_selection_loss','regret','harm','gain']} for m in cfg['rules']}
    a,b=summary['complete_prefix_mean'],summary['old_shared_secondary']
    gate = a['MAE']<b['MAE']-1e-12 and a['harm']<=b['harm']+1e-12 and a['gain']>=b['gain']-1e-12
    episodes = [e for e in read(run/'outcomes/analysis/EPISODES450.json') if e['source_family']=='CMAPSS_FD001']
    save(run/'RESULT453.json',dict(iteration='A453',rows=rows,summary=summary,ordinary_completion_control_gate=gate,
         historical_native=[dict(case=e['id'],completed=e['completed'],selected=e['selected'],final_MAE=e['selected_final_loss']) for e in episodes],
         outcome_bindings=bindings,exploratory=True,independent_confirmation=0,new_Agent_stages=0,new_PHM_fits=0,new_LLM_calls=0,
         new_prediction_processes=60,limitations=['One historically opened source, four related pools; no Agent or innovation superiority claim.',
             'Complete checks add actual evidence and cannot be credited to the original incomplete-information decisions.',
             'Three chosen development prefix fractions are a fixed ordinary tool, not a guarantee of deployment query representativeness.']))
    print(json.dumps(dict(summary=summary,ordinary_completion_control_gate=gate)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','run_check','choose','score'],required=True)
    a=p.parse_args();globals()[a.stage](a.run)
