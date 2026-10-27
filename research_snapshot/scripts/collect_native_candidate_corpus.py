"""Restore only SHA-bound candidate evidence from complete native batches.

Weights and raw signals remain immutable archive references. No submission is
executed, no candidate is selected, and no new hidden evaluation data is read.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import tarfile
import time
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def wanted(name, batch):
    if name == f'RESULT{batch}.json':
        return True
    prefixes = ('subjects/', 'reference/') + (('reused_ordinary/subjects/',) if batch == 445 else ())
    if not name.startswith(prefixes):
        return False
    if '/audit/' in name:
        return False
    base = Path(name).name
    if '/broker/' in name:
        return base in ('STATE.json', 'ACCESS.jsonl', 'FINAL_CHOICE.json') or '/lineage_receipts/' in name
    if '/primary/' in name:
        return base == 'status.json'
    if '/programs/' in name or name.startswith('reference/'):
        return (base in ('REPORT.json', 'CLAIM.json', 'ERROR.json', 'COMPOSITION.json')
                or '/source/' in name and Path(name).suffix in ('.py', '.json')
                or ('/fit/' in name or '/full_fit/' in name) and base in ('PROCESS.json', 'training_audit.json'))
    return False


def restore(run, source):
    started = time.monotonic()
    archive = Path(source['archive'])
    raw_manifest = (archive / source['manifest']).read_bytes()
    assert digest(raw_manifest) == source['manifest_sha256']
    manifest = json.loads(raw_manifest)['files']
    mapping = {r['path']: r for r in manifest}
    chosen = {k: v for k, v in mapping.items() if wanted(k, source['batch'])}
    assert chosen and f"RESULT{source['batch']}.json" in chosen
    assert sum(r['bytes'] for r in chosen.values()) <= 64*1024**2
    destination = run / 'inputs' / str(source['batch'])
    records = []
    missing = {}
    for name, expected in chosen.items():
        path = destination / name
        if path.is_file():
            raw = path.read_bytes()
            assert digest(raw) == expected['sha256'] and len(raw) == expected['bytes']
            records.append(dict(expected, batch=source['batch'], archive=str(archive),
                                copy=str(path.relative_to(run))))
        else:
            missing[name] = expected
    # Read each portable archive once. Never traverse thousands of absent files.
    if not missing:
        stream = ()
    else:
        stream = tarfile.open(archive / 'RUN_CONTENTS.tar.gz', 'r|gz')
    try:
        for member in stream:
            name = member.name.removeprefix('./')
            if name not in missing:
                continue
            expected = chosen[name]
            assert member.isfile() and member.size == expected['bytes']
            raw = stream.extractfile(member).read()
            assert digest(raw) == expected['sha256'], name
            path = destination / name
            assert '..' not in Path(name).parts
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            records.append(dict(expected, batch=source['batch'], archive=str(archive),
                                copy=str(path.relative_to(run))))
            if len(records) == len(chosen):
                break
    finally:
        if stream:
            stream.close()
    assert {r['path'] for r in records} == set(chosen)
    # Source+weight inventory permits content dedup without loading checkpoints.
    artifacts = [r for r in manifest if
                 (r['path'].startswith(('subjects/', 'reference/', 'reused_ordinary/subjects/')))
                 and '/audit/' not in r['path']
                 and ('/programs/' in r['path'] or r['path'].startswith('reference/'))]
    save(destination/'ARTIFACT_INVENTORY.json', artifacts)
    return dict(batch=source['batch'], bound_members=len(records), bytes=sum(r['bytes'] for r in records),
                bindings=records, elapsed_seconds=time.monotonic()-started)


def restore_reused_trace(run, cfg):
    source = cfg['auxiliary_artifact_sources'][0]
    archive = Path(source['archive'])
    raw = (archive/source['manifest']).read_bytes()
    assert digest(raw) == source['manifest_sha256']
    chosen = {r['path']: r for r in json.loads(raw)['files'] if r['path'].startswith('subjects/')
              and '/audit/' not in r['path'] and
              (r['path'].endswith('/broker/ACCESS.jsonl') or
               '/primary/' in r['path'] and r['path'].endswith('/status.json') or
               '/programs/' in r['path'] and r['path'].endswith('/ERROR.json'))}
    records = []
    with tarfile.open(archive/'RUN_CONTENTS.tar.gz', 'r|gz') as stream:
        for member in stream:
            name = member.name.removeprefix('./')
            if name not in chosen:
                continue
            row = chosen[name]
            raw = stream.extractfile(member).read()
            assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
            path = run/'inputs/445/reused_ordinary'/name
            if path.exists():
                assert digest(path.read_bytes()) == row['sha256']
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            records.append(dict(row, path='reused_ordinary/'+name, origin_member=name, batch=445,
                                origin_batch=444, archive=str(archive), copy=str(path.relative_to(run))))
            if len(records) == len(chosen):
                break
    assert len(records) == len(chosen)
    return records


def build(run, cfg):
    """Separate actual decision-time material from already opened outcomes."""
    started = time.monotonic()
    bindings = json.loads((run/'analysis/BINDINGS450.json').read_text())
    bound = {(r['batch'], r['path']): r for r in bindings}
    for item in bindings:
        raw = (run/item['copy']).read_bytes()
        assert len(raw) == item['bytes'] and digest(raw) == item['sha256']
    legal, outcomes, episodes, attempts = [], [], [], []
    checks = []

    def verify(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        assert condition, name

    for source in cfg['sources']:
        batch = source['batch']
        directory = run/'inputs'/str(batch)
        final = json.loads((directory/f'RESULT{batch}.json').read_text())
        files = {r['path']: dict(r, origin_archive=source['archive'], origin_member=r['path'])
                 for r in json.loads((directory/'ARTIFACT_INVENTORY.json').read_text())}
        if batch == 445:
            auxiliary = cfg['auxiliary_artifact_sources'][0]
            raw = (Path(auxiliary['archive'])/auxiliary['manifest']).read_bytes()
            verify('A444_auxiliary_manifest', digest(raw) == auxiliary['manifest_sha256'])
            for r in json.loads(raw)['files']:
                if r['path'].startswith('subjects/') and '/programs/' in r['path']:
                    files.setdefault('reused_ordinary/'+r['path'], dict(r, origin_archive=auxiliary['archive'],
                                                                      origin_member=r['path']))
        states = sorted(directory.glob('subjects/*/broker/STATE.json'))
        if batch == 445:
            states += sorted(directory.glob('reused_ordinary/subjects/*/broker/STATE.json'))
        verify(f'{batch}_all_four_subjects', len(states) == 4)
        for state_file in states:
            subject_root = state_file.parent.parent
            subject = subject_root.name
            subject_rel = str(subject_root.relative_to(directory))
            state = json.loads(state_file.read_text())
            accesses = [json.loads(line) for line in state_file.with_name('ACCESS.jsonl').read_text().splitlines()]
            native_files = list((subject_root/'primary').glob('*/status.json'))
            verify(f'{batch}_{subject}_one_native_status', len(native_files) == 1)
            native = json.loads(native_files[0].read_text())
            selected = (state.get('submission') or {}).get('program_id')
            submission_time = (state.get('submission') or {}).get('utc')
            id_prefix = f'A{batch}/{subject}'
            native_complete = native['status'] == 'finished' and selected is not None
            before = len(legal)
            successful_registrations = {}
            action_records = []
            for ordinal, event in enumerate(accesses):
                request = event['request']
                try:
                    request = json.loads(request) if isinstance(request, str) else request
                except (ValueError, TypeError):
                    request = {'action': 'unparsed_request'}
                response = event.get('response') or {}
                result = response.get('result') or {}
                action_records.append(dict(ordinal=ordinal, action=request.get('action'),
                                           ended_utc=event.get('ended_utc'), ok=response.get('ok')))
                if request.get('action') == 'register_program':
                    attempts.append(dict(episode=id_prefix, ordinal=ordinal, request=request,
                                         response=response, ended_utc=event.get('ended_utc')))
                    if isinstance(result, dict) and result.get('program_id'):
                        successful_registrations[result['program_id']] = dict(ordinal=ordinal, request=request,
                            ended_utc=event.get('ended_utc'), response_status=result.get('status'))
            for pid, program in state['programs'].items():
                initial = pid == 'p00' or pid.startswith('ref_')
                relative = f'reference/{pid}' if initial else f'{subject_rel}/programs/{pid}'
                report_path = relative+'/REPORT.json'
                verify(f'{id_prefix}/{pid}_bound_report', (batch, report_path) in bound)
                report = json.loads((directory/report_path).read_text())
                verify(f'{id_prefix}/{pid}_inline_report_equal', report == program['report'])
                verify(f'{id_prefix}/{pid}_complete', report['status'] == 'complete')
                verify(f'{id_prefix}/{pid}_final_not_read', report.get('final_data_read') is False)
                artifacts = report['artifacts']
                artifact_locations = {}
                for name, sha in artifacts.items():
                    item = files.get(relative+'/'+name)
                    verify(f'{id_prefix}/{pid}_artifact_{name}', item is not None and item['sha256'] == sha)
                    artifact_locations[name] = dict(archive=item['origin_archive'], member=item['origin_member'], sha256=sha)
                source_items = {name: sha for name, sha in artifacts.items() if name.startswith('source/')}
                if 'COMPOSITION.json' in artifacts:
                    source_items['COMPOSITION.json'] = artifacts['COMPOSITION.json']
                for name, sha in source_items.items():
                    raw = (directory/relative/name).read_bytes()
                    verify(f'{id_prefix}/{pid}_source_{name}', digest(raw) == sha)
                claim_path = directory/relative/'CLAIM.json'
                claim = json.loads(claim_path.read_text()) if claim_path.exists() else None
                trace = successful_registrations.get(pid)
                stamp = (trace or {}).get('ended_utc') or report.get('utc')
                if submission_time and stamp:
                    verify(f'{id_prefix}/{pid}_available_before_submission', stamp <= submission_time)
                cv = report.get('cv')
                if batch == 427:
                    cv_metric = sum(v['record_error'] for v in cv)/len(cv)
                    dev_metric = report['development']['record_error']
                    entry = final['all_program_results']['common_p00' if initial else subject+'__'+pid]
                    score, metric = entry['record_error'], 'record_error'
                    additional = ((state.get('probe') or {}).get('reports') or {}).get(pid)
                elif batch == 435:
                    cv_metric, dev_metric = None, report['development']['MAE']
                    entry = final['scores'][pid if initial else subject+'/'+pid]
                    score, metric = entry['MAE'], 'RUL_MAE_cycles'
                    additional = (state.get('checks') or {}).get(pid)
                else:
                    errors = [c['MAE'] for fold in cv for c in fold['cells']]
                    cv_metric, dev_metric = sum(errors)/len(errors), None
                    row = next(r for r in final['rows'] if r['id'] == subject)
                    entry = row['programs'][pid]
                    score, metric = entry['MAE'], 'RUL_MAE_cycles'
                    verify(f'{id_prefix}/{pid}_CV_47cells', len(errors) == 47)
                    verify(f'{id_prefix}/{pid}_CV_recompute', math.isclose(cv_metric, entry['CV'], abs_tol=1e-9))
                    absolute = entry.get('errors', entry.get('absolute_errors'))
                    verify(f'{id_prefix}/{pid}_external8', len(absolute) == 8)
                    verify(f'{id_prefix}/{pid}_external_MAE_recompute', math.isclose(sum(absolute)/8, score, abs_tol=1e-9))
                    additional = None
                identity = digest(json.dumps(artifacts, sort_keys=True, separators=(',', ':')).encode())
                item = dict(id=id_prefix+'/'+pid, batch=batch, source_family=source['source_family'],
                            episode=id_prefix, program_id=pid, initial=initial,
                            content_identity=identity, native_completed=native_complete,
                            code_files=[str((directory/relative/name).relative_to(run)) for name in source_items],
                            report_path=str((directory/report_path).relative_to(run)),
                            source_and_weights_manifest=artifacts, artifact_locations=artifact_locations, legal_report=report,
                            legal_cv_metric=cv_metric, legal_development_metric=dev_metric,
                            additional_legal_check=additional, registration=trace, claim=claim,
                            available_utc=stamp, parent_verified=False,
                            lexical_parent_mentions=re.findall(r'(?i)\bparent\s*[=: ]\s*((?:p\d+)|(?:ref_[a-z0-9_]+))',
                                                               (claim or {}).get('claim','')))
                legal.append(item)
                outcomes.append(dict(id=item['id'], episode=id_prefix, batch=batch, source_family=source['source_family'],
                                     content_identity=identity, metric=metric, final_loss=score,
                                     original_selected=pid == selected, native_completed=native_complete,
                                     initial=initial, program_id=pid, parent_verified=False))
            members = outcomes[-(len(legal)-before):]
            p00 = next(r for r in members if r['program_id']=='p00')
            selected_row = next((r for r in members if r['program_id']==selected), None)
            best = min(r['final_loss'] for r in members)
            errors = sorted(subject_root.glob('programs/*/ERROR.json'))
            episodes.append(dict(id=id_prefix, batch=batch, source_family=source['source_family'],
                native_status=native['status'], completed=native_complete, selected=selected,
                submission=state.get('submission'), registered_attempt_count=state['attempts'],
                broker_registration_calls=sum(r['action']=='register_program' for r in action_records),
                complete_program_members=len(members), new_complete_programs=sum(not r['initial'] for r in members),
                failed_candidate_artifacts=[str(p.relative_to(run)) for p in errors],
                selected_final_loss=selected_row['final_loss'] if selected_row else None,
                initial_p00_loss=p00['final_loss'], best_available_loss=best,
                within_pool_selection_loss=(selected_row['final_loss']-best) if selected_row and native_complete else None,
                action_history=action_records, reused_A444=batch==445 and subject.startswith('ordinary'),
                arm='ordinary' if subject.startswith('ordinary') else 'method'))
    save(run/'analysis/LEGAL_EVIDENCE450.json', legal)
    save(run/'analysis/HISTORICAL_OUTCOMES450.json', outcomes)
    save(run/'analysis/EPISODES450.json', episodes)
    save(run/'analysis/REGISTRATION_ATTEMPTS450.json', attempts)
    save(run/'analysis/BUILD_AUDIT450.json', dict(checks=checks, passed=True, count=len(checks)))
    save(run/'analysis/BUILD450.json', dict(program_memberships=len(legal), episodes=len(episodes),
        unique_artifact_contents=len({r['content_identity'] for r in legal}),
        new_complete_program_memberships=sum(not r['initial'] for r in legal),
        native_completed=sum(r['completed'] for r in episodes),
        registration_calls=len(attempts), checks=len(checks), elapsed_seconds=time.monotonic()-started,
        new_fits=0, new_model_calls=0, independent_confirmation=0))
    print((run/'analysis/BUILD450.json').read_text())


def analyze(run):
    """Independent recount from immutable records; never execute or retrain."""
    legal = json.loads((run/'analysis/LEGAL_EVIDENCE450.json').read_text())
    outcomes = json.loads((run/'analysis/HISTORICAL_OUTCOMES450.json').read_text())
    episodes = json.loads((run/'analysis/EPISODES450.json').read_text())
    attempts = json.loads((run/'analysis/REGISTRATION_ATTEMPTS450.json').read_text())
    bindings = json.loads((run/'analysis/BINDINGS450.json').read_text())
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        assert condition, name

    check('bijection', {r['id'] for r in legal} == {r['id'] for r in outcomes})
    check('unique_membership_ids', len({r['id'] for r in legal}) == len(legal))
    check('all16episodes', len(episodes) == len({r['id'] for r in episodes}) == 16)
    check('source_family_count', len({r['source_family'] for r in legal}) == 3)
    check('all_restored_hashes', all(digest((run/r['copy']).read_bytes()) == r['sha256'] for r in bindings))
    compositions = []
    for row in legal:
        check(row['id']+'_no_final_loss_field', 'final_loss' not in row)
        check(row['id']+'_final_not_read', row['legal_report']['final_data_read'] is False)
        if 'COMPOSITION.json' in row['source_and_weights_manifest']:
            contract_path = run/next(p for p in row['code_files'] if p.endswith('COMPOSITION.json'))
            contract = json.loads(contract_path.read_text())
            for member_id, member in contract['members'].items():
                member_path = (contract_path.parent/member['relative_path']/'REPORT.json').resolve()
                check(row['id']+'_member_'+member_id, digest(member_path.read_bytes()) == member['report_sha256'])
            compositions.append(row['id'])
        elif not row['initial']:
            check(row['id']+'_actual_completed_registration', row['registration']['response_status'] == 'complete')
    count = Counter((a['response'].get('result') or {}).get('status') if a['response'].get('ok') else 'rejected'
                    for a in attempts)
    check('registration_accounting', count == {'complete':45, 'authored_program_error':9, 'rejected':17})
    check('consumed_registrations', sum(e['registered_attempt_count'] for e in episodes) == 54)
    check('error_artifacts', sum(len(e['failed_candidate_artifacts']) for e in episodes) == 9)
    check('one_composition', len(compositions) == 1)
    check('all_completion_states', sum(e['completed'] for e in episodes) == 9)
    source_rows = []
    for source in sorted({r['source_family'] for r in outcomes}):
        es = [e for e in episodes if e['source_family'] == source]
        rs = [r for r in outcomes if r['source_family'] == source]
        directions = Counter()
        for row in rs:
            if row['initial']:
                continue
            anchor = next(e['initial_p00_loss'] for e in es if e['id'] == row['episode'])
            delta = row['final_loss']-anchor
            directions['unchanged' if abs(delta) < 1e-10 else 'improved' if delta < 0 else 'worse'] += 1
        ordinary = [e for e in es if e['completed'] and e['arm'] == 'ordinary']
        source_rows.append(dict(source=source, episodes=len(es), normal=sum(e['completed'] for e in es),
            unique_artifacts=len({r['content_identity'] for r in rs}), new_programs=sum(directions.values()),
            new_vs_common_p00=dict(directions), normal_ordinary=len(ordinary),
            ordinary_has_headroom=any(e['within_pool_selection_loss'] > 1e-10 for e in ordinary),
            ordinary_selection_losses=[dict(id=e['id'], loss=e['selected_final_loss'], best=e['best_available_loss'],
                                           selection_loss=e['within_pool_selection_loss']) for e in ordinary]))
    support = []
    for held in source_rows:
        other = [r for r in source_rows if r['source'] != held['source']]
        good = sum(r['new_vs_common_p00'].get('improved',0) for r in other)
        bad = sum(r['new_vs_common_p00'].get('worse',0) for r in other)
        support.append(dict(held_out=held['source'], other_sources_improved=good, other_sources_worse=bad,
                            both_directions_present=good>0 and bad>0))
    result = dict(iteration='A450', sources=source_rows, source_leaveout_label_support=support,
        memberships=len(legal), unique_artifacts=len({r['content_identity'] for r in legal}),
        new_programs=sum(not r['initial'] for r in legal), compositions=compositions,
        actual_fitted_new_programs=45, episodes=len(episodes), normal=9, incomplete=7,
        normal_positive_selection_loss=sum(e['completed'] and e['within_pool_selection_loss']>1e-10 for e in episodes),
        registration_outcomes=dict(count), restored_members=len(bindings),
        new_LLM_calls=0, new_fits=0, new_predictions=0, new_Agent_stages=0, independent_confirmation=0,
        conclusion='Historical whole-program corpus qualified for exploratory selection screening; no method superiority or causal confirmation.',
        limits=['3 source clusters, not59 independent tasks', 'Outcome labels already opened by researchers',
                'Within-pool opportunity is diagnostic, not between-method comparison',
                '7 incomplete episodes retained; they do not become completed policy successes',
                'Common-p00 contrasts are not asserted parent-child edits',
                'Full candidate code/report evidence, not full native reasoning context'])
    save(run/'analysis/RESULT450.json', result)
    save(run/'analysis/INDEPENDENT_AUDIT450.json', dict(passed=True, count=len(checks), checks=checks))
    print(json.dumps(dict(sources=source_rows, checks=len(checks), complete=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--stage', choices=['restore', 'build', 'analyze'], default='restore')
    args = ap.parse_args()
    run = args.run.resolve()
    assert Path('/tmp') not in run.parents
    cfg = json.loads((run/'PROTOCOL450.json').read_text())
    if args.stage == 'analyze':
        analyze(run)
        return
    if args.stage == 'build':
        build(run, cfg)
        return
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['budget']['workers']) as pool:
        restored = list(pool.map(lambda source: restore(run, source), cfg['sources']))
    additional = restore_reused_trace(run, cfg)
    save(run/'analysis/BINDINGS450.json', [r for batch in restored for r in batch['bindings']]+additional)
    save(run/'analysis/RESTORE450.json', dict(sources=[{k:v for k,v in r.items() if k!='bindings'} for r in restored],
                                            reused_trace_members=len(additional), elapsed_seconds=time.monotonic()-started,
                                            new_fits=0, new_model_calls=0))
    print(json.dumps({'batches':len(restored), 'members':sum(r['bound_members'] for r in restored),
                      'bytes':sum(r['bytes'] for r in restored), 'seconds':time.monotonic()-started}))


if __name__ == '__main__':
    main()
