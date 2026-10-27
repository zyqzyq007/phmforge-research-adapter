"""Post-seal provenance/compliance audit;no fitting,inference or final access.

Local-command review is a screening aid,not a proof of absent hidden work.
Only finished native builders are inspected;all unresolved calls stay flagged.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import shlex

from controlled_feedback import save,sha,utc


def read(path):return json.loads(Path(path).read_text())


def unwrap(command):
    try:
        parts=shlex.split(command)
        if len(parts)>=3 and Path(parts[0]).name in ['sh','bash','dash'] and parts[1] in ['-c','-lc','-ic','-lic']:return parts[2]
    except ValueError:pass
    return command


def python_review(command):
    shell=unwrap(command);scripts=[];unresolved=[]
    pattern=r'(?:python(?:3(?:\.12)?)?|/opt/python312/bin/python3\.12)\b[^\n]*?<<\s*[\'\"]?([A-Za-z_][A-Za-z0-9_]*)[\'\"]?\s*\n'
    for match in re.finditer(pattern,shell):
        end=re.search(r'^'+re.escape(match[1])+r'\s*$',shell[match.end():],re.MULTILINE)
        if end:scripts.append(shell[match.end():match.end()+end.start()])
        else:unresolved.append('Unterminated Python heredoc')
    calls=[]
    for script in scripts:
        try:tree=ast.parse(script)
        except SyntaxError as error:unresolved.append('Python parse:'+str(error));continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Attribute) and node.func.attr in ['fit','fit_transform','partial_fit','backward','step']:
                    calls.append(dict(method=node.func.attr,line=node.lineno,expression=ast.unparse(node)[:500],
                        category='supervised_or_optimizer_review' if node.func.attr in ['backward','step'] or len(node.args)>1 else 'fit_review'))
                if isinstance(node.func,ast.Name) and node.func.id in ['exec','eval']:unresolved.append('Dynamic Python '+node.func.id+' requires review')
    if re.search(r'python(?:3(?:\.12)?)?\s+[^\n]*\.py\b',shell) and not scripts:
        unresolved.append('Executed local Python script or script-management command;review actual invocation/source')
    return dict(executable_heredocs=len(scripts),possible_learning_calls=calls,unresolved=unresolved)


def audit(root):
    # Generic trials also create LOCAL_REVISION_PROTOCOL347 for the shared
    # evaluator. Use the legacy controller marker, not that copied protocol,
    # to distinguish them while numerical and provenance audits run together.
    generic_native=(root/'PROTOCOL337.json').exists() and not (root/'PIPELINE_STATE347.json').exists()
    if (root/'LOCAL_REVISION_PROTOCOL347.json').exists() or generic_native:
        assert (root/'ALL_CHOICES_SEALED337.json').exists(),'All actual full-coding choices must be terminal and sealed'
        if generic_native:
            assert read(root/'HANDLE337.json')['status']=='terminal'
            cfg=read(root/'PROTOCOL337.json')
        else:
            assert read(root/'PIPELINE_STATE347.json')['status']=='terminal_choices_sealed'
            cfg=read(root/'LOCAL_REVISION_PROTOCOL347.json')
        seal=read(root/'ALL_CHOICES_SEALED337.json')
        assert sha(root/'PROTOCOL337.json')==seal['protocol_sha256']
        assert sha(root/'EFFECTIVE_CODE_FREEZE337.json')==seal['code_freeze_sha256']
        for name,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/name)==digest,name
        episode_results=seal['episodes']
        scope='One full-coding stage,not separate crossed builders/selectors;all actual choices sealed.'
    else:
        assert (root/'BUILDERS_SEALED342.json').exists(),'Do not audit unfinished mutable builder state'
        assert (root/'ALL_CHOICES_SEALED342.json').exists(),'Selectors must also be terminal and sealed'
        cfg=read(root/'PROTOCOL342.json');seal=read(root/'ALL_CHOICES_SEALED342.json')
        assert sha(root/'PROTOCOL342.json')==seal['protocol_sha256']
        assert sha(root/'CODE_FREEZE.json')==seal['code_freeze_sha256']
        for name,digest in read(root/'CODE_FREEZE.json').items():assert sha(root/'code'/name)==digest,name
        episode_results=read(root/'BUILDERS_SEALED342.json')['episodes']
        scope='Crossed build/selection stages,both terminal and sealed.'
    sessions=read(root/'SESSIONS337.json');commands=[];sources=[];reports=[];fit_audits=[]
    for row in sessions:
        run=Path(row['path']);state=read(run/'broker/STATE.json')
        for ident,value in state['programs'].items():
            program=Path(value['path']);report=read(program/'REPORT.json')
            for relative,digest in report['artifacts'].items():assert sha(program/relative)==digest,(row['id'],ident,relative)
            reports.append(dict(builder=row['id'],program=ident,status=report['status'],fit_entries=report['fit_entry_executions'],
                classifier_fit_calls=report['confirmed_estimator_fits'],nonzero_optimizer_steps=report['nonzero_sampled_optimizer_steps'],
                identity_invariance_passed=report['identity_invariance_passed'],training_only=report['training_only']))
            if ident!='p00':
                for path in sorted((program/'source').rglob('*')):
                    if path.is_file():sources.append(dict(builder=row['id'],program=ident,file=str(path.relative_to(program/'source')),sha256=sha(path),bytes=path.stat().st_size))
        path=run/'primary'/row['id']/'events.jsonl'
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')=='item.completed' and item.get('type')=='command_execution':
                command=item['command'];review=python_review(command)
                commands.append(dict(builder=row['id'],event_line=line_number,event_id=item.get('id'),command=command,exit_code=item.get('exit_code'),
                    command_sha256=__import__('hashlib').sha256(command.encode()).hexdigest(),**review))
    fit_paths=root.glob('subjects/*/programs/*/**/training_audit.json') if generic_native else root.rglob('training_audit.json')
    for path in fit_paths:
        if 'evaluation' in path.relative_to(root).parts:continue
        value=read(path)
        if value.get('mode')!='fit':continue
        process=read(path.parent/'PROCESS.json');fit_audits.append(dict(path=str(path.relative_to(root)),
            process_exit_code=process.get('returncode'),status=value['status'],completed_estimator_fits=value['completed_estimator_fits'],
            optimizer_steps=value['optimizer_steps'],nonzero_sampled_optimizer_steps=value['nonzero_sampled_optimizer_steps'],
            actual_estimator_rows=[dict(class_name=e['class_name'],rows=e['rows'],status=e['status']) for e in value['estimator_fit_events']]))
    receipts=[read(p) for p in [*root.glob('subjects/*/model_gateway/*/*_receipt.json'),*root.glob('episodes/*/model_gateway/*/*_receipt.json')]]
    assert all(r.get('usage') and r['returned_model_ids']==['deepseek-flash'] for r in receipts)
    possible=[c for c in commands if c['possible_learning_calls'] or c['unresolved']]
    result=dict(utc=utc(),protocol_iteration=cfg['iteration'],budget_profile=cfg.get('budget_profile','standard'),
        audited_run=str(root),code_and_registered_artifact_hashes_passed=True,
        builder_results=episode_results,registered_program_reports=reports,stage_scope=scope,
        unique_source_files=sources,real_fit_audit_records=fit_audits,
        copied_historical_baseline_fit_records=3 if cfg.get('reuse_baseline_from') else 0,
        newly_executed_fit_audit_records=len(fit_audits)-(3 if cfg.get('reuse_baseline_from') else 0),
        command_execution_records=len(commands),local_command_screen=possible,
        local_learning_screen_requires_manual_review=bool(possible),
        local_screen_limits='Visible completed command/heredoc AST only;does not prove absence of indirect/dynamic/local fitting. Source and unresolved commands must be reviewed before an efficacy claim.',
        upstream_requests=len(receipts),HTTP200=sum(r.get('upstream_status')==200 for r in receipts),all_usage_present=True,
        input_tokens=sum(r['usage']['input_tokens'] for r in receipts),output_tokens=sum(r['usage']['output_tokens'] for r in receipts),
        final_arrays_read=False,new_fits=0,new_predictions=0)
    save(root/'audit/TRIAL_PROVENANCE_REVIEW.json',result)
    save(root/'audit/COMPLETED_COMMANDS_REVIEW.json',commands)
    print(json.dumps(dict(run=str(root),commands=len(commands),flagged=len(possible),new_fit_records=result['newly_executed_fit_audit_records'],requests=len(receipts),final_arrays_read=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();audit(args.run.resolve())
