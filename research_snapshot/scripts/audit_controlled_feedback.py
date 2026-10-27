"""Audit real fitting, full validation evidence and subject traces before scores.

No final arrays or labels are read. Trace scanning is an interpretable screen,
not a claim to detect every possible form of hidden computation.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import shutil

import numpy as np

from controlled_feedback import save, sha, utc
from evaluate_controlled_feedback import independent


def audit(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    train=dict(np.load(root/'hidden/train.npz',allow_pickle=False))
    dev=dict(np.load(root/'hidden/val.npz',allow_pickle=False))
    classes=config['classes'];cases=[];traces=[];checks=[];recipe_vectors={};repeat_checks=[]
    for case in config['cases']:
        run=root/case['id'];job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
        output=run/'primary'/job['id']/job['id'];status=json.loads((output/'status.json').read_text())
        assert status['status']!='running'
        broker=json.loads((run/'broker/STATE.json').read_text())
        events=[json.loads(x) for x in (output/'events.jsonl').read_text().splitlines()]
        items=[x['item'] for x in events if x.get('type')=='item.completed']
        commands=[x for x in items if x.get('type')=='command_execution']
        mcp=[x for x in items if x.get('type')=='mcp_tool_call']
        ws=run/job['workspace'];sources=[];models=[]
        for path in ws.rglob('*'):
            if not path.is_file():continue
            if path.suffix in ['.joblib','.pkl','.pickle','.pt','.pth']:models.append(str(path.relative_to(ws)))
            if path.suffix=='.py':
                try:
                    tree=ast.parse(path.read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
                    sources.append(dict(path=str(path.relative_to(ws)),sha256=sha(path),
                        fitting_calls=[c for c in calls if re.search(r'(?:^|\.)(fit|fit_transform|partial_fit|backward|step)$',c)]))
                except (SyntaxError,UnicodeError) as e:sources.append(dict(path=str(path.relative_to(ws)),parse_error=str(e)))
        pattern=re.compile(r'\.fit\s*\(|\.fit_transform\s*\(|\.backward\s*\(|\.step\s*\(|/hidden/|https?://|curl |wget ')
        flagged=[x for x in commands if pattern.search(x.get('command',''))]
        fitting_started=fitting_completed=0
        for trial in broker['trials']:
            folder=run/'broker'/trial['id'];eventfile=folder/'fit_events.jsonl'
            fits=[json.loads(line) for line in eventfile.read_text().splitlines()] if eventfile.exists() else []
            fitting_started+=sum(f['status']=='started' for f in fits)
            fitting_completed+=sum(f['status']=='completed' for f in fits)
            if trial['status']!='complete':continue
            assert Counter(f['role'] for f in fits if f['status']=='completed')==dict(cv_0=1,cv_1=1,cv_2=1,full_training=1)
            report=json.loads((folder/'fit_report.json').read_text());saved=np.load(folder/'predictions.npz',allow_pickle=False)
            key=json.dumps(trial['recipe'],sort_keys=True)
            vectors={part:saved[part].copy() for part in ['train','cv','development']}
            if key in recipe_vectors:
                assert all(np.array_equal(vectors[k],recipe_vectors[key]['vectors'][k]) for k in vectors)
                repeat_checks.append(dict(case=case['id'],candidate=trial['id'],prior=recipe_vectors[key]['case'],passed=True))
            else:recipe_vectors[key]=dict(case=case['id']+'/'+trial['id'],vectors=vectors)
            assert sha(folder/'checkpoint.joblib')==report['checkpoint_sha256']
            assert report['train_input_sha256']==sha(root/'hidden/train.npz') and report['development_input_sha256']==sha(root/'hidden/val.npz')
            ids=set(train['sample_ids'].tolist());seen=[]
            for fold in report['cv_folds']:
                fitting=set(fold['fitting_ids']);checking=set(fold['checking_ids'])
                assert not fitting&checking and fitting|checking==ids
                assert not set(fold['fitting_conditions'])&set(fold['checking_conditions'])
                seen.extend(fold['checking_ids'])
            assert set(seen)==ids and len(seen)==len(ids)
            for part, data in [('train',train),('cv',train),('development',dev)]:
                evidence=json.loads((folder/'evidence'/(part+'.json')).read_text())
                rows=evidence['rows'];index={r['sample_id']:r for r in rows}
                assert len(index)==len(rows)==len(data['sample_ids'])
                pred=np.asarray([index[str(i)]['predicted'] for i in data['sample_ids']])
                truth=np.asarray([index[str(i)]['actual'] for i in data['sample_ids']])
                assert np.array_equal(truth,data['labels']) and np.array_equal(pred,saved[part])
                metric=independent(truth,pred,classes)
                assert metric['confusion']==report[part]['confusion'] and abs(metric['macro_f1']-report[part]['macro_f1'])<1e-12
                assert metric['confusion']==evidence['metrics']['confusion']
                for condition in np.unique(data['condition_id']):
                    mask=data['condition_id']==condition
                    assert independent(truth[mask],pred[mask],classes)['confusion']==evidence['by_condition'][str(condition)]['confusion']
                checks.append(dict(case=case['id'],candidate=trial['id'],split=part,passed=True))
        receipts=[json.loads(x.read_text()) for x in (run/'model_gateway'/job['id']).glob('*_receipt.json')]
        gateway=json.loads((run/'model_gateway'/job['id']/'status.json').read_text())
        assert len(receipts)==gateway['requests'] and not gateway['inflight']
        access=[json.loads(x) for x in (run/'broker/ACCESS.jsonl').read_text().splitlines()]
        released=[x for x in access if x['response'].get('ok') and any(isinstance(t.get('development'),dict) for t in x['response']['result']['trials'])]
        if case['condition']=='delayed':
            for x in access:
                if not x['response'].get('ok'):continue
                result=x['response']['result'];n=len(result['trials'])
                if n<config['candidate_fits_per_case']:
                    assert all(not isinstance(t.get('development'),dict) and 'development' not in t.get('validation_evidence',{}) for t in result['trials'])
            devfiles=list((run/'feedback/evidence').glob('*/development.json'))
            if devfiles:
                assert released
                start=datetime.fromisoformat(released[0]['started_utc']).timestamp()
                assert all(p.stat().st_mtime>=start for p in devfiles)
        traces.append(dict(case=case['id'],commands=commands,mcp=mcp,flagged_commands=flagged,sources=sources,new_serialized_models=models))
        cases.append(dict(case=case['id'],status=status['status'],exit_code=status.get('exit_code'),elapsed_seconds=status['elapsed_seconds'],
            requests=len(receipts),upstream_statuses=dict(Counter(str(x.get('upstream_status')) for x in receipts)),
            returned_models=gateway['returned_model_ids'],input_tokens=gateway['input_tokens'],output_tokens=gateway['output_tokens'],
            candidates_attempted=len(broker['trials']),candidates_completed=sum(t['status']=='complete' for t in broker['trials']),
            actual_fit_started=fitting_started,actual_fit_completed=fitting_completed,
            command_count=len(commands),failed_commands=[x['id'] for x in commands if x.get('exit_code')!=0],mcp_calls=len(mcp),
            mcp_error_count=sum(bool(x.get('error')) for x in mcp),flagged_commands=[x['id'] for x in flagged],
            extra_fitting_code_flags=[s for s in sources if s.get('fitting_calls')],new_serialized_models=models,
            choice=broker['final_choice']))
    result=dict(utc=utc(),cases=cases,validation_checks=checks,validation_check_count=len(checks),
        actual_formal_fit_started=sum(c['actual_fit_started'] for c in cases),actual_formal_fit_completed=sum(c['actual_fit_completed'] for c in cases),
        new_native_qualification_fits=4,requests=sum(c['requests'] for c in cases),normal_completions=sum(c['status']=='finished' and c['exit_code']==0 for c in cases),
        unique_real_recipes=len(recipe_vectors),identical_recipe_prediction_checks=repeat_checks,
        final_scores_consulted=False,scope='Numerical evidence and trace screen. Flagged shell commands require manual inspection before asserting no extra fitting.')
    save(root/'audit/TRACE_COMMAND_SCREEN.json',dict(utc=utc(),cases=traces))
    save(root/'EXECUTION_AUDIT.json',result)
    project=Path(__file__).resolve().parents[1];save(project/config['result_directory']/'EXECUTION_AUDIT.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ['actual_formal_fit_started','actual_formal_fit_completed','requests','normal_completions','validation_check_count']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run.resolve())
