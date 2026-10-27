"""Independent outcomes for PHM tool-report reviews; no text-grading shortcut."""
import argparse
import hashlib
import json
from pathlib import Path


def score(root):
    gold = json.loads((root/'hidden/gold86.json').read_text())
    results = []
    for job in json.loads((root/'plan86.json').read_text())['jobs']:
        expected = [row for row in gold if row['job_id']==job['id']]
        output = root/job['workspace']/'results.json'
        record = {'job_id':job['id'],'expected_reports':len(expected)}
        try:
            rows = json.loads(output.read_text())['results']
            by_id = {row['report_id']:row for row in rows}
            assert len(rows)==len(by_id)==len(expected)
            assert set(by_id)=={row['report_id'] for row in expected}
            comparisons=[]
            for row in expected:
                actual = by_id[row['report_id']]
                def integer_pair(value):
                    return isinstance(value,list) and len(value)==2 and all(type(v) is int and 0<=v<=100 for v in value) and value[0]<=value[1]
                assert integer_pair(actual['fp_bounds']) and integer_pair(actual['fn_bounds'])
                assert actual['fp_admission'] in ['eligible_on_audit_sample','ineligible_on_audit_sample','indeterminate']
                comparisons.append({'report_id':row['report_id'],'correct_fp_bounds':actual['fp_bounds']==row['fp_bounds'],
                      'correct_fn_bounds':actual['fn_bounds']==row['fn_bounds'],
                      'correct_admission':actual['fp_admission']==row['fp_admission'],
                      'unsafe_admission':actual['fp_admission']=='eligible_on_audit_sample' and row['fp_bounds'][0]>10,
                      'missed_eligible':actual['fp_admission']=='ineligible_on_audit_sample' and row['fp_bounds'][1]<=10,
                      'actual':actual,'expected':{k:row[k] for k in ['fp_bounds','fn_bounds','fp_admission']}})
            record.update(scientific_status='scored',correct_reports=sum(all(v[k] for k in ['correct_fp_bounds','correct_fn_bounds','correct_admission']) for v in comparisons),
                          correct_admissions=sum(v['correct_admission'] for v in comparisons),
                          unsafe_admissions=sum(v['unsafe_admission'] for v in comparisons),
                          missed_eligible=sum(v['missed_eligible'] for v in comparisons),
                          rows=comparisons,output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
            record.update(scientific_status='missing_or_invalid',error=type(error).__name__+': '+str(error)[:300])
        results.append(record)
    return {'scope':'AI-authoredPHM tool-report audit; source reported data, not new acoustic classifier predictions',
            'task_count':len(results),'report_presentations':60,'distinct_report_pairs':30,
            'portfolio_reuses_same30':True,'tasks':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=score(args.run.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in task.items() if k!='rows'} for task in results['tasks']]))
