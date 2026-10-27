"""Independent complete-denominator scoring; no answer repair or model grading."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
RUN = None
OUTPUT = None
def save(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
def recover(text, expected_id):
 objects=[]
 try: objects=[json.loads(text)]; kind='strict_json'
 except ValueError:
  kind='unique_fenced_json'
  for match in re.finditer(r'\x60\x60\x60(?:json)?\s*\n(.*?)\n\x60\x60\x60',text,re.S):
   try: objects.append(json.loads(match[1]))
   except ValueError: pass
 if len(objects)!=1: return None, 'not_uniquely_recoverable'
 value=objects[0]
 expected={'case_id':expected_id,'analysis_code':'analysis.py','predictions_file':'predictions.json','validation_file':'validation.json','report_file':'report.md'}
 if not isinstance(value,dict) or value.get('results') != [expected]: return None,'metadata_inventory_or_paths_invalid'
 return value,kind
def main():
 gold=json.loads((RUN/'controller_only/query_gold.json').read_text())
 jobs=json.loads((RUN/'formal_plan.json').read_text())['jobs']
 rows, workflows, extras=[],[],[]
 for job in jobs:
  folder=RUN/'formal_outputs'/job['id']
  state=json.loads((folder/'status.json').read_text())
  assert state['status']!='running','Wait for scientific artifacts to become immutable'
  workspace=RUN/job['workspace']
  metadata_path=folder/'response.json'
  recovered,kind=recover(metadata_path.read_text(),job['id']) if metadata_path.exists() else (None,'absent')
  prediction_path=workspace/'predictions.json'
  raw=[]
  parse_error=None
  try:
   data=json.loads(prediction_path.read_text())
   assert isinstance(data,dict) and isinstance(data.get('results'),list)
   raw=data['results']
  except (OSError,ValueError,TypeError,AssertionError) as error: parse_error=type(error).__name__
  counts=Counter(value.get('case_id') for value in raw if isinstance(value,dict))
  raw_by_id={value.get('case_id'):value for value in raw if isinstance(value,dict)}
  planned_ids=[value['case_id'] for value in gold]
  extra_ids=[value.get('case_id') if isinstance(value,dict) else None for value in raw if not isinstance(value,dict) or value.get('case_id') not in planned_ids]
  extras.append({'job_id':job['id'],'unplanned_entries':extra_ids})
  valid=0
  for source in gold:
   case_id=source['case_id']
   row={'job_id':job['id'],'condition':job['condition'],'repeat':job['repeat'],'case_id':case_id,
        'author_fault_id':source['author_fault_id'],'alias_set':source['alias_set'],'status':'missing',
        'strict_top1':False,'strict_top3':False,'alias_top3':False,'classes':None,'confidence':None}
   value=raw_by_id.get(case_id)
   if value is not None:
    classes=value.get('ranked_class_ids')
    confidence=value.get('top_confidence')
    issues=[]
    if counts[case_id]!=1: issues.append('duplicate_case')
    if not isinstance(classes,list) or len(classes)>3 or any(type(code)!=int or not 1<=code<=15 for code in classes) or len(set(classes or []))!=len(classes or []): issues.append('invalid_classes')
    if isinstance(classes,list) and len(classes)==0:
     if confidence is not None: issues.append('abstention_confidence_not_null')
    elif type(confidence) not in [float,int] or not 0<=confidence<=1: issues.append('invalid_confidence')
    if not isinstance(value.get('rationale'),str) or not isinstance(value.get('evidence'),dict): issues.append('missing_rationale_or_evidence')
    row.update(classes=classes,confidence=confidence,issues=issues)
    if issues: row['status']='invalid'
    else:
     valid+=1
     row.update(status='answered' if classes else 'abstained',
      strict_top1=bool(classes) and classes[0]==source['author_fault_id'],
      strict_top3=source['author_fault_id'] in classes,
      alias_top3=bool(set(classes)&set(source['alias_set'])))
   rows.append(row)
  counts_status=Counter(row['status'] for row in rows if row['job_id']==job['id'])
  scored=[row for row in rows if row['job_id']==job['id']]
  workflows.append({'job_id':job['id'],'condition':job['condition'],'repeat':job['repeat'],
   'native_status':state['status'],'exit_code':state.get('exit_code'),'elapsed_seconds':state['elapsed_seconds'],
   'metadata_strict_valid':state.get('response_valid'),'metadata_recovery':kind,'metadata_recoverable':recovered is not None,
   'prediction_parse_error':parse_error,'planned':11,'valid_outputs':valid,'status_counts':dict(counts_status),
   'strict_top1':sum(row['strict_top1'] for row in scored),'strict_top3':sum(row['strict_top3'] for row in scored),
   'alias_top3':sum(row['alias_top3'] for row in scored),'prediction_order_matches':[
    value.get('case_id') for value in raw if isinstance(value,dict)]==planned_ids,
   'unplanned_entries':len(extra_ids),
   'predictions_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest() if prediction_path.exists() else None})
 by_condition={}
 for condition in ['TEPTop6','TEPFullPrefix']:
  subset=[row for row in rows if row['condition']==condition]
  by_condition[condition]={'planned':22,'status_counts':dict(Counter(row['status'] for row in subset)),
   'strict_top1':sum(row['strict_top1'] for row in subset),'strict_top3':sum(row['strict_top3'] for row in subset),
   'alias_top3':sum(row['alias_top3'] for row in subset)}
 pairs=[]
 for source in gold:
  subset=[row for row in rows if row['case_id']==source['case_id']]
  pairs.append({'case_id':source['case_id'],'author_fault_id':source['author_fault_id'],'runs':[
   {key:row[key] for key in ['job_id','status','classes','confidence','strict_top1','strict_top3','alias_top3']} for row in subset]})
 inputs=json.loads((RUN/'task_input_manifest_catalog.json').read_text())
 mutation=[value['file'] for value in inputs if hashlib.sha256((RUN/value['file']).read_bytes()).hexdigest()!=value['sha256']]
 frozen=json.loads((RUN/'protocol_frozen.json').read_text())
 frozen_mutation=[name for name,digest in frozen['frozen_sha256'].items() if hashlib.sha256((RUN/name).read_bytes()).hexdigest()!=digest]
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'planned_workflows':4,'planned_answers':44,
  'known_source_classes':15,'upstream_not_alerted':[3,4,9,15],'completed_native':sum(x['native_status']=='finished' for x in workflows),
  'by_condition':by_condition,'workflows':workflows,'input_mutation':mutation,'protocol_mutation':frozen_mutation,
  'source_families':1,'E3':False,'label_scope':'Author file labels; alias grading is a separately reported historical-family proxy, not proof of indistinguishability.',
  'original_paper_shared_misses':[5,10],'no_control_or_verifier_accuracy_claim':True}
 assert not mutation and not frozen_mutation
 out=OUTPUT;out.mkdir(parents=True,exist_ok=False)
 save(out/'summary.json',result);save(out/'all_planned_answers.json',rows);save(out/'paired_faults.json',pairs);save(out/'extras.json',extras)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 RUN=args.run.resolve();OUTPUT=args.output.resolve()
 main()


