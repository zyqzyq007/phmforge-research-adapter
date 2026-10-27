"""Qualify reported annular-pressure records; never infer health from zeros."""
import argparse,csv,hashlib,json,re
from pathlib import Path
import pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    frame=pd.read_csv(a.input,low_memory=False)
    names=list(frame.columns)
    pressures=['Bradenhead Pressure','BH Blowdown 30 min','BH After 5 min SI','Intermediate Pressure','Intermediate Blowdown 30 min','Intermediate After 5 min SI','Casing Pressure','Tubing Pressure']
    flags=[c for c in names if 'Flow -' in c]
    source=frame['Source'].fillna('<MISSING>').value_counts().to_dict()
    dates=pd.to_datetime(frame['Date of Test'],format='mixed',errors='coerce')
    before=dates.lt(pd.Timestamp('2018-01-01'))
    remarks=frame['Remarks'].fillna('')
    explicit=re.compile(r'\b(?:(?:did|could) not|unable to|cannot|can not)\s+(?:be\s+)?test(?:ed)?\b|\bnot tested\b',re.I)
    configuration=re.compile(r'\b(?:plugged|frozen|broken|inaccessible|buried)\b|\bno (?:access|valve|port)\b',re.I)
    hits=remarks.str.contains(explicit)
    configs=remarks.str.contains(configuration)
    both_zero=frame['Bradenhead Pressure'].eq(0)&frame['Intermediate Pressure'].eq(0)
    idx=frame.index[hits|configs]
    fields=['API Well Number','Date of Test','Test Year']+pressures+flags+['Remarks','Source']
    selected=frame.loc[idx,fields].copy()
    selected.insert(0,'raw_data_row',idx+1)
    selected['explicit_no_test_screen']=hits.loc[idx]
    selected['configuration_screen']=configs.loc[idx]
    selected['both_initial_annulus_pressures_zero']=both_zero.loc[idx]
    selected.to_csv(a.output/'screened_records.csv',index=False)
    counts={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'records':len(frame),'columns':names,
      'unique_well_ids':int(frame['API Well Number'].nunique()),
      'duplicate_rows_excluding_saved_index':int(frame.drop(columns=['Unnamed: 0']).duplicated().sum()),
      'source_rows':source,'date_min':str(dates.min()),'date_max':str(dates.max()),'date_unparseable_or_missing':int(dates.isna().sum()),
      'pre2018_records_with_known_date':int(before.sum()),'pre2018_unique_wells_with_known_date':int(frame.loc[before,'API Well Number'].nunique()),
      'records_with_missing_remarks':int(frame['Remarks'].isna().sum()),
      'test_year_counts':{str(k):int(v) for k,v in frame['Test Year'].value_counts().sort_index().items()},
      'pressure_fields':{c:{'missing':int(frame[c].isna().sum()),'zero':int(frame[c].eq(0).sum()),'positive':int(frame[c].gt(0).sum()),'negative':int(frame[c].lt(0).sum())} for c in pressures},
      'flow_flag_values':{c:{str(k):int(v) for k,v in frame[c].fillna('<MISSING>').value_counts().items()} for c in flags},
      'screen_counts':{'explicit_no_test':int(hits.sum()),'configuration':int(configs.sum()),'union':int((hits|configs).sum()),
       'both_initial_annulus_zero':int(both_zero.sum()),'explicit_no_test_and_both_zero':int((hits&both_zero).sum()),
       'configuration_and_both_zero':int((configs&both_zero).sum())},
      'semantic_status':'regex screens only; not adjudicated invalid measurements, not fault truth, not Agent failures',
      'maintenance_action_columns':[c for c in names if re.search(r'accept|releas|repair|remediat|action|certif|risk|limit',c,re.I)]}
    with a.input.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    independent={'records':len(rows),'unique_well_ids':len({x['API Well Number'] for x in rows}),
      'source_rows':{s:sum(x['Source']==s for x in rows) for s in sorted({x['Source'] for x in rows})},
      'flow_flag_missing':{c:sum(x[c]=='' for x in rows) for c in flags},
      'explicit_no_test_screen':sum(bool(explicit.search(x['Remarks'])) for x in rows)}
    assert independent['records']==counts['records']
    assert independent['unique_well_ids']==counts['unique_well_ids']
    assert independent['source_rows']==counts['source_rows']
    assert independent['explicit_no_test_screen']==counts['screen_counts']['explicit_no_test']
    assert all(independent['flow_flag_missing'][c]==int(frame[c].isna().sum()) for c in flags)
    for name,value in [('counts.json',counts),('independent_parser.json',independent)]:
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:counts[k] for k in ['records','unique_well_ids','pre2018_records_with_known_date','pre2018_unique_wells_with_known_date','screen_counts','maintenance_action_columns']}))
if __name__=='__main__':main()
