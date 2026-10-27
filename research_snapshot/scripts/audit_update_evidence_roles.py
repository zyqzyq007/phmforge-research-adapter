"""Post-hoc effect decomposition on already opened development/check evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
root=parser.parse_args().run.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(root/'PROTOCOL352.json');archive=Path(cfg['archive'])
assert sha(archive/'SOURCE_MANIFEST.json')==cfg['manifest_sha256']
manifest={x['path']:x for x in read(archive/'SOURCE_MANIFEST.json')['files']};bindings={}
def bound(name):
    path=archive/name;assert sha(path)==manifest[name]['sha256'];bindings[name]=sha(path);return read(path)
dev=bound('reference/p00/REPORT.json')['development']['records']
check=bound('subjects/local_revision_1/broker/STATE.json')['probe']['reports']['p00']['records']
truth={x['group']:x['actual'] for x in dev+check};programs={x['id']:x for x in cfg['programs']}
def predicted(pid,index):
    rows=read(root/'predictions'/pid/str(index)/'predictions.json')['predictions']
    values={x['sample_id']:x['predicted_label'] for x in rows};count=Counter(values.values())
    return min(count,key=lambda y:(-count[y],y)),values
rows=[]
for old,new in [('baseline','global_update'),('dual_lda','guarded_lda')]:
    roles={};details=[]
    for role in ['development','check']:
        counts=Counter()
        for index,packet in enumerate(programs[old]['packets']):
            if packet['role']!=role:continue
            a,av=predicted(old,index);b,bv=predicted(new,index);assert set(av)==set(bv)
            with np.load(archive/packet['input'],allow_pickle=False) as z:
                group=str(z['groups'][0]);condition=float(z['condition_id'][0])
            y=truth[group];counts['records']+=1;counts['record_predictions_changed']+=a!=b
            counts['any_window_predictions_changed']+=av!=bv
            counts['corrections']+=a!=y and b==y;counts['regressions']+=a==y and b!=y
            details.append(dict(role=role,condition_hz=condition,group=group,author_label=y,old_prediction=a,new_prediction=b))
        roles[role]=dict(counts)
    rows.append(dict(incumbent=old,candidate=new,roles=roles,records=details))
value=dict(status='complete',scope='Posthocdiagnosticofalreadyopenedlegalevidence;notapreplannedmethodtest.',
    rows=rows,extra_source_bindings=bindings,new_LLM_calls=0,new_fits=0,new_predictions=0,
    no_final_performance_read=True,no_A338_performance_read=True,
    result_is_not_evidence_new_acceptance_rule_generalizes=True)
(root/'EVIDENCE_ROLES352.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({r['incumbent']+'->'+r['candidate']:r['roles'] for r in rows}))
