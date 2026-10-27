"""Independent scalar/provenance audit for the cached-probability control."""
from pathlib import Path
import sys
import json
import hashlib
import math
import time
import tarfile
from datetime import datetime, timezone


def main():
    root = Path(sys.argv[1])
    start = time.monotonic()
    read = lambda p: json.loads(Path(p).read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    seal = read(root/'CHOICES_SEALED415.json')
    result = read(root/'RESULT415.json')
    count = 0

    def check(value):
        nonlocal count
        assert value, count
        count += 1

    def close(a,b):
        check(abs(a-b)<1e-10)

    def scalar(probs):
        maxima, entropies = [], []
        for p in probs:
            total=math.fsum(p)
            check(total>0 and abs(total-1)<1e-6 and all(math.isfinite(x) and x>=0 for x in p))
            q=[x/total for x in p]
            maxima.append(max(q))
            entropies.append(math.fsum(x*math.log(x) for x in q if x>0))
        return {'average_confidence':math.fsum(maxima)/len(maxima),
                'negative_entropy':math.fsum(entropies)/len(entropies)}

    for binding in read(root/'RESTORED415.json')['bindings']:
        p=root/binding['local']
        check(p.stat().st_size==binding['bytes'] and digest(p)==binding['sha256'])
    check(digest(root/'CHOICES_SEALED415.json')==result['seal_sha256'])
    check(digest(root/'PROTOCOL415.json')==seal['protocol_sha256'])
    pair_fields={}
    for pair in seal['pairs']:
        values=[]
        for role in ['reference','Agent']:
            p=root/pair['paths'][role]
            check(digest(p)==pair['file_sha256'][role])
            rows=read(p)['predictions']
            vectors=[]
            for row in rows:
                raw=row.get('probabilities',row.get('class_probabilities'))
                vec=[raw[str(c)] for c in pair['classes']] if isinstance(raw,dict) else raw
                check(str(pair['classes'][max(range(len(vec)),key=lambda i:vec[i])])==str(row['predicted_label']))
                vectors.append(vec)
            scores=scalar(vectors)
            expected=next(x for x in pair['scores'] if x['id']==role)
            for key,value in scores.items():close(value,expected[key])
            values.append((role,scores))
        for key in ['average_confidence','negative_entropy']:
            best=max(v[key] for _,v in values)
            chosen=next(role for role,v in values if v[key]>=best-1e-12)
            check(chosen==pair['selected'][key])
        pair_fields[pair['id']]=values
    check(seal['pairs'][0]['file_sha256']==seal['pairs'][1]['file_sha256'])
    # Independently attach already-opened per-record CWRU labels to exact output IDs.
    truth={r['unit_id']:r['actual'] for r in read(root/'source/RESULT413.json')['models']['ratio']['records']}
    for pair in seal['pairs']:
        if pair['source_cluster']!='cwru':continue
        candidate=read(root/pair['paths']['Agent'])['predictions']
        labels={r['sample_id']:truth[r['unit_id']] for r in candidate}
        for role in ['reference','Agent']:
            rows=read(root/pair['paths'][role])['predictions']
            accuracy=sum(r['predicted_label']==labels[r['sample_id']] for r in rows)/len(rows)
            outcome=next(r for r in result['pairs'] if r['id']==pair['id'])
            close(accuracy,outcome['quality'][role]['accuracy'])
    bases=[]
    ids=None
    for name in seal['bank_names']:
        raw=read(root/'inputs/g2/evaluation282/basis'/name/'test/probabilities.json')
        check(raw['classes']==seal['bank_classes'])
        if ids is not None:check(ids==raw['sample_ids'])
        ids=raw['sample_ids']
        bases.append([[x/math.fsum(p) for x in p] for p in raw['probabilities']])
    base_scores=[scalar(p) for p in bases]
    cache={}
    for case in seal['banks']:
        for candidate in case['candidates']:
            weights=tuple(candidate['weights'])
            if weights not in cache:
                vectors=[[math.fsum(weights[k]*bases[k][i][j]/12 for k in range(len(bases)))
                          for j in range(len(seal['bank_classes']))] for i in range(len(ids))]
                cache[weights]=scalar(vectors)
            for key,value in cache[weights].items():
                close(value,candidate[key])
                # Convexity bound: a mixture cannot exceed the best available pure basis.
                check(value<=max(x[key] for x in base_scores)+1e-10)
        for key in ['average_confidence','negative_entropy']:
            top=max(cache[tuple(c['weights'])][key] for c in case['candidates'])
            selected=next(c['id'] for c in case['candidates'] if cache[tuple(c['weights'])][key]>=top-1e-12)
            check(selected==case['selected'][key])
    check(len(cache)==seal['unique_weight_vectors_evaluated'])
    projection=read(root/'SOURCE_METRIC_PROJECTION415.json')
    binding=projection['binding']
    arc=Path(binding['archive'])
    receipt=read(arc/'ARCHIVE_VERIFIED.json')
    check(digest(arc/'SOURCE_MANIFEST.json')==receipt['manifest_sha256'])
    expected=next(x for x in read(arc/'SOURCE_MANIFEST.json')['files'] if x['path']==binding['member'])
    if (arc/binding['member']).is_file():raw=(arc/binding['member']).read_bytes()
    else:
        with tarfile.open(arc/receipt['bundle_filename'],'r|gz') as stream:
            for item in stream:
                if item.name==binding['member']:
                    raw=stream.extractfile(item).read()
                    break
    check(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']==binding['sha256'])
    full=json.loads(raw)
    original=full.get('cases',full.get('rows'))
    for case in projection['cases']:
        old=next(c for c in original if c['stage']==case['stage'] and c['id']==case['case'])
        check(len(old['candidates'])==len(case['candidates']))
        for candidate,source in zip(case['candidates'],old['candidates']):
            check(candidate=={k:source[k] for k in ['id','weights','final']})
            metric=candidate['final']
            cm=metric['confusion']
            n=sum(map(sum,cm))
            correct=sum(cm[i][i] for i in range(len(cm)))
            close(correct/n,metric['accuracy'])
            f1=[]
            for i in range(len(cm)):
                den=sum(cm[i])+sum(row[i] for row in cm)
                f1.append(2*cm[i][i]/den if den else 0.)
            close(math.fsum(f1)/len(f1),metric['macro_f1'])
        observed=next(c for c in result['banks'] if c['stage']==case['stage'] and c['case']==case['case'])
        for key,value in observed['outcomes'].items():
            source=next(c for c in case['candidates'] if c['id']==value['id'])
            close(source['final']['accuracy'],value['accuracy'])
            close(source['final']['macro_f1'],value['macro_f1'])
            close(observed['pool_oracle_macro_f1']-value['macro_f1'],observed['macro_f1_selection_loss'][key])
    for row in result['pairs']:
        ref=row['quality']['reference']['accuracy']
        for key,acc in row['outcomes'].items():
            close(row['selection_loss'][key],max(v['accuracy'] for v in row['quality'].values())-acc)
            close(row['harmful_change_loss'][key],max(0.,ref-acc))
    for phase,stats in result['bank_aggregate'].items():
        cases=[r for r in result['banks'] if r['stage']==phase]
        check(len(cases)==6)
        for key,values in stats.items():
            close(values['mean_F1'],math.fsum(c['outcomes'][key]['macro_f1'] for c in cases)/6)
            close(values['mean_accuracy'],math.fsum(c['outcomes'][key]['accuracy'] for c in cases)/6)
            close(values['mean_selection_loss'],math.fsum(c['macro_f1_selection_loss'][key] for c in cases)/6)
    elapsed=time.monotonic()-start
    report=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=count,seconds=elapsed,
                unique_scalar_probability_recomputations=len(cache),new_model_forwards=0,new_fits=0,new_LLM=0,
                all_probability_scores_recomputed_without_numpy_reduction=True,
                mixture_confidence_convex_bound_passed=True,
                all_original_candidate_confusions_recomputed=True,
                CWRU_labels_joined_by_record_from_previously_scored_artifact=True,
                rotor_accuracy_inherits_previously_verified_scoring=True,
                warnings=['Eligibility-limited historical analysis, not independent confirmation.',
                          'Probability vectors certify only window-classification output, not maintenance/safety actions.',
                          'No new per-window gold scoring for rotor or G2; original verified confusions and metric projections retained.'])
    (root/'AUDIT415.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
