"""Bound error counts from rounded domain-harmonic diagnostic metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def harmonic_pair(x, y):
    return np.divide(2*x*y, x+y,
                     out=np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y))),
                     where=(x+y) != 0)


def compatible_counts(row, positives=50, negatives=50, rounding_tolerance=.00500001):
    fp1 = np.arange(negatives+1)[:,None]
    fp2 = np.arange(negatives+1)[None,:]
    compatible = []
    for tp1 in range(positives+1):
        for tp2 in range(positives+1):
            recall = float(harmonic_pair(tp1/positives,tp2/positives))*100
            if abs(recall-row['recall_hmean']) > rounding_tolerance:
                continue
            p1 = np.divide(tp1,tp1+fp1,out=np.zeros((negatives+1,1),float),where=(tp1+fp1)!=0)
            p2 = np.divide(tp2,tp2+fp2,out=np.zeros((1,negatives+1),float),where=(tp2+fp2)!=0)
            f1 = 2*tp1/(positives+tp1+fp1)
            f2 = 2*tp2/(positives+tp2+fp2)
            mask = ((np.abs(harmonic_pair(p1,p2)*100-row['precision_hmean']) <= rounding_tolerance)
                    & (np.abs(harmonic_pair(f1,f2)*100-row['F1_hmean']) <= rounding_tolerance))
            for a,b in zip(*np.where(mask)):
                compatible.append({'tp_source':tp1,'tp_target':tp2,'fp_source':int(a),'fp_target':int(b)})
    result = {k:row[k] for k in ['submission','rank','machine','auc','pauc',
                                'F1_hmean','recall_hmean','precision_hmean']}
    result['compatible_confusion_matrices'] = len(compatible)
    if compatible:
        false_positives = [v['fp_source']+v['fp_target'] for v in compatible]
        false_negatives = [2*positives-v['tp_source']-v['tp_target'] for v in compatible]
        result.update(pooled_fp_bounds=[min(false_positives),max(false_positives)],
                      pooled_false_negative_bounds=[min(false_negatives),max(false_negatives)])
    else:
        result['status'] = 'reported_triple_incompatible_with_assumed_harmonic_counts;do_not_infer_errorrates'
    return result


def analyze(published):
    items = [compatible_counts(row) for row in published['selected']]
    return {'scope':'Conditional inference from author-reported rounded domain-harmonic metrics and exact published50normal/50anomaly per domain. Not direct per-clip reproduction.',
            'selected_by':'Six bestteams byofficial rank','items':items,'count':len(items),
            'compatible_items':sum(item['compatible_confusion_matrices']>0 for item in items),
            'new_Agent_runs':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--published',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    results = analyze(json.loads(args.published.read_text()))
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'count':results['count'],'compatible':results['compatible_items']}))
