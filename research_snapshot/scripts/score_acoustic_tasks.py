"""Score complete blinded acoustic tasks with the original DCASE metric masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix, roc_auc_score


def metric_vector(labels, domains, scores, decisions):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    decisions = np.asarray(decisions, dtype=int)
    domains = np.asarray(domains)
    assert len(labels) == len(scores) == len(domains) == len(decisions)
    assert np.isfinite(scores).all()
    assert set(decisions.tolist()).issubset({0, 1})
    output = {'n':len(labels), 'auc_all':float(roc_auc_score(labels, scores)),
              'pauc':float(roc_auc_score(labels, scores, max_fpr=0.1))}
    for domain in ['source', 'target']:
        # Official metric: normals of this domain against anomalies of BOTH domains.
        mask = (domains == domain) | (labels != 0)
        output['auc_' + domain] = float(roc_auc_score(labels[mask], scores[mask]))
        subset = domains == domain
        tn, fp, fn, tp = confusion_matrix(labels[subset], decisions[subset], labels=[0,1]).ravel()
        precision = float(tp / max(1, tp+fp))
        recall = float(tp / max(1, tp+fn))
        output[domain] = {'n':int(subset.sum()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
                          'precision':precision,'recall':recall,
                          'f1':2*precision*recall/max(np.finfo(float).eps,precision+recall)}
    output['section_hmean'] = float(hmean(np.maximum([output['auc_source'],output['auc_target'],output['pauc']],np.finfo(float).eps)))
    return output


def score_task(root, machine, artifact=None):
    gold = json.loads((root / 'hidden' / (machine + '_gold.json')).read_text())
    artifact = artifact or root / 'tasks' / machine / 'results.json'
    payload = json.loads(artifact.read_text())
    rows = payload['results'] if isinstance(payload, dict) else payload
    predictions = {row['case_id']:row for row in rows}
    expected = {row['case_id'] for row in gold}
    assert len(rows) == len(predictions) == len(gold) and set(predictions) == expected
    ordered = [predictions[row['case_id']] for row in gold]
    assert all(type(row['decision']) is int and row['decision'] in (0,1) for row in ordered)
    return metric_vector([row['label'] for row in gold], [row['domain'] for row in gold],
                         [row['anomaly_score'] for row in ordered], [row['decision'] for row in ordered])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection', default='source_selection85_v2.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.run / args.selection).read_text())
    results = []
    for machine in selection['task_types']:
        try:
            metrics = score_task(args.run, machine)
            results.append({'machine':machine, 'scientific_status':'scored', **metrics})
        except (AssertionError,KeyError,ValueError,FileNotFoundError,TypeError) as error:
            results.append({'machine':machine,'scientific_status':'invalid_or_missing',
                            'error':type(error).__name__ + ': ' + str(error)[:300]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'DCASE selected whole sections; all failures retained',
                          'results':results},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False))
