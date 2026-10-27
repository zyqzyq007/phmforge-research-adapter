"""Strict external artifact metrics and complete, untruncated evidence packaging."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(rows, key):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty or invalid submission")
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError("Duplicate sample identity: " + value)
        result[value] = row
    return result


def aligned(predictions, gold, identity):
    p, g = keyed(predictions, identity), keyed(gold, identity)
    if p.keys() != g.keys():
        raise ValueError(f"Sample coverage differs: missing={sorted(g.keys()-p.keys())[:5]}, extra={sorted(p.keys()-g.keys())[:5]}")
    ids = list(g)
    return [p[i] for i in ids], [g[i] for i in ids]


def regression(predictions, gold, identity="unit_id", prediction_key="predicted_rul", truth_key="actual_rul"):
    p, g = aligned(predictions, gold, identity)
    a = np.asarray([x[prediction_key] for x in p], float)
    b = np.asarray([x[truth_key] for x in g], float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite or malformed regression artifact")
    residual = a - b
    return dict(n=len(a), mae=float(np.abs(residual).mean()),
                rmse=float(np.sqrt(np.square(residual).mean())),
                bias=float(residual.mean()), source="external numeric evaluator",
                pass_threshold=None, positive_error_lower_bound=None)


def classification(predictions, gold, declared_classes=None):
    p, g = aligned(predictions, gold, "sample_id")
    predicted = np.asarray([x["predicted_label"] for x in p])
    actual = np.asarray([x["label"] for x in g])
    classes = sorted(set(actual.tolist()))
    if declared_classes is None:
        if not set(predicted.tolist()) <= set(classes):
            raise ValueError("Unknown predicted class")
    else:
        # A legitimate source class may be absent from a small final sample.
        # Validate against the declared task, then use the observed union for
        # primary macro F1, matching sklearn's default multiclass convention.
        for value in [x['predicted_label'] for x in p]+[x['label'] for x in g]:
            if not any(type(value) is type(c) and value==c for c in declared_classes):
                raise ValueError('Class outside declared task universe')
        classes=sorted(set(actual.tolist())|set(predicted.tolist()))
    scores = []
    for c in classes:
        tp = int(((predicted == c) & (actual == c)).sum())
        fp = int(((predicted == c) & (actual != c)).sum())
        fn = int(((predicted != c) & (actual == c)).sum())
        scores.append(dict(label=c, support=int((actual == c).sum()), true_positive=tp,
                           false_positive=fp, false_negative=fn,
                           precision=tp/max(tp+fp, 1),
                           recall=tp/max(tp+fn,1), f1=2*tp/max(2*tp+fp+fn, 1)))
    groups = {}
    for a, b in zip(p, g):
        groups.setdefault(b["group"], []).append(a["predicted_label"] == b["label"])
    return dict(n=len(p), accuracy=float((predicted == actual).mean()),
                macro_f1=float(np.mean([x["f1"] for x in scores])),
                balanced_accuracy=float(np.mean([x["recall"] for x in scores if x['support']>0])),
                class_order=classes,
                confusion_matrix=[
                    [int(((actual == truth) & (predicted == prediction)).sum())
                     for prediction in classes] for truth in classes],
                confusion_matrix_axes="rows=actual, columns=predicted",
                per_class=scores, groups=len(groups),
                mean_recording_accuracy=float(np.mean([np.mean(x) for x in groups.values()])),
                source="external numeric evaluator", pass_threshold=None)


def evidence_bundle(calls, answer, destination, max_prompt_characters=180000):
    """Full evidence or explicit capacity failure; never an invisible prefix slice."""
    business, catalog = [], []
    for index, call in enumerate(calls):
        item = dict(sequence_index=index, **call)
        (catalog if call.get("tool") in {"list_tools", "tool_schema", "full_catalog"} else business).append(item)
    bundle = dict(answer=answer, business_calls=business, discovery_calls=catalog,
                  business_calls_count=len(business), discovery_calls_count=len(catalog),
                  evidence_truncated=False)
    destination = Path(destination)
    destination.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    prompt_evidence = json.dumps(dict(answer=answer, business_calls=business), ensure_ascii=False)
    return dict(artifact=str(destination), sha256=sha(destination), business_calls=len(business),
                discovery_calls=len(catalog), prompt_characters=len(prompt_evidence),
                complete_evidence_available=True, evidence_truncated=False,
                judge_ready=len(prompt_evidence) <= max_prompt_characters,
                blocking_reason=None if len(prompt_evidence) <= max_prompt_characters else
                "Full business evidence exceeds judge budget; chunked evidence review required, no score assigned")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["rul", "classification", "forecast"], required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pred = json.loads(args.predictions.read_text())
    gold = json.loads(args.gold.read_text())
    rows = pred.get("predictions", pred) if isinstance(pred, dict) else pred
    if args.task == "classification":
        score = classification(rows, gold)
    elif args.task == "rul":
        score = regression(rows, gold)
    else:
        score = regression(rows, gold, "timestamp", "prediction", "value")
    score.update(prediction_sha256=sha(args.predictions), gold_sha256=sha(args.gold))
    args.out.write_text(json.dumps(score, indent=2) + "\n")
    print(json.dumps(score))


if __name__ == "__main__":
    main()
