"""Independent exact scoring of frozen LongHaul diagnosis JSONL submissions."""
import argparse
import hashlib
import json
from pathlib import Path


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key')
        value[key] = item
    return value


def evaluate(episodes_path, predictions_path):
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    truth = {ep['episode_id']: ep['ground_truth'] for ep in episodes}
    if len(truth) != len(episodes):
        raise ValueError('Source has duplicate episode identifiers')
    predictions, errors = {}, []
    for number, line in enumerate(predictions_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(obj, dict) or set(obj) != {'episode_id', 'diagnosis'}:
                raise ValueError('Prediction row keys differ from the task schema')
            identifier = obj['episode_id']
            if not isinstance(identifier, str):
                raise ValueError('episode_id must be a string')
            diagnosis = obj['diagnosis']
            if not isinstance(diagnosis, dict) or set(diagnosis) != {'component', 'failure_mode'} or not all(
                isinstance(diagnosis.get(k), str) and diagnosis[k]
                for k in ('component', 'failure_mode')
            ):
                raise ValueError('Diagnosis needs nonempty component and failure_mode strings')
            predictions.setdefault(identifier, []).append(diagnosis)
        except (ValueError, KeyError, TypeError) as error:
            errors.append({'line': number, 'error_type': type(error).__name__})
    missing = sorted(set(truth) - set(predictions))
    extra = sorted(set(predictions) - set(truth))
    duplicate = sorted(k for k, v in predictions.items() if len(v) != 1)
    rows = []
    for ep in episodes:
        identifier = ep['episode_id']
        candidates = predictions.get(identifier, [])
        prediction = candidates[0] if len(candidates) == 1 else {}
        target = truth[identifier]
        component = prediction.get('component') == target['component']
        exact = component and prediction.get('failure_mode') == target['failure_mode']
        rows.append({'episode_id': identifier, 'machine_id': ep['machine_id'],
                     'detailed_log': not any('no detailed log available' in x for x in ep['log_excerpt']),
                     'prediction_count': len(candidates), 'prediction': prediction,
                     'ground_truth': {k: target[k] for k in ('component', 'failure_mode')},
                     'component_correct': component, 'exact_correct': exact})
    complete = not (errors or missing or extra or duplicate)
    n = len(rows)
    def metrics(values):
        return {'n': len(values), 'exact_correct': sum(x['exact_correct'] for x in values),
                'component_correct': sum(x['component_correct'] for x in values)}
    result = {'status': 'valid_complete' if complete else 'invalid_submission',
              'denominator': n, 'exact_correct': sum(x['exact_correct'] for x in rows),
              'component_correct': sum(x['component_correct'] for x in rows),
              'exact_accuracy_all_source_cases': sum(x['exact_correct'] for x in rows) / n,
              'component_accuracy_all_source_cases': sum(x['component_correct'] for x in rows) / n,
              'invalid_submission_score_note': 'Missing or duplicate cases get no credit; scores do not make invalid submissions valid.',
              'missing': missing, 'extra': extra, 'duplicate': duplicate, 'parse_errors': errors,
              'by_machine': {k: metrics([x for x in rows if x['machine_id'] == k]) for k in sorted({x['machine_id'] for x in rows})},
              'by_log_availability': {str(v): metrics([x for x in rows if x['detailed_log'] == v]) for v in (True, False)},
              'source_sha256': hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
              'rows': rows}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to replace an existing scoring artifact')
    result = evaluate(args.episodes, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'denominator', 'exact_correct', 'component_correct')}))


if __name__ == '__main__':
    main()
