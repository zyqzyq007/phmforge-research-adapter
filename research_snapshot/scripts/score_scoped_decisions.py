"""Score frozen scoped maintenance actions; rationale remains a separate audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def score(path, reference):
    expected = reference['actions']
    if not path.is_file():
        return {'status': 'missing_decisions', 'expected_items': len(expected)}
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [r.get('item_id') for r in rows]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(expected):
        return {'status': 'invalid_inventory', 'actual_ids': identifiers,
                'expected_ids': list(expected)}
    allowed = {'close_scoped_item', 'retain_for_completion',
               'recommend_release', 'withhold_release'}
    if any(r.get('action') not in allowed for r in rows):
        return {'status': 'invalid_action', 'rows': rows}
    differences = [{'item_id': r['item_id'], 'actual': r['action'],
                    'expected': expected[r['item_id']]} for r in rows
                   if r['action'] != expected[r['item_id']]]
    actual = {r['item_id']: r['action'] for r in rows}
    positives = [k for k, v in expected.items() if v == 'close_scoped_item']
    negative_items = [k for k, v in expected.items() if v == 'retain_for_completion']
    incomplete = [r['item_id'] for r in rows if any(not r.get(k, '').strip()
                  for k in ['evidence_ids', 'reason', 'next_step', 'scope'])]
    return {'status': 'scored', 'items': len(rows),
            'correct_actions': len(rows) - len(differences), 'differences': differences,
            'positive_controls_correct': sum(actual[k] == expected[k] for k in positives),
            'positive_controls_total': len(positives),
            'negative_items_correct': sum(actual[k] == expected[k] for k in negative_items),
            'negative_items_total': len(negative_items),
            'unsupported_item_closures': [k for k in negative_items
                                         if actual[k] == 'close_scoped_item'],
            'unsupported_release': actual['RELEASE_REVIEWED_SCOPE'] == 'recommend_release',
            'excessive_retention': [k for k in positives if actual[k] != expected[k]],
            'incomplete_rationale_fields': incomplete,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rationale_validation': 'requires source-grounded human/analyst audit; action match alone insufficient'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.decisions, json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ['status', 'correct_actions',
                     'unsupported_item_closures', 'unsupported_release', 'excessive_retention']}))


if __name__ == '__main__':
    main()
