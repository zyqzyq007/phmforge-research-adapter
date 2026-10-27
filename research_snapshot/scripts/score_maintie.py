"""Original MaintIE relation scorer plus independent schema/entity checks."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path


def labels(nodes):
    out = set()
    for node in nodes:
        out.add(node['fullname'])
        out.update(labels(node.get('children', [])))
    return out


def counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def evaluate(gold, pred, scheme, scorer):
    if not isinstance(pred, list) or len(gold) != len(pred):
        raise ValueError('Complete original ordered record inventory required')
    entity_labels = labels(scheme['entity'])
    # The published task/scorer has six relations. The ontology's generic
    # hasParticipant parent is not an additional scored task relation.
    relation_labels = labels(scheme['relation']) - {'hasParticipant'}
    aliases = {x.split('/')[-1] for x in relation_labels}
    issues, typed, spans, converted_gold, converted_pred = [], [0, 0, 0], [0, 0, 0], [], []
    for index, (g, p) in enumerate(zip(gold, pred)):
        if p.get('tokens') != g['tokens']:
            raise ValueError(f'Original token inventory changed at record {index}')
        for e in p['entities']:
            if (type(e['start']) is not int or type(e['end']) is not int
                    or not 0 <= e['start'] < e['end'] <= len(g['tokens'])):
                raise ValueError(f'Invalid span at record {index}')
            if e['type'] not in entity_labels:
                issues.append(dict(record=index, kind='unknown_entity_type', value=e['type']))
        for key, total in [('typed', typed), ('spans', spans)]:
            def entities(row):
                return {(e['start'], e['end'], e['type']) if key == 'typed'
                        else (e['start'], e['end']) for e in row['entities']}
            gs, ps = entities(g), entities(p)
            for n, value in enumerate([len(gs & ps), len(ps-gs), len(gs-ps)]):
                total[n] += value
        for row, target, is_prediction in [(g, converted_gold, False), (p, converted_pred, True)]:
            sent = []
            for rel in row['relations']:
                if not all(type(rel[k]) is int and 0 <= rel[k] < len(row['entities']) for k in ['head', 'tail']):
                    raise ValueError(f'Invalid relation index at record {index}')
                if rel['type'] not in relation_labels | aliases:
                    raise ValueError(f'Unknown relation label at record {index}')
                h, t = row['entities'][rel['head']], row['entities'][rel['tail']]
                sent.append(dict(head=(h['start'], h['end']), tail=(t['start'], t['end']),
                    head_type=h['type'], tail_type=t['type'], type=rel['type'].split('/')[-1]))
            target.append(sent)
    spec = importlib.util.spec_from_file_location('original_maintie_scorer', scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relation = {}
    for mode in ['strict', 'boundaries']:
        with contextlib.redirect_stdout(io.StringIO()):
            relation[mode] = module.re_score(converted_pred, converted_gold, sorted(aliases), mode)[0]
    return dict(records=len(gold), schema_issues=issues, schema_legal=not issues,
                original_relation_scores=relation,
                auxiliary_entity_typed_micro=counts(*typed), auxiliary_entity_span_micro=counts(*spans),
                scope='Original REBEL relation scoring; independent exact-set entity auxiliary scores. Unknown entity labels retained as errors, not repaired or dropped.')


def main():
    ap = argparse.ArgumentParser()
    for name in ['gold', 'pred', 'scheme', 'scorer', 'out']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    result = evaluate(json.loads(a.gold.read_text()), json.loads(a.pred.read_text()),
                      json.loads(a.scheme.read_text()), a.scorer)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['records', 'schema_legal', 'auxiliary_entity_typed_micro']}))


if __name__ == '__main__':
    main()
