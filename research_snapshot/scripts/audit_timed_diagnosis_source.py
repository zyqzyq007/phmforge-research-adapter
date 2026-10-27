"""Recalculate a pinned public diagnosis supplement; makes no model calls.

An empty output directory is required. Author grades are not independent gold.
RDF lexical values are preserved because the source has invalid boolean lexemes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def column_number(value):
    n = 0
    for c in value:
        n = n * 26 + ord(c) - 64
    return n


def workbook(path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    sheets = {}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        for name in z.namelist():
            if not re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name):
                continue
            cells = {}
            for c in ET.fromstring(z.read(name)).findall('.//m:c', ns):
                v, f = c.find('m:v', ns), c.find('m:f', ns)
                value = v.text if v is not None else ''
                if c.get('t') == 's' and value:
                    value = strings[int(value)]
                cells[c.get('r')] = {'value': value, 'formula': f.text if f is not None else None}
            sheets[name] = cells
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    verified = []
    configs = [('sources/timed_pinned/tree.json', 'sources/timed_pinned/repository'),
               ('sources/benchmark_main_tree.json', 'sources/benchmark_main_repository'),
               ('sources/benchmark_v1_0_tree.json', 'sources/benchmark_v1_0_repository')]
    for tree_path, repo_path in configs:
        tree = json.loads((run / tree_path).read_text())
        assert not tree.get('truncated', False)
        blobs = [r for r in tree['tree'] if r['type'] == 'blob']
        for r in blobs:
            p = run / repo_path / r['path']
            body = p.read_bytes()
            assert len(body) == r['size']
            assert hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest() == r['sha']
        verified.append({'tree': tree_path, 'tree_sha': tree['sha'], 'blobs': len(blobs),
                         'bytes': sum(r['size'] for r in blobs), 'git_blob_verification': 'passed'})
    save(out / 'source_integrity.json', verified)
    repo = run / 'sources/timed_pinned/repository'
    xlsx = repo / 'SyndromeDiagnosis/DiagnosisResults.xlsx'
    sheets = workbook(xlsx)
    coded = sheets['xl/worksheets/sheet1.xml']
    readable = sheets['xl/worksheets/sheet4.xml']
    headers = {re.sub(r'\d+$', '', c): int(r['value'][1:]) for c, r in coded.items()
               if c.endswith('2') and re.fullmatch(r'S\d+', r['value'] or '')}
    rows, locations = [], {}
    for c, record in coded.items():
        if not re.fullmatch(r'B\d+', c) or not re.fullmatch(r'ds\d+', record['value'] or ''):
            continue
        row = re.search(r'\d+', c)[0]
        run_match = re.fullmatch(r'Run ([123])', coded.get('A' + row, {}).get('value') or '')
        if not run_match:
            continue
        for col, syndrome in headers.items():
            cell = col + row
            value = coded.get(cell, {}).get('value')
            if value in {'5', '2', '1', 'excluded'}:
                grade = {'5': 'C', '2': 'H', '1': 'F', 'excluded': 'excluded'}[value]
                assert {'I': 'F'}.get(readable[cell]['value'], readable[cell]['value']) == grade
                entry = dict(run=int(run_match[1]), dataset=record['value'], syndrome=syndrome, code=grade)
                rows.append(entry)
                locations[cell] = entry
    propagating = set()
    formulas = []
    for cell, record in sheets['xl/worksheets/sheet2.xml'].items():
        f = record['formula'] or ''
        if not cell.startswith('D') or not f.startswith('COUNTIF') or not re.search(r',5\)', f):
            continue
        for start_col, start_row, end_col, end_row in re.findall(r'COUNTIF\(Coded!([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?,5\)', f):
            end_col, end_row = end_col or start_col, end_row or start_row
            assert start_row == end_row
            for source_cell, entry in locations.items():
                if re.search(r'\d+', source_cell)[0] == start_row and column_number(start_col) <= column_number(re.sub(r'\d+$', '', source_cell)) <= column_number(end_col):
                    propagating.add((entry['dataset'], entry['syndrome']))
                    formulas.append(dict(source_formula_cell=cell, source_formula=f, source_cell=source_cell, **entry))
    scored = [r for r in rows if r['code'] != 'excluded']
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['syndrome'])].append(r['code'])
    assert all(len(v) == 3 for v in groups.values())
    subgroup = {}
    for name, p in [('propagating', True), ('localized', False)]:
        rr = [r for r in scored if ((r['dataset'], r['syndrome']) in propagating) == p]
        counts = Counter(r['code'] for r in rr)
        subgroup[name] = dict(denominator=len(rr), counts=dict(counts),
            top1_fraction=counts['C'] / len(rr), candidate_inclusion_fraction=(counts['C'] + counts['H']) / len(rr))
    included = {k: v for k, v in groups.items() if v[0] != 'excluded'}
    score_summary = dict(scope='author grade categories recalculated; no independent regrading or model rerun',
        xlsx_sha256=sha256(xlsx), all=dict(Counter(r['code'] for r in scored)),
        scored_repeats=len(scored), unique_scored_syndromes=len(included), excluded_syndromes=[list(k) for k, v in groups.items() if v[0] == 'excluded'],
        subgroups=subgroup, propagating_unique=len(propagating), unanimous=sum(len(set(v)) == 1 for v in included.values()),
        all_three_incorrect=[list(k) for k, v in included.items() if v == ['F'] * 3],
        majority_grade_candidate_inclusion=sum(sum(g != 'F' for g in v) >= 2 for v in included.values()))
    assert score_summary['all'] == {'C': 157, 'H': 60, 'F': 20}
    assert len(scored) == 237 and len(included) == 79 and len(propagating) == 27
    previous = json.loads((run / 'source_audit/xlsx_individual_scores.json').read_text())
    assert sorted(rows, key=lambda r: (r['run'], r['dataset'], r['syndrome'])) == sorted(previous, key=lambda r: (r['run'], r['dataset'], r['syndrome']))
    save(out / 'author_score_recalculation.json', score_summary)
    save(out / 'individual_grades.json', rows)
    save(out / 'propagating_formula_membership.json', formulas)
    sys.path.insert(0, str(run / 'dependency_site'))
    import rdflib
    from rdflib import Graph, Literal, RDF, URIRef, XSD
    rdflib.NORMALIZE_LITERALS = False
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        graph = Graph().parse(repo / 'KnowledgeBase/HaiCPPS_Complete_KnowledgeGraph.ttl', format='turtle')
        normalization_control = [{ 'input': x, 'normalized_lexical': str(Literal(x, datatype=XSD.boolean, normalize=True)),
                                   'preserved_lexical': str(Literal(x, datatype=XSD.boolean, normalize=False)),
                                   'valid_xsd_boolean_lexeme': x in {'true', 'false', '0', '1'}} for x in ['0.0', '1.0', '0', '1']]
    syndrome_type = URIRef('https://w3id.org/hsu-aut/ISO17359#Syndrom')
    symptom_type = URIRef('https://w3id.org/hsu-aut/ISO17359#Symptom')
    symptoms = set(graph.subjects(RDF.type, symptom_type))
    syndrome_rows, ownership = {}, Counter()
    for node in graph.subjects(RDF.type, syndrome_type):
        match = re.fullmatch(r'.*#Syndrome_(ds\d+)_(\d+)', str(node))
        assert match
        members = {o for _, p, o in graph.triples((node, None, None)) if str(p).endswith('#hasPart') and o in symptoms}
        ownership.update(members)
        key = (match[1], int(match[2]))
        syndrome_rows[str(node)] = dict(dataset=key[0], syndrome=key[1], symptoms=len(members),
            scope='scored' if key in included else ('excluded' if key in groups else 'extra_not_in_workbook'))
    counts = {scope: dict(syndromes=sum(r['scope'] == scope for r in syndrome_rows.values()),
                         symptoms=sum(r['symptoms'] for r in syndrome_rows.values() if r['scope'] == scope))
              for scope in ['scored', 'excluded', 'extra_not_in_workbook']}
    lexemes = Counter(str(o) for _, _, o in graph if isinstance(o, Literal) and o.datatype == XSD.boolean)
    assert len(graph) == 300301 and len(syndrome_rows) == 90 and len(symptoms) == 10329
    assert counts['scored'] == {'syndromes': 79, 'symptoms': 9247}
    assert counts['excluded'] == {'syndromes': 5, 'symptoms': 14}
    assert counts['extra_not_in_workbook'] == {'syndromes': 6, 'symptoms': 1068}
    assert len(ownership) == len(symptoms) and set(ownership.values()) == {1}
    save(out / 'graph_scope_and_adapter.json', dict(triples=len(graph), symptoms=len(symptoms), rdflib_version=rdflib.__version__,
        NORMALIZE_LITERALS=False, scope_counts=counts, boolean_lexemes=dict(lexemes),
        normalization_control=normalization_control, warning_counts=dict(Counter(str(w.message) for w in captured)), syndromes=syndrome_rows,
        limitation='Invalid source boolean lexical forms and this adapter normalization are not evidence of original GraphDB or Agent failure.'))
    paths = [p for p in (repo / 'SyndromeDiagnosis').rglob('*.txt') if re.search(r'HaiCPPSDiagnosisResults_ds\d+_S\d+\.txt$', p.name)]
    format_groups = Counter((int(re.search(r'run([123])', str(p))[1]), p.name) for p in paths)
    save(out / 'report_scope.json', dict(txt_files=len(paths), unique_run_filename_pairs=len(format_groups),
         pairs_with_two_formats=sum(v == 2 for v in format_groups.values()),
         limitation='Flat reports and nested dialogue records are two formats, not distinct native model executions. Extra/excluded cases are outside the 237 graded repeats.'))
    snippets = []
    for p in sorted((run / 'sources/model_history_content').glob('*.mo')) + [run / 'sources/benchmark_v1_0_repository/models/Mixer.mo', run / 'sources/benchmark_v1_0_repository/models/Bottling.mo', run / 'sources/benchmark_main_repository/models/Mixer.mo', run / 'sources/benchmark_main_repository/models/Bottling.mo']:
        lines = [dict(line=n, text=l.strip()) for n, l in enumerate(p.read_text().splitlines(), 1)
                 if re.search(r'condition_is_full_tank_B20[123].*waitTime|condition_is_full_tank_B20[123]\.condition|valve_out\.opening.*mod\(time', l)]
        snippets.append(dict(path=str(p.relative_to(run)), sha256=sha256(p), evidence_lines=lines))
    save(out / 'model_history_control_evidence.json', dict(files=snippets,
         interpretation='Retrieved versions use level-triggered mixer filling and periodic bottling outlet commands; exact original diagnosis-data model version is not bound.'))
    save(out / 'completed.json', dict(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
         controller='independent source reanalysis', native_LLM_calls=0, source_blobs_verified=sum(v['blobs'] for v in verified),
         script_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(status='completed', source_blobs_verified=sum(v['blobs'] for v in verified),
                         scored_repeats=len(scored), graph_triples=len(graph), native_LLM_calls=0)))


if __name__ == '__main__':
    main()
