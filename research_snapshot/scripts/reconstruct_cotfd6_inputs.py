"""Run reviewed, allowlisted offline definitions from the frozen CoT-FD6 notebook.

No notebook top-level cells, network functions, LLM calls or mock predictions execute.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFINITIONS = {
    6: ['extract_features', 'fcols', 'strat_cap'],
    8: ['load_fd001', 'build_nasa'],
    10: ['DS', 'prepare'],
    12: ['models'],
    14: ['Verifier', 'Memory'],
    16: ['zrow', 'prompt', 'actual_dir', 'grounding'],
    34: ['select_temporal_windows'],
}
CONFIG = dict(TEST_SIZE=0.25, SPLIT_SEED=42, LLM_MAX=100, NASA_WINDOW=50,
              NASA_STRIDE=10, RUL_THRESHOLD=30, PROTOTYPE_TAU=1.0,
              MEMORY_K=3, DIRECTION_EPS=0.25)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_offline_definitions(notebook):
    cells = json.loads(notebook.read_text())['cells']
    namespace = dict(globals(), **CONFIG)
    snippets, provenance = [], []
    for index, names in DEFINITIONS.items():
        source = ''.join(cells[index]['source'])
        definitions = {n.name: n for n in ast.parse(source).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            snippet = ast.get_source_segment(source, definitions[name])
            # get_source_segment omits a class's decorator; DS uses @dataclass.
            if name == 'DS':
                snippet = '@dataclass\n' + snippet
            snippets.append(snippet)
            provenance.append(dict(cell=index, definition=name,
                                   sha256=hashlib.sha256(snippet.encode()).hexdigest()))
    module = '\n\n'.join(snippets)
    exec(compile(module, str(notebook) + ':reviewed-offline-definitions', 'exec'), namespace)
    return namespace, module, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--published-results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ns, module, provenance = load_offline_definitions(args.notebook)
    (args.output / 'reviewed_offline_definitions.py').write_text(module + '\n')
    raw = ns['load_fd001'](args.data)
    assert raw['unit'].nunique() == 100 and len(raw) == 20631
    windows = ns['build_nasa'](raw)
    d = ns['prepare'](windows, 'nasa_fd001', ['healthy', 'degraded'])
    assert len(d.llm_test) == 100 and d.test['group_id'].nunique() == 25
    assert not set(d.train['group_id']) & set(d.test['group_id'])
    memory, verifier = ns['Memory'](d), ns['Verifier'](d)
    main = pd.read_csv(args.published_results / 'cotfd6_main_predictions.csv')
    main = main.loc[main['dataset'].eq('nasa_fd001')]
    published_memory = pd.read_csv(args.published_results / 'full100_memory_only_predictions.csv')
    published_memory = published_memory.loc[published_memory['dataset'].eq('nasa_fd001')].set_index('instance_id')
    published_temporal = pd.read_csv(args.published_results / 'nasa_temporal_diagnostic_stability.csv')
    temporal = ns['select_temporal_windows'](d)
    assert set(d.llm_test['instance_id']) == set(main['instance_id'])
    assert set(temporal['instance_id']) == set(published_temporal['instance_id'])
    inputs = args.output / 'agent_inputs'
    inputs.mkdir()
    audit, gold = [], []
    for i, row in d.llm_test.iterrows():
        case = f'case{i:03d}'
        z, feat = ns['zrow'](d, row)
        references = memory.retrieve(z)
        system, user = ns['prompt'](d, feat, references)
        assert 'rul_end' not in user and 'RUL_THRESHOLD' not in user
        assert '30' not in user.split('FEATURES:')[0]
        payload = dict(case_id=case, system=system, user=user)
        (inputs / (case + '.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        target = published_memory.loc[row.instance_id]
        supports = verifier.support(z)
        verifier_label = max(supports, key=supports.get)
        original_verifier = main.loc[main['instance_id'].eq(row.instance_id), 'verifier_prediction']
        assert original_verifier.nunique() == 1
        counts = pd.Series([a['label'] for a in references]).value_counts()
        majority = counts.index[0]
        expected_ids = json.loads(target.reference_ids)
        expected_similarities = json.loads(target.reference_similarities)
        meta = dict(case_id=case, instance_id=str(row.instance_id), group_id=int(row.group_id),
                    window_end=int(row.window_end), rul_end=int(row.rul_end), true_label=str(row.label),
                    reconstructed_memory_majority=majority,
                    published_memory_majority=str(target.memory_majority_prediction),
                    reference_ids_match=[a['instance_id'] for a in references] == expected_ids,
                    reference_labels_match=[a['label'] for a in references] == json.loads(target.reference_labels),
                    reference_similarity_max_error=float(np.max(np.abs(np.array([a['similarity'] for a in references]) - expected_similarities))),
                    reconstructed_verifier=verifier_label,
                    published_verifier=str(original_verifier.iloc[0]),
                    verifier_support=supports,
                    prompt_sha256=hashlib.sha256(user.encode()).hexdigest())
        audit.append(meta)
        gold.append({k:meta[k] for k in ['case_id','instance_id','group_id','rul_end','true_label']})
    frame = pd.DataFrame(audit)
    label_map = d.llm_test.set_index('instance_id')['label']
    assert main['true_label'].eq(main['instance_id'].map(label_map)).all()
    assert frame['reference_ids_match'].all() and frame['reference_labels_match'].all()
    assert frame['reference_similarity_max_error'].max() < 1e-9
    assert frame['reconstructed_memory_majority'].eq(frame['published_memory_majority']).all()
    assert frame['reconstructed_verifier'].eq(frame['published_verifier']).all()
    frame.to_csv(args.output / 'input_fidelity_audit.csv', index=False)
    (args.output / 'hidden_gold.json').write_text(json.dumps(gold, indent=2) + '\n')
    windows.to_csv(args.output / 'all_window_features_with_hidden_labels.csv', index=False)
    temporal.to_csv(args.output / 'temporal_features_with_hidden_labels.csv', index=False)
    training = args.output / 'training_evidence'
    training.mkdir()
    d.train[['instance_id', 'group_id'] + d.features + ['label']].to_csv(
        training / 'training_only_features_and_labels.csv', index=False)
    d.llm_test[['instance_id', 'group_id'] + d.features].to_csv(
        training / 'evaluation_features_without_labels.csv', index=False)
    baseline_results = []
    baseline_predictions = []
    for name, model in ns['models']().items():
        model.fit(d.train[d.features], d.train['label'])
        for scope, test in [('llm_like_for_like', d.llm_test), ('full_test', d.test)]:
            prediction = model.predict(test[d.features])
            baseline_results.append(dict(model=name, scope=scope, windows=len(test),
                                         accuracy=accuracy_score(test['label'], prediction),
                                         macro_f1=f1_score(test['label'], prediction, average='macro')))
            if scope == 'llm_like_for_like':
                probabilities = model.predict_proba(test[d.features])
                for i, (_, row) in enumerate(test.iterrows()):
                    baseline_predictions.append(dict(model=name, instance_id=str(row.instance_id),
                                                     prediction=str(prediction[i]),
                                                     healthy_probability=float(probabilities[i, list(model.classes_).index('healthy')]),
                                                     degraded_probability=float(probabilities[i, list(model.classes_).index('degraded')])) )
    pd.DataFrame(baseline_predictions).to_csv(args.output / 'task_model_predictions.csv', index=False)
    prompt_facts = dict(label_rule_defined_in_prompt=False,
                        rul_inference_prohibited=True,
                        training_references_explicitly_weak=True,
                        sensor_physical_units_defined=False,
                        feature_directions_relative_to_training_standardization=True)
    summary = dict(kind='official_offline_definition_input_reconstruction_not_llm_rerun',
                   notebook_sha256=sha(args.notebook), raw_data_sha256=sha(args.data),
                   reviewed_definitions=provenance, config=CONFIG,
                   train_devices=d.train['group_id'].nunique(), test_devices=d.test['group_id'].nunique(),
                   train_windows=len(d.train), test_windows=len(d.test), subset_windows=len(d.llm_test),
                   features=len(d.features), subset_class_counts=d.llm_test['label'].value_counts().to_dict(),
                   all_100_instance_ids_and_labels_match=True,
                   all_100_retrieval_ids_labels_and_predictions_match=True,
                   all_100_verifier_predictions_match=True,
                   max_reference_similarity_error=frame['reference_similarity_max_error'].max(),
                   constant_healthy_subset_accuracy=float(d.llm_test['label'].eq('healthy').mean()),
                   reconstructed_memory_accuracy=float(frame['reconstructed_memory_majority'].eq(frame['true_label']).mean()),
                   reconstructed_verifier_accuracy=float(frame['reconstructed_verifier'].eq(frame['true_label']).mean()),
                   conventional_baselines=baseline_results, prompt_facts=prompt_facts,
                   interpretation='Missing label semantics is a competing explanation; causal effect requires controlled model replay.',
                   environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                    scipy=scipy.__version__, sklearn=sklearn.__version__),
                   limitations=['original Python3.12 and exact patch versions not frozen',
                                'prompts reconstructed from released notebook, not captured original service requests',
                                'visible workspace separation is not OS-enforced here',
                                'no fresh model response or mechanism intervention yet'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:summary[k] for k in ['subset_windows','features','all_100_retrieval_ids_labels_and_predictions_match',
                                           'reconstructed_memory_accuracy','reconstructed_verifier_accuracy',
                                           'conventional_baselines','prompt_facts']}, indent=2))


if __name__ == '__main__':
    main()
