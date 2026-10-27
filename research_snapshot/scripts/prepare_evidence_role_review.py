"""Native, same-information review of four historical HUST program changes.

Development qualification only. Historical outcomes never enter actor mounts.
The ordinary and structured conditions share all actual evidence and tools.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import urllib.request

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from run_frozen_candidate_review import source_context

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261022T023131Z_validation274')
RECENT = Path('/data/yyf-zyq/PHM_20261010/20261025T175700Z_shadow388')
HARNESS = Path('/data/yyf-zyq/PHM_20261010/20261024T202500Z_decision340')
CLI = '/data/yyf-zyq/PHM_20261010/20261023T052522Z_rotor305/runtime/codex0164/bin/codex'

COMMON = '''Review four changes to genuinely trained HUST bearing-diagnosis programs. This is the bounded review/selection stage inside a native coding Agent, not unrestricted predictor development. Read /data/contract.json and use the phmforge evidence tool for every case. Each case has incumbent before and proposed after. They share the same measured training/development records, so four changes are not four independent datasets. Select exactly one of before/after for each case, minimizing future whole-window classification error while retaining useful improvements and avoiding harmful changes. The sensor data are real; all supplied model outputs were actually computed earlier with saved trained pipelines. This episode reuses them, with no new model fitting or prediction.

You can inspect immutable program and probe source under /data/rows/source, recompute metrics from actual public predictions, examine conflicting views, and write Python analyses. Read /data/rows/CASE.json for complete predictions, evidence identities and limitations. The same tools, rows and sources are available in both conditions. Whole-record correctness and per-class recall are informative secondary views. Repeated windows and loads do not establish independent devices. The evaluation domain is unknown, and no observed validation view is guaranteed to represent it. Do not assume a deterministic check certifies transfer.

Evidence objects differ: some refer to the full pipeline under its actual routing; some isolate a correction classifier; some use training resubstitution; some use saved group-held fits. Missing full-pipeline CV is missing, not a zero error or an automatic failure. An inactive correction can have no measured effect in a CV partition; this is not proof it is useful or useless elsewhere. A component's prediction outside its natural gate is a diagnostic, not deployment performance. Some candidates and checks were reconstructed by the researcher from recorded recipes after the original run. They are lawful shared inputs NOW, not proof the original Agent had seen those checks. Keep those distinctions when reasoning.

Final observations, labels, conditions, population counts, outcomes, historical conclusions, and other Agent sessions are unavailable. No new training, new predictors, synthetic labels, internet, additional model APIs, or spawned agents. Only the four registered binary decisions are evaluated. Each branch is a complete executable intervention defined in the evidence; no forced update or empirical acceptance threshold is imposed. You may accept or retain any case and explain uncertainty. Both choices are valid submissions.

Call phmforge select once with four rows {case,incumbent:'before',selected:'before' or 'after',reason}, then save selection.json and report.md. Evidence file fallback uses python3 /tools/program_update_tools.py client --request followed by JSON. Scratch belongs under /app/.runtime, never /tmp. Total budget per session:600 seconds,20 model requests,800000 input tokens,65536 output tokens,16384 output tokens/request,2 numerical threads. Complete the submission within that budget. External scoring runs only after ALL sessions seal; no scores are returned during review.
'''

ORDINARY = '''Independently review the four changes. Use the common evidence and source as useful, recompute relevant comparisons, consider alternative explanations and uncertainty, and decide whether to retain or replace. You may organize your analysis and use the full budget in any reasonable way. Before submitting, check that your reasons accurately describe the evidence, that all four cases are covered, and that useful changes as well as harmful ones have been considered. Your report should give actual evidence references, supporting and contrary observations, and limitations. A well-formed report alone does not establish correct choices.'''

STRUCTURED = '''Use an explicit evidence-role review workflow. First identify each actual changed computation, its routing/fit conditions and intended effect. Next write evidence_ledger.json: for EACH case record the exact before/after objects, available evidence IDs, tested population, whether the evidence evaluates the complete changed path or only a component, actual corrected/broken/unchanged counts where derivable, and what transfer claim remains unsupported. Verify ledger claims against numeric rows or source; mark absent checks unknown. Then write two plausible explanations of the observed validation difference and the observation that distinguishes them, if any; do not invent an explanation or confidence when none is identifiable. Finally decide using that ledger, explicitly discussing the strongest contrary evidence and the opportunity cost of rejecting a useful change. No particular metric, voting rule, retention preference or acceptance threshold is prescribed. This workflow may accept or reject a change; uncertainty alone is not an instruction to always retain. Submit the same four choices and report as the ordinary reviewer.'''


def prepare(root):
    import numpy as np
    assert not root.exists(), 'Unique run; never overwrite a frozen or failed attempt'
    for name in ['runtime/tmp','runtime/mpl','audit','code','scripts','shared_tools','public/rows/source','evaluation_inputs']:
        (root/name).mkdir(parents=True, exist_ok=True)
    indices = {}
    receipts = {}
    for label, directory in [('historical', SOURCE), ('recent', RECENT), ('harness', HARNESS)]:
        indices[label], receipts[label] = source_context(directory)
    bindings = []

    def bound(label, relative):
        base = {'historical': SOURCE, 'recent': RECENT, 'harness': HARNESS}[label]
        p = base/relative
        assert sha(p) == indices[label][relative]['sha256'], relative
        bindings.append(dict(source=label, path=relative, sha256=sha(p)))
        return p

    def read(label, relative):
        return json.loads(bound(label, relative).read_text())

    url = 'https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'PHM-research-provenance'}), timeout=30) as response:
        release = json.load(response)
    assert release['tag_name'] == 'rust-v0.164.0'
    assert sha(Path(CLI)) == 'b93f6811d3f653f0fb83be4050d857fb9abbb111dfa2ce160563327daff7f306'
    save(root/'audit/CLI_SOURCE389.json', dict(url=url, checked_utc=utc(), tag=release['tag_name'], published_at=release['published_at']))
    for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:
        shutil.copy2(bound('harness', name), root/name)
    policy = json.loads((root/'retention_policy.json').read_text())
    policy['shared_dependencies']['usage'] = 'Native four-change review; reuse real saved prediction evidence, zero new PHM fits or forwards'
    save(root/'retention_policy.json', policy)

    recent = read('recent', 'EVIDENCE388.json')
    val_path = bound('historical', 'b/public_data/s274_hust_geometry_r1_initial/datasets/HUST/val.npz')
    with np.load(val_path, allow_pickle=False) as z:
        labels = z['labels'].astype(int).tolist()
        ids = z['sample_ids'].astype(str).tolist()
        family = z['bearing_family'].astype(str).tolist()
        scale = z['native_record_scale'].astype(float).tolist()
        category = z['groups'].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 4200
    keys = [(category[i], family[i], scale[i]) for i in range(len(ids))]
    grouped = {}
    for i, key in enumerate(keys):
        grouped.setdefault(key, []).append(i)
    assert len(grouped) == 21

    def paired_window_outputs(before, after):
        maps = [{r['sample_id']: r['predicted_label'] for r in read('historical', p)['predictions']} for p in [before, after]]
        assert all(set(m) == set(ids) for m in maps)
        rows = []
        for n, (key, index) in enumerate(grouped.items()):
            truth = {labels[i] for i in index}
            assert len(truth) == 1
            row = dict(record='development_'+str(n), family=key[1], category_group=key[0], native_scale=key[2], label=truth.pop(), windows=len(index))
            for name, pred in zip(['before','after'], maps):
                counts = Counter(pred[ids[i]] for i in index)
                row[name+'_window_counts'] = dict(counts)
                row[name] = min(counts, key=lambda label:(-counts[label], label))
            rows.append(row)
        return rows

    prefixes = {k:'audit/early_closed_review/'+v for k,v in dict(a='a_review_component_probe_v2', b='b_component_probe_v2', c='c_component_probe', d='d_choice_probe').items()}
    descriptions = {
        'a':'Add two conditional class-correction rules around the existing SVC; auxiliary record models and gates are part of after. Before disables both correction gates, leaving the same primary predictor.',
        'b':'Add class-5 promotion only when the component-based predictor returns class4. Before and after share all feature computations and fitted component models.',
        'c':'Add the compound4/5 LDA correction to base plus healthy/IR correction. Natural compound eligibility follows the actual base prediction; it is not all true4/5 records.',
        'd':'Change the earlier recorded all-family LDA recipe to the selected normalization and42-record fit. Both fitting membership and normalization change; not a pure data-filter contrast.'}
    cases = []
    evidence_paths = {}
    for actor in 'abcd':
        case_id = 'change_'+actor
        probe = read('historical', prefixes[actor]+'/output/PROBE.json')
        if actor == 'a':
            dev = paired_window_outputs(prefixes[actor]+'/output/components/base.json', prefixes[actor]+'/output/components/full.json')
            cv = []
            provenance = 'Saved-weight component intervention of an actual ordinary-review program. Primary predictor parameters match original to floating-point precision; full correction CV unavailable.'
        elif actor == 'd':
            dev = paired_window_outputs(prefixes[actor]+'/output/val_prior_family57.json', prefixes[actor]+'/output/val_chosen_loadfamily42.json')
            cv = []
            provenance = 'Before is a real train-only reconstruction of an earlier recorded recipe; after is the actual selected checkpoint. Exact paired complete-recipe CV unavailable.'
        else:
            dev, cv = [], []
            for i, row in enumerate(recent[actor]['rows']):
                v = dict(record='record_'+str(i), fold=row['fold'], family=row['family'], category_group=row['record_id'], native_scale=row['scale'], label=row['label'], windows=row['windows'], before=row['before_target_component'], after=row['complete'], natural_gate=row['natural_gate'], component_prediction_outside_gate_allowed=row['shadow_pair_prediction'])
                v['before_window_counts'] = {v['before']:v['windows']}
                v['after_window_counts'] = {v['after']:v['windows']}
                (dev if row['fold']=='val' else cv).append(v)
            provenance = 'Real saved full and fold pipelines, replayed with all fitting disabled. This paired complete-program CV and gate audit were researcher checks, not all observed by the original Actor.'
        partitions = [('development', dev)] + ([('group_held_CV', cv)] if cv else [])
        summaries = []
        for name, rows in partitions:
            values = {}
            for candidate in ['before','after']:
                wrong = sum(sum(int(n) for y,n in r[candidate+'_window_counts'].items() if int(y)!=r['label']) for r in rows)
                values[candidate] = dict(window_error=wrong/sum(r['windows'] for r in rows), record_error=sum(r[candidate]!=r['label'] for r in rows)/len(rows))
            summaries.append(dict(id=case_id+'/'+name, kind=name, records=len(rows), windows=sum(r['windows'] for r in rows), scores=values,
                                  record_fixed=sum(r['before']!=r['label'] and r['after']==r['label'] for r in rows),record_broken=sum(r['before']==r['label'] and r['after']!=r['label'] for r in rows),complete_pipeline_routing=True))
        case = dict(id=case_id,candidate_ids=['before','after'],incumbent_ids=['before'],original_incumbent='before',description=descriptions[actor],provenance=provenance,
                    evidence=summaries,group_losses=[[v['scores'][c]['window_error'] for v in summaries] for c in ['before','after']],groups=[v['kind'] for v in summaries],development_columns=[0],missing_checks=['complete paired group-held CV'] if not cv else [],
                    source='one reused measured HUST bearing cohort, four correlated changes; no independent-device confirmation')
        details = dict(case=case,development_records=dev,CV_records=cv,original_probe=probe,source_directory='/data/rows/source/'+case_id,
                       original_actor_knowledge_not_equivalent_to_current_shared_inputs=True)
        save(root/'public/rows'/(case_id+'.json'), details)
        sources = []
        if actor == 'a':
            sources = [prefixes[actor]+'/app/'+name for name in ['solution.py','registered_a_review_components274.py','train_hybrid.py','feature_extractor.py']]
        else:
            workspace = next((SOURCE/actor/'workspaces').iterdir())
            sources = [str(p.relative_to(SOURCE)) for p in sorted(workspace.glob('*.py'))]
            if actor == 'd':
                sources += [str(p.relative_to(SOURCE)) for p in (SOURCE/prefixes[actor]/'app').glob('*.py') if p.name.startswith('probe_')]
        for relative in sources:
            p = bound('historical', relative)
            destination = root/'public/rows/source'/case_id/p.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
        evidence_paths[case_id] = sources
        cases.append(case)
    save(root/'public/cases.json', cases)
    save(root/'public/contract.json', dict(task='Four related PHM program-change decisions, before versus after',labels={'0':'healthy','1':'IR','2':'OR','3':'BALL','4':'IR+OR','5':'IR+BALL','6':'OR+BALL'},
        allowed='Immutable public validation records, metadata, actual source and checked evidence. Same materials in both conditions.',
        forbidden='Final input, labels, scores, target conditions, new predictors or synthetic data; no fitting or network.',
        limitation='78 repeated measurement records across26category groups, not78independent devices. Four changes share data. Historical outcomes known to researcher; this is development only.',
        status='All evidence consists of real saved-pipeline predictions; no online PHM model training or forward is claimed.',
        reference_rule_status='No rule is guaranteed correct. Missing CV is explicitly unknown. Current evidence is richer than some original Actor contexts.'))
    fixed = {name:[] for name in ['keep','development_best','available_view_mean','CV_nondegrade_dev_improve','strict_complete_CV_gate']}
    for c in cases:
        old,new=c['group_losses'];dv=new[0]<old[0];cv_known=len(old)>1;cv_ok=cv_known and all(b<=a+1e-12 for a,b in zip(old[1:],new[1:]))
        picks=dict(keep=False,development_best=dv,available_view_mean=sum(new)<sum(old)-1e-12,
                   CV_nondegrade_dev_improve=dv and (cv_ok if cv_known else True),strict_complete_CV_gate=dv and cv_ok)
        for name,accept in picks.items():fixed[name].append(dict(case=c['id'],incumbent='before',selected='after' if accept else 'before'))
    save(root/'FIXED_CHOICES.json',dict(utc=utc(),choices=fixed,final_scores_joined=False,tie='retain before',missing_CV='CV_nondegrade falls back to development; strict_complete requires actual CV. Both retained.'))
    config=dict(iteration='A389',result_key='evidence_review389',stage='historical_native_evidence_role_review',model='deepseek-flash',cli=CLI,cli_sha256=sha(Path(CLI)),
                selector_conditions=['ordinary','structured'],repeats=2,maximum_parallel_agents=4,episode_seconds=600,episode_requests=20,episode_input_tokens=800000,episode_output_tokens=65536,output_per_request=16384,
                preparation_file='PREPARATION389.json',qualify_submission_chain=True,reset_qualification_transport=True,
                researcher_previously_knows_final_scores=True,new_fits=0,new_PHM_predictions=0,source_archives={k:r['archive_path'] for k,r in receipts.items()},
                source_manifests={k:r['manifest_sha256'] for k,r in receipts.items()},maximum_disk_GiB=2,numerical_threads=2,
                hypothesis='Explicitly binding evidence to actual changed computations and natural routing may improve binary review decisions beyond ordinary use of the same evidence.',
                falsification='No better selection loss and gain/harm balance than ordinary and the best frozen fixed control: stop this prompting/ledger mechanism. No favorable missing-case exclusions.',
                metrics='Window-error selection regret within each before/after pool, harmful relative change, useful improvement capture. Four correlated changes averaged within ONE source; no independent confirmation.',
                primary_gate='All4native sessions submit; structured mean regret strictly lower than ordinary AND best frozen fixed control, harm<=ordinary, gain>=ordinary. Exploratory gate only, not full goal.',
                confirmation=False,novelty_confirmed=False,final22_opened=False,A33872_opened=False)
    save(root/'CAMPAIGN_CONFIG.json',config)
    save(root/'PROMPT_FREEZE.json',dict(ordinary=COMMON+'\n'+ORDINARY,structured=COMMON+'\n'+STRUCTURED))
    names=closure(['screen_program_updates','program_update_tools','run_validation_transfer','qualify_native_cli_mcp','prepare_evidence_role_review'])
    for name in names:shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in ['program_update_tools','controlled_feedback','prepare_fault_extensions','acquire_forge_pdmb_sources244','isolated_model_gateway']:
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'shared_tools'/(name+'.py'))
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    save(root/'CODE_FREEZE.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    save(root/'PREPARATION389.json',dict(passed=True,utc=utc(),source_bindings=bindings,source_paths=evidence_paths,actual_unique_public_measurements=78,correlated_program_changes=4,
        new_fits=0,new_PHM_predictions=0,new_LLM_calls=0,source_contrast='Actual saved-weight comparisons, not a claim all before artifacts were retained deployable historical submissions.',
        public_files=[dict(path=str(p.relative_to(root/'public')),sha256=sha(p)) for p in sorted((root/'public').rglob('*')) if p.is_file()]))
    save(root/'OWNERSHIP.json',dict(user_request='继续围绕程序选择与复核迭代，要求加速并保留反证',run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name))
    print(json.dumps(dict(prepared=True,cases=4,planned_native_sessions=4,maximum_parallel=4,new_fits=0,new_PHM_predictions=0,source_bindings=len(bindings))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    prepare(parser.parse_args().run.resolve())
