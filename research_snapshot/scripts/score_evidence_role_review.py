"""Join sealed native choices to already-known historical effects, once."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from controlled_feedback import save, sha, utc
from run_frozen_candidate_review import source_context

SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261022T023131Z_validation274')
MAPPING = dict(change_a='a_review_full_vs_base',change_b='b_full_vs_base',
               change_c='c_compound_added',change_d='d_all_family_prior_vs_chosen')
EXPECTED_ORIENTATION = dict(change_a=('base','full'), change_b=('base','full'),
                            change_c=('healthy_ir_only','full'),
                            change_d=('prior_family57','chosen_loadfamily42'))


def evaluate(root):
    assert (root/'ALL_CHOICES_SEALED.json').exists()
    assert not (root/'RESULT389.json').exists(), 'Do not re-evaluate a completed experiment'
    seal = json.loads((root/'ALL_CHOICES_SEALED.json').read_text())
    for item in seal['files']:
        assert sha(root/item['path']) == item['sha256'], item['path']
    index, receipt = source_context(SOURCE)
    config = json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert receipt['manifest_sha256'] == config['source_manifests']['historical']
    used = {}

    def read(relative):
        p = SOURCE/relative
        assert sha(p) == index[relative]['sha256']
        used[relative] = sha(p)
        return json.loads(p.read_text())

    contrasts = {r['name']:r for r in read('audit/REGISTERED_RECORD_CONTRASTS274.json')['contrasts']}
    predictions = read('b/evaluation/s274_hust_geometry_r1_initial/output/predictions.json')['predictions']
    total = len(predictions)
    assert total == len({r['sample_id'] for r in predictions}) == 3865
    delta = {}
    oriented_effects = {}
    for case,name in MAPPING.items():
        row = contrasts[name]
        assert row['fixed'] == sum(r['fixed'] for r in row['changed_records'])
        assert row['broken'] == sum(r['broken'] for r in row['changed_records'])
        direction = (row['before'], row['after'])
        expected = EXPECTED_ORIENTATION[case]
        assert direction in [expected, expected[::-1]], (case, direction, expected)
        fixed, broken = (row['fixed'], row['broken']) if direction == expected else (row['broken'], row['fixed'])
        delta[case] = Fraction(broken-fixed, total)
        oriented_effects[case] = dict(before=expected[0],after=expected[1],fixed=fixed,broken=broken,
                                      source_contrast_direction=direction,source_direction_reversed=direction!=expected)

    def score(choices):
        assert {r['case'] for r in choices} == set(MAPPING) and len(choices)==4
        rows = []
        for row in choices:
            assert row['incumbent']=='before' and row['selected'] in ['before','after']
            change = delta[row['case']] if row['selected']=='after' else Fraction(0)
            best = min(Fraction(0),delta[row['case']])
            gain = max(Fraction(0),-change)
            opportunity = -best
            rows.append(dict(case=row['case'],selected=row['selected'],relative_window_error=float(change),
                             selection_loss=float(change-best),harm=float(max(Fraction(0),change)),
                             gain=float(gain),gain_capture=float(gain/opportunity) if opportunity else None))
        return dict(rows=rows,mean_relative_window_error=sum(r['relative_window_error'] for r in rows)/4,
                    mean_selection_loss=sum(r['selection_loss'] for r in rows)/4,
                    mean_harm=sum(r['harm'] for r in rows)/4,mean_gain=sum(r['gain'] for r in rows)/4,
                    mean_gain_capture=sum(r['gain_capture'] for r in rows if r['gain_capture'] is not None)/sum(r['gain_capture'] is not None for r in rows))

    fixed_input = json.loads((root/'FIXED_CHOICES.json').read_text())['choices']
    fixed = {name:score(rows) for name,rows in fixed_input.items()}
    episodes = []
    for episode in json.loads((root/'AGENT_CHOICES.json').read_text())['episodes']:
        delivered = episode.get('choice') is not None
        episodes.append(dict(id=episode['id'],condition=episode['condition'],repeat=episode['repeat'],
                             completed=episode['completed'],choice_delivered=delivered,
                             scores=score(episode['choice']['choices']) if delivered else None,
                             seconds=episode['seconds']))
    arms = {}
    keys = ['mean_relative_window_error','mean_selection_loss','mean_harm','mean_gain','mean_gain_capture']
    for arm in config['selector_conditions']:
        group = [e for e in episodes if e['condition']==arm]
        assert len(group)==config['repeats']
        full = all(e['completed'] and e['choice_delivered'] for e in group)
        arms[arm] = dict(all_completed=full,normal=sum(e['completed'] for e in group),planned=len(group),
                         aggregate={k:sum(e['scores'][k] for e in group)/len(group) for k in keys} if full else None)
    primary = False
    if all(a['all_completed'] for a in arms.values()):
        old,new = arms['ordinary']['aggregate'],arms['structured']['aggregate']
        primary = (new['mean_selection_loss'] < old['mean_selection_loss']-1e-12
                   and new['mean_selection_loss'] < min(v['mean_selection_loss'] for v in fixed.values())-1e-12
                   and new['mean_harm'] <= old['mean_harm']+1e-12
                   and new['mean_gain_capture'] >= old['mean_gain_capture']-1e-12)
    usage = []
    for path in sorted((root/'episodes').glob('*/model_gateway/*/status.json')):
        usage.append(dict(episode=path.parents[2].name,status=json.loads(path.read_text())))
    result = dict(iteration='A389',utc=utc(),episodes=episodes,arms=arms,fixed=fixed,usage=usage,
                  primary_qualification_passed=primary,source_bindings=used,final_windows=total,
                  historical_effects={case:dict(oriented_effects[case],relative_after_error=float(delta[case])) for case in MAPPING},
                  historical_outcomes_already_known_to_researcher=True,new_final_model_forwards=0,
                  new_PHM_fits=0,new_PHM_predictions=0,independent_confirmation_scenarios=0,novelty_confirmed=False,
                  majority_goal_achieved=False,final22_opened=False,A33872_opened=False,
                  interpretation='Historical native review-stage qualification;four correlated component/recipe changes in one source. Relative errors subtract each incumbent, not absolute accuracies.',
                  decision='Advance only to another independent qualification, not full goal completion' if primary else 'Stop this prompt/ledger mechanism; no retuning or selectively rerunning these cases')
    save(root/'RESULT389.json',result)
    output=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/results')/config['result_key']
    save(output/'RESULT389.json',result)
    print(json.dumps(dict(primary_qualification_passed=primary,arms=arms,fixed_regrets={k:v['mean_selection_loss'] for k,v in fixed.items()},normal=sum(e['completed'] for e in episodes))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    evaluate(parser.parse_args().run.resolve())
