"""Post-seal scoring against previously opened, SHA-bound historical outcomes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import statistics as st
from pathlib import Path

from screen_score_blinding import read, save, sha


def main(root):
    seal = read(root/'CHOICES_SEALED466.json')
    assert sha((root/'CHOICES466.json').read_bytes()) == seal['choices_sha256']
    legal = read(root/'corpus/analysis/LEGAL_EVIDENCE450.json')
    lookup = {r['id']:r for r in legal}
    outcomes = {r['id']:r for r in read(root/'outcomes/analysis/HISTORICAL_OUTCOMES450.json')}
    episodes = {e['id']:e for e in read(root/'corpus/analysis/EPISODES450.json')}
    raw = {i:read(root/'outcomes/inputs'/str(i)/f'RESULT{i}.json') for i in [427,435,442,445]}
    losses = {}
    def error_vector(episode, ident):
        item = lookup[episode+'/'+ident]
        batch, subject = item['batch'], episode.split('/')[1]
        if batch == 427:
            record = raw[batch]['all_program_results']['common_p00' if item['initial'] else subject+'__'+ident]
            return {r['group']:float(r['actual'] != r['predicted']) for r in record['records']}
        if batch == 435:
            return None
        row = next(r for r in raw[batch]['rows'] if r['id'] == subject)['programs'][ident]
        values = row.get('errors', row.get('absolute_errors'))
        assert len(values) == 8
        return {str(i):e for i,e in enumerate(values)}
    for item in legal:
        vector = error_vector(item['episode'], item['program_id'])
        loss = outcomes[item['id']]['final_loss']
        if vector is not None:
            assert abs(st.mean(vector.values())-loss)<1e-8
        losses[item['id']] = dict(final_loss=loss, errors=vector)
    rows = []
    for selection in read(root/'CHOICES466.json'):
        episode=selection['episode'];meta=episodes[episode]
        members=[r for r in legal if r['episode']==episode]
        scores={r['program_id']:outcomes[r['id']]['final_loss'] for r in members}
        base=scores['p00'];best=min(scores.values());span=max(scores.values())-best
        oracle=[i for i,v in scores.items() if v<=best+1e-10]
        eligible=[i for i,d in selection['differences_to_minimum'].items() if d['mean']<=d['conditional_se']+1e-10]
        methods={}
        for name in ['keep','minimum','paired_1SE','native']:
            ident=meta['selected'] if name=='native' else selection[name]
            if name=='native' and not meta['completed']:
                methods[name]=dict(status='incomplete',selected=ident);continue
            loss=scores[ident]
            if meta['batch']==435:
                key=ident if lookup[episode+'/'+ident]['initial'] else episode.split('/')[1]+'/'+ident
                previous=raw[435]['scores'][key]
                assert previous['status']=='complete' and abs(previous['MAE']-loss)<1e-8
                harm,gain=previous['harm'],previous['gain']
                evidence='Per100engine harm/gain reused from previously audited A435 score; no replacement with positive-part ofmean'
            else:
                errors=losses[episode+'/'+ident]['errors'];anchor=losses[episode+'/p00']['errors']
                assert set(errors)==set(anchor)
                harm=st.mean(max(errors[k]-anchor[k],0) for k in anchor)
                gain=st.mean(max(anchor[k]-errors[k],0) for k in anchor)
                evidence='Recomputed from same8original entity error pairs'
            assert abs(loss-base-(harm-gain))<1e-8
            methods[name]=dict(status='complete',selected=ident,loss=loss,regret=loss-best,harm=harm,gain=gain,
                normalized_regret=(loss-best)/span if span>1e-12 else 0,
                normalized_harm=harm/span if span>1e-12 else 0,
                normalized_gain=gain/span if span>1e-12 else 0,metric_evidence=evidence)
        selected=selection['paired_1SE']
        rows.append(dict(episode=episode,family=selection['family'],view=selection['view'],
            native_completed=meta['completed'],arm=meta['arm'],methods=methods,incumbent_loss=base,
            bank_best_loss=best,bank_size=len(scores),oracle_programs=oracle,
            oracle_eligible=sorted(set(oracle)&set(eligible)),eligible_programs=eligible,
            residual_selection_loss=scores[selected]-best,
            residual_type='none' if scores[selected]<=best+1e-10 else
                ('earliest_priority_among_eligible' if set(oracle)&set(eligible) else 'oracle_excluded_by_validation_gap'),
            hypothetical_only=not meta['completed']))
    summaries=[]
    for population in ['all_complete','ordinary_complete']:
        groups=defaultdict(list)
        for row in rows:
            if row['native_completed'] and (population=='all_complete' or row['arm']=='ordinary'):
                groups[(row['family'],row['view'])].append(row)
        for (family,view),rs in sorted(groups.items()):
            values={m:{k:st.mean(r['methods'][m][k] for r in rs)
                    for k in ['loss','regret','harm','gain','normalized_regret','normalized_harm','normalized_gain']}
                    for m in ['keep','minimum','paired_1SE','native']}
            summaries.append(dict(population=population,family=family,view=view,related_banks=len(rs),metrics=values))
    result=dict(id='A466',utc=datetime.now(timezone.utc).isoformat(),rows=rows,summaries=summaries,
        episodes=16,normal=9,incomplete=7,unique_artifacts=59,source_clusters=3,
        completed_primary_residual=sum(r['native_completed'] and r['view']=='primary' and r['residual_selection_loss']>1e-10 for r in rows),
        qualification_complete=True,exploratory=True,independent_confirmation=0,goal_achieved=False,
        new_PHM_fits=0,new_LLM_calls=0,new_native_stages=0,new_predictions=0,
        limitations=['3sources;16relatedbanks are not16independent tasks','Historical results known to controller;fixedselector receives onlylegal matrices',
            'FD001completeprefixcache supplements4historicallymissingmemberships;not an equal-realized-cost native-method victory',
            'COMCVprimaryandfullfit12recordviewreportedseparately;noexternal-selection ofwinningview',
            'Fixed conditional1SEhas no adaptive-search or shift guarantee;notan innovation'])
    save(root/'FINAL_ENTITY_EVIDENCE466.json',losses)
    save(root/'RESULT466.json',result)
    print({'completed_primary_residual':result['completed_primary_residual'],'summary':[
        dict(population=s['population'],family=s['family'],view=s['view'],n=s['related_banks'],
             means={m:round(v['loss'],6) for m,v in s['metrics'].items()}) for s in summaries]},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
