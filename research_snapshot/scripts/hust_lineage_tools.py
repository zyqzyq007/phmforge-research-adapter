"""Shared battery-code tools plus a training-CV-only parent-search hint.

The same tool and frozen code archive are available in both experimental arms.
This does not certify deployment performance or modify any prediction.
"""
import argparse
import asyncio
import json
from pathlib import Path
import shutil

import numpy as np
from controlled_feedback import save, serve, utc, sha
from hust_program_tools import Broker as BatteryBroker, client
from program_revision_tools import public


def parent_hint(programs):
    ids = list(programs)
    errors = {i: {c['cell']: float(c['MAE']) for fold in v['report']['cv'] for c in fold['cells']}
              for i, v in programs.items()}
    cells = sorted(errors[ids[0]])
    assert len(cells) == 47 and all(set(e) == set(cells) for e in errors.values())
    matrix = np.array([[errors[i][c] for c in cells] for i in ids])
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    means = matrix.mean(1)
    best = min(range(len(ids)), key=lambda j: (means[j], ids[j] != 'p00', j))
    differences = matrix - matrix[best]
    se = differences.std(1, ddof=1)/np.sqrt(len(cells))
    distance = np.abs(differences).mean(1)
    near = [j for j in range(len(ids)) if j != best and means[j]-means[best] <= se[j]+1e-12]
    selected = min(near, key=lambda j: (-distance[j], j)) if near else best
    return dict(best_by_CV=ids[best], recommended_parent=ids[selected],
                alternative_available=bool(near), cells=len(cells),
                rule='Non-best parents within one paired standard error of best mean cell CV; maximize mean absolute difference of cell error profiles; registration-order ties. Fall back to best if none.',
                no_inferential_guarantee=True, no_query_RUL_or_final_labels_used=True,
                candidates=[dict(program_id=i, mean_cell_CV=float(means[j]),
                                 paired_mean_gap=float(means[j]-means[best]), paired_SE=float(se[j]),
                                 error_profile_distance=float(distance[j]), eligible_alternative=j in near)
                            for j, i in enumerate(ids)])


class Broker(BatteryBroker):
    def export_sources(self):
        for ident, item in self.state['programs'].items():
            origin = Path(item['path'])/'source'
            target = self.root/'feedback/archive'/ident/'source'
            for source in origin.rglob('*'):
                if source.is_file():
                    assert not source.is_symlink()
                    dest = target/source.relative_to(origin)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copyfile(source, dest)
                    assert sha(source) == sha(dest)

    def dispatch(self, request):
        if request['action'] == 'lineage_hint':
            self.export_sources()
            answer = parent_hint(self.state['programs'])
            answer.update(utc=utc(), code_paths={i:'/feedback/archive/'+i+'/source' for i in self.state['programs']})
            receipts = self.root/'broker/lineage_receipts';receipts.mkdir(exist_ok=True)
            save(receipts/(str(len(list(receipts.glob('*.json')))).zfill(4)+'.json'), answer)
            return answer
        result = super().dispatch(request)
        self.export_sources()
        return result


async def mcp():
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server=Server('measured-battery-lineage')
    fields=dict(info={}, lineage_hint={}, register_program=dict(
        files=dict(type='array',items=dict(type='string')),claim=dict(type='string'),rival=dict(type='string')),
        submit=dict(program_id=dict(type='string'),reason=dict(type='string')))
    descriptions=dict(info='Read complete actual programs, whole-cell training CV and remaining budget.',
        lineage_hint='Compute a training-CV-only parent-search suggestion and expose immutable prior source files. It is a search heuristic, not a final model selector or a deployment guarantee.',
        register_program='Freeze code, genuinely fit three whole-cell CV models and a full model, then cold-predict single legal queries. Counts one of four training registrations.',
        submit='Seal one complete actual program. No external-label feedback.')
    @server.list_tools()
    async def listed():
        return [types.Tool(name=n,description=descriptions[n],inputSchema=dict(type='object',properties=f,
                    required=list(f),additionalProperties=False)) for n,f in fields.items()]
    @server.call_tool()
    async def called(name, arguments):
        assert name in fields
        result=await asyncio.to_thread(client,dict(action=name,**arguments))
        return [types.TextContent(type='text',text=json.dumps(result,ensure_ascii=False))]
    async with stdio_server() as (r,w):
        await server.run(r,w,server.create_initialization_options())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['serve','mcp','client']);p.add_argument('--run',type=Path);p.add_argument('--request');a=p.parse_args()
    if a.action=='serve':serve(a.run,broker_class=Broker)
    elif a.action=='mcp':asyncio.run(mcp())
    else:print(json.dumps(client(json.loads(a.request))))
