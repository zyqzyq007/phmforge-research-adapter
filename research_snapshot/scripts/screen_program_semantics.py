"""Equal-information source-review screen; no fitting, tools or hidden outcomes."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from isolated_model_gateway import gateway
from mlestar_phm_adapter import post_socket
from screen_score_blinding import read, save, sha, restore, output_text, json_output


def render(raw):
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def prepare(run):
    cfg = read(run/'PROTOCOL464.json')
    bindings, packets, jobs = [], {}, []
    for case in cfg['cases']:
        archive = Path(case['archive'])
        assert sha((archive/'SOURCE_MANIFEST.json').read_bytes()) == case['manifest_sha256']
        prefix = case['prefix']
        bindings += restore(archive, run/'inputs'/case['id'], lambda p:
            (p.startswith(prefix) if prefix.endswith('/') else p == prefix) and p.endswith('.py'))
        root = run/'inputs'/case['id']
        files = {}
        for path in sorted(root.rglob('*.py')):
            relative = str(path.relative_to(root))
            key = relative.removeprefix(prefix) if prefix.endswith('/') else 'program.py'
            assert key not in files
            files[key] = render(path.read_text())
        assert files
        packet = dict(case=case['id'], task=case['task'], files=files)
        save(run/'visible'/(case['id']+'.json'), packet)
        packets[case['id']] = dict(files=len(files), sha256=sha((run/'visible'/(case['id']+'.json')).read_bytes()))
        jobs += [dict(id=case['id']+'_'+arm, case=case['id'], arm=arm) for arm in cfg['arms']]
    assert len(jobs) == cfg['jobs']
    save(run/'JOBS464.json', jobs)
    save(run/'INPUT_AUDIT464.json', dict(bindings=bindings, packets=packets,
         protocol_sha256=sha((run/'PROTOCOL464.json').read_bytes()),
         reference_never_sent=True, external_scores_never_sent=True))
    print(json.dumps(dict(prepared=True, packets=len(packets), jobs=len(jobs))), flush=True)


def run_jobs(run):
    cfg = read(run/'PROTOCOL464.json')
    audit = read(run/'INPUT_AUDIT464.json')
    assert sha((run/'PROTOCOL464.json').read_bytes()) == audit['protocol_sha256']
    jobs = read(run/'JOBS464.json')
    assert not (run/'model_gateway').exists(), 'Never restart dispatched calls'
    routes = [dict(job=j['id'], socket=str(run/'ipc'/(j['id']+'.sock'))) for j in jobs]
    assert all(len(r['socket'].encode()) < 108 for r in routes)
    save(run/'MODEL_GATEWAY_CONFIG.json', dict(routes=routes,
         base_url=cfg['base_url'], model=cfg['model'], max_requests=cfg['per_job_requests'],
         max_input_tokens=cfg['per_job_input_tokens'], max_output_tokens=cfg['per_request_output'],
         max_output_per_request=cfg['per_request_output'], max_inflight=1,
         buffer_verified_responses=True, denied_tool_namespaces=['collaboration'],
         denied_tool_names=['spawn_agent']))
    started = time.monotonic()
    gt = threading.Thread(target=gateway, args=(run,))
    gt.start()
    results = []
    def execute(job):
        folder = run/'jobs'/job['id']
        folder.mkdir(parents=True, exist_ok=False)
        began = time.monotonic()
        try:
            assert began-started < cfg['wall_seconds']
            packet_path = run/'visible'/(job['case']+'.json')
            assert sha(packet_path.read_bytes()) == audit['packets'][job['case']]['sha256']
            response = post_socket(run/'ipc'/(job['id']+'.sock'), dict(
                model=cfg['model'], instructions=cfg['prompts'][job['arm']],
                input=packet_path.read_text(), tools=[], stream=False,
                max_output_tokens=cfg['per_request_output']))
            save(folder/'response.json', response)
            review = json_output(output_text(response))
            assert isinstance(review.get('findings'), list)
            result = dict(**job, status='completed', review=review)
        except Exception as error:
            result = dict(**job, status='incomplete', error_type=type(error).__name__,
                          error=str(error)[:800])
        result['elapsed_seconds'] = time.monotonic()-began
        save(folder/'TERMINAL464.json', result)
        print(json.dumps(dict(job=job['id'], status=result['status'], seconds=round(result['elapsed_seconds'], 2))), flush=True)
        return result
    try:
        while not (run/'MODEL_GATEWAY_READY.json').exists():
            if not gt.is_alive() or time.monotonic()-started > 30:
                raise RuntimeError('Gateway not ready')
            time.sleep(.1)
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            for future in as_completed([pool.submit(execute, j) for j in jobs]):
                results.append(future.result())
    finally:
        (run/'MODEL_GATEWAY_STOP').touch()
        gt.join(timeout=640)
        assert not gt.is_alive()
    assert len(results) == len(jobs)
    states = [read(p) for p in (run/'model_gateway').rglob('status.json')]
    receipts = list((run/'model_gateway').rglob('*_receipt.json'))
    cost = dict(requests=sum(s['requests'] for s in states), receipts=len(receipts),
                input_tokens=sum(s['input_tokens'] for s in states),
                output_tokens=sum(s['output_tokens'] for s in states),
                returned_models=sorted({m for s in states for m in s['returned_model_ids']}),
                all_terminal=all(not s['inflight'] for s in states), PHM_fits=0, native_stages=0)
    save(run/'ALL_REVIEWS_SEALED464.json', dict(utc=datetime.now(timezone.utc).isoformat(),
         jobs=results, elapsed_seconds=time.monotonic()-started, cost=cost,
         hashes={str(p.relative_to(run)):sha(p.read_bytes()) for p in (run/'jobs').rglob('TERMINAL464.json')}))
    print(json.dumps(dict(sealed=True, cost=cost)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run_jobs)(args.run)
