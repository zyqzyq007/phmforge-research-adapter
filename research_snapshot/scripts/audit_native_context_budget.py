"""A443: recorded token accounting and zero-model native transport probes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def audit(root):
    assert not (root/'TRACE_AUDIT443.json').exists()
    rows=[]; checks=0
    for subject in sorted((root/'source/subjects').iterdir()):
        requests=sorted(subject.glob('model_gateway/*/*_request.json'))
        if not requests:continue
        records=[]; repeated=Counter(); context=defaultdict(int)
        for p in requests:
            q=read(p);receipt=read(p.with_name(p.name.replace('_request','_receipt')))
            assert sha(p)==receipt['request_sha256'];checks+=1
            usage=receipt['usage'];cached=usage['input_tokens_details']['cached_tokens']
            assert 0<=cached<=usage['input_tokens'] and receipt['returned_model_ids']==['deepseek-flash'];checks+=2
            for item in q['input']:
                raw=json.dumps(item,sort_keys=True,ensure_ascii=False).encode()
                repeated[hashlib.sha256(raw).hexdigest()]+=1
                context[item.get('type',item.get('role','unspecified'))]+=len(raw)
            records.append(dict(sequence=receipt['sequence'],input_tokens=usage['input_tokens'],
                cached_tokens=cached,uncached_tokens=usage['input_tokens']-cached,
                output_tokens=usage['output_tokens'],input_items=len(q['input']),
                compaction_items=sum(i.get('type')=='compaction' for i in q['input'])))
        total=sum(x['input_tokens'] for x in records);cached=sum(x['cached_tokens'] for x in records)
        rows.append(dict(id=subject.name,requests=len(records),input_tokens=total,cached_tokens=cached,
            uncached_tokens=total-cached,cached_fraction=cached/total,output_tokens=sum(x['output_tokens'] for x in records),
            last_request_input=records[-1]['input_tokens'],max_request_input=max(x['input_tokens'] for x in records),
            first_cumulative_budget_crossing=next((i+1 for i in range(len(records)) if sum(x['input_tokens'] for x in records[:i+1])>=4000000),None),
            repeated_item_occurrences=sum(n-1 for n in repeated.values()),context_bytes_by_type=dict(context),
            provider_compaction_items=sum(x['compaction_items'] for x in records),records=records))
    assert len(rows)==4 and sum(x['requests'] for x in rows)==146
    config=read(root/'source/model_catalog187.json')['models']
    model=next(m for m in config if m['slug']=='deepseek-flash')
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checks=checks,rows=rows,
        context_window=model['context_window'],catalog_auto_compact_token_limit=model.get('auto_compact_token_limit'),
        scientific_outcomes_read=False,new_model_calls=0,new_fits=0,
        inference='Cumulative input and active context are different limits; cached input still counts in this frozen gateway. No inferred quality or completion benefit from a hypothetical compression or higher quota.',
        source_bindings_sha256=sha(root/'SOURCE_BINDINGS443.json'))
    save(root/'TRACE_AUDIT443.json',result)
    print(json.dumps(dict(checks=checks,requests=146,input_tokens=sum(x['input_tokens'] for x in rows),cached_tokens=sum(x['cached_tokens'] for x in rows))),flush=True)


def qualify(root):
    assert not (root/'TRANSPORT_PROBES443.json').exists()
    # Reuse the verified isolation/probe implementation. Only its request-path
    # annotation and per-probe timeout change; the fixture always returns400.
    original=root/'source/code/qualify_native_cli_mcp.py'
    text=original.read_text().replace('timeout=90','timeout=60')
    old="captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))"
    new="packet=json.loads(self.rfile.read(int(self.headers['Content-Length'])));captured.append(dict(packet,_audit_http_path=self.path))"
    assert text.count(old)==1;text=text.replace(old,new)
    derivative=root/'code/probe443.py';derivative.write_text(text)
    save(root/'PROBE_DERIVATION443.json',dict(original_sha256=sha(original),derivative_sha256=sha(derivative),
        changes=['60second timeout','record HTTP path'],real_model_response_provided=False))
    sys.path.insert(0,str(root/'source/code'))
    spec=importlib.util.spec_from_file_location('probe443',derivative);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    template=root/'template';template.mkdir()
    source_protocol=read(root/'source/subjects/ordinary_1/PROTOCOL187.json')
    save(template/'PROTOCOL187.json',dict(cli=source_protocol['cli'],expected_mcp_tools=[]))
    shutil.copyfile(root/'source/model_catalog187.json',template/'model_catalog187.json')
    mounts=root/'MOUNTS443.json'
    save(mounts,dict(mounts=[dict(source='/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu',destination='/opt/python312'),
        dict(source='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps',destination='/opt/phm_python/extra')]))
    base=(root/'source/base_cli_config.toml').read_text()
    def one(variant):
        run=root/'probes'/variant;run.mkdir(parents=True)
        config=base if variant=='default' else 'model_auto_compact_token_limit = 1\n'+base
        report=module.probe(run,template,dict(id=variant,mounts=str(mounts)),variant,config)
        folder=run/'audit/native_mcp'/variant/variant
        captured=[read(p) for p in sorted(folder.glob('request_*.json'))]
        return dict(variant=variant,cli_exit_code=report['cli_exit_code'],requests_captured=len(captured),
            paths=[p['_audit_http_path'] for p in captured],
            input_types=[[i.get('type',i.get('role')) for i in p['input']] for p in captured],
            tool_counts=[len(p.get('tools',[])) for p in captured],
            real_model_calls=0,benchmark_result=False)
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,['default','compact_one']))
    save(root/'TRANSPORT_PROBES443.json',dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        fixture='Explicit HTTP400; no generated model reply, no PHM inputs, no outcome simulation.',
        semantics_not_verified=True,new_model_calls=0,new_fits=0))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','qualify']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
