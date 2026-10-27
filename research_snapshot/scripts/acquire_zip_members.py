"""Acquire a fixed set of public ZIP members through the shared range worker.

This performs source recovery only. CRC/native semantics and training admission
remain explicit later gates; downloaded compressed bytes are not Agent cases.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import requests


def utc():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);stage=path.with_name(path.name+'.new')
    stage.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');stage.replace(path)


def run(root,protocol_name='PROTOCOL287.json'):
    assert Path(protocol_name).name==protocol_name
    cfg=json.loads((root/protocol_name).read_text())
    tag=cfg['iteration'].removeprefix('A');status_file=root/('SOURCE_STATUS'+tag+'.json')
    assert not (root/'ACQUISITION_STARTED.json').exists(),'No implicit recovery or restart'
    save(root/'ACQUISITION_STARTED.json',dict(utc=utc(),pid=os.getpid(),new_LLM_calls=0,new_fits=0))
    stop_at=time.monotonic()+cfg['source_budget_seconds'];started=utc();members=[];headers=[]
    session=requests.Session();session.trust_env=cfg['route']=='proxy'
    try:
        restored=cfg.get('restored_members_file')
        if restored:
            previous=json.loads((root/restored).read_text())
            for item in cfg['members']:
                original=next(x for x in previous if x['path']==item['path'])
                assert all(original[k]==item[k] for k in ['bytes','compressed_bytes','CRC32','header_offset','compression'])
                members.append(dict(item,data_offset=original['data_offset']))
        for item in ([] if restored else cfg['members']):
            offset=item['header_offset'];body=None
            for attempt in range(1,cfg['header_attempts']+1):
                record=dict(member=item['path'],attempt=attempt,offset=offset,started_utc=utc())
                try:
                    session.headers.update({'Accept-Encoding':'identity','Range':f'bytes={offset}-{offset+511}','If-Match':cfg['etag']})
                    with session.get(cfg['url'],timeout=(12,30),stream=True) as response:
                        record.update(http_status=response.status_code,etag=response.headers.get('ETag'),content_range=response.headers.get('Content-Range'))
                        data=b''
                        for part in response.iter_content(512):
                            data+=part
                            if len(data)>512:raise ValueError('Header response exceeds frozen512-byte range')
                        assert response.status_code==206 and len(data)==512
                        assert response.headers.get('ETag')==cfg['etag']
                        assert response.headers.get('Content-Range')==f'bytes {offset}-{offset+511}/{cfg["source_total_bytes"]}'
                    fields=struct.unpack_from('<4s5H3I2H',data)
                    sig,version,flags,compression,mtime,mdate,crc,compressed,native,nlen,xlen=fields
                    assert sig==b'PK\x03\x04' and compression==item['compression']==8
                    assert 30+nlen+xlen<=512 and data[30:30+nlen].decode()==item['path']
                    cursor=30+nlen
                    while cursor<30+nlen+xlen:
                        tag,size=struct.unpack_from('<HH',data,cursor);cursor+=4
                        value=data[cursor:cursor+size];cursor+=size
                        if tag==1:
                            position=0
                            if native==0xffffffff:native=struct.unpack_from('<Q',value,position)[0];position+=8
                            if compressed==0xffffffff:compressed=struct.unpack_from('<Q',value,position)[0]
                    assert crc in [0,int(item['CRC32'],16)] and compressed in [0,item['compressed_bytes']]
                    assert native in [0,item['bytes']]
                    record.update(status='verified',http_status=206,etag=cfg['etag'],sha256=hashlib.sha256(data).hexdigest(),
                        data_offset=offset+30+nlen+xlen,compression=compression,local_crc32=f'{crc:08x}',flags=flags)
                    save(root/'source/headers'/(Path(item['path']).stem+'.json'),record)
                    (root/'source/headers'/(Path(item['path']).stem+'.bin')).write_bytes(data)
                    body=data;headers.append(record);members.append(dict(item,data_offset=record['data_offset']))
                    break
                except Exception as e:
                    record.update(status='header_error',error_type=type(e).__name__,ended_utc=utc());headers.append(record)
                    save(root/'audit/HEADER_ATTEMPTS.json',headers)
            if body is None:raise RuntimeError('Header budget exhausted for '+item['path'])
        save(root/'audit'/('HEADER_GATE'+tag+'.json'),dict(passed=True,members=members,attempts=headers,utc=utc(),
            publisher_whole_archive_md5_verified=False,member_crc_not_yet_verified=True,restored_members_file=restored))
        save(root/('MEMBER_OFFSETS'+tag+'.json'),members)
    except Exception as e:
        save(status_file,dict(status='header_gate_failed',error=repr(e),started_utc=started,ended_utc=utc(),new_LLM_calls=0,new_fits=0))
        raise
    finally:session.close()
    def acquire(item):
        name=Path(item['path']).stem.lower();out='downloads/'+name+'.deflate'
        cmd=[sys.executable,'-B',str(Path(__file__).with_name('acquire_range_source.py')),
            '--run',str(root),'--url',cfg['url'],'--bytes',str(cfg['source_total_bytes']),'--etag',cfg['etag'],
            '--name',name,'--destination',out,'--offset',str(item['data_offset']),'--length',str(item['compressed_bytes']),
            '--chunk-bytes',str(cfg['chunk_bytes']),'--workers',str(cfg['workers_per_file']),
            '--max-attempts',str(cfg['attempts_per_chunk']),'--max-failures',str(cfg['transport_failure_cap_per_file']),
            '--route',cfg['route'],'--format','raw']
        if cfg.get('proxy_wait_seconds'):cmd+=['--proxy-wait-seconds',str(cfg['proxy_wait_seconds'])]
        if cfg.get('seed_names',{}).get(name):cmd+=['--reuse-from',cfg['seed_names'][name]]
        if cfg.get('reuse_interval_manifests',{}).get(name):cmd+=['--reuse-interval-manifest',str(root/cfg['reuse_interval_manifests'][name])]
        if cfg.get('refresh_public_request'):cmd+=['--refresh-public-request']
        if cfg.get('connect_timeout_seconds'):cmd+=['--connect-timeout',str(cfg['connect_timeout_seconds'])]
        if cfg.get('read_timeout_seconds'):cmd+=['--read-timeout',str(cfg['read_timeout_seconds'])]
        if cfg.get('drop_completed_cache'):cmd+=['--drop-completed-cache']
        save(root/'commands'/(name+'.json'),cmd)
        timeout=min(cfg['per_file_seconds'],max(0,stop_at-time.monotonic()))
        if timeout<=0:return dict(member=item['path'],status='not_started_global_deadline')
        with (root/'logs'/(name+'.log')).open('wb') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'processes'/(name+'.json'),dict(pid=p.pid,member=item['path'],started_utc=utc(),timeout_seconds=timeout))
            try:
                code=p.wait(timeout=timeout);result=dict(member=item['path'],status='finished' if code==0 else 'worker_error',exit_code=code,pid=p.pid)
                if code==0 and cfg.get('audit_native_after_member'):
                    remaining=min(900,max(0,stop_at-time.monotonic()))
                    if remaining<=0:raise subprocess.TimeoutExpired(cmd,remaining)
                    native_command=[sys.executable,'-B',str(Path(__file__)), '--run',str(root),'--protocol',protocol_name,'--audit-member',name]
                    save(root/'commands'/(name+'_native_audit.json'),native_command)
                    checked=subprocess.run(native_command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                    result.update(native_audit_exit_code=checked.returncode,native_audit_passed=checked.returncode==0)
                    if checked.returncode:result['status']='member_native_audit_error'
            except subprocess.TimeoutExpired:
                running=p.poll() is None
                if running:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                result=dict(member=item['path'],status='source_timeout' if running else 'native_audit_timeout',pid=p.pid,termination_after_frozen_deadline=True)
        result['ended_utc']=utc();save(root/'outcomes'/(name+'.json'),result);return result
    outcomes=[];prior_intervals={}
    for name,relative in cfg.get('reuse_interval_manifests',{}).items():
        previous=json.loads((root/relative).read_text())['ranges']
        prior_intervals[name]=(sum(x['bytes'] for x in previous),len(previous))
    with ThreadPoolExecutor(max_workers=cfg['parallel_files']) as pool:
        pending={pool.submit(acquire,item) for item in members}
        while pending:
            done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                try:outcomes.append(future.result())
                except Exception as e:outcomes.append(dict(status='controller_task_error',error=repr(e)))
            rows=[]
            for item in members:
                name=Path(item['path']).stem.lower();receipt=root/(name+'_ranges.jsonl');verified={};fresh={};failures=0
                if receipt.exists():
                    raw=receipt.read_text();lines=raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
                    for line in lines:
                        event=json.loads(line)
                        if event['status']=='verified':
                            verified[event['index']]=event['bytes']
                            if event.get('origin')!='reused_verified_range':fresh[event['index']]=event['bytes']
                        elif event['status']=='transport_error':failures+=1
                process=root/'processes'/(name+'.json');proc=json.loads(process.read_text()) if process.exists() else {}
                baseline=prior_intervals.get(name)
                total=baseline[0]+sum(fresh.values()) if baseline else sum(verified.values())
                ranges=baseline[1]+len(fresh) if baseline else len(verified)
                rows.append(dict(member=item['path'],verified_bytes=total,compressed_bytes=item['compressed_bytes'],
                    copied_or_downloaded_worker_bytes=sum(verified.values()),reused_verified_bytes=baseline[0] if baseline else None,
                    new_verified_network_bytes=sum(fresh.values()),verified_ranges=ranges,transport_errors=failures,pid=proc.get('pid')))
            save(status_file,dict(status='acquiring_compressed_members',utc=utc(),started_utc=started,
                members=rows,outcomes=outcomes,verified_bytes=sum(r['verified_bytes'] for r in rows),
                expected_compressed_bytes=sum(x['compressed_bytes'] for x in members),new_LLM_calls=0,new_fits=0,
                scientific_source_admitted=False,remaining_controller_tasks=len(pending)))
    complete=len(outcomes)==len(members) and all(x['status']=='finished' for x in outcomes)
    native_complete=complete and cfg.get('audit_native_after_member') and all(x.get('native_audit_passed') for x in outcomes)
    final_status=('native_members_complete_waiting_condition_contract' if native_complete else 'compressed_members_complete_waiting_crc_and_native_audit') if complete else 'source_recovery_terminal_with_failures'
    value=json.loads(status_file.read_text());value.update(status=final_status,
        ended_utc=utc(),outcomes=outcomes,compressed_download_complete=complete,member_crc_verified=bool(native_complete),scientific_source_admitted=False)
    save(status_file,value)
    print(json.dumps(dict(status=value['status'],outcomes=outcomes,new_LLM_calls=0,new_fits=0)),flush=True)
    return 0 if complete else 1


def retain_terminal(root,protocol_name):
    """Preserve unique verified intervals before failed-source cache cleanup."""
    cfg=json.loads((root/protocol_name).read_text());tag=cfg['iteration'].removeprefix('A')
    status=json.loads((root/('SOURCE_STATUS'+tag+'.json')).read_text())
    assert status['status']=='source_recovery_terminal_with_failures'
    controller=json.loads((root/'ACQUISITION_STARTED.json').read_text())['pid']
    assert not Path('/proc',str(controller)).exists()
    report=[]
    for process in (root/'processes').glob('*.json'):
        assert not Path('/proc',str(json.loads(process.read_text())['pid'])).exists()
    for receipt in sorted(root.glob('*hz_ranges.jsonl')):
        name=receipt.stem.removesuffix('_ranges')
        events=[json.loads(line) for line in receipt.read_text().splitlines() if line.strip()]
        good={row['index']:row for row in events if row['status']=='verified'};files=[]
        assembled=root/'downloads'/(name+'.deflate')
        if assembled.exists():
            manifest=json.loads((root/(name+'_manifest.json')).read_text())
            with assembled.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            assert actual==manifest['sha256'] and assembled.stat().st_size==manifest['bytes']
            report.append(dict(member=name,verified_bytes=manifest['bytes'],assembled_file=str(assembled.relative_to(root)),sha256=actual,files=[]))
            continue
        for index,row in sorted(good.items()):
            prior=Path(row.get('local_verified_path','/nonexistent_verified_source'))
            if row.get('origin')=='reused_verified_range' and prior.is_file():
                selected=prior
            else:selected=root/'clients-cache'/name/f'{index:06d}.part'
            assert selected.stat().st_size==row['bytes']
            with selected.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            assert actual==row['sha256']
            if selected.is_relative_to(root/'clients-cache'):
                output=root/'retained_verified_ranges'/name/selected.name
                output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists();selected.replace(output);selected=output
            relative=selected.is_relative_to(root)
            assert relative or selected.is_relative_to(Path('/data/yyf-zyq/PHM_20261010'))
            files.append(dict(row,file=str(selected.relative_to(root)) if relative else str(selected),
                              storage='retained_in_run' if relative else 'bound_immutable_parent'))
        report.append(dict(member=name,verified_bytes=sum(x['bytes'] for x in files),files=files,
                           transport_errors=[x for x in events if x['status']=='transport_error']))
        worker_status=root/(name+'_status.json')
        if worker_status.exists():
            state=json.loads(worker_status.read_text())
            if state['status']=='running':
                original=root/'audit/raw_worker_statuses'/worker_status.name;original.parent.mkdir(exist_ok=True)
                original.write_bytes(worker_status.read_bytes())
                state.update(status='externally_observed_terminated',ended_utc=utc(),original_worker_status=str(original.relative_to(root)))
                save(worker_status,state)
    value=dict(utc=utc(),all_source_writers_stopped=True,all_retained_ranges_size_sha_verified=True,members=report,
               verified_bytes=sum(x['verified_bytes'] for x in report),new_LLM_calls=0,new_fits=0,scientific_source_admitted=False)
    save(root/'audit'/('PARTIAL_SOURCE_RECOVERY'+tag+'.json'),value)
    print(json.dumps({k:v for k,v in value.items() if k!='members'}))


def audit_member(root,protocol_name,name):
    """Complete source integrity/native scan; never defines G2 condition labels."""
    from qualify_planetary_source import decode
    from prepare_planetary_probe import native_audit
    cfg=json.loads((root/protocol_name).read_text())
    item=next(v for v in cfg['members'] if Path(v['path']).stem.lower()==name)
    manifest_path=root/(name+'_manifest.json');manifest=json.loads(manifest_path.read_text())
    raw=root/manifest['file']
    with raw.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert raw.stat().st_size==item['compressed_bytes'] and actual==manifest['sha256']
    output=root/'outputs'/Path(item['path']).name;output.parent.mkdir(exist_ok=True)
    integrity=decode(raw,output,item)
    audit=native_audit(output,minimum_duration_seconds=0,expected_carrier_unit=None)
    save(root/'audit'/(name+'_NATIVE.json'),dict(utc=utc(),integrity=integrity,native=audit,
        acquisition_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        G2_operating_condition_contract_verified=False,scientific_source_admitted=False,
        new_fits=0,new_LLM_calls=0,new_Agent_cases=0))
    print(json.dumps(dict(member=name,native_crc_verified=True,all_values_finite=True,conditions_verified=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--protocol',default='PROTOCOL287.json')
    parser.add_argument('--retain-terminal',action='store_true');parser.add_argument('--audit-member');args=parser.parse_args()
    assert not (args.retain_terminal and args.audit_member)
    if args.retain_terminal:retain_terminal(args.run.resolve(),args.protocol)
    elif args.audit_member:audit_member(args.run.resolve(),args.protocol,args.audit_member)
    else:raise SystemExit(run(args.run.resolve(),args.protocol))
