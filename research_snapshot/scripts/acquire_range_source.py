"""Acquire one public source with exact bounded ranges and an immutable receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import socket
from threading import Event, Lock, local
import time
import zipfile
from urllib.parse import urlsplit, urlunsplit
import requests

def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    source_group=parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--url')
    source_group.add_argument('--url-env',help='Runtime-only signed URL from this environment variable; never persist its query')
    parser.add_argument('--bytes',type=int,required=True)
    parser.add_argument('--etag',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--chunk-bytes',type=int,default=8*1024*1024)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-attempts',type=int,choices=[1,2,3],default=3,
                        help='Bound each newly requested range; recovery may explicitly reduce this cap')
    parser.add_argument('--max-failures',type=int,choices=range(1,21),default=20,
                        help='Remaining transport-error allowance; never exceeds 20')
    parser.add_argument('--proxy-wait-seconds',type=int,default=0,
                        help='Bounded wait for a local proxy listener before an HTTP range attempt; zero preserves legacy behavior')
    parser.add_argument('--reuse-from',help='Closed acquisition name with verified, identically bounded cached ranges')
    parser.add_argument('--reuse-interval-manifest',type=Path,
                        help='Closed, SHA-bound source intervals; request only uncovered gaps')
    parser.add_argument('--refresh-public-request',action='store_true',
                        help='Give each canonical public download request a fresh query to avoid expired redirect caches')
    parser.add_argument('--connect-timeout',type=int,default=12)
    parser.add_argument('--read-timeout',type=int,default=30)
    parser.add_argument('--drop-completed-cache',action='store_true',
                        help='Drop rebuildable pieces only after the complete assembled file and receipt exist')
    parser.add_argument('--offset',type=int,default=0,help='First source byte, for a bounded member range')
    parser.add_argument('--length',type=int,help='Number of source bytes to acquire; defaults to remainder')
    parser.add_argument('--route',choices=['direct','proxy'],default='direct')
    parser.add_argument('--format',choices=['parquet','zip','raw'],default='parquet')
    args=parser.parse_args()
    if args.url_env:
        args.url=os.environ.pop(args.url_env)
    parts=urlsplit(args.url)
    source_identity_url=urlunsplit((parts.scheme,parts.netloc,parts.path,'','')) if args.url_env else args.url
    assert 1<=args.connect_timeout<=90 and 1<=args.read_timeout<=120
    length=args.length if args.length is not None else args.bytes-args.offset
    assert 0 <= args.offset < args.bytes and 0 < length <= args.bytes-args.offset
    assert args.format=='raw' or (args.offset==0 and length==args.bytes)
    root=args.run.resolve()
    destination=root/args.destination
    assert root.is_dir() and not destination.exists()
    status_path=root/(args.name+'_status.json')
    assert not status_path.exists()
    cache=root/'clients-cache'/args.name
    cache.mkdir(parents=True,exist_ok=False)
    failures=root/'downloads'/(args.name+'_failures')
    failures.mkdir(parents=True,exist_ok=False)
    jobs=[(i,b,min(args.offset+length-1,b+args.chunk_bytes-1))
          for i,b in enumerate(range(args.offset,args.offset+length,args.chunk_bytes))]
    reused={}
    assert not (args.reuse_from and args.reuse_interval_manifest)
    if args.reuse_interval_manifest:
        previous=json.loads(args.reuse_interval_manifest.read_text())
        assert previous['writers_stopped'] is True
        assert previous['source_url']==source_identity_url and previous['source_ETag']==args.etag
        assert previous['source_total_bytes']==args.bytes
        jobs=[];cursor=args.offset
        for old in sorted(previous['ranges'],key=lambda x:x['begin']):
            assert cursor<=old['begin']<=old['end']<args.offset+length
            assert old['bytes']==old['end']-old['begin']+1 and old['ETag']==args.etag
            while cursor<old['begin']:
                end=min(old['begin']-1,cursor+args.chunk_bytes-1)
                jobs.append((len(jobs),cursor,end));cursor=end+1
            index=len(jobs);jobs.append((index,old['begin'],old['end']))
            reused[index]=dict(old,index=index,original_chunk_index=old['index'])
            cursor=old['end']+1
        while cursor<args.offset+length:
            end=min(args.offset+length-1,cursor+args.chunk_bytes-1)
            jobs.append((len(jobs),cursor,end));cursor=end+1
    if args.reuse_from:
        assert Path(args.reuse_from).name==args.reuse_from and args.reuse_from not in {'.','..',args.name}
        old=json.loads((root/(args.reuse_from+'_status.json')).read_text())
        assert old['status']!='running'
        assert old['source_url']==source_identity_url and old['source_ETag']==args.etag
        assert old['source_total_bytes']==args.bytes and old['source_offset']==args.offset
        assert old['expected_bytes']==length and old['chunk_bytes']==args.chunk_bytes
        for line in (root/(args.reuse_from+'_ranges.jsonl')).read_text().splitlines():
            row=json.loads(line)
            if row['status']=='verified':
                assert jobs[row['index']]==(row['index'],row['begin'],row['end'])
                assert row['ETag']==args.etag
                reused[row['index']]=row
    status=dict(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
                source_url=source_identity_url,expected_bytes=length,source_total_bytes=args.bytes,
                signed_URL_query_omitted=bool(args.url_env),
                source_offset=args.offset,source_ETag=args.etag,
                publisher_checksum_available=False,chunk_bytes=args.chunk_bytes,workers=args.workers,
                route=args.route,max_attempts_per_chunk=args.max_attempts,max_transport_failures=args.max_failures,new_LLM_calls=0,
                reuse_from=args.reuse_from,reusable_verified_chunks=len(reused),
                reuse_interval_manifest=str(args.reuse_interval_manifest) if args.reuse_interval_manifest else None,
                refresh_public_request=args.refresh_public_request,
                connect_timeout_seconds=args.connect_timeout,read_timeout_seconds=args.read_timeout,
                total_chunks=len(jobs),verified_chunks=0,verified_bytes=0)
    save(status_path,status)
    lock=Lock();stop=Event();error_count=0;active_requests=0
    sessions=local()
    log=(root/(args.name+'_ranges.jsonl')).open('w')
    receipts=[]
    start=time.monotonic()

    def fetch(job):
        nonlocal error_count,active_requests
        index,begin,end=job
        expected=end-begin+1
        if index in reused:
            old=reused[index]
            source_cache=Path(old['local_verified_path']) if args.reuse_interval_manifest else root/'clients-cache'/args.reuse_from/f'{index:06d}.part'
            body=source_cache.read_bytes()
            assert len(body)==expected and hashlib.sha256(body).hexdigest()==old['sha256']
            (cache/f'{index:06d}.part').write_bytes(body)
            note=dict(old,origin='reused_verified_range',reused_from=args.reuse_from,
                      elapsed_seconds=0.0,new_network_bytes=0)
            with lock:
                log.write(json.dumps(note)+'\n');log.flush()
            return note
        for attempt in range(1,args.max_attempts+1):
            if stop.is_set():
                raise RuntimeError('Transport error budget reached')
            if args.route=='proxy' and args.proxy_wait_seconds:
                proxies=requests.utils.get_environ_proxies(args.url)
                proxy=proxies.get('https',proxies.get('all'))
                if proxy:
                    endpoint=urlsplit(proxy);deadline=time.monotonic()+args.proxy_wait_seconds;wait_started=None
                    while True:
                        try:
                            with socket.create_connection((endpoint.hostname,endpoint.port or 80),timeout=2):pass
                            if wait_started is not None:
                                with lock:
                                    with (root/(args.name+'_availability.jsonl')).open('a') as event:
                                        event.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),index=index,event='local_proxy_ready',wait_seconds=time.monotonic()-wait_started,new_HTTP_requests=0))+'\n')
                            break
                        except OSError:
                            if wait_started is None:
                                wait_started=time.monotonic()
                                with lock:
                                    with (root/(args.name+'_availability.jsonl')).open('a') as event:
                                        event.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),index=index,event='local_proxy_unavailable',new_HTTP_requests=0))+'\n')
                            if stop.is_set() or time.monotonic()>=deadline:raise TimeoutError('Local proxy unavailable within frozen wait budget')
                            time.sleep(.5)
            # Reserve the possible failure before issuing a request. Near the
            # cap, in-flight requests cannot overshoot the declared total.
            while True:
                with lock:
                    if stop.is_set():raise RuntimeError('Transport error budget reached')
                    if error_count+active_requests<args.max_failures:
                        active_requests+=1;break
                time.sleep(.05)
            note=dict(index=index,begin=begin,end=end,attempt=attempt)
            body=b''
            tick=time.monotonic()
            try:
                if not hasattr(sessions,'http'):
                    sessions.http=requests.Session()
                    sessions.http.trust_env=args.route=='proxy'
                session=sessions.http
                session.headers.update({'Accept-Encoding':'identity','Range':f'bytes={begin}-{end}',
                                        'If-Match':args.etag})
                params={'phm_range':f'{begin}-{end}','phm_refresh':str(time.time_ns())} if args.refresh_public_request else None
                if params:note['public_request_query']=params
                with session.get(args.url,params=params,stream=True,timeout=(args.connect_timeout,args.read_timeout)) as response:
                    note.update(HTTP_status=response.status_code,Content_Range=response.headers.get('Content-Range'),
                                ETag=response.headers.get('ETag'))
                    for piece in response.iter_content(1024*1024):
                        body+=piece
                        if len(body)>expected:
                            raise ValueError('Response exceeds requested byte range')
                    assert response.status_code==206
                    assert response.headers.get('Content-Range')==f'bytes {begin}-{end}/{args.bytes}'
                    assert response.headers.get('ETag')==args.etag
                    assert len(body)==expected
                sha=hashlib.sha256(body).hexdigest()
                (cache/f'{index:06d}.part').write_bytes(body)
                note.update(status='verified',bytes=len(body),sha256=sha,elapsed_seconds=time.monotonic()-tick)
                with lock:
                    log.write(json.dumps(note)+'\n');log.flush()
                return note
            except Exception as exc:
                note.update(status='transport_error',error_type=type(exc).__name__,bytes=len(body),
                            sha256=hashlib.sha256(body).hexdigest(),elapsed_seconds=time.monotonic()-tick)
                if body:
                    p=failures/f'{index:06d}_attempt{attempt}.body'
                    p.write_bytes(body)
                    note['body']=str(p.relative_to(root))
                with lock:
                    error_count+=1
                    log.write(json.dumps(note)+'\n');log.flush()
                    if error_count>=args.max_failures:
                        stop.set()
                if attempt==args.max_attempts:
                    raise RuntimeError('Chunk exhausted bounded attempts') from None
                time.sleep(min(attempt,2))
            finally:
                with lock:active_requests-=1
        raise RuntimeError('No qualified response')

    try:
        # Qualify the full chosen range size before opening the worker pool.
        first_job=next((job for job in jobs if job[0] not in reused),jobs[0])
        first=fetch(first_job);receipts.append(first)
        status.update(verified_chunks=1,verified_bytes=first['bytes'])
        save(status_path,status)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(fetch,job) for job in jobs if job[0]!=first_job[0]]
            for future in as_completed(futures):
                try:note=future.result()
                except BaseException:
                    stop.set()
                    for queued in futures:queued.cancel()
                    raise
                receipts.append(note)
                status.update(verified_chunks=len(receipts),verified_bytes=sum(x['bytes'] for x in receipts),
                              transport_failures=error_count,elapsed_seconds=time.monotonic()-start)
                save(status_path,status)
        assert len(receipts)==len(jobs)
        destination.parent.mkdir(parents=True,exist_ok=True)
        sha,md5=hashlib.sha256(),hashlib.md5()
        with destination.open('xb') as output:
            for note in sorted(receipts,key=lambda x:x['index']):
                body=(cache/f'{note["index"]:06d}.part').read_bytes()
                assert len(body)==note['bytes'] and hashlib.sha256(body).hexdigest()==note['sha256']
                output.write(body);sha.update(body);md5.update(body)
        assert destination.stat().st_size==length
        if args.format=='parquet':
            with destination.open('rb') as stream:
                assert stream.read(4)==b'PAR1'
                stream.seek(-8,2);footer=stream.read(8)
            assert footer[4:]==b'PAR1'
            footer_bytes=struct.unpack('<I',footer[:4])[0]
            assert 0<footer_bytes<args.bytes-8
            format_check=dict(parquet_magic_and_footer_length_verified=True,parquet_footer_bytes=footer_bytes)
        elif args.format=='zip':
            with zipfile.ZipFile(destination) as archive:
                assert archive.testzip() is None,'ZIP member CRC mismatch'
                members=[dict(path=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,
                              CRC32=f'{x.CRC:08x}') for x in archive.infolist()]
            format_check=dict(zip_all_member_CRCS_verified=True,zip_members=members)
        else:
            format_check=dict(raw_range_only=True,whole_source_acquired=args.offset==0 and length==args.bytes,
                              decoded_format_or_member_CRC_verified=False)
        save(root/(args.name+'_manifest.json'),dict(source_url=source_identity_url,source_ETag=args.etag,
             file=str(destination.relative_to(root)),bytes=length,source_total_bytes=args.bytes,
             source_offset=args.offset,sha256=sha.hexdigest(),md5=md5.hexdigest(),
             publisher_checksum_available=False,all_ranges_size_ETag_ContentRange_verified=True,
             **format_check,
             all_values_or_PHM_truth_qualified=False,ranges=sorted(receipts,key=lambda x:x['index'])))
        status.update(status='finished',file=str(destination.relative_to(root)),full_file_bytes=length,
                      full_file_SHA256=sha.hexdigest(),full_file_MD5=md5.hexdigest(),**format_check)
        if args.drop_completed_cache:
            for piece in cache.glob('*.part'):piece.unlink()
            status['rebuildable_cache_removed_after_full_assembly']=True
    except BaseException as exc:
        stop.set();status.update(status='execution_error',error_type=type(exc).__name__)
        raise
    finally:
        log.close()
        status.update(ended_utc=datetime.now(timezone.utc).isoformat(),transport_failures=error_count,
                      elapsed_seconds=time.monotonic()-start)
        save(status_path,status)
        print(json.dumps({k:v for k,v in status.items() if k not in ['source_url','source_ETag']}),flush=True)

if __name__=='__main__':
    main()
