"""Polite byte-range recovery with immutable, SHA-checked partial-range reuse."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from fetch_zip_members import RangeReader


class RecoveringRangeReader(RangeReader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guard=Path(os.environ['PHM_RANGE_RATE_FILE'])
        self.interval=float(os.environ.get('PHM_RANGE_INTERVAL_SECONDS','2'))
        seed=json.loads(Path(os.environ['PHM_RANGE_CACHE_INDEX']).read_text())
        assert seed['url']==self.url and seed['outer_bytes']==self.size
        self.seed=seed['ranges'];assert self.interval>=2

    def rate(self,defer=0):
        self.guard.parent.mkdir(parents=True,exist_ok=True)
        while True:
            with self.guard.open('a+') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);raw=stream.read();state=json.loads(raw) if raw else {}
                now=time.time()
                if defer:
                    state['blocked_until']=max(state.get('blocked_until',0),now+defer)
                    state['next']=max(state.get('next',0),state['blocked_until']);delay=0
                else:
                    ready=max(now,state.get('next',0),state.get('blocked_until',0));state['next']=ready+self.interval
                    state['admitted_requests']=state.get('admitted_requests',0)+1;delay=ready-now
                stream.seek(0);stream.truncate();json.dump(state,stream)
            if defer:return
            if delay>0:time.sleep(delay)
            with self.guard.open() as stream:
                fcntl.flock(stream,fcntl.LOCK_SH);state=json.load(stream)
            if state.get('blocked_until',0)<=time.time():return

    def read(self,n=-1):
        wanted=min(self.size-self.position,n if n>=0 else self.size)
        if wanted<=0:return b''
        if self.received+wanted>self.budget:raise ValueError('Remote acquisition byte budget exceeded')
        parts=[]
        while wanted:
            cached=self.seed.get(str(self.position))
            if cached:
                body=Path(cached['path']).read_bytes();assert len(body)==cached['bytes'] and hashlib.sha256(body).hexdigest()==cached['sha256']
                used=body[:wanted];self.records.append(dict(status='seed_verified',range=f'bytes={self.position}-{self.position+len(used)-1}',
                    source=cached['path'],source_sha256=cached['sha256'],bytes=len(used),network_bytes=0))
                parts.append(used);self.position+=len(used);wanted-=len(used);continue
            length=min(wanted,self.range_chunk_bytes);start=self.position;end=start+length-1
            for attempt in range(1,self.max_request_attempts+1):
                self.rate();record=dict(url=self.url,range=f'bytes={start}-{end}',attempt=attempt,started_utc=datetime.now(timezone.utc).isoformat())
                try:
                    with self.session.get(self.url,headers={'Range':record['range'],'Cache-Control':'no-cache'},stream=True,timeout=(10,30)) as response:
                        record.update(status_code=response.status_code,content_range=response.headers.get('Content-Range'),
                            rate_headers={k:v for k,v in response.headers.items() if 'ratelimit' in k.lower() or k.lower()=='retry-after'})
                        if response.status_code==429:
                            now=time.time();raw_retry=response.headers.get('Retry-After','60')
                            wait=float(raw_retry) if raw_retry.isdigit() else 60
                            reset=response.headers.get('X-RateLimit-Reset','')
                            if reset.isdigit():wait=max(wait,float(reset)-now)
                            wait=max(60,wait);self.rate(defer=wait);record['retry_after_seconds']=wait
                            raise RuntimeError('Rate limit;shared admission paused')
                        if response.status_code!=206 or record['content_range']!=f'bytes {start}-{end}/{self.size}':raise ValueError('Server did not honor exact range;closed')
                        body=response.raw.read(length+1);record['received_body_bytes']=len(body)
                        if len(body)!=length:raise ValueError('Unexpected range body length')
                    filename=f'range_{len(self.records):05d}_{start}_{end}.bin';(self.folder/filename).write_bytes(body)
                    record.update(status='saved',file=filename,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                except Exception as error:
                    record.update(status='failed',error_type=type(error).__name__);self.records.append(record)
                    if attempt==self.max_request_attempts or isinstance(error,ValueError):raise
                    continue
                self.records.append(record);break
            parts.append(body);self.received+=len(body);self.position+=len(body);wanted-=len(body)
        return b''.join(parts)
