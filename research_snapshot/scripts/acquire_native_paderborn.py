"""Bounded downloads of the publisher's32 bearing archives.

The publisher supplies ETags/byte counts, not whole-file SHA256 checksums.
Local hashes are recorded and are never labelled publisher-authenticated SHA.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request

from acquire_forge_pdmb_sources244 import save,sha256


def main(run):
    report=json.loads((run/'audit/Padeborn_METADATA.json').read_text())
    prefixes={'NORMAL':'K','OR':'KA','IR':'KI','OR + IR':'KB'}
    names=sorted({prefixes[x['condition']]+str(int(x['bearing_num'])).zfill(3 if x['condition']=='NORMAL' else 2) for x in report['source_rows']})
    assert len(names)==32
    directory=run/'source/paderborn_native';directory.mkdir(exist_ok=True)
    def fetch(name):
        destination=directory/(name+'.rar');receipt=directory/(name+'.receipt.json')
        if destination.exists():
            prior=json.loads(receipt.read_text());assert sha256(destination)==prior['local_sha256'];return {'bearing':name,'passed':True,'reused':True,'bytes':destination.stat().st_size}
        url='https://groups.uni-paderborn.de/kat/BearingDataCenter/'+name+'.rar'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
        try:
            with opener.open(urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:
                length=int(response.headers['Content-Length']);etag=response.headers['ETag']
            assert 0<length<256*1024**2
            save(directory/(name+'.HEAD.json'),dict(url=url,bytes=length,etag=etag,publisher_whole_checksum_available=False))
            partial=destination.with_name(destination.name+'.partial');digest=hashlib.sha256();count=0
            with opener.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','If-Match':etag}),timeout=60) as response,partial.open('xb') as output:
                for block in iter(lambda:response.read(4*1024**2),b''):
                    count+=len(block)
                    if count>length:raise ValueError('Source exceeds advertised byte count')
                    output.write(block);digest.update(block)
            assert count==length
            partial.rename(destination)
            save(receipt,dict(url=url,bytes=count,local_sha256=digest.hexdigest(),source_etag=etag,
                publisher_whole_checksum_available=False,tls_verification_disabled_for_transport=True,
                utc=datetime.now(timezone.utc).isoformat()))
            return dict(bearing=name,passed=True,reused=False,bytes=count)
        except Exception as error:
            return dict(bearing=name,passed=False,error_type=type(error).__name__,error=str(error))
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,names))
    save(run/'audit/PU_NATIVE_ACQUISITION.json',dict(files=results,passed=all(x['passed'] for x in results),
        publisher_whole_checksums_available=False,new_LLM_calls=0))
    print(json.dumps(dict(files=len(results),passed=sum(x['passed'] for x in results),failed=[x for x in results if not x['passed']])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
