"""Source-only bounded recovery of complete nested COMFAULDA recordings.

Never fits predictors, fabricates waveforms, or computes hidden task scores.
"""
import argparse
import io
import json
import re
import struct
from pathlib import Path
import zipfile

from fetch_zip_members import RangeReader


class Slice(io.RawIOBase):
    def __init__(self, reader, start, length):
        self.reader, self.start, self.length, self.position=reader,start,length,0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        target=offset+(self.position if whence==1 else self.length if whence==2 else 0)
        if whence not in [0,1,2] or not 0<=target<=self.length:raise ValueError('Outside nested archive')
        self.position=target;return target
    def read(self, n=-1):
        count=min(self.length-self.position,n if n>=0 else self.length)
        self.reader.seek(self.start+self.position);data=self.reader.read(count);self.position+=len(data);return data


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(root, name, action, config_file='source/zip_config.json'):
    protocol=json.loads((root/'PROTOCOL332.json').read_text())
    item=next(i for i in protocol['pilot_outer_members'] if i['class_name']==name)
    outer=next(i for i in json.loads((root/'source/outer_zip_index.json').read_text()) if i['path']==item['path'])
    assert outer['compression']==0 and outer['bytes']==outer['compressed_bytes']
    config=json.loads((root/config_file).read_text())
    folder=root/'records'/name/action;folder.mkdir(parents=True,exist_ok=False);(folder/'ranges').mkdir()
    budget=protocol['per_nested_metadata_download_budget_bytes'] if action=='index' else protocol['per_raw_download_budget_bytes']
    reader=RangeReader(config['url'],config['bytes'],folder/'ranges',budget,network_route='direct',
                       range_chunk_bytes=config.get('range_chunk_bytes',512*1024),max_request_attempts=2)
    reader.session.verify='/etc/ssl/certs/ca-certificates.crt'
    report=dict(action=action,class_name=name,outer_member=item['path'],whole_ZIP_checksum_verified=False,
                outer_nested_ZIP_CRC_verified=False,new_model_calls=0,new_fits=0)
    try:
        reader.seek(outer['header_offset']);header=reader.read(30)
        fields=struct.unpack('<IHHHHHIIIHH',header);assert fields[0]==0x04034b50 and fields[3]==0
        filename=reader.read(fields[-2]).decode('utf-8');assert filename==outer['path']
        start=outer['header_offset']+30+fields[-2]+fields[-1]
        assert start+outer['bytes']<=config['bytes']
        with zipfile.ZipFile(Slice(reader,start,outer['bytes'])) as z:
            rows=[dict(path=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,CRC32=f'{i.CRC:08x}',header_offset=i.header_offset,compression=i.compress_type) for i in z.infolist()]
            save(folder/'nested_index.json',rows)
            if action=='fetch':
                registered=json.loads((root/'PILOT_RECORD_SELECTION.json').read_text())
                selected=next(i for i in registered['records'] if i['class_name']==name)
                info=z.getinfo(selected['path'])
                assert info.file_size==selected['bytes'] and f'{info.CRC:08x}'==selected['CRC32']
                assert info.file_size<=protocol['per_raw_decoded_budget_bytes']
                with z.open(info) as f:data=f.read(info.file_size+1)
                assert len(data)==info.file_size
                import hashlib,zlib
                assert f'{zlib.crc32(data):08x}'==selected['CRC32']
                p=folder/'record.csv';p.write_bytes(data)
                report.update(complete_record_CRC_verified=True,record_path=str(p.relative_to(root)),
                    original_member=info.filename,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        report.update(passed=True,nested_members=len(rows),nested_data_offset=start)
    except Exception as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error))
        raise
    finally:
        reader.session.close();report.update(received_bytes=reader.received,requests=reader.records)
        save(folder/'RECEIPT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='requests'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['index','fetch']);p.add_argument('--run',type=Path,required=True);p.add_argument('--class-name',required=True)
    p.add_argument('--config-file',default='source/zip_config.json')
    a=p.parse_args();run(a.run.resolve(),a.class_name,a.action,a.config_file)
