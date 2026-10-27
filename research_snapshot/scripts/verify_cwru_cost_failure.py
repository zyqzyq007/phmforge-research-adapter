"""Check the observed CWRU cost-task failure against primary native evidence."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import numpy as np
from scipy.io import loadmat

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import namespace, files
from run_full_benchmark_campaign import clean_env


class Links(HTMLParser):
    def __init__(self):super().__init__();self.href=None;self.text='';self.links={}
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.href=dict(attrs).get('href');self.text=''
    def handle_data(self,data):
        if self.href is not None:self.text+=data
    def handle_endtag(self,tag):
        if tag=='a' and self.href is not None:
            self.links[self.text.strip()]=self.href;self.href=None


def main(root):
    run=root/'c';job,=json.loads((run/'PLAN187.json').read_text())['jobs'];ws=run/job['workspace'];before=files(ws)
    folder=root/'audit/CWRU_FAILURE_VERIFICATION307';folder.mkdir()
    url='https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-native-label-audit'}),timeout=40) as r:
        body=r.read();status=r.status
    (folder/'official_12k.html').write_bytes(body);parser=Links();parser.feed(body.decode())
    expected={'IR007_0':105,'B007_0':118,'OR007@6_0':130,'IR014_0':169,'B014_0':185,'OR014@6_0':197,
              'IR021_0':209,'B021_0':222,'OR021@6_0':234}
    labels={};bound=[]
    for name,number in expected.items():
        link=parser.links[name];actual=int(Path(link).stem);assert actual==number,(name,actual,number)
        label='inner_race' if name.startswith('IR') else 'outer_race' if name.startswith('OR') else 'ball_race'
        labels[number]=label;bound.append(dict(table_label=name,number=number,url=link,component_label=label))
    native=json.loads((root/'audit/CWRU_NATIVE_RECORDS.json').read_text());records={v['unit_id']:v for v in native}
    assert all(labels[v['number']]==v['label'] for v in native)
    gold=json.loads((run/'hidden/CWRU/labels.json').read_text());test=dict(np.load(job['evaluation_test'],allow_pickle=False))
    by_id={str(s):i for i,s in enumerate(test['sample_ids'])};raw_by_unit={}
    for unit in {v['unit_id'] for v in gold}:
        meta=records[unit];path=root/'source/cwru/source/snapshot'/('cwru_'+str(meta['number'])+'.mat')
        assert sha256(path)==meta['file_sha256'];raw_by_unit[unit]=loadmat(path)[meta['array_key']].ravel()
    checked=0
    for row in gold:
        meta=records[row['unit_id']];assert row['label']==labels[meta['number']]
        w=raw_by_unit[row['unit_id']][row['start_sample']:row['start_sample']+2048].astype(np.float64)
        i=by_id[row['sample_id']];assert np.array_equal(w.astype(np.float32),test['waveforms'][i])
        c=w-w.mean();sd=np.sqrt(np.mean(c*c));variance=sd*sd
        time=np.array([w.mean(),sd,np.sqrt(np.mean(w*w)),np.max(np.abs(w)),w.max()-w.min(),
            np.mean(c*c*c)/max(sd,1e-12)**3,np.mean(c**4)/max(variance,1e-12)**2])
        power=np.square(np.abs(np.fft.rfft(c)))/(2048**2)
        bands=np.array([np.log1p(np.sum(v)) for v in np.array_split(power[1:],8)])
        f=np.concatenate([time,bands]).astype(np.float32)
        assert np.allclose(f,test['features'][i],rtol=1e-6,atol=1e-8)
        checked+=1
    app=folder/'app';shutil.copytree(ws,app,symlinks=True);output=folder/'output';output.mkdir()
    command=namespace(run,job,app,output,['/opt/python312/bin/python3.12','-B','/app/solution.py',
        '--input','/data/datasets/CWRU/test.npz','--output','/output/predictions.json'])
    save(folder/'second_cold_command.json',command)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
        result=subprocess.run(command,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=180)
    assert result.returncode==0
    actual=json.loads((output/'predictions.json').read_text());first=json.loads((run/'evaluation'/job['id']/'output/predictions.json').read_text())
    p={v['sample_id']:v['predicted_label'] for v in actual['predictions']};q={v['sample_id']:v['predicted_label'] for v in first['predictions']}
    assert p==q and set(p)==set(by_id)
    assert before==files(ws)
    outcome=dict(primary_source_http_status=status,primary_source_url=url,source_html_sha256=sha256(folder/'official_12k.html'),
        all_nine_native_label_bindings=bound,native_label_assignments_agree=True,
        independent_native_windows_and_feature_rows_checked=checked,
        correct_predictions=sum(p[v['sample_id']]==v['label'] for v in gold),total_windows=len(gold),
        second_independent_cold_predictions_identical=True,submission_unchanged=True,
        original_bearings=3,independent_physical_hardware_not_certified=True,
        Agent_specific_overfitting_causality_proven=False,
        scope='Verifies observed program loss and cost assignment issue; not a physical maintenance-loss experiment.')
    save(folder/'RESULT.json',outcome);print(json.dumps(outcome))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
