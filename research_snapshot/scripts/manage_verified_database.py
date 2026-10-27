"""Create/stop a private real CouchDB from the qualified, gold-free seed."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def create(runtime, workspace):
    runtime,workspace=Path(runtime).resolve(),Path(workspace).resolve()
    if not workspace.is_relative_to(Path("/srv/yyf-zyq/PHM_20261010")):
        raise ValueError("Use a managed /srv workspace")
    workspace.mkdir(parents=True,exist_ok=True)
    resource=workspace/"couch_resource.json"
    if resource.exists():
        raise FileExistsError("Existing database resource; explicitly stop or use a new workspace")
    name="phm_verified_"+hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    subprocess.run(["docker","run","-d","--name",name,"-p","127.0.0.1::5984",
                    "-e","COUCHDB_USER=phm240","-e","COUCHDB_PASSWORD=isolated-benchmark-only",
                    "couchdb:3.5"],check=True,capture_output=True)
    port=subprocess.check_output(["docker","port",name,"5984/tcp"],text=True).strip().split(":")[-1]
    url="http://127.0.0.1:"+port
    value=dict(container=name,url=url,source_runtime=str(runtime),status="initializing")
    resource.write_text(json.dumps(value,indent=2))
    auth="Basic "+base64.b64encode(b"phm240:isolated-benchmark-only").decode()
    def request(path,data=None,method=None):
        req=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),
                                   method=method,headers={"Authorization":auth,"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as response:
            return json.load(response)
    for attempt in range(30):
        try:
            request("/_up")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Database did not start")
    seed=json.loads((runtime/"database_seed.json").read_text())
    indexes=json.loads((runtime/"database_indexes.json").read_text())
    for name,docs in seed.items():
        request("/"+name,method="PUT")
        result=request("/"+name+"/_bulk_docs",dict(docs=docs),"POST")
        if any("error" in x for x in result):
            raise RuntimeError("Seed load failed")
        for index in indexes.get(name,[]):
            request("/"+name+"/_index",index,"POST")
    value.update(status="ready",counts={k:len(v) for k,v in seed.items()})
    resource.write_text(json.dumps(value,indent=2))
    return resource


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    start=sub.add_parser("start")
    start.add_argument("--runtime",type=Path,required=True)
    start.add_argument("--workspace",type=Path,required=True)
    stop=sub.add_parser("stop")
    stop.add_argument("--resource",type=Path,required=True)
    args=p.parse_args()
    if args.action=="start":
        print(create(args.runtime,args.workspace))
    else:
        data=json.loads(args.resource.read_text())
        if not data["container"].startswith("phm_verified_"):
            raise ValueError("Refusing to manage an unrelated container")
        subprocess.run(["docker","rm","-f",data["container"]],check=True)
        data["status"]="stopped"
        args.resource.write_text(json.dumps(data,indent=2))
