"""Python source-only line/arc observation; does not inspect local values."""
from collections import Counter
import json
from pathlib import Path
import sys
import time

class ProgramTrace:
    def __init__(self,source,output):
        self.prefix=str(Path(source))+'/';self.output=Path(output)
        self.lines=Counter();self.arcs=Counter();self.previous={}

    def observe(self,frame,event,arg):
        filename=frame.f_code.co_filename
        if not filename.startswith(self.prefix):return None
        key=(filename[len(self.prefix):],frame.f_code.co_name,frame.f_code.co_firstlineno)
        if event=='call':self.previous[frame]=0
        elif event=='line':
            line=frame.f_lineno;self.lines[key+(line,)]+=1
            self.arcs[key+(self.previous.get(frame,0),line)]+=1;self.previous[frame]=line
        elif event=='return':self.arcs[key+(self.previous.pop(frame,0),-1)]+=1
        return self.observe

    def __enter__(self):
        assert sys.gettrace() is None,'Existing tracer must not be displaced'
        self.started=time.monotonic();sys.settrace(self.observe);return self

    def __exit__(self,*error):
        sys.settrace(None)
        value=dict(scope='Pythonfunctionsunder/programonly;noClibraryinternalsorworkerthreadcoverage',
            local_values_recorded=False,seconds=time.monotonic()-self.started,
            lines=[dict(file=k[0],function=k[1],first_line=k[2],line=k[3],count=v) for k,v in sorted(self.lines.items())],
            arcs=[dict(file=k[0],function=k[1],first_line=k[2],start=k[3],end=k[4],count=v) for k,v in sorted(self.arcs.items())])
        self.output.write_text(json.dumps(value,indent=2)+'\n');return False
