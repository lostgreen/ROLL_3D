"""Offline matched object world/shape diagnostics, separate from v1 scoring.

Each input is a JSON list of {id, path, category?, rotation?, symmetries?}.
Paths are relative to their JSON file. Optional rotation must be a trusted
semantic local-to-world frame; do not substitute an arbitrary mesh transform.
"""
import argparse,json
from pathlib import Path
import numpy as np
from v2s_metrics import Instance
from v2s_metrics.object_quality import compare_object_sets


def load(path):
    path=Path(path)
    rows=json.loads(path.read_text())
    if not isinstance(rows,list):raise ValueError('expected instance list')
    result=[];inputs=[path.resolve()]
    for row in rows:
        p=(path.parent/row['path']).resolve();inputs.append(p)
        result.append(Instance(row['id'],np.load(p,allow_pickle=False),row.get('category'),row.get('rotation'),tuple(row.get('symmetries',()))))
    return result,inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prediction',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--yaw-steps',type=int,default=24);a=ap.parse_args()
    p,ip=load(a.prediction);g,ig=load(a.reference)
    if a.output.resolve() in ip+ig:raise ValueError('output would overwrite input')
    result=compare_object_sets(p,g,yaw_steps=a.yaw_steps)
    import hashlib
    result['inputs_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in ip+ig}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # A distinct temporary name avoids colliding with a legitimate input sibling.
    import tempfile,os
    with tempfile.NamedTemporaryFile(mode='w',dir=a.output.parent,prefix='.objects-',suffix='.tmp',delete=False) as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');temp=f.name
    os.replace(temp,a.output)
    print('object_quality_written',a.output)


if __name__=='__main__':main()
