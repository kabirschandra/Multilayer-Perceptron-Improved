#!/usr/bin/env python3
"""Download and validate the real UCI Iris and MNIST datasets. Python standard library only."""
from pathlib import Path
from urllib.request import urlopen, Request
import gzip, hashlib, struct, tempfile, os, sys
ROOT = Path(__file__).resolve().parent
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
 "train-images-idx3-ubyte": ("440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",2051,60000,28,28),
 "train-labels-idx1-ubyte": ("3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",2049,60000),
 "t10k-images-idx3-ubyte": ("8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",2051,10000,28,28),
 "t10k-labels-idx1-ubyte": ("f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",2049,10000),
}
def fetch(url):
    with urlopen(Request(url, headers={"User-Agent":"MLP-tests/1.0"}),timeout=60) as r:return r.read()
def validate_idx(raw, expected):
    header=struct.unpack(">"+"I"*len(expected),raw[:4*len(expected)])
    count=1
    for d in expected[1:]:count*=d
    if header!=expected or len(raw)!=4*len(expected)+count:raise ValueError(f"Invalid IDX header/size: {header} vs {expected}")
def main():
    iris=ROOT/'IRIS'/'iris.data'
    if not iris.exists() or len([l for l in iris.read_text().splitlines() if l.strip()])!=150:
        print('Downloading Iris...')
        raw=fetch('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
        if len([l for l in raw.splitlines() if l.strip()])!=150:raise ValueError('Iris must contain 150 samples')
        iris.write_bytes(raw)
    for name, (sha,*header) in FILES.items():
        dst=ROOT/'MNIST'/name
        if dst.exists():
            try:validate_idx(dst.read_bytes(),tuple(header));print('Found',name);continue
            except Exception:print('Replacing invalid',name)
        print('Downloading',name,flush=True)
        compressed=fetch(BASE+name+'.gz')
        if hashlib.sha256(compressed).hexdigest()!=sha:raise ValueError('SHA256 mismatch for '+name)
        raw=gzip.decompress(compressed)
        validate_idx(raw,tuple(header))
        tmp=dst.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dst)
    print('All real datasets ready.')
if __name__=='__main__':
    try:main()
    except Exception as e:print('Dataset download failed:',e,file=sys.stderr);sys.exit(1)
