"""ER-15: implicit additive response bank with exact sparse overrides."""
import argparse
import json
import resource
import time
import numpy as np

class SparseBank:
    def __init__(self, clean, first, second, ids, overrides):
        self.clean, self.first, self.second = clean, first, second
        self.ids, self.overrides = ids, overrides
        self.n, self.m = len(first), len(second)

    @property
    def payload(self):
        return sum(x.nbytes for x in (self.clean,self.first,self.second,self.ids,self.overrides))

    def query(self, probes):
        y = self.first[:,None,probes] + self.second[None,:,probes]
        y -= self.clean[probes]
        y = y.reshape(self.n*self.m,-1)
        y[self.ids] = self.overrides[:,probes]
        return y

    def dense(self):
        y = self.first[:,None,:] + self.second[None,:,:]
        y -= self.clean
        y = y.reshape(self.n*self.m,-1)
        y[self.ids] = self.overrides
        return y

def scale(n,p,k,mode,seed):
    rng = np.random.default_rng(seed)
    bank = SparseBank(rng.standard_normal(p,dtype=np.float32),
                      rng.standard_normal((n,p),dtype=np.float32),
                      rng.standard_normal((n,p),dtype=np.float32),
                      rng.choice(n*n,k,replace=False),
                      rng.standard_normal((k,p),dtype=np.float32))
    probes = [rng.choice(p,12,replace=False) for _ in range(48)]
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    if mode == 'dense':
        dense = bank.dense()
        payload = dense.nbytes
        del bank
        query = lambda a: dense[:,a]
    else:
        payload = bank.payload
        query = bank.query
    build = time.perf_counter()-t
    t = time.perf_counter()
    checksum = sum(float(query(a).sum(dtype=np.float64)) for a in probes)
    elapsed = time.perf_counter()-t
    return dict(n=n,p=p,k=k,mode=mode,seed=seed,payload=payload,
                baseline_kib=baseline,peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                build_ms=1000*build,query_ms=1000*elapsed,checksum=checksum)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('n',type=int); p.add_argument('p',type=int)
    p.add_argument('k',type=int); p.add_argument('mode',choices=['dense','sparse'])
    p.add_argument('seed',type=int)
    a = p.parse_args()
    print(json.dumps(scale(a.n,a.p,a.k,a.mode,a.seed)))
