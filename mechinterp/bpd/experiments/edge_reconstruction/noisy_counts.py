"""ER-10: exact integer sufficient posterior state for three-point noise."""
import json
import numpy as np
from noisy import world,PROBS,SUP,B

def main():
    receipt=json.load(open('noisy.json')); steps=0; maxerr=0.; R=PROBS[1]/PROBS[0]; lut=R**np.arange(B+1,dtype=float)
    for arm,v in receipt['arms'].items():
        for root,rows in zip(range(232,296),v['root_rows']):
            obs,h,noise,_=world(root); count=np.zeros(len(SUP),dtype=np.int8); w=np.full(len(SUP),1/len(SUP))
            for row in rows:
                a=row['action']; delta=int(obs[h,a]+noise[a])-obs[:,a].astype(int)
                valid=np.abs(delta)<=1; active=count>=0; count[active & (delta==0)]+=1; count[~valid]=-1
                likelihood=np.zeros(len(SUP)); likelihood[valid]=PROBS[delta[valid]+1]; w*=likelihood; w/=w.sum()
                z=np.zeros(len(SUP)); alive=count>=0; z[alive]=lut[count[alive]]; z/=z.sum()
                err=float(np.max(np.abs(w-z))); assert err<1e-12 and z[h]>0
                assert np.array_equal(w>0,alive); assert abs(z[h]-row['truth_mass'])<1e-12
                steps+=1; maxerr=max(maxerr,err)
    print(json.dumps(dict(steps=steps,max_abs_posterior_error=maxerr,raw_float64_bytes=len(SUP)*8,raw_int8_bytes=len(SUP),lut_bytes=lut.nbytes,budget=B),indent=2))
if __name__=='__main__': main()
