"""Post-pilot FUSE-02B diagnostic; independent fresh stress fixtures."""
import argparse
import json
import sys
import torch
from torch import nn
import torch.nn.functional as F


def run(TiedComponentLinear, PassMaskSchedule, AblationConfig):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    rows = []
    with torch.no_grad():
        for scale in (1., 10., 100., 10000.):
            for target in ('ordinary', 'cancellation'):
                for mask_kind in ('zero', 'soft'):
                    r = dict(scale=scale, target=target, mask=mask_kind, cases=0,
                             unguarded_fail=0, rejected=0, admitted_fail=0,
                             rejected_pass=0, guarded_fail=0, max_unguarded_error=0.,
                             max_guarded_error=0.)
                    for seed in range(2902001, 2902033):
                        torch.manual_seed(seed)
                        w = TiedComponentLinear(nn.Linear(64,64,bias=False), 32,
                                ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
                        w.V.mul_(scale); w.U.mul_(scale)
                        if target == 'cancellation':
                            w.W_target.copy_(w.reconstructed_weight())
                        x = torch.randn(1,8,64)
                        mask = torch.rand(1,1,8,32) if mask_kind == 'soft' else torch.zeros(1,1,8,32)
                        w.mode = 'component'; w.schedule = PassMaskSchedule(mask)
                        ref = w(x)
                        w.reset_pass_state(); w.use_fused_residual = True
                        fused = w(x)
                        y = F.linear(x,w.W_target)
                        z = ((mask[:,0]-1.)*(x@w.V))@w.U
                        candidate = y+z
                        torch.testing.assert_close(fused,candidate,atol=0.,rtol=0.)
                        amp = (y.abs()+z.abs()).amax(dim=-1)
                        output = candidate.abs().amax(dim=-1)
                        reject = (not bool(torch.isfinite(candidate).all()) or
                                  bool((amp>64*(output+1e-3)).any()) or
                                  float(w.V.abs().max())>2. or float(w.U.abs().max())>2.)
                        if reject:
                            w.reset_pass_state(); w.use_fused_residual=False
                            guarded=w(x)
                        else:
                            guarded=fused
                        passed=bool(torch.allclose(ref,fused,atol=5e-5,rtol=5e-4))
                        safe=bool(torch.allclose(ref,guarded,atol=5e-5,rtol=5e-4))
                        r['cases']+=1; r['unguarded_fail']+=int(not passed)
                        r['rejected']+=int(reject);r['admitted_fail']+=int(not reject and not passed)
                        r['rejected_pass']+=int(reject and passed);r['guarded_fail']+=int(not safe)
                        r['max_unguarded_error']=max(r['max_unguarded_error'],float((ref-fused).abs().max()))
                        r['max_guarded_error']=max(r['max_guarded_error'],float((ref-guarded).abs().max()))
                    rows.append(r)
    sums={k:sum(r[k] for r in rows) for k in ('cases','unguarded_fail','rejected','admitted_fail','rejected_pass','guarded_fail')}
    return {'scope':'post-pilot fresh CPU forward diagnostic only','totals':sums,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir');args=p.parse_args()
    if args.excerpt_dir:
        sys.path.insert(0,args.excerpt_dir)
        from native_excerpt import TiedComponentLinear, PassMaskSchedule, AblationConfig
    else:
        from mechinterp.bpd.components import TiedComponentLinear, PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
    print(json.dumps(run(TiedComponentLinear,PassMaskSchedule,AblationConfig),indent=2))
