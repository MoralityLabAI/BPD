"""Native BPD factorized, two-state vector TRM task adapter.

One block is shared across all tokens and both high/low visits. This is a
streaming vector adaptation of the TRM recurrence, not the attention TRM.
"""
import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig


class ComponentTRM(nn.Module):
    def __init__(self, width=32, components=29, low_cycles=1, input_dim=11, values=4):
        super().__init__()
        if min(width, components, low_cycles, input_dim, values) < 1 or components > width:
            raise ValueError('invalid TRM dimensions')
        self.width, self.components = width, components
        self.low_cycles, self.input_dim = low_cycles, input_dim
        self.input = nn.Linear(input_dim,width)
        self.up = TiedComponentLinear(nn.Linear(width,2*width,bias=False),components,
                    max_passes=512,ablation=AblationConfig(mode='zero',sphere_rescale=False),
                    init_svd=True)
        self.down = nn.Linear(2*width,width,bias=False)
        self.norm = nn.RMSNorm(width)
        self.output = nn.Linear(width,values)

    def block(self, z):
        return self.norm(z+self.down(F.silu(self.up(z))))

    def rollout(self,x,patch=None,patch_step=4,state_replacement=None):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.input_dim or not x.is_floating_point():
            raise ValueError('expected a nonempty feature sequence')
        if patch is not None and state_replacement is not None:
            raise ValueError('exclusive interventions')
        intervention = patch if patch is not None else state_replacement
        b,t,_ = x.shape
        if intervention is not None:
            if (intervention.shape != (b,2*self.width) or intervention.dtype != x.dtype
                    or intervention.device != x.device or not bool(torch.isfinite(intervention).all())
                    or type(patch_step) is not int or not 0 <= patch_step < t):
                raise ValueError('invalid joint-state intervention')
        visits = t*(self.low_cycles+1)
        self.up.reset_pass_state()
        self.up.mode = 'component'
        self.up.schedule = PassMaskSchedule(comp_mask=x.new_ones((b,visits,self.components)),
                                           delta_mask=x.new_zeros((b,visits)))
        zh = x.new_zeros((b,self.width))
        zl = torch.zeros_like(zh)
        e = self.input(x)
        states,low,high = [],[],[]
        try:
            for j in range(t):
                if intervention is not None and j == patch_step:
                    state = intervention if state_replacement is not None else torch.cat((zh,zl),-1)+patch
                    zh,zl = state.chunk(2,dim=-1)
                for _ in range(self.low_cycles):
                    zl = self.block(zl+zh+e[:,j])
                    low.append(zl)
                zh = self.block(zh+zl)
                high.append(zh)
                states.append(torch.cat((zh,zl),-1))
        finally:
            self.up.schedule = None
        return dict(logits=self.output(zh),states=torch.stack(states,1),
                    high_states=torch.stack(high,1),low_states=torch.stack(low,1),
                    block_calls=self.up.calls)

    def forward(self,x):
        return self.rollout(x)['logits']

    def prefix_state(self,x,prefix_steps=4):
        if type(prefix_steps) is not int or not 1 <= prefix_steps <= x.shape[1]:
            raise ValueError('invalid prefix')
        return self.rollout(x[:,:prefix_steps])['states'][:,-1]
