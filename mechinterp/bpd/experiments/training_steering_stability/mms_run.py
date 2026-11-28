"""MMS follow-on to RP03; original frozen experiment untouched.

Single CPU arm per process, development roots only. Uses native learned V/U
and the existing RP03 generator/model. No additional rollout per update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .data import make_batch, seed_for
from .model import ComponentStateMemory


@dataclass(frozen=True)
class Recipe:
    # RP03 subset, intentionally independent of its POSIX-only experiment runner.
    width: int = 32
    components: int = 32
    slots: int = 4
    values: int = 4
    warmup_steps: int = 6000
    training_steps: int = 240
    batch_size: int = 48
    learning_rate: float = .003
    weight_decay: float = .0001
    gradient_clip: float = 1.
    evaluation_n: int = 512
    noise: float = .03
    threads: int = 1


def new_model(recipe):
    return ComponentStateMemory(width=recipe.width, components=recipe.components,
                                input_dim=3 + recipe.slots + recipe.values,
                                values=recipe.values)


def full_task_batch(batch):
    return (torch.cat((batch.receiver, batch.donor, batch.hybrid)),
            torch.cat((batch.labels_r, batch.labels_d, batch.labels_h)))


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

from .mms import MechanisticMemory


def input_keys(x):
    # Key the complete quantized observed sequence, including its length.
    # No targets or evaluation features enter the training bank.
    rows = (x.detach().cpu() > .5).to(torch.uint8).contiguous().numpy()
    return torch.tensor([int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8],
                         'little') % (2**48) for row in rows], dtype=torch.long)


def run(args):
    if not 28100 <= args.root < 28200:
        raise ValueError('fresh MMS development root must be in [28100,28200)')
    if args.steps < 1 or args.warmup < 0 or args.threads < 1:
        raise ValueError('invalid step/thread budget')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    recipe = replace(Recipe(), threads=args.threads, warmup_steps=args.warmup,
                     training_steps=args.steps)
    torch.manual_seed(seed_for(args.root, 'mms_initialization'))
    model = new_model(recipe)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                            weight_decay=recipe.weight_decay)
    for step in range(args.warmup):
        batch = make_batch(args.root, 'mms_warmup', step, recipe.batch_size, 2, 'train', recipe)
        x, y = full_task_batch(batch)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        opt.step()
    initial = state_hash(model)
    memory = (MechanisticMemory(model.recurrent, mib=args.mms_mib, mode=args.mms_mode,
                ratio=args.ratio, rank=args.rank, topk=args.topk, refresh=args.refresh,
                max_age=args.max_age) if args.mms_mib else None)
    from contextlib import nullcontext
    wall, cpu = time.perf_counter(), time.process_time()
    stream = hashlib.sha256()
    count, loss_sum = 0, 0.
    for step in range(args.steps):
        if args.cpu_seconds is not None and time.process_time()-cpu >= args.cpu_seconds:
            break
        batch = make_batch(args.root, 'mms_train', step, recipe.batch_size, 2, 'train', recipe)
        x, y = full_task_batch(batch)
        stream.update(x.contiguous().numpy().tobytes())
        opt.zero_grad(set_to_none=True)
        context = (memory.capture(input_keys(x), step, batch.prefix_steps, x.shape[1])
                   if memory is not None else nullcontext())
        with context:
            loss = F.cross_entropy(model(x), y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        opt.step()
        count += 1
        loss_sum += float(loss.detach())
    elapsed, cpu_used = time.perf_counter()-wall, time.process_time()-cpu
    model.eval()
    evaluation = {}
    with torch.no_grad():
        for delay in (2,31):
            batch = make_batch(args.root, 'mms_development_'+str(delay), 0, recipe.evaluation_n,
                               delay, 'eval', recipe)
            x, y = full_task_batch(batch)
            logits = model(x)
            evaluation[str(delay)] = dict(correct=int((logits.argmax(-1)==y).sum()),
                                           n=len(y), ce=float(F.cross_entropy(logits,y)))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), args.out/'model.pt')
    result = dict(evidence='DEVELOPMENT_ONLY', root=args.root, initial=initial,
        final=state_hash(model), training_stream_sha256=stream.hexdigest(),
        warmup_steps=args.warmup, completed_updates=count, cpu_seconds=cpu_used,
        wall_seconds=elapsed, threads=args.threads, mean_task_ce=loss_sum/max(1,count),
        evaluation=evaluation, memory=None if memory is None else memory.receipt(),
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=int,default=28101)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=6000)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--cpu-seconds',type=float)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--topk',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.2)
    p.add_argument('--refresh',type=int,default=128)
    p.add_argument('--max-age',type=int,default=32768)
    run(p.parse_args())
