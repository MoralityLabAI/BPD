"""Local reconstruction of bounded-rejection register sampling, not full original campaign."""
import hashlib
import json
import torch

def seed_for(root, domain, index=0):
    raw = f'BPD-TRM-CURATE-02:{root}:{domain}:{index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)

def wrong_partition(task, root, domain, index, n, is_eval):
    g = torch.Generator(device='cpu').manual_seed(seed_for(root, f'gym01_{task}_{domain}', index))
    width, retries = (2, 8) if task.startswith('E27') else (4, 16)
    regs = torch.randint(0, 4, (n, width), generator=g)
    def mask():
        if width == 2:
            return ((regs[:, 0] + regs[:, 1]) % 3 == 0) != is_eval
        code = regs[:, 0] + 4 * regs[:, 1] + 16 * regs[:, 2] + 64 * regs[:, 3]
        return (code % 5 <= 1) != is_eval
    for _ in range(retries):
        bad = mask()
        if not bool(bad.any()):
            break
        regs[bad] = torch.randint(0, 4, (int(bad.sum()), width), generator=g)
    return int(mask().sum())

def main():
    out = {'torch': torch.__version__, 'scope': 'sampler prefix reconstruction; cross-version original replay unverified', 'tasks': {}}
    for task in ('E27_copy_retrieve_2slot', 'E31_2hop_pointer_chase'):
        rows = []
        for root in range(29201, 29209):
            rows.append(dict(root=root, ev=wrong_partition(task, root, 'ev', 0, 512, True),
                trev=wrong_partition(task, root, 'trev', 0, 256, False),
                fb=wrong_partition(task, root, 'fb', 0, 128, False),
                warm=sum(wrong_partition(task, root, 'warm', i, 64, False) for i in range(1, 251))))
        out['tasks'][task] = dict(rows=rows, totals={k: sum(r[k] for r in rows) for k in ('ev', 'trev', 'fb', 'warm')})
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
