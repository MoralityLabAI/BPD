"""Frozen design and provenance helpers for RP04."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = HERE.parent / 'training_steering_stability'
PARENT_COMMIT = '402ef7764e11bc7736873d66357b8dc24b996e94'
ROOTS = tuple(range(27101, 27109))
SMOKE_ROOTS = (27010, 27011)
CHECKPOINTS = ('shared_warmup', 'A_task', 'B_causal_fixed', 'C_rotated_fixed', 'D_causal_live')
DIRECTIONS = ('donor_span', 'hybrid_span', 'hybrid_live_span', 'donor_dual', 'hybrid_dual', 'hybrid_full')
PATH_DIRECTIONS = ('hybrid_live_span', 'hybrid_dual')
ROUTES = ('carry_only', 'all_write_only', 'selected_write_only', 'outside_write_only', 'carry_selected', 'carry_outside')
CONDITIONS = ('clean',) + DIRECTIONS + tuple(x+'_rotated' for x in DIRECTIONS) + ('hybrid_state_oracle',) + tuple(d+'__'+r for d in PATH_DIRECTIONS for r in ROUTES)
CONTRASTS = (
    ('hybrid_span', 'donor_span', 'donor_contamination'),
    ('hybrid_dual', 'hybrid_live_span', 'component_read_isolation'),
    ('hybrid_full', 'hybrid_live_span', 'state_span_coverage'),
    ('hybrid_dual', 'hybrid_dual_rotated', 'dual_direction_control'),
)

@dataclass(frozen=True)
class Recipe:
    base_pairs: int = 256
    smoke_base_pairs: int = 16
    slots: int = 4
    values: int = 4
    delay: int = 31
    noise: float = .03
    dose: float = .75
    threads: int = 1
    pinv_rtol: float = 1e-10
    scalar_tolerance: float = 2e-5
    scientific_min_effect: float = .05
    corruption_increase_reference: float = .05
    bootstrap_draws: int = 10000
    bootstrap_seed: int = 20261126


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')


def source_hashes():
    original = json.loads((PARENT/'SOURCE_FREEZE.json').read_text())['source_hashes']
    result = {}
    for relative, expected in original.items():
        actual = sha(REPO/relative)
        if actual != expected:
            raise ValueError('RP03 source changed: '+relative)
        result[relative] = actual
    paths = sorted(HERE.glob('*.py')) + [HERE/'PROTOCOL.md', HERE/'THEORY.md', HERE/'AMENDMENT_01_AUDIT_ARITHMETIC.md', HERE/'requirements.txt', REPO/'.github/workflows/bpd-rp04-component-selectivity.yml']
    for path in paths:
        result[str(path.relative_to(REPO))] = sha(path)
    return result


def checkpoint_paths(root, phase):
    directory = PARENT/'results'/('smoke' if phase == 'smoke' else 'development')/f'root_{root}'
    return {name: directory/(name+'.pt') for name in CHECKPOINTS} | {'geometry': directory/'geometry.pt'}


def input_hashes(phase):
    roots = SMOKE_ROOTS if phase == 'smoke' else ROOTS
    output = {}
    for root in roots:
        paths = checkpoint_paths(root, phase)
        original = json.loads((paths['geometry'].parent/'artifact_hashes.json').read_text())
        for path in paths.values():
            digest = sha(path)
            if original.get(path.name) != digest:
                raise ValueError('Parent checkpoint differs: '+str(path))
            output[str(path.relative_to(REPO))] = digest
        receipt = paths['geometry'].parent/'artifact_hashes.json'
        output[str(receipt.relative_to(REPO))] = sha(receipt)
    return output


def validate_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock['source_hashes'] != source_hashes():
        raise ValueError('RP04 frozen source mismatch')
    if lock['input_hashes'] != input_hashes('development'):
        raise ValueError('RP04 frozen checkpoint mismatch')
    if lock['recipe'] != asdict(Recipe()) or lock['roots'] != list(ROOTS):
        raise ValueError('RP04 frozen recipe or roster mismatch')
    commit = lock.get('remote_commit', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Published source commit required')
    return lock
