"""Validate the NC13 execution lock before original development training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def verify_source(*, require_runtime: bool = True) -> str:
    lock_path = HERE / 'EXECUTE_LOCK.json'
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    if lock.get('schema') != 'BPD/NC13-execution-lock/v1':
        raise ValueError('wrong execution lock schema')
    if lock.get('protocol_commit') != '069b355e8f37ec780dc6c46b82176b6f32f8e39b':
        raise ValueError('protocol first-commit mismatch')
    if lock.get('development_roots') != list(range(2601300, 2601316)):
        raise ValueError('registered root roster mismatch')
    for name, expected in lock['files_sha256'].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts:
            raise ValueError('unsafe source path in execution lock')
        path = REPO / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'source hash mismatch: {name}')
    if require_runtime:
        import numpy
        import torch
        expected = lock['runtime_recipe']
        actual = {'python_minor': f'{sys.version_info.major}.{sys.version_info.minor}',
                  'torch': torch.__version__, 'numpy': numpy.__version__}
        for name, value in actual.items():
            if value != expected[name]:
                raise ValueError(f'runtime mismatch: {name}: {value} != {expected[name]}')
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-runtime', action='store_true')
    args = p.parse_args()
    print(json.dumps({'source_lock_sha256': verify_source(require_runtime=not args.skip_runtime),
                      'source_check': 'PASS'}))


if __name__ == '__main__':
    main()
