"""Package existing RP-01 root records with deterministic ZIP metadata."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    destination = HERE / 'receipts'
    destination.mkdir(exist_ok=True)
    archives = []
    for phase in ('smoke_17000', 'pilot_results'):
        for root in sorted((HERE / phase).glob('root_*')):
            archive = destination / (phase + '_' + root.name + '.zip')
            members = {}
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_LZMA) as zipped:
                for path in sorted(root.iterdir()):
                    if not path.is_file():
                        continue
                    data = path.read_bytes()
                    relative = str(path.relative_to(HERE))
                    info = zipfile.ZipInfo(relative, date_time=(2026, 11, 25, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_LZMA
                    info.external_attr = 0o644 << 16
                    info.create_system = 3
                    zipped.writestr(info, data)
                    members[relative] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            with zipfile.ZipFile(archive) as zipped:
                assert zipped.testzip() is None
                assert len(zipped.namelist()) == len(set(zipped.namelist())) == len(members)
                for relative, identity in members.items():
                    data = zipped.read(relative)
                    assert len(data) == identity['bytes']
                    assert hashlib.sha256(data).hexdigest() == identity['sha256']
            data = archive.read_bytes()
            item = {'path': str(archive.relative_to(HERE)), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'git_blob_sha': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(),
                    'compression': 'ZIP_LZMA', 'members': members, 'byte_roundtrip_verified': True}
            archives.append(item)
            print(json.dumps({key: item[key] for key in ('path', 'bytes', 'sha256')}), flush=True)
    output = {'format': 'BPD-RP-01/root-receipts/v1', 'root_count': len(archives),
              'scope': 'Existing raw records only. Packaging makes no model or statistical changes.',
              'extraction': 'python -m zipfile -e receipts/ARCHIVE.zip .', 'archives': archives}
    (HERE / 'receipt_manifest.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
