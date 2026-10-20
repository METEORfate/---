"""Bounded read-only collection; published statistics are independently checksummed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    # Avoid the historical scripts/kaggle.py filename shadowing the installed SDK.
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent or not (Path(p or '.')/'kaggle.py').exists()]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    ref = 'meteorfate/glaucoma-p3-rfm448-profile-20261020'
    deadline = time.monotonic()+4*3600
    previous = None
    while time.monotonic() < deadline:
        try:
            status = api.kernels_status(ref).status.name
            if status != previous:
                print(status, flush=True); previous = status
            if status in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
                break
        except Exception as error:
            print('Read-only status retry:', type(error).__name__, flush=True)
        time.sleep(60)
    else:
        raise TimeoutError('Cloud profile did not finish within the collection window')
    args.destination.mkdir(parents=True, exist_ok=True)
    api.kernels_output(ref, str(args.destination), quiet=False, file_pattern=r'^(results\.zip|kaggle_run\.json)$')
    with zipfile.ZipFile(args.destination/'results.zip') as archive:
        manifest = json.loads(archive.read('plan3-result-manifest.json'))
        if manifest['run_ref'] != ref:
            raise ValueError('Profile output identity mismatch')
        for name, expected in manifest['files'].items():
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError('Profile file SHA256 mismatch: '+name)
            if Path(name).name in {'benchmark.json', 'sample_manifest.json', 'benchmark.md'}:
                (args.destination/Path(name).name).write_bytes(archive.read(name))
        (args.destination/'result_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('All cloud profile file hashes verified; original training was not altered.', flush=True)


if __name__ == '__main__':
    main()
