"""Read-only, bounded retrieval of the existing frozen Kaggle run; never cancels it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify(folder):
    """Run with the existing training runtime, after every published file hash passes."""
    import csv
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    model = torch.load(folder/'verified/model.pt', map_location='cpu', weights_only=True)
    resume = torch.load(folder/'verified/resume.pt', map_location='cpu', weights_only=True)
    history = json.loads((folder/'verified/history.json').read_text())
    assert len(history) >= 1 and resume['epoch']+1 == len(history)
    assert model['signature'] == resume['signature']
    for key in ('optimizer', 'scheduler', 'scaler', 'rng', 'ema', 'ema_updates'):
        assert key in resume, key
    assert resume['optimizer']['state'] and resume['ema_updates'] > 0
    signature = model['signature']
    frozen = folder/'frozen-source/plan3'
    hashes = {p.name: sha(p) for p in sorted(frozen.glob('*.py'))}
    expected_code = hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert signature['code_sha256'] == expected_code
    with (folder/'verified/holdout.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [r['Image_name'] for r in rows]
    metadata = model['metadata']
    assert len(names) == len(set(names)) == len(metadata['holdout_names']) == 17132
    assert set(names) == set(metadata['holdout_names'])
    assert not set(names).intersection(metadata['train_names'])
    with (folder/'train.csv').open(encoding='utf-8-sig', newline='') as stream:
        source_rows = {r['Image_name']: r for r in csv.DictReader(stream)}
        truth = {name: int(r['True_Label']) for name, r in source_rows.items()}
    with (folder/'audit_folds.csv').open(encoding='utf-8-sig', newline='') as stream:
        folds = {r['Image_name']: int(r['fold']) for r in csv.DictReader(stream)}
    assert all(int(r['True_Label']) == truth[r['Image_name']] and folds[r['Image_name']] == 0 for r in rows)
    assert set(metadata['train_names']) == {name for name, fold in folds.items() if fold != 0}
    audit = json.loads((folder/'audit_report.json').read_text())
    assert signature['data_fingerprint'] == audit['data_fingerprint']
    assert signature['fold_fingerprint'] == sha(folder/'audit_folds.csv')
    assert signature['fold'] == 0 and signature['seed'] == 42
    assert model['config'] == signature['config']
    ages = np.array([float(source_rows[n]['Age']) if source_rows[n]['Age'].strip() else float('nan') for n in metadata['train_names']])
    good = ages[np.isfinite(ages)]
    expected_age = dict(median=float(np.median(good)), lower=float(good.min()), upper=float(good.max()),
                        mean=float(good.mean()), std=max(float(good.std()), 1.0))
    assert model['age_transform'] == expected_age
    y = np.array([truth[n] for n in names]); p = np.array([float(r['Probability']) for r in rows])
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    measured = {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'n': len(rows)}
    for key in ('auroc', 'auprc'):
        assert abs(measured[key]-metadata['holdout'][key]) < 1e-10
    assert all(r['raw_holdout']['n'] == 17132 and r['ema_holdout']['n'] == 17132 for r in history)
    holdout_meta = json.loads((folder/'verified/holdout.json').read_text())
    assert sha(folder/'verified/model.pt') == holdout_meta['model_sha256']
    assert sha(folder/'verified/holdout.csv') == holdout_meta['prediction_sha256']
    result = {'verified': True, 'complete_epochs': len(history), 'best_epoch': metadata['epoch'],
              'metrics': measured, 'resume_signature_preserved': True, 'source_code_sha256': expected_code,
              'original_source_archive_sha256': sha(folder/'source.zip'),
              'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    (folder/'verified_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--kernel', default='meteorfate/glaucoma-p3-p400-rfm448-f0-20261020/1')
    parser.add_argument('--runtime', default='D:/CodexRuntimes/glaucoma-2026/.venv/Scripts/python.exe')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    folder = args.destination.resolve()
    if args.verify:
        verify(folder)
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    deadline, previous = time.monotonic()+12*3600, None
    while time.monotonic() < deadline:
        try:
            status = api.kernels_status(args.kernel).status.name
            if status != previous:
                print(datetime.now(timezone.utc).isoformat(), status, flush=True)
                previous = status
            if status in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
                break
        except Exception as error:
            print('Read-only status retry:', type(error).__name__, flush=True)
        time.sleep(60)
    else:
        raise TimeoutError('Retrieval wait expired; the cloud job was not altered')
    download = folder/'download'
    download.mkdir(exist_ok=True)
    for attempt in range(3):
        try:
            with api.build_kaggle_client() as client:
                method = client.kernels.kernels_api_client.get_kernel
                request = method.__globals__['ApiGetKernelRequest']()
                request.user_name, request.kernel_slug = args.kernel.split('/')[:2]
                current = method(request)
                if current.metadata.current_version_number != int(args.kernel.split('/')[-1]):
                    raise ValueError('Remote kernel version changed; refusing ambiguous result retrieval')
            api.kernels_output(args.kernel, str(download), quiet=False,
                               file_pattern=r'^(results\.zip|kaggle_run\.json)$')
            with zipfile.ZipFile(download/'results.zip') as archive:
                manifest = json.loads(archive.read('plan3-result-manifest.json'))
                assert manifest['run_ref'] == args.kernel.rsplit('/', 1)[0]
                required = {'model.pt', 'resume.pt', 'history.json', 'holdout.csv', 'holdout.json',
                            'signature.json', 'config.json', 'environment.json', 'status.json'}
                seen = set()
                for name, expected in manifest['files'].items():
                    path = Path(name)
                    assert not path.is_absolute() and '..' not in path.parts
                    data = archive.read(name)
                    assert hashlib.sha256(data).hexdigest() == expected
                    target = folder/'verified'/path.name
                    if path.name == 'capacity.json':
                        target = folder/'verified/capacity.json'
                    target.parent.mkdir(exist_ok=True)
                    if target.exists() and sha(target) != expected:
                        raise ValueError('Refusing to overwrite differing retained evidence')
                    target.write_bytes(data)
                    seen.add(path.name)
                assert required.issubset(seen), required-seen
                (folder/'result_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            subprocess.run([args.runtime, str(Path(__file__).resolve()), '--destination', str(folder), '--verify'], check=True)
            return
        except Exception as error:
            print('Preservation attempt failed:', repr(error), flush=True)
            if attempt == 2:
                raise
            time.sleep(45)


if __name__ == '__main__':
    main()
