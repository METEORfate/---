"""Real RFM complete-epoch recovery proof, separate from formal fold models."""
import copy
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from glaucoma.common import digest, read_json, seed_all, sha256, versions, write_json
from glaucoma.data import AgeTransform
from plan3.benchmark import FixedViews, representative
from plan3.cache import GLOBAL_RECIPE, prepare_global, source_hashes
from plan3.data import Views
from plan3.models import Classifier, Encoder
from plan3.provenance import code_hash, pretrained, safe_path
from plan3 import train


class EpochBoundary(Exception):
    pass


def tag_proof_model(folder):
    path = folder/'model.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    # A timing/proof model must never be consumed as a formal classifier.
    payload['format'] = 'plan3-speed-proof-model-v1'
    train.atomic_torch(path, payload)


def tensor_error(first, second):
    if isinstance(first, torch.Tensor):
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError('Recovered tensor layout changed')
        if first.dtype in (torch.uint8, torch.int64, torch.int32, torch.bool):
            if not torch.equal(first, second):
                raise ValueError('Recovered integral/RNG tensor changed')
            return 0.
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
        return float((first-second).abs().max()) if first.numel() else 0.
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError('Recovered state keys changed')
        return max([tensor_error(first[k], second[k]) for k in first] or [0.])
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            raise ValueError('Recovered state sequence changed')
        return max([tensor_error(a, b) for a, b in zip(first, second)] or [0.])
    if first != second:
        raise ValueError('Recovered scalar state changed')
    return 0.


def run(context, config, job, output, device='cuda'):
    output = Path(output)
    (fit_ix, check_ix), _ = representative(context)
    fit_rows = [context.rows[i] for i in fit_ix]
    check_rows = [context.rows[i] for i in check_ix]
    full_fit, _ = context.split(0)
    age = AgeTransform.fit([context.rows[i]['age'] for i in full_fit])
    cfg = {**config, 'epochs': 2, 'head_warmup_epochs': 0, 'paired_validation': True,
           'preprocess_version': GLOBAL_RECIPE, 'save_validation_members': True}
    from glaucoma.common import ARTIFACTS
    hashes = source_hashes(ARTIFACTS/'audit')
    fit_cache = prepare_global(fit_rows, 448, ARTIFACTS/'speed_proof_cache', context.audit['data_fingerprint'], hashes, 4)
    check_cache = prepare_global(check_rows, 448, ARTIFACTS/'speed_proof_cache', context.audit['data_fingerprint'], hashes, 4)
    # Exercise the actual epoch/worker augmentation streams during recovery.
    fitting = Views(fit_rows, cfg, age, augment=True, cache=fit_cache)
    checking = Views(check_rows, cfg, age, cache=check_cache)
    weight, source = pretrained(cfg)
    metadata = context.metadata(fit_ix, check_ix, 0, None)
    metadata['benchmark_only'] = True
    signature = {'config': cfg, **metadata, 'seed': 42, 'code_sha256': code_hash(),
                 'loader_workers': job.get('workers', 0), 'torch_threads': torch.get_num_threads(),
                 'proof_source_sha256': sha256(__file__), 'pretrained_sha256': sha256(weight),
                 'fit_cache_sha256': sha256(fit_cache), 'check_cache_sha256': sha256(check_cache),
                 'runtime': {k: v for k, v in versions().items() if k != 'platform'}}
    def fresh():
        seed_all(42)
        return Classifier(cfg, Encoder(cfg, weight, source)).to(device)
    run_job = {**job, 'resume': False}
    if job['proof_phase'] == 'prefix':
        reference, prefix = output/'reference', output/'prefix'
        train._train(fresh(), 'classifier', cfg, run_job, signature, metadata, fitting, checking,
                     age, None, reference, device, None)
        original = train.atomic_torch
        def durable_boundary(path, value):
            original(path, value)
            if Path(path) == prefix/'resume.pt' and value.get('epoch') == 0:
                write_json(prefix/'history.json', value['history'])
                write_json(prefix/'status.json', {'complete': False, 'reason': 'proof_complete_epoch_boundary', 'last_complete_epoch': 1})
                raise EpochBoundary()
        train.atomic_torch = durable_boundary
        try:
            train._train(fresh(), 'classifier', cfg, run_job, signature, metadata, fitting, checking,
                         age, None, prefix, device, None)
        except EpochBoundary:
            pass
        finally:
            train.atomic_torch = original
        if len(read_json(reference/'history.json')) != 2 or len(read_json(prefix/'history.json')) != 1:
            raise ValueError('Prefix proof lacks the required complete epochs')
        tag_proof_model(reference); tag_proof_model(prefix)
        write_json(output/'proof.json', {'phase': 'prefix_published', 'benchmark_only': True,
            'signature': signature, 'reference_complete_epochs': 2, 'prefix_complete_epochs': 1})
    elif job['proof_phase'] == 'suffix':
        import shutil
        prior = safe_path(job['proof_prior'])
        restored = output/'restored'
        shutil.copytree(prior/'prefix', restored)
        train._train(fresh(), 'classifier', cfg, {**run_job, 'resume': True}, signature, metadata,
                     fitting, checking, age, None, restored, device, None)
        tag_proof_model(restored)
        uninterrupted = torch.load(prior/'reference/resume.pt', map_location='cpu', weights_only=True)
        recovered = torch.load(restored/'resume.pt', map_location='cpu', weights_only=True)
        errors = {}
        for key in ('signature', 'model', 'ema', 'optimizer', 'scheduler', 'scaler', 'rng', 'epoch', 'ema_updates', 'best', 'stale'):
            errors[key] = tensor_error(uninterrupted[key], recovered[key])
        if len(uninterrupted['history']) != len(recovered['history']) or len(recovered['history']) != 2:
            raise ValueError('Recovered history has incomplete or extra epochs')
        for first, second in zip(uninterrupted['history'], recovered['history']):
            for key in ('holdout', 'raw_holdout', 'ema_holdout'):
                for metric in ('auroc', 'auprc'):
                    if abs(first[key][metric]-second[key][metric]) > 1e-10:
                        raise ValueError('Recovered complete validation differs')
        if recovered['epoch'] != 1 or recovered['ema_updates'] != 8:
            raise ValueError('Recovery did not finish both complete unfrozen epochs')
        write_json(output/'proof.json', {'verified': True, 'benchmark_only': True, 'cross_process': True,
            'backend': job.get('backend', 'kaggle'),
            'signature': signature, 'errors': errors, 'epochs': 2, 'optimizer_updates': 8,
            'reference_artifact': job['proof_prior'], 'runtime': versions()})
    else:
        raise ValueError('Unknown proof phase')


if __name__ == '__main__':
    import argparse
    from plan3.provenance import Context
    from glaucoma.common import read_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=('prefix', 'suffix'), required=True)
    parser.add_argument('--prior')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run(Context(ROOT/'glaucoma_first'), read_json(safe_path(args.config)),
        {'proof_phase': args.phase, 'proof_prior': args.prior, 'workers': args.workers,
         'backend': 'local', 'max_hours': 1, 'seed': 42, 'fold': 0}, safe_path(args.output))
