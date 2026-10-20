"""Independent steady-state validation timing, after the training matrix is idle."""
import argparse
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Subset
from glaucoma.common import ARTIFACTS, seed_all, sha256, write_json
from glaucoma.data import AgeTransform
from glaucoma.engine import autocast
from plan3.benchmark import FixedViews, sync
from plan3.cache import GLOBAL_RECIPE, prepare_global, source_hashes
from plan3.data import Views
from plan3.models import Classifier, Encoder, EMA, classification_loss
from plan3.provenance import Context, pretrained, code_hash
from plan3.train import data_loader, evaluate, evaluate_pair, move


class TimedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.first_ready = self.first_done = self.last_done = None
        self.wait_seconds = 0.

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        start = time.perf_counter()
        iterator = iter(self.loader)
        try:
            for step in range(len(self.loader)):
                before = time.perf_counter(); batch = next(iterator)
                self.wait_seconds += time.perf_counter()-before
                if step == 0:
                    self.first_ready = time.perf_counter()-start
                yield batch
                if step == 0:
                    self.first_done = time.perf_counter()-start
            self.last_done = time.perf_counter()-start
        finally:
            del iterator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    deadline = time.monotonic()+3*3600
    while time.monotonic() < deadline:
        if args.primary.exists() and json.loads(args.primary.read_text())['status'] == 'complete':
            break
        time.sleep(30)
    else:
        raise TimeoutError('Primary profiling did not finish; no overlapping GPU timing was run')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    context = Context(ROOT/'glaucoma_first')
    primary_sample = json.loads((args.primary.parent/'sample_manifest.json').read_text())
    by_name = {r['name']: i for i, r in enumerate(context.rows)}
    fit, _ = context.split(0)
    excluded = {context.groups[by_name[n]] for n in primary_sample['fit_names']+primary_sample['check_names']}
    edges = primary_sample['size_quartile_bytes']
    rng = np.random.default_rng(43); selected = []
    for band in range(4):
        for label, count in ((1, 8), (0, 120)):
            choices = [int(i) for i in fit if context.rows[i]['label'] == label and
                       np.searchsorted(edges, context.rows[i]['file_size']) == band and context.groups[i] not in excluded]
            rng.shuffle(choices)
            found = 0
            for index in choices:
                if context.groups[index] in excluded:
                    continue
                selected.append(index); excluded.add(context.groups[index]); found += 1
                if found == count:
                    break
            assert found == count
    rng.shuffle(selected)
    rows = [context.rows[i] for i in selected]
    age = AgeTransform.fit([context.rows[i]['age'] for i in fit])
    cfg = {**json.loads((ROOT/'configs/plan3/rfm448.json').read_text()), 'deterministic_execution': True}
    resized = {**cfg, 'preprocess_version': GLOBAL_RECIPE}
    hashes = source_hashes(ARTIFACTS/'audit')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = prepare_global(rows, 448, args.output/'cache', context.audit['data_fingerprint'], hashes, 4)
    seed_all(42)
    weight, source = pretrained(cfg)
    model = Classifier(cfg, Encoder(cfg, weight, source)).to('cuda').train()
    model.encoder.requires_grad_(True)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    training_rows = [context.rows[by_name[n]] for n in primary_sample['fit_names']]
    # One actual effective batch updates backbone, optimizer and EMA before timing.
    train_views = FixedViews(Views(training_rows[:32], resized, age, augment=True))
    optimizer.zero_grad(set_to_none=True)
    for images, ages, labels, _ in data_loader(train_views, 4, 0, 42):
        with autocast('cuda'):
            loss = classification_loss(model(move(images, 'cuda'), ages.cuda()), labels.cuda())
        scaler.scale(loss/8).backward()
    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update(); ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    report = {'sample_count': len(rows), 'names': [r['name'] for r in rows],
              'source_sha256': {r['name']: hashes[r['name']] for r in rows}, 'fold': 0,
              'age_transform': age.to_dict(), 'code_sha256': code_hash(), 'deterministic_execution': True,
              'validation_entry_sha256': sha256(__file__), 'cases': {},
              'scope': '512 disjoint groups inside outer training fold; complete raw and EMA predictions; no official-valid'}
    for name, config, path, workers, paired in [
        ('original_w2_separate', cfg, None, 2, False),
        ('original_w4_separate', cfg, None, 4, False),
        ('original_w8_separate', cfg, None, 8, False),
        ('resized_uncached_w2_paired', resized, None, 2, True),
        ('resized_cached_w2_paired', resized, cache, 2, True),
        ('resized_cached_w4_paired', resized, cache, 4, True),
        ('resized_cached_w8_paired', resized, cache, 8, True)]:
        print('Independent validation timing:', name, flush=True)
        views = Views(rows, config, age, cache=path)
        first = TimedLoader(data_loader(views, 4, workers, 42))
        second = None
        sync('cuda'); start = time.perf_counter()
        if paired:
            raw, averaged = evaluate_pair(model, ema.model, first, 'cuda')
        else:
            raw = evaluate(model, first, 'cuda', 'classifier')
            second = TimedLoader(data_loader(views, 4, workers, 42))
            averaged = evaluate(ema.model, second, 'cuda', 'classifier')
        sync('cuda'); seconds = time.perf_counter()-start
        loaders = [first] if paired else [first, second]
        steady = sum(t.last_done-t.first_done for t in loaders)
        report['cases'][name] = {'seconds': seconds, 'steady_seconds': steady,
            'two_member_images_per_second': (len(rows)-4)/steady,
            'startup_seconds': sum(t.first_done for t in loaders),
            'raw_probabilities': raw[1].tolist(), 'ema_probabilities': averaged[1].tolist(),
            'raw_metrics': raw[2], 'ema_metrics': averaged[2], 'workers': workers, 'paired': paired}
        reference = report['cases']['original_w2_separate']
        for member in ('raw_probabilities', 'ema_probabilities'):
            difference = float(np.max(np.abs(np.array(reference[member])-report['cases'][name][member])))
            if difference > 1e-7:
                raise ValueError('Validation changed input, prediction or order: '+name)
        write_json(args.output/'validation.json', report)
    report['status'] = 'complete'
    write_json(args.output/'validation.json', report)


if __name__ == '__main__':
    main()
