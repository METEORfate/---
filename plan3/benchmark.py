"""Train-only paired I/O and unfrozen RFM measurements, never official-valid."""
from __future__ import annotations
import argparse
import copy
import gc
import io
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from glaucoma.common import ARTIFACTS, ROOT, digest, seed_all, sha256, versions, write_json
from glaucoma.data import AgeTransform, resize_image
from glaucoma.engine import autocast
from .cache import GLOBAL_RECIPE, disk_plan, prepare_global, source_hashes
from .data import Views, augment_original, fov_geometry
from .models import Classifier, EMA, Encoder, classification_loss
from .provenance import Context, code_hash, pretrained
from .train import data_loader, evaluate, evaluate_pair, move, cpu_state


class FixedViews:
    """Per-image CPU augmentation streams independent of loader scheduling."""
    def __init__(self, views):
        self.views, self.rows = views, views.rows

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        before = random.getstate()
        random.seed(42+index)
        try:
            return self.views[index]
        finally:
            random.setstate(before)


def representative(context):
    """128 fit + 64 check images, disjoint groups, both inside outer-fold fit."""
    fit, hold = context.split(0)
    rows = context.rows
    sizes = np.array([rows[i]['file_size'] for i in fit])
    edges = np.quantile(sizes, [.25, .5, .75])
    rng, used = np.random.default_rng(42), set()
    chosen = [[], []]
    for partition, per_bin in enumerate((32, 16)):
        for band in range(4):
            for label, n in ((1, per_bin//16), (0, per_bin-per_bin//16)):
                candidates = [i for i in fit if rows[i]['label'] == label and
                              int(np.searchsorted(edges, rows[i]['file_size'])) == band and context.groups[i] not in used]
                rng.shuffle(candidates)
                count = 0
                for index in candidates:
                    if context.groups[index] in used:
                        continue
                    chosen[partition].append(int(index)); used.add(context.groups[index]); count += 1
                    if count == n:
                        break
                if count != n:
                    raise ValueError('Representative benchmark stratum has insufficient independent groups')
        rng.shuffle(chosen[partition])
    assert not set(chosen[0]+chosen[1]).intersection(hold)
    return chosen, edges.tolist()


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def cpu_stages(rows, size):
    from glaucoma.multiview import image_dataset, VIEW_VERSION
    age = AgeTransform.fit([r['age'] for r in rows])
    legacy = image_dataset(rows, {'size': size, 'native_resolution': True, 'view_version': VIEW_VERSION}, age, True)
    samples = []
    for index, row in enumerate(rows[:24]):
        start = time.perf_counter(); data = Path(row['path']).read_bytes(); read = time.perf_counter()
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB'); image.load()
        decoded = time.perf_counter()
        random.seed(42+index)
        augmented, _ = augment_original(image)
        enhanced = time.perf_counter()
        full, _ = fov_geometry(augmented); resize_image(full, size, False)
        finished = time.perf_counter()
        random.seed(42+index)
        legacy_started = time.perf_counter(); old_pixels = legacy[index][0]
        legacy_seconds = time.perf_counter()-legacy_started
        samples.append({'name': row['name'], 'source_bytes': len(data), 'source_size': list(image.size),
                        'read_seconds': read-start, 'decode_seconds': decoded-read,
                        'original_augmentation_seconds': enhanced-decoded,
                        'fov_resize_seconds': finished-enhanced, 'p1_full_pipeline_seconds': legacy_seconds})
    return {'samples': samples, 'mean_seconds': {key: float(np.mean([r[key] for r in samples]))
              for key in ('read_seconds', 'decode_seconds', 'original_augmentation_seconds', 'fov_resize_seconds', 'p1_full_pipeline_seconds')}}


def trial(cfg, initial, fit_rows, check_rows, age, cache_fit, cache_check, workers, batch, checkpoint, device, validation_mode='paired'):
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = {**cfg, 'batch_size': batch, 'accumulation': 32//batch, 'gradient_checkpointing': checkpoint}
    setup = time.perf_counter()
    seed_all(42)
    model = Classifier(cfg, Encoder(cfg)).to(device)
    model.load_state_dict(initial, strict=True)
    model.train(); model.encoder.requires_grad_(True)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder.parameters()), 'lr': cfg['backbone_lr']},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': cfg['head_lr']}], weight_decay=.01)
    ema = EMA(model, cfg.get('ema_decay', .999))
    scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda'))
    dataset = FixedViews(Views(fit_rows, cfg, age, augment=True, cache=cache_fit))
    check = Views(check_rows, cfg, age, augment=False, cache=cache_check)
    data = data_loader(dataset, batch, workers, 42, shuffle=False)
    seed_all(42)
    probe_name, probe = next((n, p) for n, p in model.encoder.named_parameters() if p.requires_grad)
    before_encoder = probe.detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    sync(device); setup_seconds = time.perf_counter()-setup
    timings = dict(wait_seconds=0., transfer_seconds=0., forward_seconds=0., backward_seconds=0., update_ema_seconds=0.)
    start = time.perf_counter(); iterator = iter(data)
    updates, losses, startup, steady_start = 0, [], None, None
    for step in range(len(data)):
        t = time.perf_counter(); images, ages, labels, _ = next(iterator)
        timings['wait_seconds'] += time.perf_counter()-t
        if startup is None:
            startup = time.perf_counter()-start
        t = time.perf_counter(); images, ages, labels = move(images, device), ages.to(device), labels.to(device)
        sync(device); timings['transfer_seconds'] += time.perf_counter()-t
        t = time.perf_counter()
        with autocast(device):
            loss = classification_loss(model(images, ages), labels)
        sync(device); timings['forward_seconds'] += time.perf_counter()-t
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite benchmark loss')
        t = time.perf_counter(); scaler.scale(loss/cfg['accumulation']).backward()
        sync(device); timings['backward_seconds'] += time.perf_counter()-t
        if (step+1) % cfg['accumulation'] == 0:
            t = time.perf_counter(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            successful = scaler.get_scale() >= scale
            ema.update(model, successful); updates += int(successful)
            optimizer.zero_grad(set_to_none=True)
            sync(device); timings['update_ema_seconds'] += time.perf_counter()-t
            if steady_start is None:
                steady_start = time.perf_counter()
        losses.append(float(loss.detach()))
    sync(device); training_finished = time.perf_counter(); seconds = training_finished-start
    changed = not torch.equal(before_encoder, dict(model.encoder.named_parameters())[probe_name].detach().cpu())
    if updates < 1 or not changed:
        raise ValueError('No real unfrozen-backbone optimizer/EMA update was verified')
    validation = data_loader(check, batch, workers, 42)
    t = time.perf_counter()
    if validation_mode == 'paired':
        raw, averaged = evaluate_pair(model, ema.model, validation, device)
    else:
        raw = evaluate(model, validation, device, 'classifier')
        averaged = evaluate(ema.model, validation, device, 'classifier')
    sync(device); validation_seconds = time.perf_counter()-t
    peak = torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None
    result = {'workers': workers, 'batch_size': batch, 'accumulation': 32//batch, 'checkpointing': checkpoint,
              'training_seconds': seconds, 'training_images': len(fit_rows), 'images_per_second': len(fit_rows)/seconds,
              'setup_seconds': setup_seconds, 'first_batch_seconds': startup, 'timings': timings,
              'validation_seconds': validation_seconds, 'validation_images': len(check_rows), 'validation_mode': validation_mode,
              'optimizer_updates': updates, 'ema_updates': ema.updates, 'backbone_changed': changed,
              'steady_images_per_second': (len(fit_rows)-32)/max(training_finished-steady_start, 1e-9),
              'peak_gpu_bytes': peak, 'loss': float(np.mean(losses)),
              'raw_metrics': raw[2], 'ema_metrics': averaged[2],
              'raw_probabilities': raw[1].tolist(), 'ema_probabilities': averaged[1].tolist(), 'indices': raw[0].tolist()}
    # One matched validation control, same trained weights and all check images.
    if validation_mode == 'paired':
        t = time.perf_counter()
        separate_raw = evaluate(model, validation, device, 'classifier')
        separate_ema = evaluate(ema.model, validation, device, 'classifier')
        sync(device)
        result['separate_validation_seconds'] = time.perf_counter()-t
        result['paired_max_probability_difference'] = max(float(np.max(np.abs(raw[1]-separate_raw[1]))), float(np.max(np.abs(averaged[1]-separate_ema[1]))))
        if result['paired_max_probability_difference'] > 1e-7:
            raise ValueError('Paired validation changed model or EMA probabilities')
    del model, ema, optimizer, data, iterator, validation, dataset, check, probe
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def run(context, config, job, output, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    (fit_ix, check_ix), edges = representative(context)
    fit_rows = [context.rows[i] for i in fit_ix]; check_rows = [context.rows[i] for i in check_ix]
    full_fit, _ = context.split(0)
    age = AgeTransform.fit([context.rows[i]['age'] for i in full_fit])
    hashes = source_hashes(ARTIFACTS/'audit')
    selected = fit_rows+check_rows
    manifest = {'fold': 0, 'seed': 42, 'fit_names': [r['name'] for r in fit_rows],
                'check_names': [r['name'] for r in check_rows], 'source_hashes': {r['name']: hashes[r['name']] for r in selected},
                'size_quartile_bytes': edges, 'sampling': 'Four source-byte strata; 6.25% positives for stable integrity checks, not scientific score estimation',
                'age_transform': age.to_dict(), 'data_fingerprint': context.audit['data_fingerprint'], 'fold_fingerprint': context.audit['fold_fingerprint']}
    write_json(output/'sample_manifest.json', manifest)
    weight, source = pretrained(config)
    seed_all(42)
    initial_model = Classifier(config, Encoder(config, weight, source))
    initial = cpu_state(initial_model); del initial_model
    resized = {**config, 'preprocess_version': GLOBAL_RECIPE}
    report = {'status': 'running', 'sample_manifest_sha256': sha256(output/'sample_manifest.json'),
              'versions': versions(), 'code_sha256': code_hash(), 'pretrained_sha256': sha256(weight),
              'gpu': torch.cuda.get_device_name(0) if str(device).startswith('cuda') else 'cpu',
              'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if str(device).startswith('cuda') else None,
              'cpu_count': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(output).free,
              'disk_plan': disk_plan(), 'cpu_stages': cpu_stages(selected, config['size']), 'cases': {},
              'scope': 'Real unfrozen backbone, optimizer, EMA and train-internal complete sample validation. No official-valid evaluation.',
              'input_semantics': 'resized-v2 changes augmentation order; only cached vs uncached resized-v2 are equivalent'}
    cache_start = time.perf_counter()
    cache_fit = prepare_global(fit_rows, config['size'], output/'cache', context.audit['data_fingerprint'], hashes, 4)
    cache_check = prepare_global(check_rows, config['size'], output/'cache', context.audit['data_fingerprint'], hashes, 4)
    report['cache_build_seconds'] = time.perf_counter()-cache_start
    report['cache_bytes'] = cache_fit.stat().st_size+cache_check.stat().st_size
    # Exact pixel/control checks before any cache speed claim.
    a = FixedViews(Views(fit_rows, resized, age, augment=True))
    b = FixedViews(Views(fit_rows, resized, age, augment=True, cache=cache_fit))
    for index in range(len(fit_rows)):
        first, second = a[index], b[index]
        for key in first[0]:
            torch.testing.assert_close(first[0][key], second[0][key], rtol=0, atol=0)
        for i in (1, 2):
            torch.testing.assert_close(first[i], second[i], rtol=0, atol=0)
        assert first[3] == second[3]
    report['cache_pixel_age_label_order_equal'] = True
    write_json(output/'benchmark.json', report)
    deadline = min(job.get('_session_deadline', float('inf'))-300, time.monotonic()+job.get('max_hours', 2)*3600-300)
    repeats = job.get('benchmark_repeats', 3)
    cases = [(f'original_w{w}_b4_gc', config, None, None, w, 4, True, 'separate') for w in (2, 4, 8)]
    cases += [('resized_uncached_w2_b4_gc', resized, None, None, 2, 4, True, 'paired')]
    cases += [(f'resized_cached_w{w}_b4_gc', resized, cache_fit, cache_check, w, 4, True, 'paired') for w in (2, 4, 8)]
    def measure(case, passes=None):
        name, cfg, fit_cache, check_cache, workers, batch, checkpoint, validation_mode = case
        entry = report['cases'].setdefault(name, {'trials': []})
        if 'rejected' in entry:
            return
        values = entry['trials']
        for repeat in range(repeats if passes is None else passes):
            if time.monotonic() >= deadline:
                raise TimeoutError('Benchmark publication reserve reached')
            print(json.dumps({'phase': 'benchmark_trial', 'name': name, 'repeat': len(values)+1}), flush=True)
            try:
                value = trial(cfg, initial, fit_rows, check_rows, age, fit_cache, check_cache, workers, batch, checkpoint, device, validation_mode)
                values.append(value)
            except torch.cuda.OutOfMemoryError:
                report['cases'][name]['rejected'] = 'CUDA OOM; no batch reduction hidden'
                gc.collect(); torch.cuda.empty_cache(); break
            write_json(output/'benchmark.json', report)
        if len(values) == repeats:
            entry = report['cases'][name]
            entry['median_images_per_second'] = float(np.median([v['images_per_second'] for v in values]))
            entry['median_training_seconds'] = float(np.median([v['training_seconds'] for v in values]))
            entry['median_validation_seconds'] = float(np.median([v['validation_seconds'] for v in values]))
        write_json(output/'benchmark.json', report)
    try:
        # Interleaving original and resized controls avoids measuring only a cold
        # original pipeline against a hot cache. Every trial uses the same samples.
        for repeat in range(repeats):
            for case in (cases if repeat % 2 == 0 else list(reversed(cases))):
                measure(case, 1)
        workers = max((2, 4, 8), key=lambda w: report['cases'][f'resized_cached_w{w}_b4_gc'].get('median_images_per_second', 0))
        for batch in (8, 16):
            measure((f'resized_cached_w{workers}_b{batch}_gc', resized, cache_fit, cache_check, workers, batch, True, 'paired'))
        options = [(name, value) for name, value in report['cases'].items() if name.startswith('resized_cached') and 'median_images_per_second' in value]
        best_name, best = max(options, key=lambda p: p[1]['median_images_per_second'])
        best_batch = best['trials'][0]['batch_size']
        measure((f'resized_cached_w{workers}_b{best_batch}_no_gc', resized, cache_fit, cache_check, workers, best_batch, False, 'paired'))
        control = report['cases']['resized_uncached_w2_b4_gc']['trials']
        cached = report['cases']['resized_cached_w2_b4_gc']['trials']
        differences = []
        for first, second in zip(control, cached):
            assert first['indices'] == second['indices'] and first['optimizer_updates'] == second['optimizer_updates']
            differences.append(max([float(np.max(np.abs(np.array(first[k])-second[k]))) for k in ('raw_probabilities', 'ema_probabilities')]))
            for member in ('raw_metrics', 'ema_metrics'):
                for metric in ('auroc', 'auprc'):
                    if abs(first[member][metric]-second[member][metric]) > .001:
                        raise ValueError('Cache changed matched internal metrics beyond tolerance')
        report['cache_training_max_probability_difference'] = max(differences)
        if report['cache_training_max_probability_difference'] > .001:
            raise ValueError('Cache changed matched training output beyond the existing numerical tolerance')
        report['status'] = 'complete'
        report['formal_training_allowed'] = True
    finally:
        write_json(output/'benchmark.json', report)
        rows = ['# 方案3真实训练测速', '', '所有数值均来自训练折内部固定样本；原图增强与缩放后增强为不同实验配置。', '',
                '|配置|训练张/秒，中位|训练秒，中位|两套验证秒，中位|结论|', '|---|---:|---:|---:|---|']
        for name, value in report['cases'].items():
            rows.append(f"|{name}|{value.get('median_images_per_second', 0):.3f}|{value.get('median_training_seconds', 0):.2f}|{value.get('median_validation_seconds', 0):.2f}|{value.get('rejected', '实测')}|")
        (output/'benchmark.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/plan3/rfm448.json').read_text())
    run(Context(ROOT/'glaucoma_first'), cfg, {'max_hours': args.hours, 'benchmark_repeats': args.repeats}, args.output, args.device)
