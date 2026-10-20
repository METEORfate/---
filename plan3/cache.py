"""Explicit new global recipe; reuse the durable, lossless project cache builder."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageOps

from glaucoma.common import digest, read_json, sha256, write_json
from glaucoma.data import resize_image
from glaucoma.pixel_cache import build_pixel_cache
from .data import fov_geometry, original

GLOBAL_RECIPE = 'original-fov-pad-then-augment-v2'


def global_dtype(size):
    return np.dtype([('pixels', np.uint8, (size, size, 3)),
                     ('crop', np.int32, (4,)), ('source_size', np.int32, (2,))])


def global_record(path, size):
    image = original(path)
    return image_record(image, size)


def image_record(image, size):
    full, bounds = fov_geometry(image)
    value = np.zeros((), dtype=global_dtype(size))
    value['pixels'] = resize_image(full, size, False)
    value['crop'], value['source_size'] = bounds, image.size
    return value


def cache_identity(rows, size, fingerprint, source_hashes):
    return {'format': 'plan3-global-cache-v2', 'recipe': GLOBAL_RECIPE, 'size': size,
            'data_fingerprint': fingerprint, 'names': [r['name'] for r in rows],
            'source_sha256': {r['name']: source_hashes[r['name']] for r in rows}}


def prepare_global(rows, size, folder, fingerprint, source_hashes, workers=4):
    identity = cache_identity(rows, size, fingerprint, source_hashes)
    folder = Path(folder)
    path = folder / ('global_'+str(size)+'_'+digest(identity)[:16]+'.npy')
    # Construction hashes and decodes the same bytes once. Reuse still verifies
    # source bytes, concurrently; the formal cloud audit remains mandatory.
    expected = {str(row['path']): source_hashes[row['name']] for row in rows}
    def decode_checked(filename, target_size):
        payload = Path(filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[str(filename)]:
            raise ValueError('Cache source image SHA256 differs: '+str(filename))
        with Image.open(io.BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB'); image.load()
        return image_record(image, target_size)
    if path.exists():
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
            actual_sources = list(pool.map(sha256, [r['path'] for r in rows]))
        if any(actual != source_hashes[row['name']] for row, actual in zip(rows, actual_sources)):
            raise ValueError('Cache source image SHA256 differs')
    build_pixel_cache(rows, size, path, workers=workers, decoder=decode_checked,
                      signature=identity, item_shape=(), dtype=global_dtype(size))
    if read_json(path.with_suffix('.json')) != identity:
        raise ValueError('Cache identity changed')
    done = np.load(path.with_suffix('.done.npy'), mmap_mode='r', allow_pickle=False)
    if done.shape != (len(rows),) or not done.all():
        raise ValueError('Incomplete global cache')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.shape != (len(rows),) or value.dtype != global_dtype(size):
        raise ValueError('Global cache layout differs')
    del value, done
    actual = sha256(path)
    certificate = path.with_suffix('.sha256.json')
    if certificate.exists() and read_json(certificate)['sha256'] != actual:
        raise ValueError('Global cache bytes changed')
    write_json(certificate, {'sha256': actual, 'identity_sha256': digest(identity), 'bytes': path.stat().st_size})
    return path


def source_hashes(audit_folder):
    result = {}
    with (Path(audit_folder)/'images.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['key'].startswith('train:'):
                result[row['key'].split(':', 1)[1]] = row['sha256']
    return result


def disk_plan(count=51396):
    return {'count': count, 'global448_bytes': count*global_dtype(448).itemsize+128,
            'roi3_512_and_roi5_672_pixel_bytes': count*3*(512**2+672**2),
            'minimum_free_reserve_bytes': 30*1024**3,
            'roi_policy': 'Original-coordinate, fold-locator-bound source windows only. No crops from the global cache. Jitter/rotation support and review hashes must be bound before enabling a local cache.'}
