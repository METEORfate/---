"""Small correctness checks only; scientific speed evidence comes from real profiling."""
import random
from unittest.mock import patch
from types import SimpleNamespace
import torch

from test_plan3 import Fixture, TinyEncoder
from glaucoma.common import sha256
from glaucoma.data import AgeTransform
from plan3 import benchmark, cache, data, jobs, models, train


class SpeedChecks(Fixture):
    def make_cache(self):
        hashes = {r['name']: sha256(r['path']) for r in self.rows}
        with patch('glaucoma.pixel_cache.shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)):
            path = cache.prepare_global(self.rows, 16, self.root/'cache', 'fixture', hashes, 2)
        return path, hashes

    def test_new_recipe_cache_has_identical_pixels_ages_labels_order(self):
        path, _ = self.make_cache()
        cfg = {**self.cfg, 'preprocess_version': cache.GLOBAL_RECIPE}
        age = AgeTransform.fit([r['age'] for r in self.rows])
        first = benchmark.FixedViews(data.Views(self.rows, cfg, age, augment=True))
        second = benchmark.FixedViews(data.Views(self.rows, cfg, age, augment=True, cache=path))
        for index in range(len(self.rows)):
            a, b = first[index], second[index]
            for key in a[0]:
                torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)
            for k in (1, 2):
                torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
            self.assertEqual(a[3], b[3])
        self.assertIsNone(second.views.__getstate__()['cache_array'])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            data.Views(list(reversed(self.rows)), cfg, age, cache=path)

    def test_cache_rejects_modified_source_or_pixels(self):
        path, hashes = self.make_cache()
        with path.open('r+b') as stream:
            stream.seek(-1, 2);stream.write(b'\xff')
        with patch('glaucoma.pixel_cache.shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)), \
                self.assertRaisesRegex(ValueError, 'bytes changed'):
            cache.prepare_global(self.rows, 16, self.root/'cache', 'fixture', hashes, 2)
        with open(self.rows[0]['path'], 'ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'source image'):
            cache.prepare_global(self.rows, 16, self.root/'cache', 'fixture', hashes, 2)

    def test_config_does_not_equate_original_and_resized_augmentation(self):
        with self.assertRaisesRegex(ValueError, 'original augmentation'):
            jobs.validate_config({**self.cfg, 'pixel_cache': True})
        with self.assertRaisesRegex(ValueError, 'independent global'):
            jobs.validate_config({**self.cfg, 'view': 'cross', 'preprocess_version': cache.GLOBAL_RECIPE})
        jobs.validate_config({**self.cfg, 'preprocess_version': cache.GLOBAL_RECIPE, 'pixel_cache': True})

    def test_two_models_share_decode_without_changing_complete_validation(self):
        age = AgeTransform.fit([r['age'] for r in self.rows])
        views = data.Views(self.rows, self.cfg, age)
        first = models.Classifier(self.cfg, TinyEncoder(self.cfg))
        averaged = models.EMA(first)
        with torch.no_grad():
            averaged.model.global_head.bias.add_(.5)
        loader = train.data_loader(views, 4, 0, 42)
        raw = train.evaluate(first, loader, 'cpu', 'classifier')
        ema = train.evaluate(averaged.model, loader, 'cpu', 'classifier')
        paired = train.evaluate_pair(first, averaged.model, loader, 'cpu')
        for a, b in zip((raw, ema), paired):
            self.assertEqual(a[2]['n'], len(self.rows))
            self.assertEqual(a[2], b[2])
            self.assertTrue((a[0] == b[0]).all())
            self.assertTrue((a[1] == b[1]).all())

    def test_timing_workload_really_updates_encoder_optimizer_and_ema(self):
        cfg = {**self.cfg, 'preprocess_version': cache.GLOBAL_RECIPE}
        model = models.Classifier(cfg, TinyEncoder(cfg))
        initial = train.cpu_state(model)
        age = AgeTransform.fit([r['age'] for r in self.rows])
        with patch.object(benchmark, 'Encoder', TinyEncoder):
            result = benchmark.trial(cfg, initial, self.rows[:32]*2, self.rows[32:], age,
                                     None, None, 0, 4, True, 'cpu')
        self.assertTrue(result['backbone_changed'])
        self.assertEqual(result['optimizer_updates'], 2)
        self.assertEqual(result['ema_updates'], 2)
        self.assertEqual(result['paired_max_probability_difference'], 0)
