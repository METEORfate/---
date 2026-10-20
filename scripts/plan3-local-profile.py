"""Run the existing real profiler with explicit deterministic GPU execution."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from glaucoma.common import read_json, seed_all, write_json, sha256
from glaucoma.data import AgeTransform
from plan3 import benchmark
from plan3.cache import GLOBAL_RECIPE, prepare_global, source_hashes
from plan3.models import Classifier, Encoder
from plan3.provenance import Context, pretrained
from plan3.train import cpu_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--diagnose-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {**read_json(ROOT/'configs/plan3/rfm448.json'), 'deterministic_execution': True}
    context = Context(ROOT/'glaucoma_first')
    if not args.diagnose_only:
        original = benchmark.write_json
        def record(path, value):
            if Path(path).name == 'benchmark.json':
                value['deterministic_execution'] = True
                value['cublas_workspace_config'] = os.environ['CUBLAS_WORKSPACE_CONFIG']
                value['execution_entry_sha256'] = sha256(__file__)
            original(path, value)
        benchmark.write_json = record
        return benchmark.run(context, config, {'max_hours': 3, 'benchmark_repeats': 3}, args.output)
    (fit, hold), _ = benchmark.representative(context)
    full_fit, _ = context.split(0)
    age = AgeTransform.fit([context.rows[i]['age'] for i in full_fit])
    fit_rows, hold_rows = [context.rows[i] for i in fit], [context.rows[i] for i in hold]
    weight, source = pretrained(config)
    seed_all(42)
    model = Classifier(config, Encoder(config, weight, source))
    initial = cpu_state(model); del model
    config['preprocess_version'] = GLOBAL_RECIPE
    hashes = source_hashes(ROOT/'artifacts/audit')
    cache = ROOT/'artifacts/plan3_speed_20261020/local-nondeterministic-failed/cache'
    fit_cache = prepare_global(fit_rows, 448, cache, context.audit['data_fingerprint'], hashes, 4)
    hold_cache = prepare_global(hold_rows, 448, cache, context.audit['data_fingerprint'], hashes, 4)
    report = {'status': 'running', 'deterministic_execution': True, 'trials': []}
    for repeat in range(2):
        print('Deterministic real cache/uncached paired training', repeat+1, flush=True)
        first = benchmark.trial(config, initial, fit_rows, hold_rows, age, None, None, 2, 4, True, 'cuda')
        second = benchmark.trial(config, initial, fit_rows, hold_rows, age, fit_cache, hold_cache, 2, 4, True, 'cuda')
        error = max(float(np.max(np.abs(np.array(first[k])-second[k]))) for k in ('raw_probabilities', 'ema_probabilities'))
        if error > .001:
            raise ValueError('Deterministic paired probability comparison failed')
        for member in ('raw_metrics', 'ema_metrics'):
            for metric in ('auroc', 'auprc'):
                if abs(first[member][metric]-second[member][metric]) > .001:
                    raise ValueError('Deterministic paired metric comparison failed')
        report['trials'].append({'uncached': first, 'cached': second, 'max_probability_difference': error})
        write_json(args.output, report)
    report['status'] = 'complete'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
