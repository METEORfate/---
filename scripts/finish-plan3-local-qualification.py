"""One-off sequential release: measured matrix -> validation -> resume proof -> one fold.

Defaults reproduce the original fold-0 release. A second machine passes --fold N and
--pin-config <already qualified config> so it runs the identical recipe (same micro-batch,
same checkpointing) and the folds remain comparable.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.common import write_json

BASE = ROOT/'artifacts/plan3_speed_20261020'
RECIPE = ROOT/'configs/plan3/rfm448-local-speed-v2.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def step(phase, command=None, logstem=None):
    write_json(BASE/'release-status.json', {'phase': phase, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('LOCAL RELEASE:', phase, flush=True)
    if command:
        if logstem:
            with (BASE/(logstem+'.log')).open('w', encoding='utf-8') as out, (BASE/(logstem+'.err.log')).open('w', encoding='utf-8') as err:
                subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True, stdout=out, stderr=err)
        else:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)


def wait_complete(path, hours=4):
    deadline = time.monotonic()+hours*3600
    while time.monotonic() < deadline:
        if path.exists() and read(path).get('status') == 'complete':
            return
        time.sleep(30)
    raise TimeoutError('No complete measurement: '+str(path))


def pin(accepted, pin_config):
    if pin_config is None:
        return accepted
    spec = read(pin_config)
    want = (int(spec['batch_size']), bool(spec.get('gradient_checkpointing', False)))
    kept = {k: v for k, v in accepted.items() if (k[0], k[1]) == want}
    if not kept:
        raise ValueError('Pinned micro-batch %d / gradient_checkpointing=%s did not pass this machine\'s real '
                         'measurements; refuse to run a fold with a different recipe' % (want[0], want[1]))
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, choices=(0, 1, 2))
    parser.add_argument('--pin-config', type=Path, default=None,
                        help='Config already qualified elsewhere; keeps micro-batch/checkpointing identical')
    parser.add_argument('--qualify-only', action='store_true',
                        help='Stop after writing the fold job; do not start training')
    args = parser.parse_args()
    fold = args.fold
    pin_config = args.pin_config.resolve() if args.pin_config else None
    pin_args = ['--pin-config', str(pin_config)] if pin_config else []
    primary = BASE/'local/benchmark.json'
    validation = BASE/'validation/validation.json'
    output = BASE/'qualified'
    step('waiting_for_deterministic_real_paired_diagnostic')
    diagnostic = BASE/'determinism-diagnostic.json'
    wait_complete(diagnostic)
    if len(read(diagnostic)['trials']) != 2 or any(t['max_probability_difference'] != 0 for t in read(diagnostic)['trials']):
        raise ValueError('Deterministic paired diagnostic was not exact; no automatic training release')
    if not primary.exists() or read(primary).get('status') != 'complete':
        step('real_deterministic_training_matrix', ['scripts/plan3-local-profile.py', '--output', str(primary.parent)], 'local')
    if not validation.exists() or read(validation).get('status') != 'complete':
        step('independent_512_image_validation', ['scripts/plan3-validation-timing.py', '--primary', str(primary),
             '--output', str(validation.parent)], 'validation')
    step('select_execution_candidate', ['scripts/qualify-plan3-speed.py', '--local', str(primary),
         '--validation', str(validation), '--select-only', *pin_args, '--output', str(output)])
    trials = read(primary)['cases']
    candidate = read(output/'candidate-config.json')
    # Use the same deterministic selection and workers as the independent inspector.
    import importlib.util
    spec = importlib.util.spec_from_file_location('qualify_plan3', ROOT/'scripts/qualify-plan3-speed.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _, accepted = module.inspect(primary)
    chosen = pin(accepted, pin_config)[module.select(read(primary), pin(accepted, pin_config), read(validation))]
    common = ['scripts/plan3-speed-resume-workload.py', '--config', str((output/'candidate-config.json').relative_to(ROOT).as_posix()),
              '--workers', str(chosen['workers'])]
    step('resume_proof_uninterrupted_and_prefix', [*common, '--phase', 'prefix',
         '--output', 'artifacts/plan3_speed_20261020/resume_prefix'])
    step('resume_proof_second_process', [*common, '--phase', 'suffix',
         '--prior', 'artifacts/plan3_speed_20261020/resume_prefix',
         '--output', 'artifacts/plan3_speed_20261020/resume_suffix'])
    step('final_execution_qualification', ['scripts/qualify-plan3-speed.py', '--local', str(primary),
         '--validation', str(validation), '--resume-proof', str(BASE/'resume_suffix/proof.json'),
         *pin_args, '--output', str(output)])
    qualification = read(output/'qualification.json')
    if qualification.get('qualified') is not True:
        raise ValueError('No formal training without all real execution checks')
    if RECIPE.exists() and read(RECIPE) != candidate:
        raise ValueError('Refusing to overwrite a different local experimental recipe: '+str(RECIPE))
    write_json(RECIPE, candidate)
    job = {**read(ROOT/'kaggle/plan3/pilot400-rfm448-f0.json'),
           'preflight_capacity': False, 'workers': chosen['workers'], 'audit_workers': 4,
           'max_hours': 10, 'config': RECIPE.relative_to(ROOT).as_posix(),
           'fold': fold,
           'output': 'artifacts/plan3/pilot400-rfm448-local-speed-v2-f%d' % fold}
    job_path = BASE/('local-fold%d-job.json' % fold)
    write_json(job_path, job)
    if args.qualify_only:
        step('qualified_ready_fold%d_not_started' % fold)
        print(json.dumps(job, indent=2), flush=True)
        return
    step('qualified_starting_full_local_fold%d' % fold, ['scripts/plan3_local.py', '--job', str(job_path),
         '--qualification', str(output/'qualification.json'), '--continue-until-complete'])
    step('full_local_fold%d_complete' % fold)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        step('stopped_for_inspection_'+type(error).__name__)
        raise
