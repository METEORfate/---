"""Local sessions with a measured configuration and strict, unchanged resume checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def seal(output):
    from glaucoma.common import write_json
    selected = [p for p in output.rglob('*') if p.is_file() and p.name not in
                ('local_manifest.json', '.lock') and not p.name.endswith('.tmp')]
    write_json(output/'local_manifest.json', {'files': {p.relative_to(output).as_posix():
        {'sha256': sha(p), 'bytes': p.stat().st_size} for p in selected},
        'status': read(output/'status.json') if (output/'status.json').exists() else None})


def session(job_path, qualification_path, resume):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from glaucoma.common import write_json, versions
    from plan3.jobs import validate, validate_config
    from plan3.provenance import code_hash, safe_path, pretrained
    from plan3 import train
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    job = read(job_path)
    # Cloud jobs retain their original dependency-only recovery rules. Local
    # recovery is a CLI action after validating the canonical fresh-run job.
    if job.get('resume') or job.get('resume_from') or job.get('dependencies'):
        raise ValueError('Local entry expects a fresh canonical job without cloud recovery/dependencies')
    validate(job)
    if job['task'] != 'train':
        raise ValueError('Measured local entry currently supports classifier folds only')
    config = validate_config(read(safe_path(job['config'])))
    qualified = read(qualification_path)
    if (qualified.get('qualified') is not True or qualified['code_sha256'] != code_hash()
            or config != qualified['selected_config']
            or job['workers'] != qualified['chosen']['local']['workers']):
        raise ValueError('This code/configuration/workers did not pass real execution qualification')
    weight, _ = pretrained(config)
    if sha(weight) != qualified['pretrained_sha256']:
        raise ValueError('Qualified pretrained weight changed')
    output = safe_path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    identity = {'canonical_job': job, 'config': config, 'code_sha256': code_hash(),
                'runner_sha256': sha(__file__), 'qualification_sha256': sha(qualification_path),
                'torch_threads': 4, 'runtime': versions()}
    receipt = output/'local_execution.json'
    if resume:
        if not receipt.exists() or read(receipt) != identity or not (output/'resume.pt').exists():
            raise ValueError('Local execution or recovery artifact changed; no forced loading is allowed')
    else:
        if receipt.exists() or (output/'resume.pt').exists():
            raise ValueError('Output already exists; use --resume after reviewing its identity')
        write_json(receipt, identity)
        snapshot = output/'frozen_source'
        for package in ('plan3', 'glaucoma'):
            for source in (ROOT/package).glob('*.py'):
                target = snapshot/package/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        shutil.copy2(__file__, snapshot/Path(__file__).name)
        for filename in ('plan3-local-profile.py', 'plan3-validation-timing.py',
                         'plan3-speed-resume-workload.py', 'qualify-plan3-speed.py'):
            shutil.copy2(ROOT/'scripts'/filename, snapshot/filename)
        shutil.copy2(job_path, output/'canonical_job.json')
        shutil.copy2(qualification_path, output/'qualification.json')
    train.SESSION_DEADLINE = time.monotonic()+job['max_hours']*3600
    try:
        train.run_training(ROOT/'glaucoma_first', config, {**job, 'resume': resume}, 'cuda')
    finally:
        seal(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue-until-complete', action='store_true')
    args = parser.parse_args()
    if not args.continue_until_complete:
        session(args.job, args.qualification, args.resume)
        return
    job = read(args.job)
    output = ROOT/job['output']
    resume = args.resume
    previous = -1
    for index in range(30):
        command = [sys.executable, str(Path(__file__).resolve()), '--job', str(args.job.resolve()),
                   '--qualification', str(args.qualification.resolve())]
        if resume:
            command.append('--resume')
        print('Local full-fold session', index+1, 'resume', resume, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        status = read(output/'status.json')
        if status['complete']:
            print('Local fold completed with retained model and complete validation.', flush=True)
            return
        current = status['last_complete_epoch']
        if current <= previous or current == 0:
            raise RuntimeError('Session made no complete-epoch progress; stopping for inspection')
        previous, resume = current, True
    raise RuntimeError('Local session limit reached; preserved results require inspection')


if __name__ == '__main__':
    main()
