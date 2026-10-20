"""Finish and verify CPU-only preparation; this entry never launches training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from glaucoma.common import read_json, sha256, write_json
from glaucoma.data import AgeTransform
from plan3.cache import prepare_global, source_hashes, global_record
from plan3.jobs import validate, validate_config
from plan3.provenance import Context, safe_path, code_hash, pretrained


def main():
    parser = argparse.ArgumentParser(description='仅校验方案3准备阶段，完成后退出')
    parser.add_argument('--job', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = safe_path(args.output)
    job = validate(read_json(safe_path(args.job)))
    cfg = validate_config(read_json(safe_path(job['config'])))
    qualified = read_json(safe_path(args.qualification))
    if not qualified.get('qualified') or qualified['code_sha256'] != code_hash() or cfg != qualified['selected_config']:
        raise ValueError('Prepared configuration/source differs from the measured qualified implementation')
    import torch
    if torch.cuda.is_available():
        raise ValueError('Preparation must run without a visible CUDA device')
    torch.set_num_threads(4)
    started = time.monotonic()
    state = {'status': 'verifying_preparation_only', 'training_authorized': False,
             'automatic_training_enabled': False, 'preparation_entry_sha256': sha256(__file__)}
    write_json(output/'preparation.json', state)
    context = Context(ROOT/'glaucoma_first')
    annotation, _ = context.annotations(safe_path(job['annotations']), job['annotation_policy'])
    weight, _ = pretrained(cfg)
    if sha256(weight) != qualified['pretrained_sha256']:
        raise ValueError('Qualified pretrained weight changed')
    fit, hold = context.split(job['fold'])
    hashes = source_hashes(ROOT/'artifacts/audit')
    files = {}
    for label, selected in [('train', fit), ('holdout', hold)]:
        rows = [context.rows[i] for i in selected]
        print('Preparation-only verification:', label, len(rows), flush=True)
        path = prepare_global(rows, cfg['size'], ROOT/'artifacts/plan3_cache',
                              context.audit['data_fingerprint'], hashes, job.get('audit_workers', 4))
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        # Check the original-coordinate records, including cache records completed
        # before the automatic execution chain was stopped.
        for start in range(0, len(rows), 512):
            crop, size = values['crop'][start:start+512], values['source_size'][start:start+512]
            if not (np.all(size > 0) and np.all(crop[:, :2] >= 0)
                    and np.all(crop[:, 2:] > crop[:, :2]) and np.all(crop[:, 2:] <= size)):
                raise ValueError('Invalid original-coordinate cache transform records')
        sample = np.random.default_rng(42).choice(len(rows), min(16, len(rows)), replace=False)
        for index in sample:
            if not np.array_equal(values[int(index)], global_record(rows[int(index)]['path'], cfg['size'])):
                raise ValueError('Cache pixel/coordinate record differs from original image')
        del values
        certificate = read_json(path.with_suffix('.sha256.json'))
        files[label] = {'path': path.relative_to(ROOT).as_posix(), 'count': len(rows),
                        'bytes': path.stat().st_size, 'sha256': certificate['sha256'],
                        'identity_sha256': certificate['identity_sha256'], 'original_pixel_checks': len(sample)}
    destination = safe_path(job['output'])
    if (destination/'resume.pt').exists() or (destination/'model.pt').exists():
        raise ValueError('Unexpected formal training checkpoint; inspect before declaring preparation-only completion')
    state.update({'status': 'prepared_waiting_for_user_training_command', 'cache': files,
        'data_fingerprint': context.audit['data_fingerprint'], 'fold_fingerprint': context.audit['fold_fingerprint'],
        'code_sha256': code_hash(), 'config_sha256': sha256(safe_path(job['config'])),
        'qualification_sha256': sha256(safe_path(args.qualification)), 'pretrained_sha256': sha256(weight),
        'annotation_version': annotation['version'], 'annotation_sha256': annotation['boxes_sha256'],
        'age_transform': AgeTransform.fit([context.rows[i]['age'] for i in fit]).to_dict(),
        'fold': job['fold'], 'seed': job['seed'], 'formal_training_updates': 0,
        'verified_seconds': time.monotonic()-started, 'completed_local': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    write_json(output/'preparation.json', state)
    write_json(ROOT/'artifacts/plan3_speed_20261020/release-status.json',
               {'phase': state['status'], 'automatic_training_enabled': False, 'time': state['completed_local']})
    (output/'准备完成说明.md').write_text(
        '# 方案3本机训练准备完成，等待指令\n\n'
        '自动训练链已停止。本次仅完成准备核验，不执行训练或official-valid评测。\n\n'
        f"训练缓存{len(fit):,}张、内部验证缓存{len(hold):,}张，全部来源图像和缓存哈希、输入版本、分折、名称顺序、像素及原图坐标抽查通过。\n\n"
        '测速、512张完整内部验证及跨进程恢复证据均已保留。未开始本机正式训练参数更新。\n\n'
        '配置：448全图，seed42，微批量4×累积8，读取进程2，混合精度，梯度检查点，最多24轮，早停耐心4轮。\n\n'
        '后续必须收到用户新的训练命令，才运行以下命令；本准备入口不会运行它：\n\n'
        '```powershell\n'
        "& 'D:\\CodexRuntimes\\glaucoma-2026\\.venv\\Scripts\\python.exe' -u scripts/plan3_local.py "
        f"--job {args.job} --qualification {args.qualification} --continue-until-complete\n"
        '```\n\n'
        '既有本机输出只有执行凭据和冻结源码，已归档到pre_training_receipt_20261020；没有正式模型或训练恢复断点。'
        '后续从预训练权重开始，复用已完成缓存并保留原哈希检查，不使用--resume强行加载。\n', encoding='utf-8')
    print('PREPARATION COMPLETE; exited without launching training.', flush=True)


if __name__ == '__main__':
    main()
