"""Independent release checks for measured execution profiles, not score selection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(path):
    report = read(path)
    if report['status'] != 'complete':
        raise ValueError('Real GPU timing matrix is incomplete: '+str(path))
    if not report.get('deterministic_execution'):
        raise ValueError('The local implementation requires explicitly verified deterministic GPU execution')
    if report.get('execution_entry_sha256') != sha(Path(__file__).with_name('plan3-local-profile.py')):
        raise ValueError('Measured local execution entry changed')
    if not report.get('cache_pixel_age_label_order_equal') or report.get('cache_training_max_probability_difference', 1) > .001:
        raise ValueError('Cache correctness gate did not pass')
    reference = report['cases']['resized_cached_w2_b4_gc']['trials']
    control = report['cases']['resized_uncached_w2_b4_gc']['trials']
    if len(reference) != 3 or len(control) != 3:
        raise ValueError('Require three real repetitions')
    sample = read(Path(path).parent/'sample_manifest.json')
    if sha(Path(path).parent/'sample_manifest.json') != report['sample_manifest_sha256']:
        raise ValueError('Benchmark sample manifest changed')
    accepted = {}
    for name, case in report['cases'].items():
        trials = case['trials']
        if not name.startswith('resized_cached') or len(trials) != 3 or case.get('rejected'):
            continue
        passed = True
        for candidate, baseline in zip(trials, reference):
            passed &= (candidate['backbone_changed'] and candidate['optimizer_updates'] == candidate['ema_updates'] == 4
                       and candidate['raw_metrics']['n'] == candidate['ema_metrics']['n'] == 64
                       and candidate['batch_size']*candidate['accumulation'] == 32
                       and candidate['paired_max_probability_difference'] <= 1e-7
                       and candidate['peak_gpu_bytes']+1024**3 < report['gpu_memory_bytes'])
            for member in ('raw_probabilities', 'ema_probabilities'):
                passed &= max(abs(a-b) for a, b in zip(candidate[member], baseline[member])) <= .001
            for member in ('raw_metrics', 'ema_metrics'):
                for metric in ('auroc', 'auprc'):
                    passed &= abs(candidate[member][metric]-baseline[member][metric]) <= .001
        if passed:
            key = (trials[0]['batch_size'], trials[0]['checkpointing'], trials[0]['workers'])
            summary = {'case': name, 'workers': trials[0]['workers'], 'batch_size': key[0],
                       'accumulation': 32//key[0], 'checkpointing': key[1],
                       'steady_images_per_second': statistics.median(t['steady_images_per_second'] for t in trials),
                       'peak_gpu_bytes': max(t['peak_gpu_bytes'] for t in trials)}
            if key not in accepted or summary['steady_images_per_second'] > accepted[key]['steady_images_per_second']:
                accepted[key] = summary
    if not accepted:
        raise ValueError('No measured candidate passed unfrozen/EMA/numerical/memory checks')
    return report, accepted


def round_seconds(trials, validation):
    speed = statistics.median(t['steady_images_per_second'] for t in trials)
    overhead = statistics.median(max(0., t['training_seconds']-128/t['steady_images_per_second']) for t in trials)
    return 34264/speed+overhead+validation['startup_seconds']+17132/validation['two_member_images_per_second']


def select(report, accepted, validation):
    if validation.get('status') != 'complete' or validation.get('sample_count') != 512:
        raise ValueError('Execution selection requires completed independent validation timing')
    return min(accepted, key=lambda key: round_seconds(report['cases'][accepted[key]['case']]['trials'],
        validation['cases'][f"resized_cached_w{accepted[key]['workers']}_paired"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', required=True, type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--resume-proof', type=Path)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--pin-config', type=Path, default=None,
                        help='Restrict selection to the micro-batch/checkpointing of an already qualified config '
                             'so a second machine runs the identical recipe and folds stay comparable')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    local, local_ok = inspect(args.local)
    if args.pin_config is not None:
        pinned = read(args.pin_config)
        want = (int(pinned['batch_size']), bool(pinned.get('gradient_checkpointing', False)))
        local_ok = {k: v for k, v in local_ok.items() if (k[0], k[1]) == want}
        if not local_ok:
            raise ValueError('Pinned micro-batch %d / gradient_checkpointing=%s did not pass this machine\'s '
                             'real measurements; do not mix recipes across folds' % (want[0], want[1]))
    if args.validation is None:
        raise ValueError('Selection requires full validation timing as well as training throughput')
    validation = read(args.validation)
    selected = select(local, local_ok, validation)
    chosen = {'local': local_ok[selected]}
    args.output.mkdir(parents=True, exist_ok=True)
    base = read(Path(__file__).resolve().parents[1]/'configs/plan3/rfm448-resized-aug-v2.json')
    candidate = {**base, 'batch_size': selected[0], 'accumulation': 32//selected[0],
                 'gradient_checkpointing': selected[1], 'validation_batch_size': 4, 'deterministic_execution': True}
    (args.output/'candidate-config.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    if args.select_only:
        print(json.dumps({'qualified': False, 'selected_for_resume_proof': chosen}), flush=True)
        return
    if args.validation is None or args.resume_proof is None:
        raise ValueError('Final qualification requires independent validation and real cross-process recovery')
    if (validation.get('status') != 'complete' or validation['sample_count'] != 512
            or not validation.get('deterministic_execution')
            or validation.get('validation_entry_sha256') != sha(Path(__file__).with_name('plan3-validation-timing.py'))):
        raise ValueError('Independent 512-image validation timing is incomplete')
    proof = read(args.resume_proof)
    expected_proof_config = {**candidate, 'epochs': 2, 'head_warmup_epochs': 0}
    if (proof.get('verified') is not True or proof.get('backend') != 'local' or not proof.get('cross_process')
            or proof['signature']['config'] != expected_proof_config or proof.get('epochs') != 2
            or proof['signature']['code_sha256'] != local['code_sha256']
            or proof['signature']['loader_workers'] != chosen['local']['workers']
            or proof['signature']['torch_threads'] != 4
            or validation['code_sha256'] != local['code_sha256']):
        raise ValueError('Recovery or validation does not qualify this exact measured implementation/configuration')
    v = validation['cases'][f"resized_cached_w{chosen['local']['workers']}_paired"]
    selected_trials = local['cases'][chosen['local']['case']]['trials']
    train_seconds_local = 34264/chosen['local']['steady_images_per_second']+statistics.median(
        max(0., t['training_seconds']-128/t['steady_images_per_second']) for t in selected_trials)
    valid_seconds_local = v['startup_seconds']+17132/v['two_member_images_per_second']
    original_workers = min((2, 4, 8), key=lambda workers: round_seconds(
        local['cases'][f'original_w{workers}_b4_gc']['trials'], validation['cases'][f'original_w{workers}_separate']))
    original_local = local['cases'][f'original_w{original_workers}_b4_gc']['trials']
    original_val = validation['cases'][f'original_w{original_workers}_separate']
    baseline_seconds_local = round_seconds(original_local, original_val)
    result = {'qualified': True, 'code_sha256': local['code_sha256'],
              'sample_manifest_sha256': local['sample_manifest_sha256'], 'pretrained_sha256': local['pretrained_sha256'],
              'execution_entry_sha256': local['execution_entry_sha256'],
              'preprocess_version': 'original-fov-pad-then-augment-v2', 'chosen': chosen,
              'report_sha256': {'local': sha(args.local), 'validation': sha(args.validation), 'resume_proof': sha(args.resume_proof)},
              'selected_config': candidate,
              'local_round_estimate_seconds': train_seconds_local+valid_seconds_local,
              'local_original_round_estimate_seconds': baseline_seconds_local,
              'local_original_comparison_workers': original_workers,
              'estimated_local_cache_build_seconds': local['cache_build_seconds']*51396/192,
              'estimate_scope': 'Steady unfrozen training + full raw/EMA validation, extrapolated from train-internal samples; calibration by the first optimized full epoch remains required.',
              'input_change': 'Independent resized-augmentation configuration. Cached/uncached v2 match; no equality claim to original augmented inputs.'}
    if result['local_round_estimate_seconds'] >= .9*baseline_seconds_local:
        raise ValueError('Measured local saving is below the existing 10% execution qualification threshold')
    result['maximum_24_epoch_three_fold_seconds'] = 72*result['local_round_estimate_seconds']+result['estimated_local_cache_build_seconds']*3
    (args.output/'qualification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 方案3优化实测', '', '两种增强顺序为独立配置。所有测速均在固定外折训练部分，未调用official-valid。', '',
             '|设备/配置|完整短段训练张/秒|稳定段训练张/秒|原模型+EMA验证秒（64张，含启动）|', '|---|---:|---:|---:|']
    for device, report in [('RTX4060', local)]:
        for name, case in report['cases'].items():
            if len(case['trials']) == 3:
                lines.append(f"|{device}/{name}|{case['median_images_per_second']:.2f}|{statistics.median(t['steady_images_per_second'] for t in case['trials']):.2f}|{case['median_validation_seconds']:.2f}|")
            else:
                lines.append(f"|{device}/{name}|—|—|{case.get('rejected', '未完成')}|")
    lines += ['', '主矩阵的短段计时保留进程冷启动；其验证数字还包含未释放训练读取进程的占用，因此完整轮预测改用独立512图验证测量，不把64图冷启动时间直接按比例放大。', '',
              f"本机配置：微批量{selected[0]}、累积{32//selected[0]}、梯度检查点{selected[1]}，读取进程{chosen['local']['workers']}。Kaggle优化测速因磁盘保护检查停止，未获得T4速度，不参与选择。", '',
              f"本机原入口单轮估计：{baseline_seconds_local/60:.1f}分钟；候选单轮估计：{result['local_round_estimate_seconds']/60:.1f}分钟。均为主干解冻训练、完整17132图原模型和EMA验证的外推。", '',
              '缩放前后的增强不是同一训练输入；小样本数值检查只证明v2缓存执行正确，不证明该新配置的最终AUROC收益。', '',
              f"三折各跑满24轮的稳态外推加三次缓存准备：{result['maximum_24_epoch_three_fold_seconds']/3600:.1f}小时。早停取决于真实完整折结果；此估计尚未纳入保存、哈希复核和会话边界重做的不完整轮开销，待第一完整轮校准。", '',
              '当前单折训练34264张、验证17132张；正式训练不会缩小样本数量。旧作业与旧断点继续使用冻结代码和原镜像。真实恢复对照完成两轮主干解冻训练及原模型/EMA完整小样本验证，参数、优化器、调度器、缩放器和随机状态逐项核验。']
    stages = local['cpu_stages']['mean_seconds']
    lines += ['', '## 瓶颈与修改', '',
              '旧P1已使用缓存并在缩放后增强；方案3旧入口每次解码原图，在原始大图上旋转/颜色增强，再裁视野和缩放，原模型与EMA验证又分别读图。', '',
              '|原管线CPU阶段（24张真实图片均值）|毫秒/张|', '|---|---:|']
    for key, label in [('read_seconds', '读文件'), ('decode_seconds', 'EXIF转正及解码'),
                       ('original_augmentation_seconds', '原图尺寸增强'), ('fov_resize_seconds', '裁视野及缩放'),
                       ('p1_full_pipeline_seconds', 'P1同为448输入的完整CPU处理')]:
        lines.append(f'|{label}|{stages[key]*1000:.2f}|')
    lines += ['', f'|真实128张训练累计阶段|原入口w{original_workers}|选中缓存配置|', '|---|---:|---:|']
    selected_trials = local['cases'][chosen['local']['case']]['trials']
    for key in original_local[0]['timings']:
        before = statistics.median(t['timings'][key] for t in original_local)
        after = statistics.median(t['timings'][key] for t in selected_trials)
        lines.append(f'|{key}（秒）|{before:.3f}|{after:.3f}|')
    lines += ['', '## 独立完整验证测速', '',
              '512张额外图像来自外折训练部分，与主测速样本按相似性分组隔离。以下每项均完整运行原模型与EMA；各样本概率和顺序与原验证方式逐项比较，容差1e-7。', '',
              '|方式|总秒数（含启动）|稳定合计双模型张/秒|', '|---|---:|---:|']
    for name, item in validation['cases'].items():
        lines.append(f"|{name}|{item['seconds']:.2f}|{item['two_member_images_per_second']:.2f}|")
    plan = local['disk_plan']
    lines += ['', '## 正确性与磁盘', '',
              '130项自动检查已通过。另有真实RFM三次重复验证：128张主干解冻训练、64张完整内部验证、每次4次优化器及EMA更新；固定有效批量32。缓存/未缓存v2像素、年龄、标签和顺序一致；缓存训练概率及AUROC/AUPRC差均不超过0.001。', '',
              '首次非确定性GPU测速因64图AUROC数值漂移超过门槛被拒绝，完整证据保留在local-nondeterministic-failed/。未放宽门槛。本报告仅采用显式确定性GPU运算重新实测且通过的结果；正式本机训练和恢复使用同一设置。', '',
              f"本机GPU为{local['gpu']}，可见显存{local['gpu_memory_bytes']/1024**3:.2f}GiB；选中配置实测峰值{chosen['local']['peak_gpu_bytes']/1024**3:.2f}GiB，已预留至少1GiB。使用现有混合精度，不引入新训练框架。", '',
              f"全train的448缓存约{plan['global448_bytes']/1024**3:.2f}GiB（含裁黑边边界、转正原图尺寸），缓存最小空闲预留30GiB。数据/逐图来源哈希、名称顺序、处理版本和缓存文件哈希均绑定。模型、完整恢复断点及临时写入另留数GiB；仅在E盘构建。", '',
              f"3d512+5d672两个局部的全量原始像素缓存约{plan['roi3_512_and_roi5_672_pixel_bytes']/1024**3:.2f}GiB，当前不构建。局部缓存必须来自原图坐标窗口，保存定位器/人工复核版本、裁视野与缩放映射和无效局部标记；待人工漏盘检查通过再安排局部训练。", '',
              '第一折使用两个分区缓存合计28.82GiB。后续折顺序进行，先确认正式结果和恢复断点可取回，再管理本轮新生成的缓存空间；本次不删除历史P1/P2缓存。缓存准备外推约'+f"{result['estimated_local_cache_build_seconds']/60:.1f}分钟/折。", '',
              'official-valid台账维持原有baseline，测速、恢复对照和本机单折训练均不使用official-valid。旧Kaggle作业不取消，按原时限收尾并由独立回收任务下载、逐文件校验；未生成verified_result.json前不声称其成果已经回收。', '',
              '## 结果位置', '',
              '- 原始测速：artifacts/plan3_speed_20261020/local/benchmark.json。',
              '- 独立验证：artifacts/plan3_speed_20261020/validation/validation.json。',
              '- 跨进程恢复：artifacts/plan3_speed_20261020/resume_suffix/proof.json。',
              '- 正式本机第0折：artifacts/plan3/pilot400-rfm448-local-speed-v2-f0/。',
              '- 旧Kaggle冻结代码与回收：C:/Users/admins/Documents/Codex/kaggle-plan3-optimization-20261020/preserved/rfm-f0-v1/。']
    (args.output/'测速报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
